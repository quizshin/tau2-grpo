"""Continue an authorized SFT run through CPU export and bounded independent eval.

Consumes a frozen run plan and reuses existing CLIs and owned process lifecycle.
It never retries failed trials, starts another training stage, or kills other jobs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import shutil
import socket
import time
import urllib.request
from pathlib import Path

from tau3_grpo.training.services import launch_process, stop_process


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_training(root, audit, *, expected_steps=13, expected_epochs=1):
    if (root / 'train.exit').read_text().strip() != '0':
        raise ValueError('Training did not exit successfully')
    summary = read(root / 'adapter/train_summary.json')
    state = read(root / 'adapter/trainer_state.json')
    if (summary['actual_optimizer_steps'] != expected_steps
            or state['global_step'] != expected_steps
            or not math.isclose(state['epoch'], expected_epochs)):
        raise ValueError('Training did not complete the approved epochs/updates')
    for field in ('train_loss',):
        if not math.isfinite(summary[field]):
            raise ValueError('Nonfinite training result')
    if not math.isfinite(summary['validation_metrics']['eval_loss']):
        raise ValueError('Nonfinite validation loss')
    for row in state.get('log_history', []):
        for key in ('loss', 'grad_norm', 'eval_loss'):
            if key in row and not math.isfinite(row[key]):
                raise ValueError('Nonfinite recorded training metric')
    for name in ('train', 'validation'):
        if summary[name] != audit['token_stats'][name]:
            raise ValueError('Actual training token statistics changed')
        if digest(root / 'adapter' / f'effective_{name}.jsonl') != audit['files'][f'{name}_effective.jsonl']:
            raise ValueError('Actual training masks/messages differ from frozen effective data')
    return summary


def preserve_base_tokenizer(base, merged, provenance):
    """Restore original bytes only after loaded tokenizer semantics match."""
    from transformers import AutoTokenizer

    left = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    right = AutoTokenizer.from_pretrained(str(merged), local_files_only=True)
    if (left.get_vocab() != right.get_vocab()
            or read_tokenizer_backend(left) != read_tokenizer_backend(right)
            or left.chat_template != right.chat_template):
        raise ValueError('Exported tokenizer semantics changed')
    for attr in ('eos_token_id', 'bos_token_id', 'pad_token_id',
                 'all_special_tokens', 'all_special_ids'):
        if getattr(left, attr) != getattr(right, attr):
            raise ValueError(f'Exported tokenizer property changed: {attr}')
    provenance.mkdir(parents=True, exist_ok=False)
    receipt = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        receipt[name] = {'original_export': digest(merged / name), 'base': digest(base / name)}
        shutil.copy2(merged / name, provenance / name)
        shutil.copy2(base / name, merged / name)
    (provenance / 'equivalence.json').write_text(json.dumps(receipt, indent=2) + '\n')


def read_tokenizer_backend(tokenizer):
    return json.loads(tokenizer.backend_tokenizer.to_str())


class Continuation:
    def __init__(self, path):
        self.plan = read(path)
        self.root = Path(self.plan['run_dir']).resolve()
        self.cwd = Path(self.plan['code_root'])
        self.children = []
        self.state = {'status': 'waiting_for_training', 'plan_sha256': digest(path), 'pid': os.getpid()}

    def status(self, value, **fields):
        self.state.update(status=value, updated_at=time.time(), **fields)
        path = self.root / 'continuation.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=2) + '\n')
        tmp.replace(path)

    def launch(self, name):
        task = self.plan['commands'][name]
        env = dict(os.environ)
        env.update(self.plan['environment'])
        env.update(task.get('environment', {}))
        process = launch_process(task['argv'], cwd=self.cwd, env=env,
                                 log_path=self.root / f'{name}.log', pid_path=self.root / f'{name}.pid')
        self.children.append(process)
        return process

    def wait(self, process, deadline):
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('Approved phase deadline reached')
            time.sleep(2)
        if process.returncode:
            raise RuntimeError(f'Owned child {process.pid} exited {process.returncode}')

    def run(self):
        with (self.root / '.continuation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.status('waiting_for_training')
            try:
                deadline = time.monotonic() + self.plan['wait_train_seconds']
                while not (self.root / 'train.exit').exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Training exit receipt not produced by deadline')
                    time.sleep(5)
                summary = validate_training(
                    self.root, read(self.plan['data_audit']),
                    expected_steps=self.plan.get('expected_optimizer_steps', 13),
                    expected_epochs=self.plan.get('expected_epochs', 1))
                self.status('exporting', actual_optimizer_steps=summary['actual_optimizer_steps'])
                self.wait(self.launch('merge'), time.monotonic() + 1800)
                base, merged = Path(self.plan['base_model']), self.root / 'merged'
                preserve_base_tokenizer(base, merged, self.root / 'provenance/export_tokenizer')
                for name in ('tokenizer.json', 'chat_template.jinja'):
                    if digest(base / name) != digest(merged / name):
                        raise ValueError(f'Exported tokenizer identity changed: {name}')
                for path, expected in self.plan['baseline_source_hashes'].items():
                    if digest(self.cwd / path) != expected:
                        raise ValueError(f'Baseline protocol source changed: {path}')
                for port in (8000, 8100):
                    with socket.socket() as sock:
                        if sock.connect_ex(('127.0.0.1', port)) == 0:
                            raise ValueError(f'Port {port} is owned by an existing service')
                deadline = time.monotonic() + self.plan['eval_seconds']
                self.status('starting_evaluation_services', evaluation_deadline_unix=time.time() + self.plan['eval_seconds'])
                services = {name: self.launch(name) for name in ('policy', 'user')}
                ready = set()
                while len(ready) < 2:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Evaluation service startup exceeded budget')
                    for name, process in services.items():
                        if process.poll() is not None:
                            raise RuntimeError(f'{name} service exited before readiness')
                        if name in ready:
                            continue
                        endpoint = self.plan['endpoints'][name]
                        try:
                            with urllib.request.urlopen(endpoint['url'] + '/models', timeout=2) as response:
                                packet = json.load(response)
                            if endpoint['model'] not in {m['id'] for m in packet['data']}:
                                raise ValueError(f'{name} service model name differs')
                            ready.add(name)
                        except (OSError, TimeoutError):
                            pass
                    time.sleep(2)
                self.status('evaluating_selection60x4')
                self.wait(self.launch('evaluation'), deadline)
                evaluated = read(self.root / 'eval/summary.json')
                if not evaluated['metrics_valid'] or evaluated['completed_trajectories'] != 240:
                    raise ValueError('Incomplete evaluation; no valid aggregate comparison')
                # Free both GPUs before the CPU statistical comparison.
                for process in services.values():
                    stop_process(process)
                self.status('comparing_with_base')
                self.wait(self.launch('compare'), time.monotonic() + 600)
                self.status('complete', metrics=evaluated['metrics'])
            except Exception as exc:
                self.status('failed', error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                for process in reversed(self.children):
                    stop_process(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    def interrupted(signum, frame):
        raise InterruptedError(f'Continuation received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Continuation(parser.parse_args().plan).run()


if __name__ == '__main__':
    main()
