"""Prepare CPU artifacts, then evaluate frozen policies after the RL queue exits.

This controller never imports or changes the running trainer, never starts RL,
and owns only its own service process groups. A failed/paused RL queue cannot
trigger evaluation. Official final tasks are deliberately outside this plan.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau3_grpo.evaluation.harness import LEGACY, PROTOCOLS, TOKENS_V4, protocol_metadata
from tau3_grpo.training.rl.checkpoints import validate_checkpoint
from tau3_grpo.training.services import launch_process, stop_process

CODE = Path(__file__).resolve().parents[2]
ARMS = ('e0', 'e1', 'e2', 'e3')
# Put the SFT and GRPO baselines and combined method on the first three slots.
SLOTS = (('sft',), ('e0',), ('e3',), ('e1',))
GIB = 2**30


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def validate_finished_arm(root, arm):
    root = Path(root)
    if (root / f'{arm}.exit').read_text().strip() != '0':
        raise ValueError(f'{arm} did not exit successfully')
    run = root / f'{arm}_seed42'
    done = read(run / 'completion.json')
    if done['arm'] != arm or done['target_step'] != 20 or done['swanlab']['cloud_steps_verified'] != 20:
        raise ValueError(f'{arm} completion has not been verified')
    for step in (10, 20):
        rows = [json.loads(line) for line in (run / f'validation/{step}.jsonl').open() if line.strip()]
        if len(rows) != 240 or set(Counter(row['task_id'] for row in rows).values()) != {4}:
            raise ValueError(f'{arm}/{step} evaluation is incomplete')
    return validate_checkpoint(run)


def queue_ready(root, controller_alive):
    root = Path(root)
    state = read(root / 'controller-state.json')
    if (root / 'STOP_AFTER_BOUNDARY').exists() or state['status'] in ('failed', 'paused_at_boundary', 'waiting_for_storage'):
        raise RuntimeError(f'RL queue stopped: {state["status"]}')
    exit_file = root / 'controller.exit'
    if exit_file.exists() and exit_file.read_text().strip() != '0':
        raise RuntimeError('RL controller exited with an error')
    if state['status'] != 'complete' or not exit_file.exists() or controller_alive:
        return False
    if state.get('target_step') != 20 or {x['arm'] for x in state.get('completed', [])} != set(ARMS):
        raise ValueError('RL completion does not contain all four 20-step runs')
    for arm in ARMS:
        validate_finished_arm(root, arm)
    return True


def validate_hf(path):
    path = Path(path)
    config = read(path / 'config.json')
    if config.get('model_type') != 'qwen3_5':
        raise ValueError('Expected dense Qwen3.5 checkpoint')
    weights = sorted(path.glob('*.safetensors'))
    if not weights or not all(p.stat().st_size > 0 for p in weights):
        raise ValueError('Merged HF checkpoint lacks weights')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        for name in set(read(index)['weight_map'].values()):
            if not (path / name).is_file():
                raise ValueError(f'Missing HF shard: {name}')
    if not (path / 'tokenizer_config.json').is_file():
        raise ValueError('Merged HF checkpoint lacks tokenizer')
    from verl.utils.qwen35_checkpoint import qwen35_tensor_manifest
    qwen35_tensor_manifest(path)


def evaluation_source_hashes():
    return {str(p.relative_to(CODE)): sha(p)
            for p in sorted((CODE / 'tau3_grpo/evaluation').rglob('*.py'))}


def validate_sft_reuse(previous, plan):
    """Reuse scored SFT only; the RL attempts must have failed before any trials."""
    previous = Path(previous)
    old_plan = read(previous / 'plan.json')
    ignored = {'slots', 'first_free_slot_queue', 'reuse_sft_from'}
    if {k: v for k, v in old_plan.items() if k not in ignored} != {
        k: v for k, v in plan.items() if k not in ignored
    }:
        raise ValueError('Cannot reuse SFT with a changed evaluation protocol')
    if read(previous / 'software.json')['evaluation_source_sha256'] != evaluation_source_hashes():
        raise ValueError('Cannot reuse SFT after changing evaluation source code')
    if (previous / 'controller.exit').read_text().strip() != '1':
        raise ValueError('Recovery requires an exited, failed prior attempt')
    state = read(previous / 'state.json')
    if state['status'] != 'incomplete' or state['models']['sft']['status'] != 'complete':
        raise ValueError('Previous SFT evaluation is not complete')
    for arm in ARMS:
        if state['models'][arm]['status'] != 'failed' or any(
            (previous / stage / arm).exists() for stage in ('smoke', 'selection')
        ):
            raise ValueError('Recovery cannot replay an RL model that already attempted trials')
    return check_summary(previous / 'selection/sft', plan['task_ids'])


def persist_complete(root, output, arm):
    """Copy immutable final recovery files; publish only after SHA256 equality."""
    root, output = Path(root), Path(output)
    cp = validate_finished_arm(root, arm)
    dest = root / 'persistent-checkpoints' / f'{arm}_seed42'
    receipt_file = output / f'{arm}-backup.json'
    if dest.exists():
        receipt = read(receipt_file)
        validate_checkpoint(dest)
        if file_hashes(dest) != receipt['files_sha256']:
            raise ValueError(f'Existing {arm} backup is corrupt')
        return dest
    if shutil.disk_usage(root).free < sum(p.stat().st_size for p in cp.rglob('*') if p.is_file()) + 10 * GIB:
        raise RuntimeError('Insufficient persistent space for complete backup')
    staging = dest.with_name(dest.name + '.copying')
    if staging.exists():
        raise RuntimeError(f'Unfinished backup needs inspection: {staging}')
    staging.mkdir(parents=True)
    shutil.copytree(cp, staging / cp.name)
    # Preserve the original schedule, dataloader pointer, and SwanLab identity.
    for item in cp.parent.iterdir():
        if item.is_file():
            shutil.copy2(item, staging / item.name)
    source_hashes = {str(Path(cp.name) / k): v for k, v in file_hashes(cp).items()}
    source_hashes.update({p.name: sha(p) for p in cp.parent.iterdir() if p.is_file()})
    if file_hashes(staging) != source_hashes:
        raise ValueError('Backup SHA256 mismatch; source retained')
    validate_checkpoint(staging)
    atomic_json(receipt_file, {'arm': arm, 'step': 20, 'includes_optimizer': True,
                             'source': str(cp.parent.resolve()), 'destination': str(dest),
                             'files_sha256': source_hashes})
    staging.rename(dest)
    return dest


def check_summary(directory, task_ids, trials=4):
    directory = Path(directory)
    summary = read(directory / 'summary.json')
    plan = read(directory / 'run.json')['planned']
    expected = {(task, trial, 42 + trial) for task in task_ids for trial in range(trials)}
    actual = [(x['task_id'], x['trial'], x['seed']) for x in plan]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Evaluation task/trial/seed plan differs from frozen manifest')
    if not summary['metrics_valid'] or summary['completed_trajectories'] != len(expected):
        raise ValueError('Evaluation has unscored trials; aggregate ranking withheld')
    if set(summary['per_task']) != set(task_ids):
        raise ValueError('Evaluation task set differs from frozen manifest')
    return summary


class Controller:
    def __init__(self, root, output, reuse_sft_from=None, harness_protocol=LEGACY):
        self.harness_protocol = harness_protocol
        protocol_metadata(harness_protocol)
        self.root, self.output = Path(root), Path(output)
        self.env = dict(os.environ)
        self.lock = threading.Lock()
        self.children = []
        self.stop_event = threading.Event()
        self.models = {}
        self.plan = {}
        self.reuse_sft_from = Path(reuse_sft_from).resolve() if reuse_sft_from else None
        self.slots = tuple((arm,) for arm in ARMS) if self.reuse_sft_from else SLOTS
        self.pending = [] if self.reuse_sft_from else ['e2']
        self.reused_results = {}
        self.state = {'status': 'preparing_cpu', 'models': {}, 'backups': {}}

    def status(self, **updates):
        with self.lock:
            self.state.update(updates, updated_at=time.time())
            atomic_json(self.output / 'state.json', self.state)
            print(json.dumps(updates), flush=True)

    def model_status(self, model, **updates):
        with self.lock:
            self.state['models'][model] = dict(self.state['models'].get(model, {}), **updates)
            self.state['updated_at'] = time.time()
            atomic_json(self.output / 'state.json', self.state)

    def launch(self, args, env, logname):
        if self.stop_event.is_set():
            raise InterruptedError('Controller is stopping; no child launched')
        proc = launch_process(args, cwd=CODE, env=env,
                              log_path=self.output / f'{logname}.log')
        with self.lock:
            self.children.append(proc)
        return proc

    def stop(self, proc):
        stop_process(proc, timeout=25)

    def cleanup(self):
        self.stop_event.set()
        for proc in reversed(self.children):
            self.stop(proc)

    def run_command(self, args, env, logname, timeout=1800, dependencies=()):
        proc = self.launch(args, env, logname)
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if self.stop_event.wait(2) or time.monotonic() > deadline or any(p.poll() is not None for p in dependencies):
                self.stop(proc)
                raise RuntimeError(f'{logname} stopped or exceeded its timeout; inspect log')
        if proc.returncode:
            raise RuntimeError(f'{logname} exited {proc.returncode}; inspect log')

    def freeze_plan(self):
        data = Path(self.env['TAU3_DATA_ROOT'])
        manifest = data / 'manifests/areal_airline_selection_seed42.jsonl'
        entries = [json.loads(line) for line in manifest.open() if line.strip()]
        ids = [x['task_id'] for x in entries]
        if len(ids) != 60 or len(set(ids)) != 60 or any(x['split'] != 'selection' for x in entries):
            raise ValueError('Expected the frozen selection60 manifest')
        for x in entries:
            p = (data / 'raw/areal_tau2' / x['db_path']).resolve()
            if (data / 'raw/areal_tau2').resolve() not in p.parents or sha(p) != x['db_hash']:
                raise ValueError('Selection database hash mismatch')
        plan = {'target': 'selection', 'target_step': 20, 'models': ['sft', 'e0', 'e3', 'e1', 'e2'],
                'slots': [list(x) for x in self.slots], 'first_free_slot_queue': list(self.pending),
                'trials': 4, 'seed': 42, 'data_seed': 42,
                'harness_protocol': protocol_metadata(self.harness_protocol),
                'task_ids': ids, 'manifest_sha256': sha(manifest), 'manifest': str(manifest),
                'policy_temperature': 0.7,
                'user_temperature': 0.7, 'max_steps': 30,
                'max_errors': 10, 'per_model_concurrency': 4, 'global_max_concurrency': 16,
                'dtype': 'bfloat16', 'enable_thinking': False, 'tool_execution': 'sequential',
                'policy_generation_config': 'vllm', 'policy_engine_seed': 42,
                'policy_max_model_len': 24576, 'simulator_max_model_len': 16384,
                'simulator': self.env['TAU3_USER_MODEL'], 'root': str(self.root),
                'sft': str(Path(self.env['TAU3_ROOT']) / 'code/checkpoints/sft-merged/new-off'),
                'no_official_final': True, 'no_shutdown': True}
        if self.reuse_sft_from:
            if self.reuse_sft_from == self.output.resolve():
                raise ValueError('Recovery output must differ from the previous attempt')
            plan['reuse_sft_from'] = str(self.reuse_sft_from)
            validate_sft_reuse(self.reuse_sft_from, plan)
        path = self.output / 'plan.json'
        if path.exists() and read(path) != plan:
            raise ValueError('Frozen plan differs from current configuration')
        if not path.exists():
            atomic_json(path, plan)
        self.plan = plan
        return plan

    def prepare_model(self, arm):
        dest = self.output / 'models' / arm
        receipt = self.output / f'{arm}-export.json'
        if arm == 'sft':
            dest = Path(self.plan['sft'])
            validate_hf(dest)
            hashes = file_hashes(dest)
            if receipt.exists() and read(receipt)['files_sha256'] != hashes:
                raise ValueError('SFT checkpoint changed')
            atomic_json(receipt, {'source': str(dest), 'files_sha256': hashes})
        elif dest.exists():
            validate_hf(dest)
            if file_hashes(dest) != read(receipt)['files_sha256']:
                raise ValueError(f'{arm} merged model hash mismatch')
        else:
            cp = validate_finished_arm(self.root, arm)
            staging = dest.with_name(arm + '.merging')
            if staging.exists():
                raise RuntimeError(f'Unfinished merge requires inspection: {staging}')
            self.model_status(arm, status='merging_cpu')
            env = dict(self.env, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            self.run_command([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                              '--local_dir', str(cp / 'actor'), '--target_dir', str(staging)],
                             env, f'merge-{arm}')
            validate_hf(staging)
            verification = read(staging / 'export-verification.json')
            if not verification['native_format'] or verification['verified_tensors'] <= 0:
                raise ValueError('Merged model has not passed exact FSDP tensor verification')
            atomic_json(receipt, {'source': str(cp), 'source_model_sha256': {
                p.name: sha(p) for p in (cp / 'actor').glob('model_world_size_4_rank_*.pt')},
                'dtype': 'bfloat16', 'tensor_verification': verification,
                'files_sha256': file_hashes(staging)})
            staging.rename(dest)
        self.models[arm] = dest
        self.model_status(arm, status='prepared', checkpoint=str(dest))

    def backup(self, arm):
        self.status(status=f'backing_up_{arm}')
        dest = persist_complete(self.root, self.reuse_sft_from or self.output, arm)
        if self.reuse_sft_from:
            atomic_json(self.output / f'{arm}-backup.json', read(self.reuse_sft_from / f'{arm}-backup.json'))
        with self.lock:
            self.state['backups'][arm] = str(dest)
        self.status(status='preparing_cpu')

    def prepare_existing(self):
        self.prepare_model('sft')
        if self.reuse_sft_from:
            summary = validate_sft_reuse(self.reuse_sft_from, self.plan)
            if read(self.output / 'sft-export.json')['files_sha256'] != read(
                self.reuse_sft_from / 'sft-export.json'
            )['files_sha256']:
                raise ValueError('SFT checkpoint changed since the completed evaluation')
            source = self.reuse_sft_from / 'selection/sft'
            atomic_json(self.output / 'sft-reuse.json', {
                'source': str(source), 'files_sha256': file_hashes(source),
                'metrics': summary['metrics'], 'new_sft_trials': 0})
            self.reused_results['sft'] = summary['metrics']
            self.model_status('sft', status='complete', reused_from=str(source), metrics=summary['metrics'])
        for arm in ARMS:
            if (self.root / f'{arm}.exit').exists() and (self.root / f'{arm}_seed42/completion.json').exists():
                self.prepare_model(arm)
                if arm in ('e2', 'e3'):
                    self.backup(arm)

    def controller_alive(self):
        pid = int((self.root / 'controller.pid').read_text())
        cmd = Path(f'/proc/{pid}/cmdline')
        return cmd.exists() and b'run_e1_after_e0.py' in cmd.read_bytes()

    def wait_for_training(self):
        self.status(status='waiting_for_rl', trigger='complete + exit0 + controller exited + GPU idle')
        while not queue_ready(self.root, self.controller_alive()):
            if not self.controller_alive():
                raise RuntimeError('RL controller disappeared without complete queue')
            if self.stop_event.wait(15):
                raise InterruptedError('Stopped while waiting for RL')
        deadline = time.monotonic() + 600
        while self.gpu_pids():
            if time.monotonic() > deadline:
                raise RuntimeError('GPU still occupied after RL completion; no processes killed')
            if self.stop_event.wait(5):
                raise InterruptedError('Stopped waiting for GPU release')

    @staticmethod
    def gpu_pids(devices='0,1,2,3,4'):
        text = subprocess.check_output(['nvidia-smi', '-i', devices,
            '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        return {int(line.strip()) for line in text.splitlines() if line.strip().isdigit()}

    @staticmethod
    def port_free(port):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))

    def next_pending(self):
        with self.lock:
            return self.pending.pop(0) if self.pending else None

    def wait_health(self, process, port, model):
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if process.poll() is not None or self.stop_event.is_set():
                raise RuntimeError(f'Service {model} exited during startup')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                    names = [x['id'] for x in json.load(response)['data']]
                if names != [model]:
                    raise ValueError(f'Unexpected model at port {port}: {names}')
                return
            except (OSError, TimeoutError):
                self.stop_event.wait(3)
        raise TimeoutError(f'Service {model} startup timeout')

    def evaluation_command(self, arm, port, output, manifest_dir, trials):
        return [sys.executable, '-m', 'tau3_grpo.evaluation.run', '--target', 'selection',
                '--checkpoint', str(self.models[arm]), '--manifest-dir', str(manifest_dir),
                '--results-dir', str(self.output), '--policy-model', f'postrl-{arm}',
                '--policy-base-url', f'http://127.0.0.1:{port}/v1', '--policy-attestation',
                str(self.output / f'{arm}-attestation.json'), '--user-model', self.env['TAU3_USER_SERVED_MODEL_NAME'],
                '--user-base-url', 'http://127.0.0.1:8210/v1', '--seed', '42', '--data-seed', '42',
                '--trials', str(trials), '--ks', *(['1', '2', '4'] if trials == 4 else ['1']),
                '--include-pass-hat', '--max-concurrency', '4', '--max-steps', '30',
                '--harness-protocol', self.harness_protocol,
                '--token-request-timeout', str(self.plan.get('token_request_timeout', 120)),
                '--policy-temperature', '0.7',
                '--user-temperature', '0.7', '--output-dir', str(output)]

    def evaluate_slot(self, slot, arms, simulator):
        results = {}
        port = 8200 + slot
        arms = list(arms)
        while arms:
            arm = arms.pop(0)
            policy = None
            try:
                deadline = time.monotonic() + 120
                while self.gpu_pids(str(slot)):
                    if time.monotonic() >= deadline or self.stop_event.wait(2):
                        raise RuntimeError(f'GPU {slot} still occupied; no process killed')
                self.port_free(port)
                checkpoint = str(self.models[arm])
                self.model_status(arm, status='starting_service', gpu=slot, port=port)
                args = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', checkpoint,
                        '--served-model-name', f'postrl-{arm}', '--host', '127.0.0.1', '--port', str(port),
                        '--tensor-parallel-size', '1', '--dtype', 'bfloat16', '--max-model-len', '24576',
                        '--gpu-memory-utilization', '.80', '--max-num-seqs', '8', '--enforce-eager',
                        '--generation-config', 'vllm', '--seed', '42',
                        '--language-model-only', '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
                        '--reasoning-parser', 'qwen3', '--default-chat-template-kwargs', '{"enable_thinking":false}',
                        '--enable-prefix-caching']
                if self.harness_protocol == TOKENS_V4:
                    args += ['--logprobs-mode', 'processed_logprobs']
                env = dict(self.env, CUDA_VISIBLE_DEVICES=str(slot), VLLM_WORKER_MULTIPROC_METHOD='spawn',
                           VLLM_CACHE_ROOT=str(self.output / 'cache' / arm), PYTHONUNBUFFERED='1')
                policy = self.launch(args, env, f'policy-{arm}')
                self.wait_health(policy, port, f'postrl-{arm}')
                # Bind the exact process/checkpoint using the existing verifier.
                from tau3_grpo.evaluation.service_attestation import write_service_attestation
                write_service_attestation(checkpoint_path=checkpoint, served_model_name=f'postrl-{arm}',
                    base_url=f'http://127.0.0.1:{port}/v1', pid=policy.pid,
                    output=self.output / f'{arm}-attestation.json')
                cpu_env = dict(self.env, CUDA_VISIBLE_DEVICES='')
                smoke = self.output / 'smoke' / arm
                self.model_status(arm, status='smoke', gpu=slot)
                self.run_command(self.evaluation_command(arm, port, smoke, self.output / 'smoke-manifests', 1),
                                 cpu_env, f'smoke-{arm}', timeout=1800, dependencies=(policy, simulator))
                check_summary(smoke, self.plan['task_ids'][:4], trials=1)
                result = self.output / 'selection' / arm
                self.model_status(arm, status='evaluating', output=str(result), gpu=slot, started_at=time.time())
                self.run_command(self.evaluation_command(arm, port, result, self.output / 'manifests', 4),
                                 cpu_env, f'eval-{arm}', timeout=4 * 3600, dependencies=(policy, simulator))
                summary = check_summary(result, self.plan['task_ids'])
                results[arm] = summary['metrics']
                self.model_status(arm, status='complete', metrics=summary['metrics'], finished_at=time.time())
            except Exception as exc:
                results[arm] = {'error': str(exc)}
                self.model_status(arm, status='failed', error=str(exc), finished_at=time.time())
            finally:
                if policy is not None:
                    self.stop(policy)
            pending = self.next_pending()
            if pending:
                arms.append(pending)
        return results

    def evaluate(self):
        if self.gpu_pids():
            raise RuntimeError('GPU became occupied before evaluation; no services launched')
        for port in (8200, 8201, 8202, 8203, 8210):
            self.port_free(port)
        self.status(status='starting_simulator')
        env = dict(self.env, TAU3_USER_CUDA_DEVICES='4', TAU3_USER_PORT='8210',
                   TAU3_USER_MAX_NUM_SEQS='16', TAU3_USER_MAX_MODEL_LEN='16384',
                   TAU3_USER_GPU_MEMORY_UTILIZATION='.65', TAU3_USER_ENFORCE_EAGER='1')
        simulator = self.launch(['bash', str(CODE / 'scripts/serve/simulator_qwen38.sh')], env, 'simulator')
        self.wait_health(simulator, 8210, env['TAU3_USER_SERVED_MODEL_NAME'])
        count = sum(len(slot) for slot in self.slots) + len(self.pending)
        self.status(status='evaluating', planned_trajectories=count * 240, smoke_trajectories=count * 4)
        results = dict(self.reused_results)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.evaluate_slot, slot, arms, simulator) for slot, arms in enumerate(self.slots)]
            for future in as_completed(futures):
                results.update(future.result())
        complete = all('error' not in results.get(arm, {'error': 'missing'}) for arm in self.plan['models'])
        self.status(status='complete' if complete else 'incomplete', results=results,
                    safe_to_shutdown=complete and set(self.state['backups']) == {'e2', 'e3'})
        return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('dry-run', 'prepare', 'watch'), default='dry-run')
    parser.add_argument('--harness-protocol', choices=PROTOCOLS, default=LEGACY)
    parser.add_argument('--reuse-sft-from', type=Path,
                        help='Explicit recovery of RL startup failures; reuse the completed SFT evaluation')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = Controller(args.root, args.output, args.reuse_sft_from, args.harness_protocol)
        def interrupted(signum, frame):
            controller.stop_event.set()
            raise KeyboardInterrupt('Post-RL controller interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        if args.mode == 'watch' and ((args.output / 'selection').exists() or (args.output / 'controller.exit').exists()):
            raise RuntimeError('Existing evaluation attempt; inspect without automatic replay')
        (args.output / 'controller.pid').write_text(str(os.getpid()))
        try:
            plan = controller.freeze_plan()
            atomic_json(args.output / 'software.json', {
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
                'controller_sha256': sha(__file__),
                'evaluation_source_sha256': evaluation_source_hashes(),
                'versions': {name: importlib.metadata.version(name) for name in ('torch', 'vllm', 'transformers')},
            })
            for sub, lines in [('manifests', None), ('smoke-manifests', 4)]:
                dest = args.output / sub / 'areal_airline_selection_seed42.jsonl'
                content = Path(plan['manifest']).read_text().splitlines(keepends=True)
                text = ''.join(content if lines is None else content[:lines])
                if dest.exists() and dest.read_text() != text:
                    raise ValueError('Staged selection manifest changed')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text)
            if args.mode == 'dry-run':
                print(json.dumps(plan, indent=2))
                return 0
            controller.prepare_existing()
            if args.mode == 'prepare':
                controller.status(status='prepared_cpu')
                return 0
            controller.wait_for_training()
            if 'e3' not in controller.models:
                controller.prepare_model('e3')
                controller.backup('e3')
            code = controller.evaluate()
            controller.cleanup()
            (args.output / 'controller.exit').write_text(str(code))
            return code
        except BaseException as exc:
            controller.status(status='failed', error=f'{type(exc).__name__}: {exc}', safe_to_shutdown=False)
            if args.mode == 'watch':
                (args.output / 'controller.exit').write_text('1')
            raise
        finally:
            controller.cleanup()


if __name__ == '__main__':
    raise SystemExit(main())
