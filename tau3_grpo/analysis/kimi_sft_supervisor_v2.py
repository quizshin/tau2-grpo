"""Bounded Kimi review: three concurrent candidates, at most three preserved attempts each."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tau3_grpo.analysis.kimi_sft_review import save
from tau3_grpo.utils.hashing import sha256_file

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = {2: 15, 3: 45}


def attempts_for(row, review_root, retry_root):
    cid = row['candidate_id']
    paths = [(1, review_root / cid)] + [(n, retry_root / cid / f'attempt_{n:02d}')
                                       for n in (2, 3)]
    found = []
    for number, path in paths:
        if not path.exists():
            if any(later.exists() for _, later in paths[number:]):
                raise ValueError(f'Gap in immutable Kimi attempt ledger for {cid}')
            break
        state_path = path / 'state.json'
        if not state_path.exists():
            raise ValueError(f'Attempt directory without state for {cid}')
        state = json.loads(state_path.read_text())
        if (state.get('candidate_id') != cid or state.get('task_hash') != row['task_hash']
                or state.get('candidate_sha256') != row['candidate_sha256']
                or state.get('reviewer', {}).get('model') != 'Kimi-K3'):
            raise ValueError(f'Attempt identity mismatch for {cid}')
        found.append((number, path, state))
    completed = [s for _, _, s in found if s['status'] == 'review_complete']
    if len(completed) > 1 or (completed and found[-1][2] is not completed[0]):
        raise ValueError(f'Review continued after semantic decision for {cid}')
    if not completed and any(s['status'] not in ('review_failed_no_retry', 'request_started_usage_unknown')
                             for _, _, s in found):
        raise ValueError(f'Unknown attempt state for {cid}')
    return found


def assert_worker_terminated(row):
    """A prior unknown may be retried only after its local request process has exited."""
    output = subprocess.check_output(['ps', '-ax', '-o', 'command='], text=True)
    marker = '-m tau3_grpo.analysis.kimi_sft_review '
    candidate_arg = '--candidate ' + row['candidate_file']
    if any(marker in line and candidate_arg in line for line in output.splitlines()):
        raise ValueError('Prior Kimi request process still running for ' + row['candidate_id'])


def validate_sources(plan, frozen_ids):
    rows = plan['candidates']
    if (not 1 <= len(rows) <= 100 or len({r['candidate_id'] for r in rows}) != len(rows)
            or len({r['task_hash'] for r in rows}) != len(rows)):
        raise ValueError('Plan must contain 1-100 unique candidates')
    if {r['candidate_id'] for r in rows} & frozen_ids:
        raise ValueError('Frozen Codex review must not repeat')
    for row in rows:
        for kind in ('candidate', 'rubrics'):
            if sha256_file(row[kind + '_file']) != row[kind + '_sha256']:
                raise ValueError('Plan source identity changed')
        candidate = json.loads(Path(row['candidate_file']).read_text())
        if (candidate['candidate_id'] != row['candidate_id'] or
                candidate['task']['task_hash'] != row['task_hash'] or
                candidate['status'] != 'user_stop'):
            raise ValueError('Candidate task identity/status mismatch')
    return rows


async def run(args):
    args.review_root.mkdir(parents=True, exist_ok=True)
    args.retry_root.mkdir(parents=True, exist_ok=True)
    with (args.review_root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.runtime.exists():
            raise FileExistsError('Runtime exists; never restart this bounded run')
        plan = json.loads(args.plan.read_text())
        frozen = {r['id'] for r in json.loads(args.frozen_index.read_text())}
        rows = validate_sources(plan, frozen)
        started = datetime.now(timezone.utc).isoformat()
        semaphore = asyncio.Semaphore(args.concurrency)
        active = {}
        completed = []
        failed_attempts = []
        errors = []
        stop = asyncio.Event()

        def snapshot(reason=None):
            states = []
            for root, pattern in ((args.review_root, '*/state.json'),
                                  (args.retry_root, '*/*/state.json')):
                for path in root.glob(pattern):
                    state = json.loads(path.read_text())
                    if state.get('status') == 'review_complete' and state.get('quality_accepted'):
                        states.append(state)
            if len({s['task_hash'] for s in states}) != len(states):
                raise ValueError('Duplicate accepted task hash')
            save(args.runtime, dict(
                plan_sha256=sha256_file(args.plan), pid=os.getpid(), started_at=started,
                updated_at=datetime.now(timezone.utc).isoformat(), plan_count=len(rows),
                active=active, completed_candidates=completed, failed_attempts=failed_attempts,
                remaining_candidates=len(rows)-len(completed)-len(active),
                accepted_count=len(states), reviewer='Kimi-K3', concurrency=args.concurrency,
                max_attempts_per_candidate=MAX_ATTEMPTS, backoff_seconds=BACKOFF_SECONDS,
                no_usage_fabrication=True, stopped_on_internal_error=stop.is_set(),
                internal_errors=errors, reason=reason,
            ))

        def accepted_task_hashes():
            result = {r['task_hash'] for r in json.loads(args.frozen_index.read_text())
                      if r.get('task_hash')}
            for root, pattern in ((args.review_root, '*/state.json'),
                                  (args.retry_root, '*/*/state.json')):
                for path in root.glob(pattern):
                    state = json.loads(path.read_text())
                    if state.get('status') == 'review_complete' and state.get('quality_accepted'):
                        result.add(state['task_hash'])
            return result

        async def review(row):
            cid = row['candidate_id']
            try:
                prior = attempts_for(row, args.review_root, args.retry_root)
                while not stop.is_set():
                    if row['task_hash'] in accepted_task_hashes() and not (
                        prior and prior[-1][2]['status'] == 'review_complete' and
                        prior[-1][2]['quality_accepted']):
                        completed.append(dict(candidate_id=cid, status='already_accepted_other_candidate',
                                              accepted=False, attempts=len(prior)))
                        snapshot()
                        return
                    if prior and prior[-1][2]['status'] == 'review_complete':
                        completed.append(dict(candidate_id=cid, status='review_complete',
                                              accepted=prior[-1][2]['quality_accepted'],
                                              attempts=len(prior), already_reviewed=True))
                        snapshot()
                        return
                    if len(prior) >= MAX_ATTEMPTS:
                        completed.append(dict(candidate_id=cid, status='attempts_exhausted',
                                              accepted=False, attempts=len(prior)))
                        snapshot()
                        return
                    number = len(prior) + 1
                    if prior:
                        await asyncio.sleep(BACKOFF_SECONDS[number])
                    async with semaphore:
                        if stop.is_set():
                            return
                        if row['task_hash'] in accepted_task_hashes():
                            completed.append(dict(candidate_id=cid, status='already_accepted_other_candidate',
                                                  accepted=False, attempts=len(prior)))
                            snapshot()
                            return
                        assert_worker_terminated(row)
                        if attempts_for(row, args.review_root, args.retry_root) != prior:
                            raise ValueError('Attempt ledger changed during scheduling')
                        destination = (args.review_root / cid if number == 1 else
                                       args.retry_root / cid / f'attempt_{number:02d}')
                        if destination.exists():
                            raise ValueError('Attempt output already exists')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        log_path = args.runtime.parent / f'{cid}_attempt_{number:02d}_kimi.log'
                        with log_path.open('xb') as log:
                            proc = await asyncio.create_subprocess_exec(
                                sys.executable, '-m', 'tau3_grpo.analysis.kimi_sft_review',
                                '--candidate', row['candidate_file'], '--rubrics', row['rubrics_file'],
                                '--db-root', str(args.db_root), '--output', str(destination),
                                stdout=log, stderr=log,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                         OPENBLAS_NUM_THREADS='1'),
                            )
                            active[cid] = dict(pid=proc.pid, attempt=number)
                            snapshot()
                            code = await proc.wait()
                        active.pop(cid)
                        prior = attempts_for(row, args.review_root, args.retry_root)
                        if len(prior) != number:
                            raise ValueError('Kimi subprocess omitted its immutable attempt state')
                        result = prior[-1][2]
                        if result['status'] == 'review_complete':
                            if code:
                                raise ValueError('Completed review process returned nonzero')
                            completed.append(dict(candidate_id=cid, status='review_complete',
                                                  accepted=result['quality_accepted'], attempts=number))
                            snapshot()
                            return
                        failed_attempts.append(dict(candidate_id=cid, attempt=number,
                                                    status=result['status'],
                                                    usage_resolved=result.get('usage_resolved', False),
                                                    state_file=str(prior[-1][1] / 'state.json')))
                        snapshot()
            except Exception as exc:
                stop.set()
                errors.append(dict(candidate_id=cid, error_type=type(exc).__name__, reason=str(exc)))
                snapshot('internal error; in-flight requests drain without new dispatch')
                return

        snapshot('bounded review started')
        try:
            await asyncio.gather(*(review(row) for row in rows))
        finally:
            snapshot('stopped on internal error' if stop.is_set() else 'bounded plan complete')
        if errors:
            raise RuntimeError('Kimi supervisor internal error; see preserved runtime')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'runtime', 'review-root', 'retry-root', 'frozen-index', 'db-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=range(1, 4), default=3)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
