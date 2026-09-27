"""Bounded, resumable Kimi-only review scheduling; never grades a dialogue."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tau3_grpo.analysis.kimi_sft_review import save
from tau3_grpo.utils.hashing import sha256_file


def select_pending(plan, review_root, frozen_ids):
    rows = plan['candidates']
    if not 1 <= len(rows) <= 100:
        raise ValueError('A review plan must contain 1–100 candidates')
    ids = [r['candidate_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate candidate in plan')
    if set(ids) & frozen_ids:
        raise ValueError('Frozen Codex200 must never be reviewed again')
    pending = []
    for row in rows:
        if (review_root / row['candidate_id']).exists():
            # Even an incomplete/unknown paid attempt is immutable, never retried.
            continue
        for key in ('candidate', 'rubrics'):
            if sha256_file(row[key + '_file']) != row[key + '_sha256']:
                raise ValueError('Plan source identity changed')
        record = json.loads(Path(row['candidate_file']).read_text())
        if record['candidate_id'] != row['candidate_id'] or record['task']['task_hash'] != row['task_hash']:
            raise ValueError('Candidate/task identity changed')
        if record['status'] != 'user_stop':
            raise ValueError('Incomplete generation cannot be semantically accepted')
        pending.append(row)
    return pending


async def run(args):
    args.review_root.mkdir(parents=True, exist_ok=True)
    with (args.review_root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads(args.plan.read_text())
        frozen = {r['id'] for r in json.loads(args.frozen_index.read_text())}
        pending = select_pending(plan, args.review_root, frozen)
        queue = asyncio.Queue()
        for row in pending:
            queue.put_nowait(row)
        stop = asyncio.Event()
        active = {}
        completed = []
        started = datetime.now(timezone.utc).isoformat()

        def snapshot(reason=None):
            states = []
            for path in args.review_root.glob('*/state.json'):
                state = json.loads(path.read_text())
                if state.get('status') == 'review_complete':
                    states.append(state)
            accepted = [s for s in states if s.get('quality_accepted') is True]
            if len({s['task_hash'] for s in accepted}) != len(accepted):
                raise ValueError('Duplicate accepted task hash: stop and reconcile')
            save(args.runtime, dict(
                plan_sha256=sha256_file(args.plan), pid=os.getpid(), started_at=started,
                updated_at=datetime.now(timezone.utc).isoformat(),
                pending=queue.qsize(), active=active, completed_this_run=completed,
                kimi_complete_count=len(states), kimi_accepted_count=len(accepted),
                original_codex200_unchanged=True, semantic_reviewer='Kimi-K3',
                stopped_on_new_unresolved_usage=stop.is_set(), reason=reason,
            ))

        async def worker():
            while not stop.is_set() and not queue.empty():
                row = queue.get_nowait()
                cid = row['candidate_id']
                destination = args.review_root / cid
                if destination.exists():
                    raise ValueError('Attempt appeared after plan validation; no duplicate call')
                log_path = args.runtime.parent / (cid + '_kimi.log')
                with log_path.open('ab') as log:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, '-m', 'tau3_grpo.analysis.kimi_sft_review',
                        '--candidate', row['candidate_file'], '--rubrics', row['rubrics_file'],
                        '--db-root', str(args.db_root), '--output', str(destination),
                        stdout=log, stderr=log,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                    )
                    active[cid] = proc.pid
                    snapshot()
                    code = await proc.wait()
                active.pop(cid)
                state_path = destination / 'state.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                if code and not state.get('usage_resolved', False):
                    stop.set()
                completed.append(dict(candidate_id=cid, exit_code=code,
                                      status=state.get('status', 'pre_request_failure'),
                                      accepted=state.get('quality_accepted', False)))
                queue.task_done()
                snapshot()

        snapshot('bounded review started; no retry of existing attempt directories')
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        snapshot('stopped for unresolved request' if stop.is_set() else 'bounded plan complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'runtime', 'review-root', 'frozen-index', 'db-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=range(1, 4), default=3)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
