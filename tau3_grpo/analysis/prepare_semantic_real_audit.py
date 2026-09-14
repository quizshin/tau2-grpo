"""Freeze pre-action semantic inputs from hash-verified trajectory replays."""
import argparse
import json
from pathlib import Path

from tau3_grpo.analysis.replay_decisions import visible_message
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def run(replay_dir, lock_path, output):
    replay_dir, lock_path, output = Path(replay_dir), Path(lock_path), Path(output)
    lock = json.loads(lock_path.read_text())
    replay = json.loads((replay_dir / 'summary.json').read_text())
    source = Path(lock['source'])
    if sha256_file(source) != lock['source_sha256'] or replay['source_sha256'].get(str(source)) != lock['source_sha256']:
        raise ValueError('Source hash differs from lock/replay')
    raw_rows = [json.loads(line) for line in source.open()]
    cases, metadata, missing = [], {}, []
    for line in (replay_dir / 'trajectories.jsonl').open():
        row = json.loads(line)
        if row['status'] != 'verified' or row['source'] != str(source):
            raise ValueError('Only fully verified locked-source trajectories are accepted')
        if row['task_id'] not in [lock['pilot_task'], *lock['audit_tasks']]:
            raise ValueError('Task outside locked cohort')
        raw = raw_rows[row['source_line']-1]
        if (raw['task_id'], raw['trial']) != (row['task_id'], row['trial']):
            raise ValueError('Replay identity mismatch')
        found = set()
        for snap in row['snapshots']:
            step = snap['assistant_step']
            if step not in lock['positions']:
                continue
            prefix = [visible_message(m) for m in raw['simulation']['messages'][:snap['message_index']]]
            if not snap['certified_prefix'] or sha256_json(prefix) != snap['prefix_hash']:
                raise ValueError('Pre-action prefix hash mismatch')
            found.add(step)
            cid = f"{row['task_id']}/trial{row['trial']}/step{step}"
            if cid in metadata:
                raise ValueError('Duplicate real decision identity')
            cases.append({'id': cid, 'task_id': row['task_id'], 'messages': prefix,
                          'db_hash': snap['db_hash'], 'policy_hash': snap['policy_hash'],
                          'remaining_turns': None})
            metadata[cid] = {'source': str(source), 'source_line': row['source_line'],
                             'trial': row['trial'], 'assistant_step': step,
                             'message_index': snap['message_index'], 'reward': raw['reward'],
                             'prefix_sha256': snap['prefix_hash'], 'task_id': row['task_id'],
                             'db_hash': snap['db_hash'], 'policy_hash': snap['policy_hash']}
        missing.extend({'task': row['task_id'], 'trial': row['trial'], 'step': step}
                       for step in set(lock['positions'])-found)
    cases.sort(key=lambda c: c['id'])
    pilot_all = [c for c in cases if c['task_id'] == lock['pilot_task']]
    first_trial = min(metadata[c['id']]['trial'] for c in pilot_all)
    pilot = [c for c in pilot_all if metadata[c['id']]['trial'] == first_trial
             and metadata[c['id']]['assistant_step'] in (min(lock['positions']), max(lock['positions']))]
    audit = [c for c in cases if c['task_id'] in lock['audit_tasks']]
    if len(pilot) != 2 or not audit:
        raise ValueError('Locked pilot/audit positions missing')
    output.mkdir(parents=True, exist_ok=False)
    for name, selected in [('pilot', pilot), ('audit', audit)]:
        (output / f'{name}_cases.json').write_text(json.dumps({'cases': selected, 'pairs': []}, ensure_ascii=False, indent=2)+'\n')
    (output / 'outcome_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    result = {'pilot_cases': len(pilot), 'audit_cases': len(audit), 'missing_positions': missing,
              'lock_sha256': sha256_file(lock_path), 'source_sha256': lock['source_sha256'],
              'replay_sha256': sha256_file(replay_dir / 'trajectories.jsonl'),
              'cases_sha256': {name: sha256_file(output / f'{name}_cases.json') for name in ('pilot','audit')},
              'scope': 'Outcome metadata is separate and never sent to extractor. Same-step evaluation trials are not original training groups.'}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--replay-dir', required=True)
    p.add_argument('--lock', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(run(args.replay_dir, args.lock, args.output), indent=2))


if __name__ == '__main__':
    main()
