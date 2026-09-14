import json

import pytest

from tau3_grpo.analysis.prepare_semantic_real_audit import run
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def fixture(tmp_path):
    source = tmp_path / 'source.jsonl'
    raw, replay = [], []
    for i, task in enumerate(['pilot_task', 'audit_task']):
        messages = [{'role': 'assistant' if n % 2 == 0 else 'user', 'content': f'message {n}'} for n in range(14)]
        raw.append({'task_id': task, 'trial': 0, 'reward': i,
                    'simulation': {'messages': messages}})
        snapshots = [{'assistant_step': step, 'message_index': step*2,
                      'prefix_hash': sha256_json(messages[:step*2]), 'certified_prefix': True,
                      'db_hash': 'db', 'policy_hash': 'policy'} for step in (1,3,6)]
        replay.append({'status': 'verified', 'source': str(source), 'source_line': i+1,
                       'task_id': task, 'trial': 0, 'snapshots': snapshots})
    source.write_text(''.join(json.dumps(r)+'\n' for r in raw))
    directory = tmp_path / 'replay'
    directory.mkdir()
    (directory / 'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in replay))
    (directory / 'summary.json').write_text(json.dumps({'source_sha256': {str(source): sha256_file(source)}}))
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({'source': str(source), 'source_sha256': sha256_file(source),
                                'pilot_task': 'pilot_task', 'audit_tasks': ['audit_task'],
                                'positions': [1,3,6]}))
    return directory, lock, source


def test_pre_action_inputs_exclude_rewards_and_future_messages(tmp_path):
    directory, lock, source = fixture(tmp_path)
    result = run(directory, lock, tmp_path / 'out')
    assert result['pilot_cases'] == 2 and result['audit_cases'] == 3
    cases = json.loads((tmp_path / 'out/audit_cases.json').read_text())['cases']
    assert all('reward' not in c for c in cases)
    for c in cases:
        step = int(c['id'].split('step')[-1])
        assert len(c['messages']) == step*2
        assert c['messages'][-1]['content'] == f'message {step*2-1}'
    assert json.loads((tmp_path / 'out/outcome_metadata.json').read_text())['audit_task/trial0/step1']['reward'] == 1


def test_changed_source_is_rejected(tmp_path):
    directory, lock, source = fixture(tmp_path)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError, match='Source hash'):
        run(directory, lock, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_future_prefix_or_uncertified_snapshot_is_rejected(tmp_path):
    directory, lock, source = fixture(tmp_path)
    path = directory / 'trajectories.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['snapshots'][0]['message_index'] += 1
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='prefix hash'):
        run(directory, lock, tmp_path / 'out')
