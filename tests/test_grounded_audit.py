"""Audit accepts pinned pre-action receipts and never reads future text as input."""
import json
from pathlib import Path

import pytest

from tau3_grpo.analysis.audit_grounded import run
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def setup_receipt(tmp_path):
    source = tmp_path / 'source.jsonl'
    prefix = [{'role': 'user', 'content': 'My user ID is alice_123.'}]
    raw = {'task_id': 'task', 'trial': 0, 'reward': 1,
           'simulation': {'messages': prefix + [{'role': 'assistant', 'content': 'Future refund $999'}]}}
    source.write_text(json.dumps(raw) + '\n')
    replay = tmp_path / 'replay'; replay.mkdir()
    (replay / 'summary.json').write_text(json.dumps({'source_sha256': {str(source): sha256_file(source)}}))
    row = {'task_id': 'task', 'trial': 0, 'source': str(source), 'source_line': 1, 'status': 'verified',
           'snapshots': [{'message_index': 1, 'certified_prefix': True, 'prefix_hash': sha256_json(prefix),
                          'db_hash': 'db', 'policy_hash': 'policy', 'assistant_step': 1}]}
    (replay / 'trajectories.jsonl').write_text(json.dumps(row) + '\n')
    return source, replay


def test_audit_rejects_future_information_and_records_zero_training_eligibility(tmp_path):
    _, replay = setup_receipt(tmp_path)
    result = run([replay], tmp_path / 'out', enabled=True)
    assert result['counts']['evidence_validation_passed'] == 1
    assert result['counts']['with_communicated_money'] == 0
    assert result['counts']['training_eligible'] == 0
    decision = json.loads((tmp_path / 'out/decisions.jsonl').read_text())
    assert 'reward' not in decision and len(decision['claims']) == 1


@pytest.mark.parametrize('mutation', ['source', 'prefix', 'partial'])
def test_changed_source_or_uncertified_prefix_is_rejected(tmp_path, mutation):
    source, replay = setup_receipt(tmp_path)
    if mutation == 'source':
        source.write_text(source.read_text().replace('alice_123', 'bob_456'))
    else:
        path = replay / 'trajectories.jsonl'; row = json.loads(path.read_text())
        if mutation == 'prefix': row['snapshots'][0]['prefix_hash'] = 'wrong'
        else: row['status'] = 'partial'
        path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError):
        run([replay], tmp_path / 'out', enabled=True)


def test_disabled_has_no_io(tmp_path):
    result = run(['/does/not/exist'], tmp_path / 'out')
    assert not result['enabled'] and not (tmp_path / 'out').exists()


def test_review_label_scoring_requires_source_witness_and_keeps_misses(tmp_path):
    from tau3_grpo.analysis.score_grounded_labels import score
    from copy import deepcopy
    source, _ = setup_receipt(tmp_path)
    raw = json.loads(source.read_text()); prefix = raw['simulation']['messages'][:1]
    text = prefix[0]['content']; start = text.index('alice_123')
    label = {'id': 'test', 'source': str(source), 'source_line': 1, 'task_id': 'task', 'trial': 0,
             'prefix_end': 1, 'prefix_sha256': sha256_json(prefix), 'kind': 'user_id_mention', 'value': 'alice_123',
             'evidence': {'message_index': 0, 'role': 'user', 'start': start, 'end': start+9, 'quote': 'alice_123'}}
    labels = {'source_sha256': {str(source): sha256_file(source)}, 'labels': [label]}
    assert score(labels)['extracted'] == 1
    missing = deepcopy(labels); missing['labels'][0]['kind'] = 'unsupported_goal'
    assert score(missing)['extracted'] == 0
    invalid = deepcopy(labels); invalid['labels'][0]['evidence']['quote'] = 'bob_456'
    with pytest.raises(ValueError): score(invalid)
