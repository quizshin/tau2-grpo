import pytest

from tau3_grpo.analysis.score_semantic_real_audit import score
from tau3_grpo.utils.hashing import sha256_json


def sample():
    cases = [{'id': str(i), 'messages': [], 'task_id': 'task'} for i in range(4)]
    metadata = {str(i): {'source': 'source', 'task_id': 'task', 'assistant_step': 3,
                        'trial': i, 'reward': i % 2, 'prefix_sha256': sha256_json([])} for i in range(4)}
    rows = [{'id': str(i), 'key': 'a' if i < 2 else 'b', 'error': None} for i in range(4)]
    labels = {'pairs': [{'a': '0', 'b': '1', 'expected': 'merge', 'category': 'semantic_positive'},
                        {'a': '0', 'b': '2', 'expected': 'separate', 'category': 'knowledge_difference'}]}
    return cases, rows, labels, metadata


def test_abstention_is_not_success_and_full_group_is_not_new_subgroup():
    cases, rows, labels, metadata = sample()
    for row in rows:
        row['key'] = None
        row['error'] = 'rejected'
    result = score(cases, rows, labels, metadata)
    assert result['semantic_checks']['passed'] == 0
    assert result['semantic_checks']['abstained'] == 2
    assert result['comparable_groups'] == []
    for row in rows:
        row.update(key='same', error=None)
    result = score(cases, rows, labels, metadata)
    assert result['variable_return_groups'] == 1
    assert result['variable_return_proper_subgroups'] == 0
    assert result['semantic_checks']['false_merges'] == 1


def test_proper_subgroups_require_multiple_trials_and_report_incomplete_coverage():
    cases, rows, labels, metadata = sample()
    result = score(cases, rows, labels, metadata)
    assert result['fully_observed_variable_return_proper_subgroups'] == 2
    rows[3].update(key=None, error='timeout')
    result = score(cases, rows, labels, metadata)
    assert result['variable_return_proper_subgroups'] == 1
    assert result['fully_observed_variable_return_proper_subgroups'] == 0


def test_hash_mismatch_and_incomplete_results_are_rejected():
    cases, rows, labels, metadata = sample()
    with pytest.raises(ValueError, match='exactly one'):
        score(cases, rows[:-1], labels, metadata)
    metadata['0']['prefix_sha256'] = 'changed'
    with pytest.raises(ValueError, match='prefix mismatch'):
        score(cases, rows, labels, metadata)
