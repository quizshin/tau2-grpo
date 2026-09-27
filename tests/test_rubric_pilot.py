import copy

import pytest

from tau3_grpo.analysis.rubric_pilot import (
    DIMENSIONS,
    Budget,
    validate_audit,
    validate_rubric,
    visible_events,
)


def test_visible_projection_removes_private_labels_and_reasoning():
    events = visible_events([{'role': 'assistant', 'content': 'Visible.',
                              'thinking': 'private', 'reasoning': 'private',
                              'reward': 1, 'metadata': {'correct': 1}}])
    assert events == [{'event_id': 'm000', 'role': 'assistant', 'content': 'Visible.'}]


def fixture():
    events = visible_events([{'role': 'user', 'content': 'Cancel if eligible.'},
                             {'role': 'assistant', 'content': 'Need reservation ID.'}])
    rubric = {'requirements': [{'id': 'R1', 'criterion': 'Use the correct reservation',
                                'source_refs': ['m000'], 'available_from': 'm000',
                                'applies_when': 'Cancelling', 'severity': 'critical',
                                'verification_method': 'User and tool evidence'}]}
    audit = {'requirements': [{'id': 'R1', 'status': 'unknown', 'evidence_refs': [], 'reason': 'No ID yet'}],
             'dimensions': {d: {'score': None, 'status': 'unknown', 'requirement_ids': ['R1'],
                                'reason': 'Insufficient evidence'} for d in DIMENSIONS},
             'issues': [], 'recommendation': 'review', 'difficulty': {'level': 'unknown'}}
    return events, rubric, audit


def test_evidence_identity_and_unknown_are_validated():
    events, rubric, audit = fixture()
    validate_rubric(rubric, events[:1])
    validate_audit(audit, rubric, events)
    invalid = copy.deepcopy(audit)
    invalid['requirements'][0].update(status='satisfied', evidence_refs=['m999'])
    with pytest.raises(ValueError, match='Invented evidence'):
        validate_audit(invalid, rubric, events)
    invalid = copy.deepcopy(audit)
    invalid['dimensions']['intent']['score'] = 2
    with pytest.raises(ValueError, match='null'):
        validate_audit(invalid, rubric, events)


def test_author_cannot_cite_hidden_actor_messages():
    events, rubric, _ = fixture()
    rubric['requirements'][0]['source_refs'] = ['m001']
    with pytest.raises(ValueError, match='evidence reference'):
        validate_rubric(rubric, events[:1])


def test_budget_holds_unknown_spend_and_refuses_call_before_cap(tmp_path):
    budget = Budget(tmp_path / 'budget.json', limit=0.02)
    row = budget.reserve('one', 'system', {'input': 'x'}, 128)
    reserved = budget.accounted
    budget.settle(row, {}, 'failed')
    assert budget.accounted == reserved > 0
    with pytest.raises(ValueError, match='Budget cap'):
        budget.reserve('too-big', 'system', {'input': 'x' * 10000}, 8192)
    assert len(budget.state['calls']) == 1
    resumed = Budget(tmp_path / 'budget.json', limit=0.02)
    assert resumed.accounted == reserved


def test_usage_cost_and_overrun_stop_are_explicit(tmp_path):
    budget = Budget(tmp_path / 'budget.json')
    row = budget.reserve('one', 'system', {}, 128)
    budget.settle(row, {'usage': {'prompt_tokens': 20, 'completion_tokens': 10},
                       'response_model': 'deepseek-flash'}, 'received')
    assert budget.accounted == pytest.approx(0.00012)
    row = budget.reserve('two', 'system', {}, 128)
    with pytest.raises(RuntimeError, match='bound exceeded'):
        budget.settle(row, {'usage': {'prompt_tokens': 100000, 'completion_tokens': 10}}, 'received')


@pytest.mark.parametrize('field,value', [('difficulty', 'hard'), ('issues', ['bad']),
                                       ('requirements', ['bad'])])
def test_malformed_audit_fails_as_validation_error(field, value):
    events, rubric, audit = fixture()
    audit[field] = value
    with pytest.raises(ValueError):
        validate_audit(audit, rubric, events)


def test_continuation_uses_same_cumulative_budget(tmp_path):
    path = tmp_path / 'budget.json'
    old = Budget(path, max_calls=1)
    old.reserve('v1', 'system', {}, 128)
    spent = old.accounted
    with pytest.raises(ValueError, match='cap reached'):
        old.reserve('v2', 'system', {}, 128)
    resumed = Budget(path, max_calls=3)
    resumed.reserve('v2', 'system', {}, 128)
    assert resumed.accounted > spent
    with pytest.raises(ValueError, match='Duplicate'):
        resumed.reserve('v2', 'system', {}, 128)
