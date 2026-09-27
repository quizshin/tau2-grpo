import copy

import pytest

from tau3_grpo.analysis.sft_coldstart_audit import audit_record
from tau3_grpo.data.curriculum_split import build_split


def label(sid, group=None, **kwargs):
    return dict(source_id=sid, group_id=group or sid, group_status='reviewed',
                label_status='reviewed', difficulty='medium', difficulty_basis='task_constraints',
                needs_replanning=False, needs_dependent_subgoals=False, **kwargs)


def test_group_and_history_protection_and_reproducibility():
    labels = [label(str(i)) for i in range(40)]
    labels.extend([label('same1', 'family'), label('same2', 'family')])
    labels[0]['locked_split'] = 'train'
    labels[1]['locked_split'] = 'validation'
    result = build_split(labels, protected_groups={'3'})
    assert result == build_split(list(reversed(labels)), protected_groups={'3'})
    assert '0' in result['splits']['train'] and '1' in result['splits']['validation']
    assert all('3' not in ids for ids in result['splits'].values())
    assert any({'same1', 'same2'} <= set(ids) for ids in result['splits'].values())
    bad = [label('a', 'family', locked_split='train'), label('b', 'family', locked_split='validation')]
    with pytest.raises(ValueError, match='family collision'):
        build_split(bad)


def test_unreviewed_member_blocks_whole_family_and_length_not_difficulty():
    labels = [label('a', 'family'), label('b', 'family'), label('c')]
    labels[0]['difficulty_basis'] = 'trajectory_length'
    labels[2]['label_status'] = 'unknown'
    result = build_split(labels)
    assert not result['splits']['train'] and not result['splits']['validation']
    assert len(result['held']) == 3


def test_same_batch_cannot_use_future_profile_but_next_turn_can():
    profile = {'role': 'tool', 'name': 'get_user_details',
               'content': '{"user_id":"u","payment_methods":{"card":{}}}'}
    lookup = {'function': {'name': 'get_user_details', 'arguments': {'user_id': 'u'}}}
    booking = {'function': {'name': 'book_reservation', 'arguments':
                           {'user_id': 'u', 'payment_methods': [{'payment_id': 'card', 'amount': 10}]}}}
    row = {'metadata': {'source_dialog_id': 'test'}, 'supervision': {'message_indices': [1]},
           'messages': [{'role': 'user', 'content': 'u, book using card'},
                        {'role': 'assistant', 'tool_calls': [lookup, booking]}, profile,
                        {'role': 'tool', 'name': 'book_reservation', 'content': '{}'}]}
    assert audit_record(row)['checks'][0]['status'] == 'unknown'
    ordered = copy.deepcopy(row)
    ordered['messages'][1]['tool_calls'] = [lookup]
    ordered['messages'].insert(3, {'role': 'assistant', 'tool_calls': [booking]})
    ordered['supervision']['message_indices'].append(3)
    result = audit_record(ordered)
    assert result['checks'][0]['status'] == 'satisfied'
    assert not any(x['kind'] == 'tool_receipt_mismatch' for x in result['findings'])


def test_tool_call_id_mismatch_never_updates_evidence_state():
    row = {'metadata': {'source_dialog_id': 'test'}, 'supervision': {'message_indices': [0]},
           'messages': [{'role': 'assistant', 'tool_calls': [
               {'id': 'expected', 'function': {'name': 'get_user_details', 'arguments': {'user_id': 'u'}}}]},
               {'role': 'tool', 'name': 'get_user_details', 'tool_call_id': 'wrong',
                'content': '{"user_id":"u","payment_methods":{"card":{}}}'}]}
    assert any(x['kind'] == 'tool_receipt_mismatch' for x in audit_record(row)['findings'])
