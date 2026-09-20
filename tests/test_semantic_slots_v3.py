"""Counterexamples for typed identity, passenger scope, money and consent."""
from copy import deepcopy
import json

import pytest

from tau3_grpo.algorithms.anchors.semantic_slots_v3 import VERSION, identity_answer, passenger_scope
from tau3_grpo.algorithms.anchors.semantic_state import SCHEMA, SemanticError, compile_state
from tau3_grpo.models.semantic_extractor import build_request
from tau3_grpo.models.semantic_incremental import append_delta, delta_request
from tau3_grpo.utils.hashing import sha256_json


def evidence(messages, at):
    text = messages[at]['content']
    return {'message_index': at, 'start': 0, 'end': len(text), 'quote': text}


def event(messages, at, kind, data, eid=None, supporting=()):
    return {'id': eid or f'm{at}:{kind}', 'at': at, 'kind': kind, 'data': data,
            'evidence': [evidence(messages, i) for i in [at, *supporting]]}


def compile_events(messages, events, version=VERSION):
    return compile_state(messages, {'schema': SCHEMA, 'prefix_sha256': sha256_json(messages), 'events': events},
                         task_id='task', db_hash='db', policy_hash='policy', slot_schema=version)


def offer(text='Original fare: USD 100. Refund: USD 20.'):
    messages = [{'role': 'user', 'content': 'Cancel reservation ABC123.'},
                {'role': 'assistant', 'content': text}]
    goal = event(messages, 0, 'goal', {'operation': 'cancel', 'target': {'reservation_id': 'ABC123'},
                                     'relation': 'add', 'replaces': []}, 'g')
    proposal = event(messages, 1, 'proposal', {'goal_ids': ['g'], 'supersedes': [], 'operations': [
        {'operation': 'cancel', 'target': {'reservation_id': 'ABC123'},
         'terms': {'quoted_refund': '20', 'currency': 'USD'}, 'goal_ids': ['g']}]}, 'p', supporting=(0,))
    return messages, [goal, proposal]


@pytest.mark.parametrize('text', ['My user ID is alice and my reservation ID is ABC123.',
                                  'Sure, reservation ID: ABC123, user ID: alice!'])
def test_id_only_answer_is_a_claim_not_a_goal(text):
    messages = [{'role': 'user', 'content': text}]
    result = compile_events(messages, [event(messages, 0, 'identity', {'values': identity_answer(text)})])
    assert result['state']['identity_claims'] == {'user_id_claim': 'alice', 'reservation_id': 'ABC123'}
    assert result['state']['goals'] == result['state']['consents'] == []
    assert not result['training_enabled'] and not result['semantic_accuracy_verified']


def test_id_answer_wording_does_not_change_claim_key_but_values_do():
    keys = []
    for text in ['My user ID is alice and my reservation ID is ABC123.',
                 'Sure, reservation ID: ABC123, user ID: alice!', 'My user ID is bob and my reservation ID is ABC123.']:
        messages = [{'role': 'user', 'content': text}]
        keys.append(compile_events(messages, [event(messages, 0, 'identity', {'values': identity_answer(text)})])['key'])
    assert keys[0] == keys[1] != keys[2]


@pytest.mark.parametrize('text', ['My user ID is alice, but do not cancel.', 'My user ID is alice and cancel my booking.',
                                  'My user ID is alice and user ID is bob.', 'My reservation ID is ABC123 if the fee is zero.'])
def test_id_event_cannot_erase_a_qualification(text):
    assert identity_answer(text) is None
    messages = [{'role': 'user', 'content': text}]
    with pytest.raises(SemanticError, match='unsupported_identity_answer'):
        compile_events(messages, [event(messages, 0, 'identity', {'values': {'user_id_claim': 'alice'}})])


def test_identity_requires_user_and_is_opt_in():
    messages = [{'role': 'assistant', 'content': 'My user ID is alice.'}]
    events = [event(messages, 0, 'identity', {'values': {'user_id_claim': 'alice'}})]
    with pytest.raises(SemanticError, match='event_requires_user'):
        compile_events(messages, events)
    messages[0]['role'] = 'user'
    for version in ('airline_slots_v1', 'airline_slots_v2'):
        with pytest.raises(SemanticError):
            compile_events(messages, events, version)


def test_id_only_answer_cannot_be_disguised_as_lookup():
    messages = [{'role': 'user', 'content': 'My user ID is alice.'}]
    with pytest.raises(SemanticError, match='identity_answer_must_not_change_goal'):
        compile_events(messages, [event(messages, 0, 'goal', {'operation': 'lookup',
            'target': {'user_id_claim': 'alice'}, 'relation': 'add', 'replaces': []})])


def test_identity_change_preserves_goal_and_invalidates_consent():
    messages, events = offer('Refund: USD 20.')
    messages.append({'role': 'user', 'content': 'Yes.'})
    consent = {'proposal_id': 'p', 'operation_indices': [0], 'status': 'approved', 'binding': 'reply'}
    events.append(event(messages, 2, 'consent', consent))
    assert compile_events(messages, events)['state']['consents']
    messages.append({'role': 'user', 'content': 'My user ID is bob.'})
    events.append(event(messages, 3, 'identity', {'values': {'user_id_claim': 'bob'}}))
    state = compile_events(messages, events)['state']
    assert state['goals'] == [{'operation': 'cancel', 'target': {'reservation_id': 'ABC123'}}]
    assert not state['consents']
    assert any(h.get('invalidation') == 'identity_claim_changed' for h in state['history'])
    messages.append({'role': 'user', 'content': 'Yes.'})
    events.append(event(messages, 4, 'consent', consent))
    with pytest.raises(SemanticError, match='ambiguous_reply'):
        compile_events(messages, events)


def test_passenger_count_never_becomes_all_or_an_unqualified_matching_set():
    text = 'New total: USD 100 times 3 passengers.'
    messages = [{'role': 'assistant', 'content': text}]
    e = event(messages, 0, 'context', {'text': text})
    count = passenger_scope({'kind': 'count', 'count': 3}, e, messages)
    assert count['kind'] == 'count' and count['count'] == 3 and count['reference_guard']
    with pytest.raises(SemanticError, match='unsupported_all_passengers'):
        passenger_scope({'kind': 'all'}, e, messages)
    changed = [*messages, {'role': 'assistant', 'content': text}]
    e2 = event(changed, 1, 'context', {'text': text})
    assert passenger_scope({'kind': 'count', 'count': 3}, e2, changed) != count
    with pytest.raises(SemanticError, match='unsupported_passenger_count'):
        passenger_scope({'kind': 'count', 'count': 2}, e, messages)


@pytest.mark.parametrize('text,accepted', [('Change all 3 passengers.', True),
                                         ('Change all 3 passengers except Alice.', False),
                                         ('Do not change all passengers.', False)])
def test_explicit_universal_scope_preserves_exceptions(text, accepted):
    messages = [{'role': 'user', 'content': text}]
    e = event(messages, 0, 'context', {'text': text})
    if accepted:
        assert passenger_scope({'kind': 'all'}, e, messages) == {'kind': 'all'}
    else:
        with pytest.raises(SemanticError, match='qualified_all_passengers'):
            passenger_scope({'kind': 'all'}, e, messages)


def test_financial_reference_prices_are_preserved_without_becoming_charges():
    messages, events = offer()
    before = deepcopy(events)
    result = compile_events(messages, events)
    terms = result['state']['proposals'][0]['operations'][0]['terms']
    assert terms == {'quoted_refund': '20', 'currency': 'USD'}
    assert result['state']['financial_context'][0]['text'] == messages[1]['content']
    assert events == before
    with pytest.raises(SemanticError, match='unrepresented_communicated_money'):
        compile_events(messages, events, 'airline_slots_v2')
    other_messages, other_events = offer('Original fare: USD 101. Refund: USD 20.')
    assert compile_events(other_messages, other_events)['key'] != result['key']


@pytest.mark.parametrize('text', ['**Refund amount:** USD 20.', 'You will receive a refund of **USD 20**.'])
def test_markdown_refund_keeps_its_transaction_role(text):
    assert compile_events(*offer(text))['state']['proposals'][0]['operations'][0]['terms']['quoted_refund'] == '20'


@pytest.mark.parametrize('text', ['Original price: USD 20.', 'There is no refund of USD 20.',
                                  'Refund amount: USD 20 - USD 10 = USD 10.'])
def test_reference_negated_or_intermediate_amount_cannot_be_a_refund(text):
    with pytest.raises(SemanticError, match='unsupported_money_role'):
        compile_events(*offer(text))


def test_missing_identifier_citation_is_still_rejected():
    messages, events = offer()
    events[1]['evidence'] = [evidence(messages, 1)]
    with pytest.raises(SemanticError, match='unsupported_identity_slot:reservation_id'):
        compile_events(messages, events)


def test_refund_destination_statement_does_not_turn_confirmation_into_a_choice():
    messages, events = offer('Refund: USD 20. It will go to your original payment method. '
                             'Would you like to proceed with this cancellation? Please confirm.')
    events.append(event(messages, 1, 'question', {'intent': 'confirm_proposal', 'proposal_id': 'p'}, 'q'))
    assert compile_events(messages, events)['state']['pending_question']['topic'] == 'confirm_proposal'


@pytest.mark.parametrize('text,accepted', [('| **Refund Amount** | - | USD 20 |', True),
                                         ('| **Refund Amount** | USD 10 | USD 20 |', False),
                                         ('| **Original fare** | - | USD 20 |', False)])
def test_table_refund_requires_a_single_value_and_correct_row_label(text, accepted):
    if accepted:
        assert compile_events(*offer(text))['state']['proposals'][0]['operations'][0]['terms']['quoted_refund'] == '20'
    else:
        with pytest.raises(SemanticError, match='unsupported_money_role'):
            compile_events(*offer(text))


def test_legacy_other_operation_cannot_bypass_passenger_scope_guards():
    messages = [{'role': 'user', 'content': 'Remove all passengers except Alice from ABC123.'}]
    e = event(messages, 0, 'goal', {'operation': 'other',
        'target': {'action': 'remove', 'passengers': 'all', 'reservation_id': 'ABC123'},
        'relation': 'add', 'replaces': []})
    with pytest.raises(SemanticError, match='qualified_all_passengers'):
        compile_events(messages, [e])


def test_choice_question_cannot_authorize_execution():
    messages, events = offer('Refund: USD 20. Which payment method would you prefer?')
    events.append(event(messages, 1, 'question', {'intent': 'choose_option', 'proposal_id': None}, 'q'))
    messages.append({'role': 'user', 'content': 'Yes.'})
    events.append(event(messages, 2, 'consent', {'proposal_id': 'p', 'operation_indices': [0],
                                              'status': 'approved', 'binding': 'reply'}))
    with pytest.raises(SemanticError, match='ambiguous_reply'):
        compile_events(messages, events)
    events[2]['data'] = {'intent': 'confirm_proposal', 'proposal_id': 'p'}
    with pytest.raises(SemanticError, match='confirmation_question_scope_conflict'):
        compile_events(messages, events)


def test_incremental_v3_identity_event_keeps_history_immutable():
    messages = [{'role': 'user', 'content': 'My user ID is alice.'}]
    prior = {'schema': SCHEMA, 'prefix_sha256': sha256_json([]), 'events': []}
    _, payload = delta_request(messages, prior, slot_schema=VERSION)
    delta = {'schema': 'semantic_delta_v1', 'prefix_sha256': sha256_json(messages), 'events': [
        {'id': 'm0:identity', 'kind': 'identity', 'data': {'values': {'user_id_claim': 'alice'}}, 'evidence': ['m0']}]}
    snapshot = deepcopy(delta)
    packet = append_delta(messages, prior, delta, payload['evidence_catalog'], slot_schema=VERSION)
    assert packet['events'][0]['data']['values'] == {'user_id_claim': 'alice'}
    assert prior['events'] == [] and delta == snapshot
    assert 'VOCABULARY v3' in build_request([], slot_schema=VERSION)['system']


def test_saved_delta_replay_never_invents_a_missing_suffix(tmp_path):
    from tau3_grpo.analysis.replay_semantic_incremental import run
    one = [{'role': 'user', 'content': 'My user ID is alice.'}]
    two = one + [{'role': 'assistant', 'content': 'Which option?'}]
    source = tmp_path / 'source'; source.mkdir()
    delta = {'schema': 'semantic_delta_v1', 'prefix_sha256': sha256_json(one), 'events': [
        {'id': 'm0:identity', 'kind': 'identity', 'data': {'values': {'user_id_claim': 'alice'}}, 'evidence': ['m0']}]}
    raw = json.dumps({'prefix_sha256': sha256_json(one), 'at': 0, 'delta': delta, 'error': 'old_rejection'}) + '\n'
    (source / 'deltas.jsonl').write_text(raw)
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps({'cases': [
        {'id': str(i), 'messages': messages, 'task_id': 't', 'db_hash': 'd', 'policy_hash': 'p'}
        for i, messages in enumerate((one, two))]}))
    output = tmp_path / 'replay'
    result = run({'slot_schema': VERSION, 'cases': str(cases)}, source, output)
    assert result['new_api_calls'] == 0 and result['valid_cases'] == 1
    assert result['errors'] == {'no_saved_delta': 1}
    assert (source / 'deltas.jsonl').read_text() == raw
    with pytest.raises(FileExistsError):
        run({'slot_schema': VERSION, 'cases': str(cases)}, source, output)
