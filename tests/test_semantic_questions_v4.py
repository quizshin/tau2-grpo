"""Late-turn counterexamples, not evidence of model extraction accuracy."""
from copy import deepcopy

import pytest

from tau3_grpo.algorithms.anchors.semantic_state import SCHEMA, SemanticError, compile_state
from tau3_grpo.models.semantic_incremental import delta_request
from tau3_grpo.utils.hashing import sha256_json

VERSION = 'airline_slots_v4'


def ev(messages, at, kind, data, eid, support=()):
    return {'id': eid, 'at': at, 'kind': kind, 'data': data, 'evidence': [
        {'message_index': i, 'start': 0, 'end': len(messages[i]['content']), 'quote': messages[i]['content']}
        for i in (at, *support)]}


def packet(messages, events):
    return {'schema': SCHEMA, 'prefix_sha256': sha256_json(messages), 'events': events}


def compile_case(messages, events, **kwargs):
    return compile_state(messages, packet(messages, events), task_id='task', db_hash='db',
                         policy_hash='policy', slot_schema=VERSION, **kwargs)


def setup(compound=True):
    confirm = 'Would you like to proceed with this cancellation?'
    choice = 'Which payment method would you like to use?'
    text = 'Refund: USD 20. ' + confirm + (' ' + choice if compound else '')
    messages = [{'role': 'user', 'content': 'Cancel reservation ABC123.'},
                {'role': 'assistant', 'content': text}]
    events = [ev(messages, 0, 'goal', {'operation': 'cancel', 'target': {'reservation_id': 'ABC123'},
        'relation': 'add', 'replaces': []}, 'g'),
        ev(messages, 1, 'proposal', {'goal_ids': ['g'], 'operations': [
            {'operation': 'cancel', 'target': {'reservation_id': 'ABC123'},
             'terms': {'quoted_refund': '20', 'currency': 'USD'}, 'goal_ids': ['g']}], 'supersedes': []}, 'p', (0,)),
        ev(messages, 1, 'question', {'intent': 'confirm_proposal', 'proposal_id': 'p', 'focus': confirm}, 'q_confirm')]
    if compound:
        events.append(ev(messages, 1, 'question', {'intent': 'choose_payment_method', 'proposal_id': None,
                                                 'focus': choice}, 'q_payment'))
    return messages, events


def add_consent(messages, events, text='Yes.', binding='reply', pid='p'):
    messages.append({'role': 'user', 'content': text})
    events.append(ev(messages, len(messages) - 1, 'consent', {'proposal_id': pid,
        'operation_indices': [0], 'status': 'approved', 'binding': binding}, 'c'))


def test_compound_questions_are_order_independent_and_bare_yes_is_ambiguous():
    messages, events = setup()
    a = compile_case(messages, events)
    reverse = events[:2] + list(reversed(events[2:]))
    assert compile_case(messages, reverse)['key'] == a['key']
    assert len(a['state']['pending_questions']) == 2
    for sequence in (events, reverse):
        m, e = deepcopy(messages), deepcopy(sequence)
        add_consent(m, e)
        with pytest.raises(SemanticError, match='ambiguous_compound_reply'):
            compile_case(m, e)


def test_choice_answer_does_not_grant_consent_or_change_proposal():
    messages, events = setup()
    before = compile_case(messages, events)['state']['proposals']
    messages.append({'role': 'user', 'content': 'Use card_2.'})
    events.append(ev(messages, 2, 'question_answer', {'question_id': 'q_payment', 'value': 'Use card_2.'}, 'answer'))
    state = compile_case(messages, events)['state']
    assert state['consents'] == [] and state['proposals'] == before
    assert [q['topic'] for q in state['pending_questions']] == ['confirm_proposal']
    assert state['question_answers'][0]['answer'] == 'Use card_2.'
    add_consent(messages, events)
    with pytest.raises(SemanticError, match='ambiguous_compound_reply'):
        compile_case(messages, events)


@pytest.mark.parametrize('reverse', [False, True])
def test_same_turn_choice_answer_cannot_disambiguate_bare_yes(reverse):
    messages, events = setup()
    add_consent(messages, events, 'Yes, use card_2.')
    answer = ev(messages, 2, 'question_answer', {'question_id': 'q_payment', 'value': messages[2]['content']}, 'answer')
    if reverse:
        events.insert(-1, answer)
    else:
        events.append(answer)
    with pytest.raises(SemanticError, match='combined_answer_and_consent_unsupported'):
        compile_case(messages, events)


def test_omitting_second_question_is_rejected_even_with_full_message_citation():
    messages, events = setup()
    with pytest.raises(SemanticError, match='unaccounted_question_clause'):
        compile_case(messages, events[:-1])


def test_invented_focus_and_choice_disguised_as_approval_are_rejected():
    messages, events = setup()
    events[-1]['data']['focus'] = 'Please approve this cancellation.'
    with pytest.raises(SemanticError, match='unsupported_question_focus'):
        compile_case(messages, events)
    messages, events = setup()
    events[-1]['data'].update(intent='confirm_proposal', proposal_id='p')
    with pytest.raises(SemanticError, match='scope_conflict'):
        compile_case(messages, events)


def test_single_confirmation_accepts_yes_and_keeps_reference_metadata_out_of_key():
    messages, events = setup(False)
    add_consent(messages, events)
    plain = compile_case(messages, events)
    traced = compile_case(messages, events, include_references=True)
    assert plain['key'] == traced['key']
    assert plain['state']['consents'][0]['operations'][0]['status'] == 'approved'
    assert not plain['state']['pending_questions']


@pytest.mark.parametrize('text', ['Use card_2.', 'I do not approve this cancellation.',
                                  'Yes, only if there is no fee.'])
def test_selection_negation_and_condition_are_not_unconditional_approval(text):
    messages, events = setup(False)
    add_consent(messages, events, text)
    with pytest.raises(SemanticError):
        compile_case(messages, events)


def test_explicit_label_cannot_rescue_ambiguous_yes_but_named_approval_is_distinct():
    messages, events = setup()
    add_consent(messages, events, 'Yes.', binding='explicit')
    with pytest.raises(SemanticError, match='explicit_approval_requires_operation'):
        compile_case(messages, events)
    messages, events = setup()
    add_consent(messages, events, 'I approve this cancellation.', binding='explicit')
    state = compile_case(messages, events)['state']
    assert state['consents'] and len(state['pending_questions']) == 1
    assert state['pending_questions'][0]['topic'] == 'choose_payment_method'


def test_later_payment_answer_invalidates_prior_approval_without_editing_offer():
    messages, events = setup()
    add_consent(messages, events, 'I approve this cancellation.', binding='explicit')
    before = compile_case(messages, events)['state']['proposals']
    messages.append({'role': 'assistant', 'content': 'Which payment method would you like to use?'})
    events.append(ev(messages, 3, 'question', {'intent': 'choose_payment_method', 'proposal_id': None,
                                            'focus': messages[3]['content']}, 'q_payment_new'))
    messages.append({'role': 'user', 'content': 'Use card_2, only if there is no fee.'})
    events.append(ev(messages, 4, 'question_answer', {'question_id': 'q_payment_new',
                                                    'value': messages[4]['content']}, 'answer'))
    state = compile_case(messages, events)['state']
    assert not state['consents'] and state['proposals'] == before
    assert state['question_answers'][0]['answer'] == messages[4]['content']
    assert any(h.get('invalidation') == 'question_answer_changed_context' for h in state['history'])


def test_goal_replacement_deactivates_offer_in_reference_snapshot_and_cannot_resurrect_it():
    messages, events = setup(False)
    messages.append({'role': 'user', 'content': 'Change the cabin on ABC123 instead.'})
    events.append(ev(messages, 2, 'goal', {'operation': 'change_cabin',
        'target': {'reservation_id': 'ABC123'}, 'relation': 'replace', 'replaces': ['g']}, 'g2'))
    result = compile_case(messages, events, include_references=True)
    assert [g['id'] for g in result['references']['active_goals']] == ['g2']
    assert result['references']['active_proposals'] == []
    assert result['references']['inactive_proposals'] == [{'id': 'p', 'reason': 'goal_replaced', 'by_event': 'g2'}]
    prior = packet(messages, events)
    next_messages = messages + [{'role': 'assistant', 'content': 'I will search cabin prices.'}]
    _, payload = delta_request(next_messages, prior, slot_schema=VERSION)
    assert payload['reference_state'] == result['references']
    assert 'reward' not in payload and 'current_action' not in payload
    add_consent(messages, events, 'I approve this cancellation.', binding='explicit')
    with pytest.raises(SemanticError, match='consent_to_inactive_proposal'):
        compile_case(messages, events)


def test_changed_amount_requires_new_approval_and_original_offer_stays_in_history():
    messages, events = setup(False)
    add_consent(messages, events)
    old = compile_case(messages, events)['key']
    messages.append({'role': 'assistant', 'content': 'Refund: USD 30. Please confirm this cancellation.'})
    new_offer = deepcopy(events[1]['data'])
    new_offer['supersedes'] = ['p']
    new_offer['operations'][0]['terms']['quoted_refund'] = '30'
    events.append(ev(messages, 3, 'proposal', new_offer, 'p2', (0,)))
    events.append(ev(messages, 3, 'question', {'intent': 'confirm_proposal', 'proposal_id': 'p2',
                                            'focus': 'Please confirm this cancellation.'}, 'q2'))
    state = compile_case(messages, events, include_references=True)
    assert not state['state']['consents'] and state['key'] != old
    assert state['references']['inactive_proposals'][0]['id'] == 'p'
    assert any('prior_consent' in row for row in state['state']['history'])
    add_consent(messages, events, 'I approve this cancellation.', binding='explicit', pid='p')
    events[-1]['id'] = 'new_c'
    with pytest.raises(SemanticError, match='consent_to_inactive_proposal'):
        compile_case(messages, events)


def test_question_answer_requires_live_reference_and_exact_user_text():
    messages, events = setup()
    messages.append({'role': 'user', 'content': 'Use card_2, only if there is no fee.'})
    for qid, value in [('q_missing', messages[2]['content']), ('q_payment', 'Use card_2.'),
                       ('q_confirm', messages[2]['content'])]:
        candidate = events + [ev(messages, 2, 'question_answer', {'question_id': qid, 'value': value}, 'a')]
        with pytest.raises(SemanticError):
            compile_case(messages, candidate)


def test_offer_without_confirmation_question_does_not_bind_yes():
    messages, events = setup(False)
    messages[1]['content'] = 'Refund: USD 20.'
    events = [events[0], ev(messages, 1, 'proposal', events[1]['data'], 'p', (0,))]
    add_consent(messages, events)
    with pytest.raises(SemanticError, match='ambiguous_compound_reply'):
        compile_case(messages, events)


@pytest.mark.parametrize('text', [
    'Yes, that looks correct. Please go ahead and process the cancellation.',
    'Please go ahead and process the cancellation.',
    'Go ahead and execute this cancellation.',
])
def test_explicit_execution_directive_is_approval(text):
    messages, events = setup(False)
    add_consent(messages, events, text, binding='explicit')
    state = compile_case(messages, events)['state']
    assert state['consents'][0]['operations'][0]['status'] == 'approved'


@pytest.mark.parametrize('text', [
    'Please do not go ahead and process the cancellation.',
    'Go ahead and process the cancellation only if there is no fee.',
    'Yes, go ahead and explain the cancellation.',
    'Go ahead and check the cancellation.',
])
def test_execution_wording_does_not_approve_information_or_qualified_requests(text):
    messages, events = setup(False)
    add_consent(messages, events, text, binding='explicit')
    with pytest.raises(SemanticError):
        compile_case(messages, events)


def test_goal_replacement_and_question_answer_are_order_independent():
    messages, events = setup()
    messages.append({'role': 'user', 'content': 'Change the cabin on ABC123 instead; use card_2.'})
    goal = ev(messages, 2, 'goal', {'operation': 'change_cabin',
        'target': {'reservation_id': 'ABC123'}, 'relation': 'replace', 'replaces': ['g']}, 'g2')
    answer = ev(messages, 2, 'question_answer', {'question_id': 'q_payment',
        'value': messages[2]['content']}, 'a')
    forward = compile_case(messages, events + [goal, answer], include_references=True)
    reverse = compile_case(messages, events + [answer, goal], include_references=True)
    assert forward['key'] == reverse['key']
    assert not forward['state']['proposals'] and not forward['state']['consents']
    assert not forward['state']['pending_questions']
    assert forward['state']['question_answers'][0]['answer'] == messages[2]['content']
    assert forward['references']['inactive_proposals'] == [
        {'id': 'p', 'reason': 'goal_replaced', 'by_event': 'g2'}]
    duplicate = deepcopy(answer)
    duplicate['id'] = 'a_duplicate'
    with pytest.raises(SemanticError, match='answer_to_inactive_question'):
        compile_case(messages, events + [goal, answer, duplicate])
    messages.append({'role': 'user', 'content': 'Use card_2.'})
    stale = ev(messages, 3, 'question_answer', {'question_id': 'q_payment',
        'value': messages[3]['content']}, 'a_stale')
    with pytest.raises(SemanticError, match='answer_to_inactive_question'):
        compile_case(messages, events + [goal, answer, stale])


@pytest.mark.parametrize('context_first', [False, True])
def test_background_keeps_new_nonapproval_question_but_blocks_bare_yes(context_first):
    messages, events = setup()
    context = ev(messages, 1, 'context', {'text': messages[1]['content']}, 'background')
    events.insert(2 if context_first else len(events), context)
    state = compile_case(messages, events)['state']
    assert [q['topic'] for q in state['pending_questions']] == ['choose_payment_method']
    approved_messages, approved_events = deepcopy(messages), deepcopy(events)
    add_consent(approved_messages, approved_events)
    with pytest.raises(SemanticError, match='ambiguous_compound_reply'):
        compile_case(approved_messages, approved_events)
    messages.append({'role': 'user', 'content': 'Change the cabin on ABC123 instead; use card_2.'})
    goal = ev(messages, 2, 'goal', {'operation': 'change_cabin',
        'target': {'reservation_id': 'ABC123'}, 'relation': 'replace', 'replaces': ['g']}, 'g2')
    answer = ev(messages, 2, 'question_answer', {'question_id': 'q_payment',
        'value': messages[2]['content']}, 'a')
    forward = compile_case(messages, events + [goal, answer])
    reverse = compile_case(messages, events + [answer, goal])
    assert forward['key'] == reverse['key']
    assert forward['state']['question_answers'][0]['answer'] == messages[2]['content']
    assert not forward['state']['consents'] and not forward['state']['proposals']


def test_background_on_later_turn_does_not_keep_old_nonapproval_questions():
    messages, events = setup()
    messages.append({'role': 'user', 'content': 'I have another issue.'})
    events.append(ev(messages, 2, 'context', {'text': messages[2]['content']}, 'user_context'))
    assert not compile_case(messages, events)['state']['pending_questions']
    messages.append({'role': 'user', 'content': 'Use card_2.'})
    events.append(ev(messages, 3, 'question_answer', {'question_id': 'q_payment',
        'value': messages[3]['content']}, 'a'))
    with pytest.raises(SemanticError, match='answer_to_inactive_question'):
        compile_case(messages, events)
