"""Adversarial evidence tests: source integrity is not semantic entailment."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tau3_grpo.algorithms.anchors.grounded import extract, validate


def example():
    user = {'user_id': 'alice_123', 'payment_methods': {
        'credit_card_1': {'id': 'credit_card_1', 'last_four': '1234'},
        'credit_card_2': {'id': 'credit_card_2', 'last_four': '5678'}}}
    reservation = {'reservation_id': 'ABC123', 'user_id': 'alice_123', 'cabin': 'business',
                   'flights': [{'flight_number': 'HAT216', 'date': '2024-05-21', 'price': 824}]}
    return [
        {'role': 'user', 'content': 'Hi, my user ID is alice_123. I do not want economy for reservation ABC123.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'u', 'name': 'get_user_details', 'arguments': {'user_id': 'alice_123'}},
            {'id': 'r', 'name': 'get_reservation_details', 'arguments': {'reservation_id': 'ABC123'}}]},
        {'role': 'tool', 'id': 'u', 'content': json.dumps(user)},
        {'role': 'tool', 'id': 'r', 'content': json.dumps(reservation)},
        {'role': 'assistant', 'content': 'Refund: **$654** to Visa ending in **5678**. Shall I proceed?'},
        {'role': 'user', 'content': 'Yes, but only if there is no fee.'},
    ]


def test_mixed_sentence_mentions_do_not_erase_negation_or_create_consent():
    messages = example(); report = extract(messages)
    assert not validate(messages, report)
    assert any(c['kind'] == 'cabin_mention' and c['value'] == 'economy' for c in report['claims'])
    assert report['guard'][0]['content'] == messages[0]['content']
    assert report['decision_view']['response_candidates'][-1]['approved_operation'] is None
    assert not report['semantic_complete'] and not report['training_eligible']
    assert report['decision_view']['observed_associations'][0]['reservation_id'] == 'ABC123'


def test_tail_resolution_uses_tool_evidence_and_keeps_quote_separate():
    report = extract(example())
    link = next(x for x in report['links'] if report['claims'][x['claim']]['kind'] == 'card_tail_mention')
    assert link['status'] == 'matched'
    assert report['facts'][link['facts'][0]]['evidence']['pointer'] == '/payment_methods/credit_card_2/last_four'
    money = [c['value']['amount'] for c in report['claims'] if c['kind'] == 'money_mention']
    assert money == ['654']
    assert any(f['value'] == 824 for f in report['facts'])


def test_duplicate_tail_is_ambiguous():
    messages = example()
    messages[2]['content'] = messages[2]['content'].replace('1234', '5678')
    report = extract(messages)
    link = next(x for x in report['links'] if report['claims'][x['claim']]['kind'] == 'card_tail_mention')
    assert link['status'] == 'ambiguous' and len(link['facts']) == 2


@pytest.mark.parametrize('mutation', ['span', 'quote', 'value', 'role', 'future', 'negative_index', 'pointer', 'guard', 'drop_claim', 'eligibility', 'link'])
def test_tampered_or_future_evidence_fails_closed(mutation):
    messages = example(); report = extract(messages)
    ev = report['claims'][0]['evidence']
    if mutation == 'span': ev['start'] += 1
    elif mutation == 'quote': ev['quote'] = 'bob_999'
    elif mutation == 'value': report['claims'][0]['value'] = 'bob_999'
    elif mutation == 'role': ev['role'] = 'assistant'
    elif mutation == 'future': ev['message_index'] = len(messages)
    elif mutation == 'negative_index': ev['message_index'] = -1
    elif mutation == 'pointer': report['facts'][0]['evidence']['pointer'] = '/missing'
    elif mutation == 'guard': report['guard'][0]['content'] = 'I want economy.'
    elif mutation == 'drop_claim': report['claims'].pop()
    elif mutation == 'eligibility': report['training_eligible'] = True
    elif mutation == 'link': report['links'][0]['facts'] = []
    assert validate(messages, report)


def test_future_tool_cannot_verify_current_user_claim():
    messages = example(); before = extract(messages[:1]); after = extract(messages)
    assert not before['facts'] and not before['links']
    assert after['links']
    assert validate(messages[:1], after)


@pytest.mark.parametrize('change', ['error', 'wrong_lookup', 'requestor', 'order', 'missing'])
def test_bad_tool_lineage_cannot_silently_pass(change):
    messages = example()
    if change == 'error': messages[2]['error'] = True
    elif change == 'wrong_lookup': messages[1]['tool_calls'][0]['arguments']['user_id'] = 'bob_999'
    elif change == 'requestor': messages[1]['tool_calls'][0]['requestor'] = 'user'
    elif change == 'order': messages[2], messages[3] = messages[3], messages[2]
    elif change == 'missing': del messages[2]
    report = extract(messages)
    assert report['issues']
    assert not report['decision_view']['observed_associations']


def test_old_quotes_and_new_payment_conditions_are_retained():
    messages = example() + [
        {'role': 'assistant', 'content': 'Correction: refund EUR 272.50 to credit_card_1.'},
        {'role': 'user', 'content': 'No. I asked for credit_card_2, only if no fee.'}]
    report = extract(messages)
    assert [c['value'] for c in report['claims'] if c['kind'] == 'money_mention'] == [
        {'currency': 'USD', 'amount': '654'}, {'currency': 'EUR', 'amount': '272.5'}]
    assert len(report['decision_view']['communicated_quotes']) == 2
    assert report['guard'][-1]['content'] == messages[-1]['content']


def test_search_yes_has_no_write_authorization():
    messages = [{'role': 'assistant', 'content': 'Should I search for flights?'}, {'role': 'user', 'content': 'Yes'}]
    response = extract(messages)['decision_view']['response_candidates'][0]
    assert response['phrase_kind'] == 'approve'
    assert response['responding_to_message_index'] == 0
    assert response['approved_operation'] is None and response['requires_scope_review']


def test_money_sign_currency_and_number_separators():
    report = extract([{'role': 'assistant', 'content': 'Original -$1,395.00; refund $-990; EUR 272.50; £5.'}])
    assert [c['value'] for c in report['claims'] if c['kind'] == 'money_mention'] == [
        {'currency': 'USD', 'amount': '-1395'}, {'currency': 'USD', 'amount': '-990'},
        {'currency': 'EUR', 'amount': '272.5'}, {'currency': 'GBP', 'amount': '5'}]


def test_literal_guard_distinguishes_unknown_conditions():
    a = example(); b = deepcopy(a)
    b[-1]['content'] = 'Yes, but only if the ticket remains refundable.'
    assert extract(a)['guard_sha256'] != extract(b)['guard_sha256']


def test_no_input_mutation_or_tool_ids_in_text_guard():
    a = example(); saved = deepcopy(a)
    extract(a)
    assert a == saved
    b = deepcopy(a)
    b[1]['tool_calls'][0]['id'] = 'new'; b[2]['id'] = 'new'
    assert extract(a)['guard_sha256'] == extract(b)['guard_sha256']


def test_real_prefix_field_fixtures():
    fixture = json.loads(Path('tests/fixtures/grounded_evidence_real_20260914.json').read_text())
    for case in fixture['cases']:
        report = extract(case['messages'])
        assert not validate(case['messages'], report)
        for expected in case['expected_mentions']:
            matches = [c for c in report['claims'] if c['kind'] == expected['kind']
                       and c['value'] == expected['value']
                       and c['evidence']['message_index'] == expected['message_index']]
            assert matches, (case['id'], expected)
        for expected in case.get('expected_payment_pointers', []):
            matching = [link for link in report['links']
                        if report['claims'][link['claim']]['value'] == expected['tail']]
            assert any(report['facts'][fi]['evidence']['pointer'] == expected['pointer']
                       for link in matching for fi in link['facts'])
        assert not report['training_eligible']


def test_evidence_validation_preserves_json_scalar_types():
    messages = example()
    user = json.loads(messages[2]['content']); user['count'] = 1
    messages[2]['content'] = json.dumps(user)
    report = extract(messages)
    fact = next(f for f in report['facts'] if f['evidence']['pointer'] == '/count')
    fact['value'] = True
    assert validate(messages, report)
