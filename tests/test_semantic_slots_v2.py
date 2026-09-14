"""Typed semantic comparisons, evidence guards and historical compatibility."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tau3_grpo.algorithms.anchors.semantic_slots_v2 import (
    VERSION,
    fee_predicate,
    passenger_scope,
    question,
)
from tau3_grpo.algorithms.anchors.semantic_state import SemanticError, compile_state
from tau3_grpo.models.semantic_extractor import build_request


def sample(name='approved'):
    cases = json.loads(Path('tests/fixtures/semantic_model_cases_20260914.json').read_text())['cases']
    c = next(c for c in cases if c['id'] == name)
    prefix = build_request(c['messages'])['prefix_sha256']
    packets = json.loads(Path('tests/fixtures/semantic_model_responses_20260914.json').read_text())['packets']
    return c, next(p for p in packets if p['prefix_sha256'] == prefix)


def compile_case(c, p, version=VERSION):
    return compile_state(c['messages'], p, task_id=c['task_id'], db_hash=c['db_hash'],
                         policy_hash=c['policy_hash'], remaining_turns=c['remaining_turns'], slot_schema=version)


def event(text):
    return {'at': 0, 'evidence': [{'message_index': 0, 'start': 0, 'end': len(text), 'quote': text}]}


@pytest.mark.parametrize('text,scope', [('I would prefer no fee.', 'all'),
                                      ('Only if there is no change fee.', 'change'),
                                      ('No cancellation fee, please.', 'cancellation')])
def test_no_fee_has_no_currency_and_preserves_scope(text, scope):
    value = {'fee': {'kind': 'none', 'scope': scope}}
    assert fee_predicate(value, event(text)) == value
    legacy_null = {'fee': {'op': 'eq', 'amount': '0', 'currency': None}}
    assert fee_predicate(legacy_null, event(text)) == value


def test_change_fee_does_not_mean_all_fees():
    with pytest.raises(SemanticError, match='unsupported_no_fee_scope'):
        fee_predicate({'fee': {'kind': 'none', 'scope': 'all'}}, event('No change fee.'))


def test_old_fixture_inferred_usd_is_rejected_without_rewriting_history():
    c, p = sample('preference')
    before = deepcopy(p)
    compile_case(c, p, 'airline_slots_v1')
    with pytest.raises(SemanticError, match='unsupported_fee_bound_money'):
        compile_case(c, p)
    assert p == before


def test_currency_specific_zero_is_not_universal_no_fee():
    zero = fee_predicate({'fee': {'kind': 'amount_bound', 'scope': 'all', 'op': 'eq',
                                  'amount': '0', 'currency': 'USD'}}, event('The fee must be USD 0.'))
    none = fee_predicate({'fee': {'kind': 'none', 'scope': 'all'}}, event('No fees.'))
    assert zero != none


@pytest.mark.parametrize('currency,amount', [('EUR', '20'), ('USD', '30')])
def test_fee_bound_requires_its_own_currency_and_amount(currency, amount):
    with pytest.raises(SemanticError, match='unsupported_fee_bound_money'):
        fee_predicate({'fee': {'kind': 'amount_bound', 'scope': 'all', 'op': 'lte',
                               'amount': amount, 'currency': currency}}, event('At most USD 20 in fees.'))


def corrected_fee_sample(name):
    c, p = sample(name)
    corrections = json.loads(Path('tests/fixtures/semantic_slots_v2_reference_corrections_20260914.json').read_text())['corrections']
    for e in p['events']:
        if e['kind'] == 'constraint':
            correction = next(r for r in corrections if r['prefix_sha256'] == p['prefix_sha256']
                              and r['event_id'] == e['id'])
            assert e['data']['predicate'] == correction['old_predicate']
            e['data']['predicate'] = correction['corrected_predicate']
    return c, p


def test_preference_requirement_and_later_condition_remain_distinct():
    results = [compile_case(*corrected_fee_sample(n)) for n in ('preference', 'conditional', 'later_condition')]
    assert len({r['key'] for r in results}) == 3
    assert results[0]['state']['consents'][0]['operations'][0]['status'] == 'approved'
    assert results[1]['state']['consents'][0]['operations'][0]['status'] == 'conditional'
    assert not results[2]['state']['consents']


def test_all_and_everyone_are_same_but_ids_and_ordinal_are_not_all():
    messages = [{'role': 'user', 'content': 'Downgrade everyone on ABC123 to economy.'}]
    e = event(messages[0]['content'])
    assert passenger_scope('all', e, messages) == passenger_scope('everyone', e, messages)
    with pytest.raises(SemanticError, match='unsupported_passenger_ordinal'):
        passenger_scope({'kind': 'ordinal', 'position': 2}, e, messages)
    ids = passenger_scope({'kind': 'ids', 'values': ['P2']}, event('Passenger P2 is included.'), [])
    assert ids != passenger_scope('all', e, messages)
    with pytest.raises(SemanticError, match='unsupported_passenger_id'):
        passenger_scope({'kind': 'ids', 'values': ['ABC123']}, e, messages)


def test_ordinal_cannot_cross_contexts_or_become_an_id():
    text = 'Remove the second passenger from ABC123.'
    a = passenger_scope({'kind': 'ordinal', 'position': 2}, event(text), [{'role': 'user', 'content': text}])
    b = passenger_scope({'kind': 'ordinal', 'position': 2}, event(text), [{'role': 'user', 'content': text + ' Previous list changed.'}])
    assert a != b
    assert a['kind'] == 'ordinal' and 'reference_guard' in a
    with pytest.raises(SemanticError, match='unsupported_passenger_id'):
        passenger_scope({'kind': 'ids', 'values': ['P2']}, event(text), [])


def test_remove_goal_decompositions_merge_and_history_is_preserved():
    c, p = sample('replace_goal')
    alias = deepcopy(p)
    first = alias['events'][0]['data']['target']
    first.update(action='remove', passengers='second passenger')
    alias['events'][1]['data']['target']['passengers'] = 'everyone'
    a, b = compile_case(c, p), compile_case(c, alias)
    assert a['key'] == b['key']
    assert a['state']['history'][0]['replaced_goal']['operation'] == 'remove_passenger'
    assert a['key'] != compile_case(*sample('add_goal'))['key']


def test_conflicting_passenger_scope_is_not_overwritten():
    c, p = sample('replace_goal')
    p['events'][0]['data']['target']['passenger_scope'] = {'kind': 'all'}
    with pytest.raises(SemanticError):
        compile_case(c, p)


def test_question_description_and_typed_intent_merge():
    c, p = sample('changed_pending')
    alias = deepcopy(p)
    last = alias['events'][-1]
    proposal_id = last['id']
    alias['events'].append({'id': 'qextra', 'kind': 'question', 'at': last['at'],
                            'data': {'intent': 'confirm_proposal', 'proposal_id': proposal_id},
                            'evidence': deepcopy(last['evidence'])})
    a = compile_case(c, p)
    b = compile_case(c, alias)
    assert a['key'] == b['key']
    alias['events'][-1]['data'] = {'topic': 'confirm revised cancellation proposal', 'proposal_id': proposal_id}
    assert compile_case(c, alias)['key'] == a['key']


@pytest.mark.parametrize('text', ['Please confirm your user ID.', 'Please confirm permission to search.'])
def test_identity_and_search_cannot_be_bound_as_write_approval(text):
    with pytest.raises(SemanticError, match='scope_conflict'):
        question({'intent': 'confirm_proposal', 'proposal_id': 'p1'}, event(text))


def test_confirmation_requires_a_proposal_and_unknown_intent_abstains():
    with pytest.raises(SemanticError, match='unbound_confirmation'):
        question({'intent': 'confirm_proposal', 'proposal_id': None}, event('Please confirm.'))
    with pytest.raises(SemanticError, match='unsupported_question_intent'):
        question({'intent': 'other', 'proposal_id': None}, event('Which city?'))


@pytest.mark.parametrize('text', ['Please confirm.', 'Do you approve this proposal?',
                                 'Shall I proceed with this cancellation?'])
def test_confirmation_wording_has_one_intent(text):
    assert question({'intent': 'confirm_proposal', 'proposal_id': 'p1'}, event(text)) == {
        'topic': 'confirm_proposal', 'proposal_id': 'p1'}


def test_tool_identifier_must_be_a_passenger_field():
    e = {'at': 0, 'evidence': [{'message_index': 0, 'pointer': '/reservation_id', 'value': 'P2'}]}
    with pytest.raises(SemanticError, match='unsupported_passenger_id'):
        passenger_scope({'kind': 'ids', 'values': ['P2']}, e, [])
    e['evidence'][0]['pointer'] = '/passengers/0/passenger_id'
    assert passenger_scope({'kind': 'ids', 'values': ['P2']}, e, []) == {'kind': 'ids', 'values': ['P2']}


def test_changed_offer_cannot_inherit_old_approval():
    pending = compile_case(*sample('changed_pending'))
    approved = compile_case(*sample('changed_approved'))
    assert not pending['state']['consents']
    assert approved['state']['consents'] and approved['key'] != pending['key']
    with pytest.raises(SemanticError):
        compile_case(*sample('stale_approval'))


def test_v2_prompt_and_key_are_versioned():
    assert 'VOCABULARY v2' in build_request([], slot_schema=VERSION)['system']
    assert compile_case(*sample())['key'] != compile_case(*sample(), version='airline_slots_v1')['key']
