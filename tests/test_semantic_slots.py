"""Adversarial slot/financial checks and opt-in compatibility."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tau3_grpo.algorithms.anchors.semantic_slots import VERSION, normalize_packet, slots
from tau3_grpo.algorithms.anchors.semantic_state import SemanticError, compile_state
from tau3_grpo.models.semantic_extractor import build_request


def sample(name='approved'):
    cases = json.loads(Path('tests/fixtures/semantic_model_cases_20260914.json').read_text())['cases']
    c = next(c for c in cases if c['id'] == name)
    prefix = build_request(c['messages'])['prefix_sha256']
    packets = json.loads(Path('tests/fixtures/semantic_model_responses_20260914.json').read_text())['packets']
    return c, next(p for p in packets if p['prefix_sha256'] == prefix)


def compile_case(c, p, strict=True):
    return compile_state(c['messages'], p, task_id=c['task_id'], db_hash=c['db_hash'],
                         policy_hash=c['policy_hash'], remaining_turns=c['remaining_turns'],
                         slot_schema=VERSION if strict else None)


def terms(p):
    return next(e for e in p['events'] if e['kind'] == 'proposal')['data']['operations'][0]['terms']


def test_aliases_merge_without_mutating_packet():
    c, p = sample()
    alias = deepcopy(p)
    for e in alias['events']:
        d = e['data']
        if 'target' in d:
            d['target']['booking_id'] = d['target'].pop('reservation_id')
        for op in d.get('operations', []):
            op['target']['booking_id'] = op['target'].pop('reservation_id')
    t = terms(alias)
    t['refund_amount'] = t.pop('quoted_refund') + '.00'
    t['refund_currency'] = t.pop('currency').lower()
    before = deepcopy(alias)
    assert compile_case(c, alias)['key'] == compile_case(c, p)['key']
    assert alias == before
    assert compile_case(c, p, strict=False)['key'] != compile_case(c, p)['key']


@pytest.mark.parametrize('name', ['amount', 'currency', 'card', 'object', 'conditional',
                                 'later_condition', 'changed_pending', 'changed_approved'])
def test_material_differences_remain_separate(name):
    assert compile_case(*sample(name))['key'] != compile_case(*sample())['key']


@pytest.mark.parametrize('alias', ['quoted_refund', 'refund_amount', 'refund'])
def test_wrong_money_rejected_through_every_refund_alias(alias):
    c, p = sample()
    t = terms(p)
    del t['quoted_refund']
    t[alias] = '999'
    with pytest.raises(SemanticError, match='unsupported_money_role'):
        compile_case(c, p)


def test_same_amount_wrong_financial_role_rejected():
    c, p = sample()
    t = terms(p)
    t['charge_amount'] = t.pop('quoted_refund')
    with pytest.raises(SemanticError, match='unsupported_money_role'):
        compile_case(c, p)


@pytest.mark.parametrize('value,is_terms', [
    ({'booking_id': 'A', 'reservation_id': 'B'}, False),
    ({'refund_currency': 'USD', 'charge_currency': 'EUR'}, True),
    ({'refund_amount': '100', 'quoted_refund': '200', 'currency': 'USD'}, True),
])
def test_conflicting_aliases_abstain(value, is_terms):
    with pytest.raises(SemanticError, match='conflicting_slot_alias'):
        slots(value, terms=is_terms)


@pytest.mark.parametrize('value', [{'price': '100'}, {'money': {'amount': '100'}},
                                  {'refund_amount': '100'}, {'quoted_charge': True, 'currency': 'USD'},
                                  {'quoted_refund': 'NaN', 'currency': 'USD'}])
def test_unknown_or_untyped_money_cannot_bypass_checks(value):
    with pytest.raises(SemanticError):
        slots(value, terms=True)


def test_unknown_conditions_are_not_dropped():
    _, p = sample('conditional')
    e = next(e for e in p['events'] if e['kind'] == 'constraint')
    e['data']['predicate']['must_arrive_before'] = '10:00'
    with pytest.raises(SemanticError, match='unsupported_constraint_predicate'):
        normalize_packet(p, VERSION)


def test_previous_quote_cannot_support_current_amount():
    c, p = sample('changed_approved')
    proposals = [e for e in p['events'] if e['kind'] == 'proposal']
    proposals[-1]['data']['operations'][0]['terms']['quoted_refund'] = '100'
    proposals[-1]['evidence'].extend(deepcopy(proposals[0]['evidence']))
    with pytest.raises(SemanticError, match='unsupported_money_role'):
        compile_case(c, p)


def test_strict_prompt_is_opt_in():
    assert 'STRICT AIRLINE' not in build_request([])['system']
    assert 'STRICT AIRLINE' in build_request([], slot_schema=VERSION)['system']


def test_free_does_not_prove_usd_currency():
    c, p = sample('partial')
    compile_case(c, p, strict=False)  # Historical fixture contract remains unchanged.
    with pytest.raises(SemanticError, match='unsupported_money_role'):
        compile_case(c, p)


def test_omitting_all_financial_terms_cannot_evade_amount_check():
    c, p = sample()
    terms(p).clear()
    with pytest.raises(SemanticError, match='unrepresented_communicated_money'):
        compile_case(c, p)


def test_invented_payment_id_is_not_supported_by_full_message_span():
    c, p = sample()
    terms(p)['refund_payment_id'] = 'card_9'
    with pytest.raises(SemanticError, match='unsupported_identity_slot'):
        compile_case(c, p)


def test_invented_goal_identifier_is_rejected():
    c, p = sample()
    p['events'][0]['data']['target']['booking_id'] = 'ZZZ999'
    del p['events'][0]['data']['target']['reservation_id']
    with pytest.raises(SemanticError, match='unsupported_identity_slot'):
        compile_case(c, p)
