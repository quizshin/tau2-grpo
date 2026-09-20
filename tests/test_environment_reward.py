from copy import deepcopy

import pytest

from tau3_grpo.evaluation.environment_reward import typed_json
from tau3_grpo.evaluation.process_reward import reward_settings, score_turns

ENV = {"mode": "paper", "version": "paper_env_v2"}
FLIGHT = {"name": "update_reservation_flights", "arguments": {
    "reservation_id": "00123", "cabin": "economy", "payment_id": "P",
    "flights": [{"flight_number": "F1", "date": "2026-09-18"}]}}
READ = {"name": "get_reservation_details", "arguments": {"reservation_id": "A"}}


def event(gold, observation='{"status":"active"}', **extra):
    return {**deepcopy(gold), "observation": observation, "error": False, **extra}


def score(events, gold=None, config=None):
    records = [{"schema": "tau3_turn_v1", "turn_index": k, "token_span": [k, k + 1],
                "tool_calls": [e]} for k, e in enumerate(events)]
    return score_turns(records, gold or [], ["DB"], config or ENV, official_outcome=0)


def tiers(result):
    return [t['tool_calls'][0]['reward_type'] for t in result['turn_records']]


def test_tool_ignored_fields_match_gold_but_legacy_paper_stays_soft():
    actual = event(FLIGHT)
    actual['arguments']['flights'][0].update(price=999, origin='SFO')
    result = score([actual], [FLIGHT])
    assert tiers(result) == ['gold_exact']
    assert result['turn_rewards'] == [1]
    assert tiers(score([actual], [FLIGHT], {'mode':'paper', 'version':'paper_v1'})) == ['soft_match']
    assert score_turns(result['turn_records'], result['golden_actions'], ['DB'], result['settings'],
                       official_outcome=0) == result


def test_identity_and_flight_order_are_preserved():
    actual = event(FLIGHT)
    actual['arguments']['reservation_id'] = '123'
    assert tiers(score([actual], [FLIGHT])) != ['gold_exact']
    gold = deepcopy(FLIGHT)
    gold['arguments']['flights'].append({'flight_number':'F2', 'date':'2026-09-19'})
    actual = event(gold)
    actual['arguments']['flights'].reverse()
    assert tiers(score([actual], [gold])) != ['gold_exact']
    assert typed_json({'n':True}) != typed_json({'n':1})
    assert typed_json({'n':'001'}) != typed_json({'n':1})
    assert typed_json({'empty':None}) != typed_json({})


def test_changed_read_is_neutral_and_never_renews_consumed_gold():
    calls = [event(READ), event(READ, '{"status":"cancelled"}'), event(READ, '{ "status": "cancelled" }')]
    original = deepcopy(calls)
    result = score(calls, [READ])
    assert tiers(result) == ['gold_exact', 'read_only', 'duplicate']
    assert result['turn_rewards'] == [1, 0, -.2]
    assert calls == original
    assert tiers(score(calls, [READ], {'mode':'paper','version':'paper_v1'})) == ['gold_exact','duplicate','duplicate']
    assert tiers(score(calls, [READ, READ])) == ['gold_exact','gold_exact','duplicate']


@pytest.mark.parametrize('missing', [None, '', '   '])
def test_unavailable_repeated_read_is_unknown(missing):
    assert tiers(score([event(READ), event(READ, missing)], [READ])) == ['gold_exact','unknown']
    assert tiers(score([event(READ, missing), event(READ)], [READ])) == ['gold_exact','unknown']


def test_explicitly_truncated_response_cannot_certify_duplicate():
    assert tiers(score([event(READ), event(READ, observation_truncated=True)])) == ['read_only','unknown']


def test_unknown_read_evidence_blocks_calibration_alignment():
    from tau3_grpo.analysis.calibrate_paper_rewards import (
        DEFAULT_IRC,
        alignment_issues,
        rescore,
        summarize,
    )
    from tau3_grpo.evaluation.process_reward import DEFAULT_WEIGHTS

    process = score([event(READ), event(READ, None)], [READ])
    observations = rescore([{'processes':[process,process], 'outcomes':[0,1], 'uids':['g','g'],
                             'task_ids':['a','a'], 'algorithm':{'gamma':.9,'lambda_outcome':.3,'eps':1e-6,'min_group_size':2}}],
                           reward_settings(ENV))
    config = {**DEFAULT_IRC, 'intended_signs':{}, 'fixed_weights':{k:0 for k in DEFAULT_WEIGHTS}}
    assert 'unknown_tool_tier_present' in alignment_issues(summarize(observations, config), config)


def test_invalid_error_does_not_consume_gold_or_history():
    invalid = event(FLIGHT, error=True)
    invalid['arguments']['flights'] = 'invalid'
    result = score([invalid, event(FLIGHT)], [FLIGHT])
    assert tiers(result) == ['error','gold_exact']
    invalid['error'] = False
    assert tiers(score([invalid, event(FLIGHT)], [FLIGHT])) == ['unknown','gold_exact']


def test_write_duplicates_ignore_changed_response_and_extra_fields():
    a, b = event(FLIGHT, 'first'), event(FLIGHT, 'second')
    b['arguments']['flights'][0]['price'] = 999
    result = score([a,b], [FLIGHT])
    assert tiers(result) == ['gold_exact','duplicate']


def test_non_gold_read_changed_response_does_not_repeat_soft_reward():
    gold = {'name':'search_direct_flight', 'arguments':{'origin':'A','destination':'B','date':'D'}}
    a = event(gold, 'first')
    a['arguments']['date'] = 'D2'
    b = {**deepcopy(a),'observation':'changed'}
    assert tiers(score([a,b], [gold])) == ['soft_match','read_only']


@pytest.mark.parametrize('version,options', [
    ('paper_v1', {'normalization':'execution_v1'}),
    ('paper_env_v2', {'normalization':'recursive_v1'}),
    ('paper_env_v2', {'duplicate_scope':'successful_call'}),
])
def test_versions_cannot_silently_change_matching(version, options):
    with pytest.raises(ValueError, match='unsupported paper option'):
        reward_settings({'mode':'paper','version':version,'paper_options':options})
