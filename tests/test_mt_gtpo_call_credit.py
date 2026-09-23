"""Independent numerical checks and strict call/response identity contracts."""
import json
from copy import deepcopy
from statistics import mean, pstdev

import numpy as np
import pytest

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
from tau3_grpo.algorithms.mt_gtpo_call_credit import compute_mt_gtpo_call_credit
from tau3_grpo.data.call_attribution import retained_call_attribution
from tau3_grpo.data.call_credit import call_credit_inputs
from tau3_grpo.evaluation.process_reward import reward_settings


def numeric_batch(q=None):
    q = q or [[[-.4, .6, .6]], [[-.4]]]
    return dict(outcomes=[0., 0.], uids=['g', 'g'],
                turn_rewards=[[sum(x) for x in row] for row in q],
                turn_spans=[[[0, 9]], [[0, 9]]], response_mask=np.ones((2, 11), dtype=int),
                call_rewards=q, call_spans=[[[[1+2*j, 3+2*j] for j in range(len(row[0]))]] for row in q],
                eligible_turns=[[True], [True]])


def batch(q=None):
    value = numeric_batch(q)
    value['response_mask'][:, 9:] = 0
    return value


def baseline(value):
    return compute_mt_gtpo(**{k: v for k, v in value.items()
                             if k not in {'call_rewards', 'call_spans', 'eligible_turns'}})


def test_mixed_calls_match_independent_formula_and_preserve_text_returns():
    value = batch()
    original = deepcopy(value)
    old, old_returns, _ = baseline(value)
    adv, returns, details = compute_mt_gtpo_call_credit(**value)
    center = mean([mean(row[0]) for row in value['call_rewards']])
    scale = pstdev([sum(row[0]) for row in value['call_rewards']]) + 1e-6
    for i, row in enumerate(value['call_rewards']):
        for j, reward in enumerate(row[0]):
            a, b = value['call_spans'][i][0][j]
            np.testing.assert_allclose(adv[i, a:b], (reward-center)/scale, atol=1e-12)
    assert adv[0, 1] < 0 < adv[0, 3]
    assert adv[0, 0] == old[0, 0] and adv[0, 8] == old[0, 8]
    assert not adv[:, 9:].any()
    np.testing.assert_array_equal(returns, old_returns)
    np.testing.assert_array_equal(value['response_mask'], original['response_mask'])
    assert value['call_rewards'] == original['call_rewards']
    assert details['call_credit_stats']['applied_calls'] == 4


def test_single_call_equivalence_and_future_credit():
    value = batch([[[.6]], [[-.4]]])
    value.update(outcomes=[1., 0.], turn_rewards=[[.6, .3], [-.4, .1]],
                 turn_spans=[[[0, 9], [9, 10]], [[0, 9], [9, 10]]],
                 call_rewards=[[[.6], []], [[-.4], []]],
                 call_spans=[[[[1, 3]], None], [[[1, 3]], None]],
                 eligible_turns=[[True, False], [True, False]])
    value['response_mask'][:, 9] = 1
    old, _, _ = baseline(value)
    adv, _, details = compute_mt_gtpo_call_credit(**value)
    np.testing.assert_allclose(adv, old, atol=1e-12)
    assert details['call_credit'][0][0]['future'] == pytest.approx(.9*.3 + .9**2)


@pytest.mark.parametrize('kind', ['ineligible', 'zero_variance', 'support', 'padding', 'empty'])
def test_fallbacks_and_padding(kind):
    value = batch()
    if kind == 'ineligible':
        value['eligible_turns'] = [[False], [False]]
        value['call_spans'] = [[None], [None]]
    elif kind == 'zero_variance':
        value = batch([[[.6]], [[.6]]])
    elif kind == 'support':
        value['uids'] = ['g', 'h']
    elif kind == 'padding':
        value['response_mask'][1] = 0
    else:
        value['response_mask'][:] = 0
    adv, _, _ = compute_mt_gtpo_call_credit(**value)
    np.testing.assert_array_equal(adv, baseline(value)[0])


@pytest.mark.parametrize('kind', ['sum', 'overlap', 'outside', 'nan', 'eligibility'])
def test_bad_numeric_contracts_fail(kind):
    value = batch()
    if kind == 'sum':
        value['turn_rewards'][0][0] = 9
    elif kind == 'overlap':
        value['call_spans'][0][0][1] = [2, 4]
    elif kind == 'outside':
        value['call_spans'][0][0][2] = [8, 10]
    elif kind == 'nan':
        value['call_rewards'][0][0][0] = float('nan')
    else:
        value['eligible_turns'][0][0] = 1
    with pytest.raises(ValueError):
        compute_mt_gtpo_call_credit(**value)


def test_missing_later_turn_and_call_support_are_not_filled_with_zeros():
    value = batch()
    value['turn_rewards'][1] = [0.]
    value['call_rewards'][1] = [[]]
    value['call_spans'][1] = [None]
    value['eligible_turns'][1] = [False]
    adv, _, details = compute_mt_gtpo_call_credit(**value)
    np.testing.assert_array_equal(adv, baseline(value)[0])
    assert details['call_credit'][0][0]['fallback_reason'] == 'insufficient_call_support'


def identity_batch():
    value = batch()
    ids = np.tile(np.arange(11), (2, 1))
    processes, facts = [], []
    for i in range(2):
        spans = value['call_spans'][i][0]
        calls = [dict(id=f'{i}/{j}', name='calculate', arguments={'expression': '1+1'},
                      error=q < 0, db_hash_after='db', reward=q)
                 for j, q in enumerate(value['call_rewards'][i][0])]
        turn = dict(turn_index=0, token_span=[0, 9], tool_calls=calls,
                    reward=value['turn_rewards'][i][0])
        process = dict(schema='tau3_process_reward_v1', trajectory_id=f't{i}', settings=reward_settings(), turn_records=[turn],
                       turn_spans=[[0, 9]], turn_rewards=value['turn_rewards'][i])
        attr = dict(schema='tau3_call_attribution_v1', emitted_span=[0, 9],
                    coordinate_system='response_token_offset_half_open',
                    emitted_token_ids=ids[i, :9].tolist(), scan_status='exact', parse_status='parsed',
                    blocks=[dict(block_index=j, emitted_span=s, closure='closed') for j, s in enumerate(spans)],
                    calls=[dict(call_id=c['id'], block_index=j, alignment='exact',
                                execution_status='executed', name=c['name'], error=c['error'], db_hash_after='db')
                           for j, c in enumerate(calls)])
        attr = retained_call_attribution(attr, ids[i, :9].tolist(), [1]*9)
        fact = dict(schema='tau3_trajectory_facts_v1', identity=dict(trajectory_id=f't{i}', sample_group_uid='g'),
                    tokens=dict(response_ids=ids[i, :9].tolist(), response_mask=[1]*9),
                    turns=[{**deepcopy(turn), 'call_attribution': attr}])
        processes.append(process)
        facts.append(fact)
    return value, ids, processes, facts


def test_join_uses_stable_ids_for_identical_named_calls():
    value, ids, processes, facts = identity_batch()
    inputs, receipts = call_credit_inputs(processes, list(map(json.dumps, facts)), value['uids'], ids, value['response_mask'])
    assert inputs['call_rewards'] == value['call_rewards']
    assert inputs['call_spans'] == value['call_spans']
    assert receipts[0][0]['call_ids'] == ['0/0', '0/1', '0/2']


@pytest.mark.parametrize('kind', ['uid', 'trajectory', 'ids', 'mask', 'missing', 'duplicate', 'reward_id', 'receipt', 'coordinates', 'mean'])
def test_join_rejects_corrupted_identity_and_metadata(kind):
    value, ids, processes, facts = identity_batch()
    f = facts[0]
    if kind == 'uid':
        f['identity']['sample_group_uid'] = 'other'
    elif kind == 'trajectory':
        f['identity']['trajectory_id'] = 'other'
    elif kind == 'ids':
        ids[0, 0] = 123
    elif kind == 'mask':
        value['response_mask'][0, 10] = 1
    elif kind == 'missing':
        del f['turns'][0]['call_attribution']
    elif kind == 'duplicate':
        f['turns'][0]['tool_calls'][1]['id'] = '0/0'
    elif kind == 'reward_id':
        processes[0]['turn_records'][0]['tool_calls'][0]['id'] = 'other'
    elif kind == 'receipt':
        f['turns'][0]['call_attribution']['calls'][0]['error'] = False
    elif kind == 'coordinates':
        f['turns'][0]['call_attribution']['coordinate_system'] = 'characters'
    else:
        processes[0]['settings'] = reward_settings({'mode': 'paper', 'version': 'paper_env_split_v4', 'paper_options': {'aggregation': 'mean'}})
    with pytest.raises(ValueError):
        call_credit_inputs(processes, list(map(json.dumps, facts)), value['uids'], ids, value['response_mask'])


def test_ambiguous_alignment_preserves_rewards_but_has_no_trainable_spans():
    value, ids, processes, facts = identity_batch()
    attr = facts[0]['turns'][0]['call_attribution']
    attr['calls'][0]['alignment'] = 'ambiguous'
    facts[0]['turns'][0]['call_attribution'] = retained_call_attribution(attr, ids[0, :9].tolist(), [1]*9)
    inputs, _ = call_credit_inputs(processes, list(map(json.dumps, facts)), value['uids'], ids, value['response_mask'])
    assert inputs['call_rewards'][0] == value['call_rewards'][0]
    assert inputs['call_spans'][0] == [None]
    assert inputs['eligible_turns'][0] == [False]
