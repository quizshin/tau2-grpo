"""Residual credit preservation, length effects and denominator stress cases."""
from copy import deepcopy
from statistics import mean, pstdev

import numpy as np
import pytest
from test_mt_gtpo_call_credit import baseline, batch

from tau3_grpo.algorithms.mt_gtpo_call_credit import compute_mt_gtpo_call_credit


def residual(value):
    return compute_mt_gtpo_call_credit(**value, credit_mode='call_residual_v1')


def test_residual_matches_oracle_and_preserves_noncall_tokens_and_returns():
    value = batch()
    original = deepcopy(value)
    old, old_returns, _ = baseline(value)
    adv, returns, details = residual(value)
    scale = pstdev([sum(row[0]) for row in value['call_rewards']]) + 1e-6
    for i, row in enumerate(value['call_rewards']):
        qs = row[0]
        for j, q in enumerate(qs):
            a, b = value['call_spans'][i][0][j]
            np.testing.assert_allclose(adv[i, a:b], old[i, a] + (q-mean(qs))/scale, atol=1e-12)
    assert adv[0, 1] < 0 < adv[0, 3]
    np.testing.assert_array_equal(adv[1], old[1])
    np.testing.assert_array_equal(adv[0, [0, 7, 8, 9, 10]], old[0, [0, 7, 8, 9, 10]])
    np.testing.assert_array_equal(returns, old_returns)
    np.testing.assert_array_equal(value['response_mask'], original['response_mask'])
    assert value['call_rewards'] == original['call_rewards']
    assert details['call_credit_stats']['applied_calls'] == 3
    assert details['call_credit_stats']['call_mean_drift_abs_max'] < 1e-12
    assert details['call_credit'][0][0]['baseline_scope'] == 'within_turn'
    assert details['call_credit'][1][0]['fallback_reason'] == 'single_call'


def test_equal_calls_are_bitwise_unchanged_despite_future_and_peer_difference():
    value = batch([[[1., 1.]], [[1.]]])
    value.update(outcomes=[1., 1.], turn_rewards=[[2., 0.], [1., 1.]],
                 turn_spans=[[[0, 9], [9, 10]], [[0, 9], [9, 10]]],
                 call_rewards=[[[1., 1.], []], [[1.], [1.]]],
                 call_spans=[[[[1, 3], [3, 5]], None], [[[1, 3]], [[9, 10]]]],
                 eligible_turns=[[True, False], [True, True]])
    value['response_mask'][:, 9] = 1
    old, returns, _ = baseline(value)
    adv, result_returns, details = residual(value)
    np.testing.assert_array_equal(adv, old)
    np.testing.assert_array_equal(result_returns, returns)
    assert details['call_credit_stats']['changed_tokens'] == 0
    # Old candidate differs; retaining equal-call turns is not inherited behavior.
    local, _, _ = compute_mt_gtpo_call_credit(**value)
    assert np.any(local != old)


def test_unequal_lengths_preserve_call_mean_but_not_token_weighted_sum():
    value = batch([[[1., -1.]], [[1.]]])
    value['call_spans'][0][0] = [[1, 2], [2, 8]]
    old, _, _ = baseline(value)
    adv, _, details = residual(value)
    delta = adv-old
    assert delta[0, 1] == pytest.approx(-delta[0, 2])
    assert delta.sum() != pytest.approx(0.)
    assert details['call_credit_stats']['token_weighted_correction_sum'] == pytest.approx(delta.sum())
    assert details['call_credit_stats']['call_mean_drift_abs_max'] < 1e-12


@pytest.mark.parametrize('kind', ['ineligible', 'zero_variance', 'turn_support', 'call_support', 'padding', 'empty'])
def test_residual_preserves_existing_guards(kind):
    value = batch()
    if kind == 'ineligible':
        value['eligible_turns'] = [[False], [False]]
        value['call_spans'] = [[None], [None]]
    elif kind == 'zero_variance':
        value = batch([[[1., -1.]], [[2., -2.]]])
    elif kind == 'turn_support':
        value['uids'] = ['g', 'h']
    elif kind == 'call_support':
        value['turn_rewards'][1] = [0.]
        value['call_rewards'][1] = [[]]
        value['call_spans'][1] = [None]
        value['eligible_turns'][1] = [False]
    elif kind == 'padding':
        value['response_mask'][1] = 0
    else:
        value['response_mask'][:] = 0
    np.testing.assert_array_equal(residual(value)[0], baseline(value)[0])


def test_near_zero_nonzero_variance_is_visible_not_silently_clipped():
    value = batch([[[1., -1.]], [[1e-8]]])
    adv, _, details = residual(value)
    assert np.isfinite(adv).all()
    stats = details['call_credit_stats']
    assert stats['near_zero_scale_turns'] == 1
    assert stats['correction_abs_max'] > 900_000
    assert stats['applied_return_std_min'] == pytest.approx(5e-9)


def test_residual_retains_exact_span_validation():
    value = batch()
    value['call_spans'][0][0][1] = [2, 4]
    with pytest.raises(ValueError, match='overlapping'):
        residual(value)
