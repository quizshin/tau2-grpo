import copy

import pytest

from tau3_grpo.analysis.rubric_pilot import Budget, flash_usage_estimate


def test_actual_baseline_does_not_erase_estimates_and_future_cost_is_added(tmp_path):
    budget = Budget(tmp_path / 'budget.json')
    row = budget.reserve('old', 'system', {}, 128)
    budget.settle(row, {'usage': {'prompt_tokens': 10, 'completion_tokens': 10}}, 'received')
    old = copy.deepcopy(budget.state['calls'])
    budget.reconcile(13.93, source='user_confirmed')
    assert budget.state['calls'] == old
    assert budget.accounted == pytest.approx(13.93)
    new = budget.reserve('new', 'system', {}, 128)
    assert budget.accounted == pytest.approx(13.93 + new['reserved_cny'])
    restored = Budget(tmp_path / 'budget.json')
    assert restored.accounted == budget.accounted
    restored.state['calls'][0]['accounted_cny'] = 0
    with pytest.raises(ValueError, match='history changed'):
        _ = restored.accounted


def test_weekend_cache_prices_and_weekday_boundary():
    usage = {'prompt_tokens': 1000, 'completion_tokens': 100,
             'prompt_cache_hit_tokens': 600, 'prompt_cache_miss_tokens': 400}
    sat = '2026-09-26T02:00:00+00:00'
    cost, details = flash_usage_estimate(usage, sat, sat)
    assert cost == pytest.approx((600 * .02 + 400 * 1 + 100 * 4) / 1e6)
    assert details['time_band'] == 'off_peak'
    cost, details = flash_usage_estimate(usage, '2026-09-25T00:59:00+00:00', '2026-09-25T01:01:00+00:00')
    assert details['time_band'] == 'peak'
    assert cost == pytest.approx((600 * .04 + 400 * 2 + 100 * 8) / 1e6)


def test_missing_or_inconsistent_cache_usage_is_not_free():
    sat = '2026-09-26T02:00:00+00:00'
    usage = {'prompt_tokens': 1000, 'completion_tokens': 100,
             'prompt_cache_hit_tokens': 600, 'prompt_cache_miss_tokens': 900}
    cost, details = flash_usage_estimate(usage, sat, sat)
    assert cost == pytest.approx(.0014)
    assert details['cache_basis'] == 'uncached_upper_estimate'


def test_actual_baseline_still_enforces_total_cap(tmp_path):
    budget = Budget(tmp_path / 'budget.json')
    budget.reconcile(99.9999, source='user_confirmed')
    with pytest.raises(ValueError, match='Budget cap'):
        budget.reserve('blocked', 'system', {}, 128)
    assert not budget.state['calls']
