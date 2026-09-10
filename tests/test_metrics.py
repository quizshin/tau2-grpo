"""pass@k and paired bootstrap."""

from __future__ import annotations

import pytest

from tau3_grpo.evaluation.metrics import (
    aggregate_pass_at_k,
    align_by_task,
    paired_bootstrap,
    paired_bootstrap_by_task,
    pass_at_k,
    pass_at_k_from_rewards,
    solve_rate,
)


def test_pass_at_1_equals_success_fraction():
    assert pass_at_k(8, 2, 1) == pytest.approx(0.25)


def test_pass_at_k_is_one_when_all_correct():
    assert pass_at_k(8, 8, 4) == 1.0


def test_pass_at_k_is_zero_when_none_correct():
    assert pass_at_k(8, 0, 4) == 0.0


def test_pass_at_k_known_value():
    # n=4, c=1, k=2: 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)


def test_pass_at_k_is_monotonic_in_k():
    values = [pass_at_k(8, 2, k) for k in (1, 2, 4, 8)]
    assert values == sorted(values)


def test_pass_at_k_one_when_failures_below_k():
    assert pass_at_k(8, 6, 4) == 1.0


@pytest.mark.parametrize(
    "n,c,k",
    [(8, 2, 0), (0, 0, 1), (8, 9, 2), (8, -1, 2), (4, 1, 5)],
)
def test_pass_at_k_rejects_bad_input(n, c, k):
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_pass_at_k_from_rewards_uses_threshold():
    rewards = [1.0, 0.0, 0.0, 0.0]
    assert pass_at_k_from_rewards(rewards, 1) == pytest.approx(0.25)
    assert pass_at_k_from_rewards([0.5] * 4, 1, threshold=0.6) == 0.0


def test_aggregate_pass_at_k_skips_k_above_rollouts():
    per_task = {"a": [1.0, 0.0], "b": [0.0, 0.0]}
    out = aggregate_pass_at_k(per_task, ks=(1, 2, 4))
    assert "pass@1" in out
    assert "pass@2" in out
    assert "pass@4" not in out


def test_aggregate_pass_at_k_averages_across_tasks():
    per_task = {"a": [1.0, 1.0], "b": [0.0, 0.0]}
    out = aggregate_pass_at_k(per_task, ks=(1,))
    assert out["pass@1"] == pytest.approx(0.5)


def test_solve_rate():
    assert solve_rate([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25)
    assert solve_rate([]) == 0.0


def test_bootstrap_is_deterministic_for_a_seed():
    baseline = [0.0, 1.0, 0.0, 1.0, 0.0]
    treatment = [1.0, 1.0, 0.0, 1.0, 1.0]
    a = paired_bootstrap(baseline, treatment, resamples=500, seed=42)
    b = paired_bootstrap(baseline, treatment, resamples=500, seed=42)
    assert a.to_dict() == b.to_dict()


def test_bootstrap_seed_changes_interval():
    baseline = [0.0, 1.0, 0.0, 1.0, 0.0]
    treatment = [1.0, 1.0, 0.0, 1.0, 1.0]
    a = paired_bootstrap(baseline, treatment, resamples=500, seed=42)
    b = paired_bootstrap(baseline, treatment, resamples=500, seed=43)
    # Percentile endpoints are discrete for a five-task sample and may happen
    # to coincide; the seeded resampling should still affect some statistic.
    assert (a.ci_low, a.ci_high, a.p_value) != (b.ci_low, b.ci_high, b.p_value)
    assert a.difference == pytest.approx(b.difference)


def test_bootstrap_reports_observed_difference():
    result = paired_bootstrap([0.0, 0.0], [1.0, 1.0], resamples=200, seed=42)
    assert result.difference == pytest.approx(1.0)
    assert result.baseline_mean == 0.0
    assert result.treatment_mean == 1.0


def test_identical_arms_give_zero_difference_and_no_significance():
    scores = [0.0, 1.0, 0.5]
    result = paired_bootstrap(scores, scores, resamples=500, seed=42)
    assert result.difference == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert not result.significant


def test_large_consistent_improvement_is_significant():
    baseline = [0.0] * 30
    treatment = [1.0] * 30
    result = paired_bootstrap(baseline, treatment, resamples=1000, seed=42)
    assert result.significant
    assert result.ci_low > 0.0


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="must align"):
        paired_bootstrap([0.0, 1.0], [1.0])


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError, match="at least one task"):
        paired_bootstrap([], [])


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_rejects_bad_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap([0.0], [1.0], confidence=confidence)


def test_align_by_task_uses_intersection_sorted():
    tasks, base, treat = align_by_task(
        {"b": 1.0, "a": 0.0, "c": 1.0}, {"a": 1.0, "b": 0.0}
    )
    assert tasks == ["a", "b"]
    assert base == [0.0, 1.0]
    assert treat == [1.0, 0.0]


def test_align_by_task_requires_overlap():
    with pytest.raises(ValueError, match="no shared task ids"):
        align_by_task({"a": 1.0}, {"b": 1.0})


def test_paired_bootstrap_by_task_pairs_correctly():
    result = paired_bootstrap_by_task(
        {"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}, resamples=200, seed=42
    )
    assert result.num_tasks == 2
    assert result.difference == pytest.approx(1.0)
