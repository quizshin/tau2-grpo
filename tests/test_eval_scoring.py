"""Independent benchmark metric semantics and completeness checks."""

import pytest

from tau3_grpo.evaluation.scoring import resolve_ks, summarize_trials


def plan(tasks=("a", "b"), trials=4):
    return [{"task_id": task, "trial": i, "seed": 42 + i} for task in tasks for i in range(trials)]


def score(rewards, **kwargs):
    jobs = plan(tasks=tuple(rewards))
    rows = [dict(job, reward=rewards[job["task_id"]][job["trial"]]) for job in jobs]
    return summarize_trials(planned=jobs, results=rows, errors=[], trials=4, **kwargs)


def test_task_macro_average_and_optional_consistency():
    result = score({"a": [1, 1, 0, 0], "b": [0, 0, 0, 0]}, include_pass_hat=True)
    assert result["metrics"] == pytest.approx({
        "pass@1": 0.25, "pass@2": 5 / 12, "pass@4": 0.5,
        "pass^1": 0.25, "pass^2": 1 / 12, "pass^4": 0,
    })
    assert result["metrics_valid"]
    assert result["solve_rate"] == result["metrics"]["pass@1"]
    assert result["per_task"]["a"]["successes"] == 2


def test_partial_rewards_are_not_official_successes():
    result = score({"a": [0.5, 1 - 0.5e-6, 1 + 0.5e-6, 1 - 2e-6]})
    assert result["metrics"]["pass@1"] == 0.5
    assert "pass^1" not in result["metrics"]


def test_missing_or_failed_trials_never_shrink_denominator():
    jobs = plan()
    result = summarize_trials(
        planned=jobs, results=[dict(job, reward=1) for job in jobs[:4]],
        errors=[dict(jobs[4], error="endpoint unavailable")], trials=4,
    )
    assert result["metrics"] is None
    assert result["solve_rate"] is None
    assert not result["metrics_valid"]
    assert result["tasks"] == 2
    assert result["complete_tasks"] == 1
    assert result["missing_trajectories"] == 3
    assert result["failed_trajectories"] == 1
    assert result["per_task"]["a"]["metrics"]["pass@4"] == 1
    assert result["per_task"]["b"]["metrics"] is None


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected", "seed", "nan", "infra"])
def test_invalid_evidence_is_rejected(mutation):
    jobs = plan(tasks=("a",), trials=1)
    rows = [dict(jobs[0], reward=1)]
    if mutation == "duplicate":
        rows *= 2
    elif mutation == "unexpected":
        rows[0]["task_id"] = "unknown"
    elif mutation == "seed":
        rows[0]["seed"] = 99
    elif mutation == "nan":
        rows[0]["reward"] = float("nan")
    else:
        rows[0]["termination_reason"] = "infrastructure_error"
    with pytest.raises(ValueError):
        summarize_trials(planned=jobs, results=rows, errors=[], trials=1)


def test_duplicate_error_and_success_is_rejected():
    jobs = plan(tasks=("a",), trials=1)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_trials(planned=jobs, results=[dict(jobs[0], reward=1)], errors=jobs, trials=1)


@pytest.mark.parametrize("jobs", [[], plan()[:-1], plan() + plan()])
def test_invalid_plan_is_rejected(jobs):
    with pytest.raises(ValueError):
        summarize_trials(planned=jobs, results=[], errors=[], trials=4)


def test_default_ks_respect_trial_budget():
    assert resolve_ks(1) == (1,)
    assert resolve_ks(3) == (1, 2)
    assert resolve_ks(8) == (1, 2, 4)
    assert resolve_ks(8, (8, 2, 2)) == (2, 8)
    for ks in ([], [0], [5]):
        with pytest.raises(ValueError):
            resolve_ks(4, ks)


@pytest.mark.tau3
def test_success_and_consistency_match_pinned_official_implementation(requires_tau2):
    from tau2.metrics.agent_metrics import is_successful
    from tau2.metrics.agent_metrics import pass_hat_k as official_pass_hat_k

    from tau3_grpo.evaluation.metrics import is_benchmark_success, pass_hat_k

    for reward in (0, 0.5, 1 - 2e-6, 1 - 1e-6, 1, 1 + 1e-6, 1 + 2e-6):
        assert is_benchmark_success(reward) == is_successful(reward)
    for n in range(1, 9):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert pass_hat_k(n, c, k) == pytest.approx(official_pass_hat_k(n, c, k))
