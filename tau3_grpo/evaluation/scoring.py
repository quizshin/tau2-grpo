"""Deterministic benchmark scoring, independent of model serving and training."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

from tau3_grpo.evaluation.metrics import is_benchmark_success, pass_at_k, pass_hat_k


def resolve_ks(trials: int, ks: Sequence[int] | None = None) -> tuple[int, ...]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    values = tuple(sorted(set(ks if ks is not None else (k for k in (1, 2, 4) if k <= trials))))
    if not values or any(k <= 0 or k > trials for k in values):
        raise ValueError("metric ks must be nonempty and within [1, trials]")
    return values


def summarize_trials(
    *,
    planned: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, Any]],
    trials: int,
    ks: Sequence[int] | None = None,
    include_pass_hat: bool = False,
) -> dict[str, Any]:
    """Score the complete planned task set, never silently drop failed trials.

    Metrics are unavailable until every planned trial has a scored trajectory.
    Complete tasks still receive diagnostic per-task metrics in incomplete runs.
    Endpoint failures are not scored as model failures, nor as missing successes.
    """

    metric_ks = resolve_ks(trials, ks)
    if not planned:
        raise ValueError("evaluation needs at least one task")
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for job in planned:
        key = (job["task_id"], job["trial"])
        if key in expected:
            raise ValueError(f"duplicate planned task/trial: {key}")
        if not isinstance(key[0], str) or not isinstance(key[1], int) or not 0 <= key[1] < trials:
            raise ValueError(f"invalid planned task/trial: {key}")
        expected[key] = job
    task_ids = sorted({key[0] for key in expected})
    counts = Counter(key[0] for key in expected)
    if any(count != trials for count in counts.values()):
        raise ValueError("every task must have exactly trials planned attempts")

    seen: set[tuple[str, int]] = set()
    per_task_rewards: dict[str, list[float]] = {task: [] for task in task_ids}
    for rows, scored in ((results, True), (errors, False)):
        for row in rows:
            key = (row["task_id"], row["trial"])
            if key not in expected or key in seen:
                raise ValueError(f"unexpected or duplicate task/trial: {key}")
            if row["seed"] != expected[key]["seed"]:
                raise ValueError(f"trial seed does not match plan: {key}")
            seen.add(key)
            if scored:
                if row.get("termination_reason") == "infrastructure_error":
                    raise ValueError("infrastructure errors belong in errors.jsonl")
                reward = float(row["reward"])
                if not math.isfinite(reward):
                    raise ValueError(f"non-finite reward for {key}")
                per_task_rewards[key[0]].append(reward)

    per_task: dict[str, Any] = {}
    for task_id, rewards in per_task_rewards.items():
        n = len(rewards)
        c = sum(is_benchmark_success(reward) for reward in rewards)
        metrics = None
        if n == trials:
            metrics = {f"pass@{k}": pass_at_k(n, c, k) for k in metric_ks}
            if include_pass_hat:
                metrics.update({f"pass^{k}": pass_hat_k(n, c, k) for k in metric_ks})
        per_task[task_id] = {"completed_trials": n, "successes": c, "metrics": metrics}

    complete = len(results) == len(expected)
    metrics = None
    if complete:
        names = next(iter(per_task.values()))["metrics"]
        metrics = {
            name: sum(item["metrics"][name] for item in per_task.values()) / len(task_ids)
            for name in names
        }
    rewards = [reward for values in per_task_rewards.values() for reward in values]
    return {
        "status": "complete" if complete else "incomplete",
        "metrics_valid": complete,
        "primary_metric_family": "pass@k",
        "metric_ks": list(metric_ks),
        "include_pass_hat": include_pass_hat,
        "success_rule": "1 - 1e-6 <= reward <= 1 + 1e-6 (official tau3 v1.0.1)",
        "incomplete_policy": "withhold aggregate metrics until all planned trials are scored",
        "tasks": len(task_ids),
        "trials_per_task": trials,
        "planned_trajectories": len(expected),
        "completed_trajectories": len(results),
        "failed_trajectories": len(errors),
        "missing_trajectories": len(expected) - len(seen),
        "complete_tasks": sum(item["completed_trials"] == trials for item in per_task.values()),
        "mean_reward": sum(rewards) / len(rewards) if complete else None,
        "solve_rate": sum(is_benchmark_success(r) for r in rewards) / len(rewards) if complete else None,
        "metrics": metrics,
        "per_task": per_task,
    }
