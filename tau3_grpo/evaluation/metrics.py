"""pass@k, pass^k and paired bootstrap.

Milestone D14. Both estimators are deterministic given their inputs: pass@k uses
the unbiased combinatorial estimator (no resampling at all), and the paired
bootstrap seeds its own generator so a report regenerates byte-identically.

Paired, not unpaired: every arm is evaluated on the same task ids, so the
bootstrap resamples *tasks* and keeps both arms' scores for a resampled task
together. That removes task difficulty from the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_CONFIDENCE = 0.95


def _validate_counts(num_samples: int, num_correct: int, k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError(f"num_correct {num_correct} outside [0, {num_samples}]")
    if k > num_samples:
        raise ValueError(f"k={k} exceeds num_samples={num_samples}")


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k: ``1 - C(n-c, k) / C(n, k)`` for one task."""

    _validate_counts(num_samples, num_correct, k)
    if num_samples - num_correct < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - product


def pass_hat_k(num_samples: int, num_correct: int, k: int) -> float:
    """pass^k: ``C(c, k) / C(n, k)``, matching the pinned official benchmark."""

    _validate_counts(num_samples, num_correct, k)
    if num_correct < k:
        return 0.0
    product = 1.0
    for i in range(k):
        product *= (num_correct - i) / (num_samples - i)
    return product


def is_benchmark_success(reward: float) -> bool:
    """Same full-success tolerance as tau2.metrics.agent_metrics.is_successful.

    Legacy training telemetry helpers below intentionally retain their explicit
    reward-threshold API; independent benchmark evaluation uses this predicate.
    """

    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def pass_at_k_from_rewards(rewards: Sequence[float], k: int, *, threshold: float = 0.0) -> float:
    """pass@k for one task given its per-rollout rewards."""

    successes = sum(1 for value in rewards if float(value) > threshold)
    return pass_at_k(len(rewards), successes, k)


def aggregate_pass_at_k(
    per_task_rewards: Mapping[str, Sequence[float]],
    ks: Sequence[int] = (1, 2, 4, 8),
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Mean pass@k across tasks, skipping k larger than a task's rollout count."""

    out: dict[str, float] = {}
    for k in ks:
        values = [
            pass_at_k_from_rewards(rewards, k, threshold=threshold)
            for rewards in per_task_rewards.values()
            if len(rewards) >= k
        ]
        if values:
            out[f"pass@{k}"] = float(np.mean(values))
    return out


@dataclass(frozen=True)
class BootstrapResult:
    """Paired bootstrap outcome for `treatment - baseline`."""

    baseline_mean: float
    treatment_mean: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    resamples: int
    confidence: float
    num_tasks: int
    seed: int

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""

        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_mean": self.baseline_mean,
            "treatment_mean": self.treatment_mean,
            "difference": self.difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "resamples": self.resamples,
            "confidence": self.confidence,
            "num_tasks": self.num_tasks,
            "seed": self.seed,
            "significant": self.significant,
        }


def paired_bootstrap(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap over tasks.

    `baseline[i]` and `treatment[i]` must be the same task. Resampling draws task
    indices with replacement and keeps both arms together.
    """

    base = np.asarray([float(value) for value in baseline], dtype=np.float64)
    treat = np.asarray([float(value) for value in treatment], dtype=np.float64)
    if base.shape != treat.shape:
        raise ValueError(f"paired arrays must align: {base.shape} vs {treat.shape}")
    if base.size == 0:
        raise ValueError("paired bootstrap needs at least one task")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    deltas = treat - base
    observed = float(deltas.mean())

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, base.size, size=(resamples, base.size))
    resampled = deltas[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(resampled, alpha))
    ci_high = float(np.quantile(resampled, 1.0 - alpha))

    # Two-sided p-value by centring the bootstrap distribution on zero.
    centred = resampled - observed
    p_value = float(np.mean(np.abs(centred) >= abs(observed)))

    return BootstrapResult(
        baseline_mean=float(base.mean()),
        treatment_mean=float(treat.mean()),
        difference=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        resamples=resamples,
        confidence=confidence,
        num_tasks=int(base.size),
        seed=seed,
    )


def align_by_task(
    baseline: Mapping[str, float], treatment: Mapping[str, float]
) -> tuple[list[str], list[float], list[float]]:
    """Intersect two task->score maps into paired, task-sorted vectors."""

    shared = sorted(set(baseline) & set(treatment))
    if not shared:
        raise ValueError("no shared task ids between baseline and treatment")
    return shared, [baseline[t] for t in shared], [treatment[t] for t in shared]


def paired_bootstrap_by_task(
    baseline: Mapping[str, float],
    treatment: Mapping[str, float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap over the intersection of two task->score maps."""

    _, base, treat = align_by_task(baseline, treatment)
    return paired_bootstrap(
        base, treat, resamples=resamples, confidence=confidence, seed=seed
    )


def mean_reward(rewards: Sequence[float]) -> float:
    values = [float(value) for value in rewards]
    return float(np.mean(values)) if values else 0.0


def solve_rate(rewards: Sequence[float], *, threshold: float = 0.0) -> float:
    values = [float(value) for value in rewards]
    if not values:
        return 0.0
    return float(np.mean([1.0 if value > threshold else 0.0 for value in values]))
