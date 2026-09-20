"""Dynamic Filtering (fixed-rollout main configuration).

Milestone D4–D5. The main configuration is **fixed rollout**: every update draws
`groups_per_update` uids at `group_size=8` rollouts each, detects degenerate
groups, and zeroes their response mask. It never resamples to refill a batch and
never carries a group across updates, so the optimizer step count and the token
budget stay fixed and comparable across E0/E1/E2/E3.

A group is:

- ``all_zero``     : every rollout scored 0 -> no gradient signal, drop.
- ``all_one``      : every rollout scored the max -> no contrast, drop.
- ``informative``  : mixed rewards -> keep.

Reported per update:

- ``candidate_groups``  : groups drawn.
- ``effective_groups``  : groups kept.
- ``d_bar``             : fraction of groups dropped.
- ``inverse_one_minus_d_bar`` : 1/(1-d_bar), the variance inflation factor.

``fixed_informative`` is the explicitly optional alternative that keeps sampling
until a target number of informative groups is reached. Its cap is **off by
default** and must be switched on deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import numpy as np

DEFAULT_GROUP_SIZE = 8
ALL_ONE_TOLERANCE = 1e-9


class GroupKind(str, Enum):
    ALL_ZERO = "all_zero"
    ALL_ONE = "all_one"
    INFORMATIVE = "informative"
    INCOMPLETE = "incomplete"


class FilterMode(str, Enum):
    FIXED_ROLLOUT = "fixed_rollout"
    FIXED_INFORMATIVE = "fixed_informative"


@dataclass
class GroupVerdict:
    uid: str
    kind: GroupKind
    size: int
    rewards: tuple[float, ...]
    indices: tuple[int, ...]

    @property
    def keep(self) -> bool:
        return self.kind is GroupKind.INFORMATIVE


@dataclass
class FilterStats:
    """Telemetry for one update. Never aggregated across updates."""

    mode: str = FilterMode.FIXED_ROLLOUT.value
    group_size: int = DEFAULT_GROUP_SIZE
    candidate_groups: int = 0
    effective_groups: int = 0
    all_zero_groups: int = 0
    all_one_groups: int = 0
    incomplete_groups: int = 0
    candidate_rollouts: int = 0
    effective_rollouts: int = 0
    verdicts: list[GroupVerdict] = field(default_factory=list)

    @property
    def d_bar(self) -> float:
        """Fraction of candidate groups dropped this update."""

        if self.candidate_groups == 0:
            return 0.0
        return 1.0 - (self.effective_groups / self.candidate_groups)

    @property
    def inverse_one_minus_d_bar(self) -> float:
        """1/(1-d_bar); inf when every group was dropped."""

        remaining = 1.0 - self.d_bar
        if remaining <= 0.0:
            return float("inf")
        return 1.0 / remaining

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "group_size": self.group_size,
            "candidate_groups": self.candidate_groups,
            "effective_groups": self.effective_groups,
            "all_zero_groups": self.all_zero_groups,
            "all_one_groups": self.all_one_groups,
            "incomplete_groups": self.incomplete_groups,
            "candidate_rollouts": self.candidate_rollouts,
            "effective_rollouts": self.effective_rollouts,
            "d_bar": self.d_bar,
            "inverse_one_minus_d_bar": self.inverse_one_minus_d_bar,
        }


def group_indices(uids: Sequence[Any]) -> dict[str, list[int]]:
    """Group row indices by uid, preserving first-seen uid order."""

    groups: dict[str, list[int]] = {}
    for index, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(index)
    return groups


def classify_group(
    rewards: Sequence[float],
    *,
    expected_size: int = DEFAULT_GROUP_SIZE,
    max_reward: float = 1.0,
) -> GroupKind:
    """Classify one uid group by its terminal rewards."""

    values = [float(value) for value in rewards]
    if not values:
        return GroupKind.INCOMPLETE
    if expected_size > 0 and len(values) != expected_size:
        return GroupKind.INCOMPLETE
    if all(value <= 0.0 for value in values):
        return GroupKind.ALL_ZERO
    if all(value >= max_reward - ALL_ONE_TOLERANCE for value in values):
        return GroupKind.ALL_ONE
    if max(values) - min(values) <= ALL_ONE_TOLERANCE:
        # Degenerate but non-zero, non-max: still no contrast within the group.
        return GroupKind.ALL_ONE
    return GroupKind.INFORMATIVE


def evaluate_groups(
    rewards: Sequence[float],
    uids: Sequence[Any],
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    max_reward: float = 1.0,
) -> FilterStats:
    """Classify every uid group without touching any mask."""

    if len(rewards) != len(uids):
        raise ValueError(f"rewards/uids length mismatch: {len(rewards)} vs {len(uids)}")
    stats = FilterStats(group_size=group_size)
    for uid, indices in group_indices(uids).items():
        group_rewards = tuple(float(rewards[i]) for i in indices)
        kind = classify_group(group_rewards, expected_size=group_size, max_reward=max_reward)
        verdict = GroupVerdict(
            uid=uid, kind=kind, size=len(indices), rewards=group_rewards, indices=tuple(indices)
        )
        stats.verdicts.append(verdict)
        stats.candidate_groups += 1
        stats.candidate_rollouts += len(indices)
        if kind is GroupKind.INFORMATIVE:
            stats.effective_groups += 1
            stats.effective_rollouts += len(indices)
        elif kind is GroupKind.ALL_ZERO:
            stats.all_zero_groups += 1
        elif kind is GroupKind.ALL_ONE:
            stats.all_one_groups += 1
        else:
            stats.incomplete_groups += 1
    return stats


def apply_dynamic_filter(
    response_mask: np.ndarray,
    rewards: Sequence[float],
    uids: Sequence[Any],
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    max_reward: float = 1.0,
) -> tuple[np.ndarray, FilterStats]:
    """Zero the response mask of degenerate groups.

    Only the mask changes. Rewards, advantages and the batch layout are left
    alone, so a filtered rollout contributes no gradient while still occupying
    its slot; that is what keeps the token budget fixed.
    """

    mask = np.asarray(response_mask)
    if mask.shape[0] != len(rewards):
        raise ValueError(f"mask rows {mask.shape[0]} != rewards {len(rewards)}")
    stats = evaluate_groups(rewards, uids, group_size=group_size, max_reward=max_reward)
    filtered = mask.copy()
    for verdict in stats.verdicts:
        if not verdict.keep:
            for index in verdict.indices:
                filtered[index] = 0
    return filtered, stats


def select_fixed_informative(
    rewards: Sequence[float],
    uids: Sequence[Any],
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    target_informative_groups: int,
    max_candidate_groups: Optional[int] = None,
    max_reward: float = 1.0,
) -> tuple[list[str], FilterStats]:
    """Optional fixed-informative selection.

    Not the main configuration. `max_candidate_groups` is the cap and defaults to
    None (disabled) so it can never silently truncate an update.
    """

    stats = evaluate_groups(rewards, uids, group_size=group_size, max_reward=max_reward)
    stats.mode = FilterMode.FIXED_INFORMATIVE.value
    kept: list[str] = []
    for position, verdict in enumerate(stats.verdicts):
        if max_candidate_groups is not None and position >= max_candidate_groups:
            break
        if verdict.keep:
            kept.append(verdict.uid)
        if len(kept) >= target_informative_groups:
            break
    return kept, stats


def merge_stats(batches: Iterable[FilterStats]) -> dict[str, Any]:
    """Summarise several updates for the final report.

    Explicitly a reporting helper: the filter itself never carries state across
    updates.
    """

    items = list(batches)
    if not items:
        return {"updates": 0}
    d_bars = [item.d_bar for item in items]
    return {
        "updates": len(items),
        "candidate_groups": sum(item.candidate_groups for item in items),
        "effective_groups": sum(item.effective_groups for item in items),
        "all_zero_groups": sum(item.all_zero_groups for item in items),
        "all_one_groups": sum(item.all_one_groups for item in items),
        "d_bar_mean": float(np.mean(d_bars)),
        "d_bar_max": float(np.max(d_bars)),
        "inverse_one_minus_d_bar_mean": float(np.mean([1.0 / max(1e-9, 1.0 - d) for d in d_bars])),
    }


def apply_advantage_filter(response_mask, advantages, uids, *, group_size, tolerance=1e-12,
                           padding=None):
    """Optional fixed-rollout filter AFTER hybrid advantages are computed.

    A group survives if any valid policy token has nonzero advantage. Complete
    real groups are required; padding never participates in the decision.
    """
    mask, adv = np.asarray(response_mask), np.asarray(advantages)
    if (mask.shape != adv.shape or mask.shape[0] != len(uids)
            or not np.isfinite(adv).all() or not np.isfinite(tolerance) or tolerance < 0
            or group_size < 2):
        raise ValueError("invalid advantage filter inputs")
    padded = np.zeros(len(uids), dtype=bool) if padding is None else np.asarray(padding, dtype=bool)
    if padded.shape != (len(uids),):
        raise ValueError("invalid padding marker")
    result = mask.copy()
    result[padded] = 0
    decisions = {}
    for uid, indices in group_indices(uids).items():
        real = [i for i in indices if not padded[i]]
        if not real:
            continue
        if len(real) != group_size:
            raise ValueError(f"incomplete mt_gtpo group {uid}: {len(real)} != {group_size}")
        keep = bool(np.any((np.abs(adv[real]) > tolerance) & (mask[real] > 0)))
        decisions[uid] = keep
        if not keep:
            result[real] = 0
    kept = sum(decisions.values())
    return result, decisions, {"candidate_groups": len(decisions), "effective_groups": kept,
                               "zero_signal_groups": len(decisions) - kept,
                               "d_bar": 1 - kept / len(decisions) if decisions else 0.}
