"""Tau-GiGPO advantage: A = A_episode + omega * A_step.

Milestone D6–D7. Frozen constants:

- ``A = A_episode + omega * A_step``
- ``Fnorm = 1``   -> group advantages are mean-centred only, never divided by a
  group std. A fixed normaliser keeps the episode and step terms on one scale so
  ``omega`` means the same thing in every run.
- ``gamma = 0.95`` -> the step return of an assistant step is the terminal reward
  discounted by the number of assistant steps still to come.
- If no anchor group has ``size >= 2`` then ``A_step = 0`` exactly, and the
  estimator degrades to episode-level GRPO rather than inventing a step signal
  from a singleton group.

Everything here is a pure function over numpy arrays and plain lists so the unit
tests can pin the arithmetic without a GPU, a tokenizer or Ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

GAMMA = 0.95
FNORM = 1.0
DEFAULT_OMEGA = 1.0
MIN_ANCHOR_GROUP_SIZE = 2


@dataclass(frozen=True)
class StepRecord:
    """One assistant generation inside a trajectory."""

    trajectory_index: int
    step_index: int
    anchor_id: Optional[str]
    span: Optional[tuple[int, int]]


@dataclass
class GiGPOStats:
    """Telemetry describing how much step signal was actually available."""

    trajectories: int = 0
    steps: int = 0
    anchored_steps: int = 0
    anchor_groups: int = 0
    usable_anchor_groups: int = 0
    steps_in_usable_groups: int = 0
    step_advantage_active: bool = False
    omega: float = DEFAULT_OMEGA
    gamma: float = GAMMA
    fnorm: float = FNORM
    group_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def anchor_coverage(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.anchored_steps / self.steps

    @property
    def usable_step_coverage(self) -> float:
        """Fraction of assistant steps that receive a non-singleton step signal."""

        if self.steps == 0:
            return 0.0
        return self.steps_in_usable_groups / self.steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "steps": self.steps,
            "anchored_steps": self.anchored_steps,
            "anchor_groups": self.anchor_groups,
            "usable_anchor_groups": self.usable_anchor_groups,
            "steps_in_usable_groups": self.steps_in_usable_groups,
            "step_advantage_active": self.step_advantage_active,
            "anchor_coverage": self.anchor_coverage,
            "usable_step_coverage": self.usable_step_coverage,
            "omega": self.omega,
            "gamma": self.gamma,
            "fnorm": self.fnorm,
        }


def episode_advantages(
    returns: Sequence[float],
    uids: Sequence[Any],
    *,
    fnorm: float = FNORM,
) -> np.ndarray:
    """Group-mean-centred episode advantage, one scalar per trajectory."""

    values = np.asarray([float(value) for value in returns], dtype=np.float64)
    if len(values) != len(uids):
        raise ValueError(f"returns/uids length mismatch: {len(values)} vs {len(uids)}")
    if fnorm <= 0.0:
        raise ValueError("fnorm must be positive")
    advantages = np.zeros_like(values)
    buckets: dict[str, list[int]] = {}
    for index, uid in enumerate(uids):
        buckets.setdefault(str(uid), []).append(index)
    for indices in buckets.values():
        subset = values[indices]
        advantages[indices] = (subset - subset.mean()) / fnorm
    return advantages


def step_returns(
    steps: Sequence[StepRecord],
    episode_returns: Sequence[float],
    *,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Discounted return of each assistant step.

    The environment pays only a terminal reward, so a step's return is the
    terminal reward discounted by how many assistant steps remain after it.
    """

    counts: dict[int, int] = {}
    for step in steps:
        counts[step.trajectory_index] = counts.get(step.trajectory_index, 0) + 1
    out = np.zeros(len(steps), dtype=np.float64)
    for position, step in enumerate(steps):
        total = counts[step.trajectory_index]
        remaining = total - 1 - step.step_index
        out[position] = (gamma**remaining) * float(episode_returns[step.trajectory_index])
    return out


def step_advantages(
    steps: Sequence[StepRecord],
    episode_returns: Sequence[float],
    *,
    gamma: float = GAMMA,
    fnorm: float = FNORM,
    min_group_size: int = MIN_ANCHOR_GROUP_SIZE,
) -> tuple[np.ndarray, GiGPOStats]:
    """Anchor-grouped step advantage, one scalar per step.

    Steps whose anchor group is smaller than `min_group_size` get exactly 0.
    """

    returns = step_returns(steps, episode_returns, gamma=gamma)
    advantages = np.zeros(len(steps), dtype=np.float64)

    groups: dict[str, list[int]] = {}
    for position, step in enumerate(steps):
        if step.anchor_id is None:
            continue
        groups.setdefault(step.anchor_id, []).append(position)

    usable = 0
    for anchor_id, positions in groups.items():
        if len(positions) < min_group_size:
            continue
        usable += 1
        subset = returns[positions]
        advantages[positions] = (subset - subset.mean()) / fnorm

    stats = GiGPOStats(
        trajectories=len({step.trajectory_index for step in steps}),
        steps=len(steps),
        anchored_steps=sum(1 for step in steps if step.anchor_id is not None),
        anchor_groups=len(groups),
        usable_anchor_groups=usable,
        steps_in_usable_groups=sum(
            len(positions) for positions in groups.values() if len(positions) >= min_group_size
        ),
        step_advantage_active=usable > 0,
        gamma=gamma,
        fnorm=fnorm,
        group_sizes={key: len(value) for key, value in groups.items()},
    )
    if usable == 0:
        # Frozen rule: no anchor group with size >= 2 -> A_step is exactly zero.
        advantages[:] = 0.0
    return advantages, stats


def combine_advantages(
    episode_adv: np.ndarray,
    step_adv: np.ndarray,
    steps: Sequence[StepRecord],
    *,
    num_trajectories: int,
    response_length: int,
    response_mask: Optional[np.ndarray] = None,
    omega: float = DEFAULT_OMEGA,
) -> np.ndarray:
    """Broadcast A_episode + omega*A_step onto a token-level advantage matrix.

    A_episode covers every response token of its trajectory. omega*A_step is
    added only on the token span of the assistant generation that produced it, so
    tool and user observation tokens carry the episode term alone.
    """

    out = np.zeros((num_trajectories, response_length), dtype=np.float64)
    for index in range(num_trajectories):
        out[index, :] = episode_adv[index]
    for position, step in enumerate(steps):
        if step.span is None or step.anchor_id is None:
            continue
        start, end = step.span
        start = max(0, int(start))
        end = min(response_length, int(end))
        if end <= start:
            continue
        out[step.trajectory_index, start:end] += omega * step_adv[position]
    if response_mask is not None:
        out = out * np.asarray(response_mask)
    return out


def compute_tau_gigpo_advantage(
    episode_returns: Sequence[float],
    uids: Sequence[Any],
    steps: Sequence[StepRecord],
    *,
    response_length: int,
    response_mask: Optional[np.ndarray] = None,
    omega: float = DEFAULT_OMEGA,
    gamma: float = GAMMA,
    fnorm: float = FNORM,
    min_group_size: int = MIN_ANCHOR_GROUP_SIZE,
) -> tuple[np.ndarray, GiGPOStats]:
    """Full Tau-GiGPO advantage: token-level matrix plus telemetry."""

    episode_adv = episode_advantages(episode_returns, uids, fnorm=fnorm)
    step_adv, stats = step_advantages(
        steps,
        episode_returns,
        gamma=gamma,
        fnorm=fnorm,
        min_group_size=min_group_size,
    )
    stats.omega = omega
    token_adv = combine_advantages(
        episode_adv,
        step_adv,
        steps,
        num_trajectories=len(episode_returns),
        response_length=response_length,
        response_mask=response_mask,
        omega=omega,
    )
    return token_adv, stats


def steps_from_anchor_payload(
    anchor_ids: Sequence[Sequence[Optional[str]]],
    anchor_spans: Sequence[Sequence[Optional[Sequence[int]]]],
) -> list[StepRecord]:
    """Rebuild `StepRecord`s from what the veRL patch put in `non_tensor_batch`.

    `ToolAgentLoop` emits one entry per generated segment and writes ``None`` at
    tool and user observation segments, so the two lists stay index-aligned with
    the token stream.
    """

    records: list[StepRecord] = []
    for trajectory_index, (ids, spans) in enumerate(
        zip(anchor_ids, anchor_spans, strict=True)
    ):
        step_index = 0
        for anchor_id, span in zip(ids, spans, strict=True):
            if anchor_id is None:
                continue
            parsed = None
            if span is not None:
                start, end = span
                parsed = (int(start), int(end))
            records.append(
                StepRecord(
                    trajectory_index=trajectory_index,
                    step_index=step_index,
                    anchor_id=str(anchor_id),
                    span=parsed,
                )
            )
            step_index += 1
    return records
