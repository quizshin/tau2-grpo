"""Register `tau_gigpo` as a veRL custom advantage estimator.

Milestone D6–D7. veRL's `AdvantageEstimator` docstring states that new estimators
should be registered by string name rather than added to the enum, so this module
calls `register_adv_est("tau_gigpo")` and keeps the algorithm itself outside the
veRL tree. Importing this module is the registration entrypoint; the training
launcher imports it before `verl.trainer.main_ppo` runs.

The pure arithmetic lives in `tau3_grpo.algorithms.tau_gigpo`; this file only converts
tensors and reads `non_tensor_batch`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from tau3_grpo.algorithms.anchors.encoder import (
    DEFAULT_SIMILARITY_THRESHOLD,
    resolve_similarity_candidates,
)
from tau3_grpo.algorithms.tau_gigpo import (
    DEFAULT_OMEGA,
    FNORM,
    GAMMA,
    MIN_ANCHOR_GROUP_SIZE,
    compute_tau_gigpo_advantage,
    steps_from_anchor_payload,
)

ESTIMATOR_NAME = "tau_gigpo"
ANCHOR_IDS_KEY = "anchor_ids"
ANCHOR_SPANS_KEY = "anchor_spans"

_REGISTERED = False
_LAST_STATS: dict[str, Any] = {}


def last_stats() -> dict[str, Any]:
    """Telemetry from the most recent estimator call, for the trainer's logger."""

    return dict(_LAST_STATS)


def _gigpo_settings(config: Any) -> dict[str, Any]:
    """Read the `algorithm.gigpo` block, falling back to the frozen defaults."""

    block = None
    if config is not None:
        block = getattr(config, "gigpo", None)
        if block is None and hasattr(config, "get"):
            block = config.get("gigpo", None)

    def pick(name: str, default: Any) -> Any:
        if block is None:
            return default
        if hasattr(block, name):
            value = getattr(block, name)
            return default if value is None else value
        if hasattr(block, "get"):
            value = block.get(name, None)
            return default if value is None else value
        return default

    return {
        "omega": float(pick("omega", DEFAULT_OMEGA)),
        "gamma": float(pick("gamma", GAMMA)),
        "fnorm": float(pick("fnorm", FNORM)),
        "min_group_size": int(pick("min_anchor_group_size", MIN_ANCHOR_GROUP_SIZE)),
        # Missing setting preserves historical checkpoints. New controlled
        # comparisons explicitly choose grpo, including its singleton behavior.
        "episode_normalization": str(pick("episode_normalization", "legacy_mean")),
        "similarity_threshold": float(
            pick("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        ),
    }


def _resolve_similarity_payload(
    ids: list[list[Optional[str]]], *, threshold: float
) -> list[list[Optional[str]]]:
    """Resolve rollout-local similarity candidates across the gathered batch."""

    positions: list[tuple[int, int]] = []
    candidates: list[str] = []
    for row_index, row in enumerate(ids):
        for step_index, anchor_id in enumerate(row):
            if anchor_id is not None:
                positions.append((row_index, step_index))
                candidates.append(anchor_id)
    resolved = resolve_similarity_candidates(candidates, threshold=threshold)
    output = [list(row) for row in ids]
    for (row_index, step_index), anchor_id in zip(positions, resolved, strict=True):
        output[row_index][step_index] = anchor_id
    return output


def _episode_returns(token_level_rewards: Any, response_mask: Any) -> np.ndarray:
    """Collapse token-level rewards to one terminal return per trajectory."""

    rewards = token_level_rewards.detach().to("cpu").numpy()
    return rewards.sum(axis=1).astype(np.float64)


def _anchor_payload(
    non_tensor_batch: Optional[dict[str, Any]], batch_size: int
) -> tuple[list[list[Optional[str]]], list[list[Optional[tuple[int, int]]]]]:
    """Pull the per-trajectory anchor ids/spans emitted by the ToolAgentLoop patch."""

    if not non_tensor_batch:
        return [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    raw_ids = non_tensor_batch.get(ANCHOR_IDS_KEY)
    raw_spans = non_tensor_batch.get(ANCHOR_SPANS_KEY)
    if raw_ids is None or raw_spans is None:
        return [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]

    ids: list[list[Optional[str]]] = []
    spans: list[list[Optional[tuple[int, int]]]] = []
    for row in range(batch_size):
        row_ids = raw_ids[row] if row < len(raw_ids) else None
        row_spans = raw_spans[row] if row < len(raw_spans) else None
        ids.append(list(row_ids) if row_ids else [])
        spans.append(list(row_spans) if row_spans else [])
    return ids, spans


def compute_tau_gigpo_verl(
    token_level_rewards: Any,
    response_mask: Any,
    index: Any = None,
    config: Any = None,
    non_tensor_batch: Optional[dict[str, Any]] = None,
    batch: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """veRL-facing estimator: returns `(advantages, returns)` as torch tensors."""

    import torch

    global _LAST_STATS

    batch_size, response_length = token_level_rewards.shape
    returns_np = _episode_returns(token_level_rewards, response_mask)
    uids = (
        [str(value) for value in index]
        if index is not None
        else [str(i) for i in range(batch_size)]
    )
    mask_np = response_mask.detach().to("cpu").numpy()

    settings = _gigpo_settings(config)
    episode_normalization = settings.pop("episode_normalization")
    if episode_normalization not in ("legacy_mean", "grpo"):
        raise ValueError(f"unknown episode_normalization: {episode_normalization!r}")
    ids, spans = _anchor_payload(non_tensor_batch, batch_size)
    ids = _resolve_similarity_payload(ids, threshold=settings.pop("similarity_threshold"))
    # v4 state equality is necessary but not sufficient: comparisons remain
    # inside the original sampled episode group, including repeated task IDs.
    from tau3_grpo.utils.hashing import sha256_json
    ids = [[("structured:v4:" + sha256_json([uids[i], aid]))
            if isinstance(aid, str) and aid.startswith("structured:v4:") else aid
            for aid in row] for i, row in enumerate(ids)]
    steps = steps_from_anchor_payload(ids, spans)

    if episode_normalization == "grpo":
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        from tau3_grpo.algorithms.tau_gigpo import combine_advantages, step_advantages

        normalize = kwargs.get("norm_adv_by_std_in_grpo")
        if normalize is None:
            normalize = config.get("norm_adv_by_std_in_grpo", True) if config is not None else True
        episode, _ = compute_grpo_outcome_advantage(
            token_level_rewards, response_mask,
            np.asarray(index if index is not None else uids),
            norm_adv_by_std_in_grpo=normalize,
        )
        step_values, stats = step_advantages(
            steps, returns_np, gamma=settings["gamma"], fnorm=settings["fnorm"],
            min_group_size=settings["min_group_size"],
        )
        step_tokens = combine_advantages(
            np.zeros(batch_size), step_values, steps, num_trajectories=batch_size,
            response_length=response_length, response_mask=mask_np, omega=settings["omega"],
        )
        advantages = episode + torch.as_tensor(
            step_tokens, dtype=episode.dtype, device=episode.device,
        )
        stats.omega = settings["omega"]
        _LAST_STATS = stats.to_dict()
        _LAST_STATS["episode_grpo_normalization"] = 1
        if kwargs.get("diagnostics_out") is not None:
            kwargs["diagnostics_out"].update(estimator="tau_gigpo", stats=dict(_LAST_STATS))
        # Match both outputs of the stock estimator when omega=0. The actor
        # consumes advantages; no critic is used in these GRPO experiments.
        return advantages, advantages.clone()
    advantages_np, stats = compute_tau_gigpo_advantage(
        returns_np,
        uids,
        steps,
        response_length=response_length,
        response_mask=mask_np,
        **settings,
    )
    _LAST_STATS = stats.to_dict()

    advantages = torch.as_tensor(
        advantages_np, dtype=token_level_rewards.dtype, device=token_level_rewards.device
    )
    _LAST_STATS["episode_grpo_normalization"] = 0
    if kwargs.get("diagnostics_out") is not None:
        kwargs["diagnostics_out"].update(estimator="tau_gigpo", stats=dict(_LAST_STATS))
    # Terminal-reward setting: the trajectory return broadcast over its tokens.
    returns = torch.as_tensor(
        np.repeat(returns_np[:, None], response_length, axis=1),
        dtype=token_level_rewards.dtype,
        device=token_level_rewards.device,
    ) * response_mask
    return advantages, returns


def register(force: bool = False) -> bool:
    """Register the estimator with veRL. Idempotent; returns True when registered."""

    global _REGISTERED
    if _REGISTERED and not force:
        return False
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, register_adv_est

    if ESTIMATOR_NAME in ADV_ESTIMATOR_REGISTRY and not force:
        _REGISTERED = True
        return False
    if force:
        ADV_ESTIMATOR_REGISTRY.pop(ESTIMATOR_NAME, None)
    register_adv_est(ESTIMATOR_NAME)(compute_tau_gigpo_verl)
    _REGISTERED = True
    return True


def is_registered() -> bool:
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY

    return ESTIMATOR_NAME in ADV_ESTIMATOR_REGISTRY


# Public batch readers shared with the signal auditor.
settings_from_config = _gigpo_settings
anchor_payload = _anchor_payload
episode_returns = _episode_returns
resolve_similarity_payload = _resolve_similarity_payload
