"""veRL adapter and replay records for the opt-in mt_gtpo estimator."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
from tau3_grpo.evaluation.process_reward import SPLIT_VERSION, reward_settings

_LAST_STATS = {}
_LAST_FILTER_STATS = {}


def settings_from_config(config):
    config = config or {}
    if config.get("use_kl_in_reward", False):
        raise ValueError("mt_gtpo requires separate actor KL loss, not KL in rewards")
    df = config.get("dynamic_filter") or {}
    if df.get("enable", False) and (
        df.get("mode", "fixed_rollout") != "fixed_rollout"
        or df.get("metric", "advantage") != "advantage"
    ):
        raise ValueError("mt_gtpo filtering requires fixed_rollout and metric=advantage")
    if (config.get("filter_groups") or {}).get("enable", False):
        raise ValueError("mt_gtpo does not support outcome group filtering")
    settings = {"gamma": 0.9, "lambda_outcome": 0.3, "eps": 1e-6, "min_group_size": 2}
    supplied = dict(config.get("mt_gtpo") or {})
    if set(supplied) - set(settings):
        raise ValueError("unknown mt_gtpo setting")
    settings.update(supplied)
    return settings


def compute_mt_gtpo_verl(
    token_level_rewards, response_mask, index=None, config=None, non_tensor_batch=None, **kwargs
):
    import torch

    global _LAST_STATS, _LAST_FILTER_STATS
    settings = settings_from_config(config)
    expected_reward = reward_settings((config or {}).get("process_reward"))
    metadata = non_tensor_batch
    if metadata is None or index is None or "process_reward_json" not in metadata:
        raise ValueError("mt_gtpo requires uid and process_reward_json")
    mask = response_mask.detach().cpu().numpy().copy()
    outcomes = token_level_rewards.detach().cpu().numpy().sum(axis=1)
    rewards, spans, audit = [], [], []
    for i, raw in enumerate(metadata["process_reward_json"]):
        if not mask[i].any():
            rewards.append([])
            spans.append([])
            audit.append(None)
            continue
        payload = json.loads(raw)
        if (
            payload.get("schema") != "tau3_process_reward_v1"
            or payload.get("settings") != expected_reward
        ):
            raise ValueError("process reward schema/config mismatch between worker and trainer")
        if (expected_reward["mode"] in {"paper", "reference_write"}
                and payload.get("official_outcome") != float(outcomes[i])):
            raise ValueError("process reward official outcome mismatch between worker and trainer")
        rewards.append(payload["turn_rewards"])
        spans.append(payload["turn_spans"])
        audit.append(payload)
    advantages, returns, details = compute_mt_gtpo(
        outcomes, index, rewards, spans, mask, **settings
    )
    _LAST_STATS = {k: details[k] for k in ("active_rows", "turns", "nonzero_turns")}
    _LAST_STATS["zero_turn_fraction"] = 1 - details["nonzero_turns"] / max(details["turns"], 1)
    events = [[event for turn in p["turn_records"] for event in turn["tool_calls"]]
              for p in audit if p is not None]
    _LAST_STATS.update(
        positive_process_rows=sum(any(e["reward"] > 0 for e in row) for row in events),
        positive_process_calls=sum(e["reward"] > 0 for row in events for e in row),
        negative_process_calls=sum(e["reward"] < 0 for row in events for e in row),
        process_reward_sum=sum(sum(p["turn_rewards"]) for p in audit if p is not None),
        process_call_reward_sum=sum(e["reward"] for row in events for e in row),
    )
    if expected_reward["version"] == SPLIT_VERSION:
        # Candidate-call statistics before optional DF, uploaded by the existing
        # per-step telemetry path. Neutral reads remain visible even at weight 0.
        tier_counts = Counter(e["reward_type"] for row in events for e in row)
        tier_rewards = Counter()
        for row in events:
            for event in row:
                tier_rewards[event["reward_type"]] += event["reward"]
        for tier in expected_reward["weights"]:
            _LAST_STATS[f"candidate_calls/{tier}"] = tier_counts[tier]
            _LAST_STATS[f"candidate_call_reward_sum/{tier}"] = tier_rewards[tier]
    df = dict((config or {}).get("dynamic_filter") or {})
    filtered, decisions, _LAST_FILTER_STATS = mask, {}, {}
    if df.get("enable", False):
        from tau3_grpo.algorithms.dynamic_filtering import apply_advantage_filter

        filtered, decisions, _LAST_FILTER_STATS = apply_advantage_filter(
            mask,
            advantages,
            index,
            group_size=int(df.get("group_size", 8)),
            tolerance=float(df.get("tolerance", 1e-12)),
            padding=metadata.get("tau3_is_padding"),
        )
        if kwargs.get("batch") is not None:
            kwargs["batch"]["mt_gtpo_candidate_mask"] = response_mask.clone()
        response_mask.copy_(
            torch.as_tensor(filtered, device=response_mask.device, dtype=response_mask.dtype)
        )
        advantages *= filtered
        returns *= filtered
    # Stored by ray_trainer in the usual rollout JSONL, with original uid/mask.
    replay = np.empty(len(outcomes), dtype=object)
    group_sizes = Counter(str(index[i]) for i in range(len(index)) if mask[i].any())
    for i in range(len(outcomes)):
        replay[i] = json.dumps(
            {
                "schema": "mt_gtpo_replay_v1",
                "uid": str(index[i]),
                "sampled_group_size": group_sizes[str(index[i])],
                "outcome": float(outcomes[i]),
                "settings": settings,
                "response_mask": mask[i].tolist(),
                "process": audit[i],
                "filtered_response_mask": filtered[i].tolist(),
                "dynamic_filter": df,
                "filter_keep": decisions.get(str(index[i])),
                "episode_advantage": details["episode_advantages"][i],
                "turn_returns": details["turn_returns"][i],
                "turn_advantages": details["turn_advantages"][i],
            },
            allow_nan=False,
        )
    metadata["mt_gtpo_replay_json"] = replay
    if kwargs.get("diagnostics_out") is not None:
        kwargs["diagnostics_out"].update(
            estimator="mt_gtpo", stats=dict(_LAST_STATS), filter_stats=last_filter_stats(),
        )
    return (
        torch.as_tensor(
            advantages, device=token_level_rewards.device, dtype=token_level_rewards.dtype
        ),
        torch.as_tensor(
            returns, device=token_level_rewards.device, dtype=token_level_rewards.dtype
        ),
    )


def last_stats():
    return dict(_LAST_STATS)


def last_filter_stats():
    return {f"dynamic_filter/{k}": v for k, v in _LAST_FILTER_STATS.items()}


def register():
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, register_adv_est

    if "mt_gtpo" not in ADV_ESTIMATOR_REGISTRY:
        register_adv_est("mt_gtpo")(compute_mt_gtpo_verl)
