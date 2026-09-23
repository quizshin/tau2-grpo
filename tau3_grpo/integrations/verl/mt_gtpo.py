"""veRL adapter and replay records for the opt-in mt_gtpo estimator."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
from tau3_grpo.evaluation.process_reward import SPLIT_VERSIONS, reward_settings

_LAST_STATS = {}
_LAST_FILTER_STATS = {}


def credit_mode_from_config(config):
    mode = ((config or {}).get('mt_gtpo') or {}).get('credit_mode', 'turn_v1')
    if mode not in {'turn_v1', 'call_local_v1', 'call_residual_v1'}:
        raise ValueError('Unknown MT-GTPO credit mode')
    return mode


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
    credit_mode_from_config(config)
    supplied.pop('credit_mode', None)
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
    credit_mode = credit_mode_from_config(config)
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
    receipts = None
    if credit_mode != 'turn_v1':
        from tau3_grpo.algorithms.mt_gtpo_call_credit import compute_mt_gtpo_call_credit
        from tau3_grpo.data.call_credit import call_credit_inputs

        batch = kwargs.get('batch')
        if batch is None or 'responses' not in batch or 'trajectory_facts_json' not in metadata:
            raise ValueError(f'{credit_mode} requires actual batch responses and trajectory_facts_json')
        inputs, receipts = call_credit_inputs(
            audit, metadata['trajectory_facts_json'], index,
            batch['responses'].detach().cpu().numpy(), mask)
        advantages, returns, details = compute_mt_gtpo_call_credit(
            outcomes, index, rewards, spans, mask, **settings, **inputs, credit_mode=credit_mode)
    else:
        advantages, returns, details = compute_mt_gtpo(
            outcomes, index, rewards, spans, mask, **settings)
    _LAST_STATS = {k: details[k] for k in ("active_rows", "turns", "nonzero_turns")}
    _LAST_STATS["zero_turn_fraction"] = 1 - details["nonzero_turns"] / max(details["turns"], 1)
    if receipts is not None:
        _LAST_STATS.update({f'call_credit/{key}': value
                            for key, value in details['call_credit_stats'].items()})
    events = [[event for turn in p["turn_records"] for event in turn["tool_calls"]]
              for p in audit if p is not None]
    if receipts is not None:
        counts = Counter(dict(error_calls=0, error_positive_before=0, error_positive_after=0,
                              positive_reward_calls=0, positive_reward_signal_lost=0))
        for i, process in enumerate(audit):
            if process is None:
                continue
            for k, turn in enumerate(process['turn_records']):
                by_id = {e['id']: e for e in turn['tool_calls']}
                detail = details['call_credit'][i][k]
                old = detail['baseline_advantage']
                new = detail['call_advantages'] or [old] * len(receipts[i][k]['call_ids'])
                for call_id, value in zip(receipts[i][k]['call_ids'], new, strict=True):
                    event = by_id[call_id]
                    if event['error']:
                        counts['error_calls'] += 1
                        counts['error_positive_before'] += int(old > 0)
                        counts['error_positive_after'] += int(value > 0)
                    if event['reward'] > 0:
                        counts['positive_reward_calls'] += 1
                        counts['positive_reward_signal_lost'] += int(old > 0 and value <= 0)
        _LAST_STATS.update({f'call_credit/{key}': value for key, value in counts.items()})
    _LAST_STATS.update(
        positive_process_rows=sum(any(e["reward"] > 0 for e in row) for row in events),
        positive_process_calls=sum(e["reward"] > 0 for row in events for e in row),
        negative_process_calls=sum(e["reward"] < 0 for row in events for e in row),
        process_reward_sum=sum(sum(p["turn_rewards"]) for p in audit if p is not None),
        process_call_reward_sum=sum(e["reward"] for row in events for e in row),
    )
    if expected_reward["version"] in SPLIT_VERSIONS:
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
    if receipts is not None:
        # Legacy replay readers reject this schema rather than interpreting the
        # baseline turn constants as the actual mixed token advantages.
        for i, raw in enumerate(replay):
            row = json.loads(raw)
            row.update(schema='mt_gtpo_call_replay_v1', credit_mode=credit_mode,
                       baseline_turn_advantages=row.pop('turn_advantages'),
                       token_advantages=advantages[i].tolist(),
                       returns_semantics='original_discounted_turn_return',
                       call_credit=details['call_credit'][i], call_receipts=receipts[i],
                       trajectory_facts=(json.loads(metadata['trajectory_facts_json'][i])
                                         if mask[i].any() else None))
            replay[i] = json.dumps(row, allow_nan=False)
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
