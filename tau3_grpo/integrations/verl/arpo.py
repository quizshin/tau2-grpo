"""ARPO soft attribution delegates arithmetic to the pinned GRPO implementation."""
from __future__ import annotations

import json

from tau3_grpo.algorithms.arpo import group_rows, settings


def validate_training_config(config):
    cfg = settings(config['algorithm'])
    rollout = config['actor_rollout_ref']['rollout']
    actor = config['actor_rollout_ref']['actor']
    if rollout.get('name') != 'vllm' or not rollout.get('calculate_log_probs', False):
        raise ValueError('ARPO requires vLLM and recorded rollout logprobs')
    if not 1 <= cfg.initial_rollouts <= int(rollout['n']) or int(rollout['n']) < 2:
        raise ValueError('ARPO roots must fit the rollout group')
    if actor.get('loss_agg_mode') != 'seq-mean-token-mean':
        raise ValueError('ARPO requires seq-mean-token-mean loss aggregation')
    if not actor.get('use_kl_loss', False):
        raise ValueError('ARPO requires separate actor KL loss')
    if (actor.get('policy_loss') or {}).get('loss_mode', 'vanilla') != 'vanilla':
        raise ValueError('ARPO v1 requires vanilla clipped PPO loss')
    if (rollout.get('generation_guard') or {}).get('mode', 'off') != 'off':
        raise ValueError('ARPO v1 requires generation guard off')
    if rollout.get('enable_rollout_routing_replay', False):
        raise ValueError('ARPO v1 does not support routing replay')
    return cfg


def compute_arpo_verl(token_level_rewards, response_mask, index=None, config=None,
                      non_tensor_batch=None, diagnostics_out=None, **kwargs):
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    cfg = settings(config)
    metadata = non_tensor_batch or {}
    records = metadata.get('arpo_rollout_json')
    if index is None or records is None or len(records) != len(index):
        raise ValueError('ARPO requires grouped rollout lineage, not flat GRPO samples')
    decoded = [json.loads(raw) for raw in records]
    if not decoded:
        raise ValueError('ARPO requires a nonempty rollout batch')
    size = int(decoded[0]['group_size'])
    if size < 2 or cfg.initial_rollouts > size:
        raise ValueError('Invalid ARPO group budget')
    groups = group_rows(index, size)
    for rows in groups:
        seen = set()
        for i in rows:
            r = decoded[i]
            if (r['schema'] != 'arpo_tau_rollout_v1' or r['config_sha256'] != cfg.identity
                    or r['group_uid'] != str(index[i]) or r['group_size'] != size):
                raise ValueError('ARPO rollout/config identity mismatch')
            seen.add(r['node'])
        if seen != set(range(size)):
            raise ValueError('ARPO group has missing or duplicate leaves')
        by_node = {decoded[i]['node']: decoded[i] for i in rows}
        if len({(decoded[i]['seed'], decoded[i]['policy_step']) for i in rows}) != 1:
            raise ValueError('ARPO group mixes sampling or policy identities')
        for node, record in by_node.items():
            parent = record['parent']
            if parent is None:
                if record['root'] != node or record['shared_response_tokens'] != 0:
                    raise ValueError('ARPO root lineage is invalid')
            elif (node < cfg.initial_rollouts or parent not in by_node or parent >= node
                  or record['root'] != by_node[parent]['root'] or record['shared_response_tokens'] <= 0):
                raise ValueError('ARPO branch lineage is invalid')
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards, response_mask=response_mask,
        index=index, norm_adv_by_std_in_grpo=True)
    # Enough estimator inputs for offline recomputation; exact tokens and old
    # logprobs remain in the existing trajectory facts / rollout buffers.
    import numpy as np

    metadata['arpo_replay_json'] = np.array([
        json.dumps(dict(schema='arpo_soft_replay_v1', rollout=decoded[i],
                        reward=float(token_level_rewards[i].sum().item()),
                        response_mask=response_mask[i].detach().cpu().tolist(),
                        advantages=advantages[i].detach().cpu().tolist()), allow_nan=False)
        for i in range(len(index))], dtype=object)
    if diagnostics_out is not None:
        diagnostics_out.update(estimator='arpo', stats={
            'groups': len(groups),
            'branches': sum(r['parent'] is not None for r in decoded),
            'new_generated_tokens': sum(r['new_tokens'] for r in decoded),
            'new_tool_calls': sum(r.get('new_tool_calls', 0) for r in decoded),
            'new_user_turns': sum(r.get('new_user_turns', 0) for r in decoded),
            'shared_response_tokens': sum(r['shared_response_tokens'] for r in decoded),
        })
    return advantages, returns


def register():
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, register_adv_est

    if 'arpo' not in ADV_ESTIMATOR_REGISTRY:
        register_adv_est('arpo')(compute_arpo_verl)


def resume_identity(config):
    cfg = validate_training_config(config)
    rollout = config['actor_rollout_ref']['rollout']
    return dict(config_sha256=cfg.identity, group_size=int(rollout['n']),
                seed=config['data']['seed'], loss_agg_mode=config['actor_rollout_ref']['actor']['loss_agg_mode'],
                sampling={key: rollout.get(key) for key in ('temperature', 'top_p', 'top_k',
                    'repetition_penalty', 'logprobs_mode', 'response_length')})
