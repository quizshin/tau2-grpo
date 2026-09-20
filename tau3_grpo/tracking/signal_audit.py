"""Opt-in, driver-side GiGPO evidence; never alters an optimizer batch.

Masks and rewards are stored losslessly as intervals/sparse entries. Signal
magnitudes describe advantages, not gradients or counterfactual model quality.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from tau3_grpo.algorithms.tau_gigpo import step_advantages, steps_from_anchor_payload
from tau3_grpo.integrations.verl.gigpo import (
    anchor_payload, episode_returns, settings_from_config, resolve_similarity_payload,
)


def capture_mask(data, config):
    if not os.environ.get('TAU3_GRPO_SIGNAL_AUDIT_DIR') or str(config.get('adv_estimator')) != 'tau_gigpo':
        return None
    return data.batch['response_mask'].detach().cpu().numpy().copy()


def intervals(mask):
    mask = np.asarray(mask)
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError('signal audit requires binary policy masks')
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)]


def restore_mask(rows, length):
    mask = np.zeros((len(rows), length), dtype=np.int64)
    for i, ranges in enumerate(rows):
        for start, end in ranges:
            mask[i, start:end] = 1
    return mask


@lru_cache(maxsize=1)
def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ['tau3_grpo/integrations/verl/gigpo.py', 'tau3_grpo/algorithms/tau_gigpo.py',
             'tau3_grpo/tracking/signal_audit.py', 'tau3_grpo/algorithms/dynamic_filtering.py',
             'tau3_grpo/integrations/anchor_hook.py', 'tau3_grpo/algorithms/anchors/features.py',
             'tau3_grpo/algorithms/anchors/encoder.py', 'tau3_grpo/algorithms/anchors/evidence.py', 'tau3_grpo/algorithms/anchors/semantic.py', 'verl/verl/trainer/ppo/ray_trainer.py',
             'verl/verl/trainer/ppo/core_algos.py']
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def analyze(data, before, config):
    """Build one replayable payload from the actual gathered driver batch."""
    from omegaconf import OmegaConf
    after = data.batch['response_mask'].detach().cpu().numpy()
    if before.shape != after.shape or np.any(after > before):
        raise ValueError('filter mask must be a subset of candidate mask')
    nt = data.non_tensor_batch
    size, length = before.shape
    padding = np.asarray(nt.get('tau3_is_padding', np.zeros(size)), dtype=bool)
    # Padding rows must not contribute to either the audit or the estimator.
    if np.any(before[padding]) or np.any(after[padding]):
        raise ValueError('padding rows have policy tokens')
    raw_ids, spans = anchor_payload(nt, size)
    settings = settings_from_config(config)
    ids = resolve_similarity_payload(raw_ids, threshold=settings['similarity_threshold'])
    steps = steps_from_anchor_payload(ids, spans)
    if any(padding[s.trajectory_index] for s in steps):
        raise ValueError('padding rows have anchors')
    rewards = episode_returns(data.batch['token_level_rewards'], data.batch['response_mask'])
    values, _ = step_advantages(steps, rewards, gamma=settings['gamma'], fnorm=settings['fnorm'],
                                min_group_size=settings['min_group_size'])
    groups = defaultdict(set)
    for s in steps:
        groups[s.anchor_id].add(s.trajectory_index)
    metrics = {'omega': settings['omega'], 'real_trajectories': int((~padding).sum()), 'steps': len(steps),
               'cross_trajectory_steps': sum(len(groups[s.anchor_id]) >= 2 for s in steps),
               'noninitial_steps': sum(s.step_index > 0 for s in steps)}
    step_rows = []
    for label, mask in [('before', before), ('after', after)]:
        counters = dict(policy_tokens=int(mask.sum()), nonzero_steps=0, nonzero_step_tokens=0,
                        cross_noninitial_nonzero_steps=0, weighted_abs_step_advantage=0.)
        for s, value in zip(steps, values, strict=True):
            start, end = s.span if s.span is not None else (0, 0)
            start, end = max(0, start), min(length, end)
            n = int(mask[s.trajectory_index, start:end].sum()) if end > start else 0
            active = abs(value) > 1e-12 and n > 0
            counters['nonzero_steps'] += int(active)
            counters['nonzero_step_tokens'] += n if active else 0
            counters['cross_noninitial_nonzero_steps'] += int(active and s.step_index > 0 and len(groups[s.anchor_id]) >= 2)
            counters['weighted_abs_step_advantage'] += abs(settings['omega'] * float(value)) * n
        metrics.update({f'{label}_{key}': value for key, value in counters.items()})
    for s, value in zip(steps, values, strict=True):
        step_rows.append({'trajectory': s.trajectory_index, 'step': s.step_index, 'initial': s.step_index == 0,
                          'anchor': s.anchor_id, 'span': s.span, 'step_advantage': float(value),
                          'distinct_trajectories': len(groups[s.anchor_id])})
    for label in ('before', 'after'):
        metrics[f'{label}_applied_nonzero_steps'] = metrics[f'{label}_nonzero_steps'] if settings['omega'] != 0 else 0
    metrics['masked_nonzero_steps'] = metrics['before_nonzero_steps'] - metrics['after_nonzero_steps']
    metrics['masked_nonzero_step_tokens'] = metrics['before_nonzero_step_tokens'] - metrics['after_nonzero_step_tokens']
    denom = metrics['noninitial_steps']
    metrics['after_cross_noninitial_nonzero_coverage'] = metrics['after_cross_noninitial_nonzero_steps'] / denom if denom else 0.
    # Preserve actual post-filter combined advantages separately from raw step
    # contributions. A nonzero raw step term at omega=0 contributes no update.
    actual = data.batch['advantages'].detach().cpu().numpy()
    metrics['actual_nonzero_advantage_tokens'] = int(np.count_nonzero(actual))
    metrics['actual_abs_advantage'] = float(np.abs(actual).sum(dtype=np.float64))
    raw_rewards = data.batch['token_level_rewards'].detach().cpu().numpy()
    rows = []
    for i in range(size):
        nz = np.flatnonzero(raw_rewards[i])
        metadata = {key: str(nt[key][i]) for key in ('uid', 'task_id', 'session_id', 'trajectory_json') if key in nt}
        rows.append({'metadata': metadata, 'padding': bool(padding[i]), 'return': float(rewards[i]),
                     'reward_entries': [[int(j), float(raw_rewards[i,j])] for j in nz],
                     'anchor_ids': raw_ids[i], 'anchor_spans': spans[i],
                     'mask_before': intervals(before[i]), 'mask_after': intervals(after[i])})
    cfg = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    payload = {'schema_version': 1, 'response_length': length, 'reward_dtype': str(data.batch['token_level_rewards'].dtype),
               'mask_dtype': str(data.batch['response_mask'].dtype), 'algorithm': cfg, 'settings': settings,
               'algorithm_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
               'rows': rows, 'steps': step_rows, 'metrics': metrics,
               'actual_advantage_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
               'source_sha256': source_hashes(),
               'scope': 'actual masks/rewards/anchors; raw step counts are before omega scaling; not gradient attribution'}
    # Validate JSON before attempting to write. Numeric metrics remain SwanLab-safe.
    json.dumps(payload, allow_nan=False)
    return payload


def audit_update(data, before, config, update):
    if before is None:
        return {}
    payload = analyze(data, before, config)
    payload['update'] = int(update)
    directory = Path(os.environ['TAU3_GRPO_SIGNAL_AUDIT_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    # Unique attempt names retain evidence if an update is retried after a crash.
    path = directory / f'update_{int(update):06d}_{uuid.uuid4().hex}.json'
    temporary = path.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {f'gigpo_signal/{k}': v for k,v in payload['metrics'].items()}
