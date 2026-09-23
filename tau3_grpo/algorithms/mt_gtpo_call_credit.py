"""Opt-in call credit variants; preserve original turn returns.

The local baseline gives each call-bearing trajectory one vote at the original
(UID, turn position). This is a research estimator, not a causal value estimate.
Residual v1 instead adds within-turn centered rewards with fixed beta=1.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo


def compute_mt_gtpo_call_credit(
    outcomes, uids, turn_rewards, turn_spans, response_mask, *,
    call_rewards, call_spans, eligible_turns,
    gamma=0.9, lambda_outcome=0.3, eps=1e-6, min_group_size=2,
    credit_mode='call_local_v1',
):
    if credit_mode not in {'call_local_v1', 'call_residual_v1'}:
        raise ValueError('Unknown call credit mode')
    residual = credit_mode == 'call_residual_v1'
    settings = dict(gamma=gamma, lambda_outcome=lambda_outcome,
                    eps=eps, min_group_size=min_group_size)
    baseline, returns, details = compute_mt_gtpo(
        outcomes, uids, turn_rewards, turn_spans, response_mask, **settings)
    mask = np.asarray(response_mask)
    n = len(mask)
    if any(len(x) != n for x in (call_rewards, call_spans, eligible_turns)):
        raise ValueError('Call credit batch length mismatch')
    groups, records = {}, [[] for _ in range(n)]
    for i in np.flatnonzero(mask.any(axis=1)):
        count = len(turn_rewards[i])
        if any(len(x[i]) != count for x in (call_rewards, call_spans, eligible_turns)):
            raise ValueError('Call credit turn length mismatch')
        for k, reward in enumerate(turn_rewards[i]):
            q = np.asarray(call_rewards[i][k], dtype=np.float64)
            eligible = eligible_turns[i][k]
            spans = call_spans[i][k]
            if q.ndim != 1 or not np.isfinite(q).all() or type(eligible) is not bool:
                raise ValueError('Invalid call rewards/eligibility')
            if len(q) and not np.isclose(q.sum(), reward, rtol=1e-10, atol=1e-12):
                raise ValueError('Call credit requires sum aggregation of paid call rewards')
            if eligible:
                if not len(q) or spans is None or len(q) != len(spans):
                    raise ValueError('Eligible calls require complete exact spans')
                start, end = turn_spans[i][k]
                previous = start
                for span in spans:
                    if (len(span) != 2 or any(type(v) not in (int, np.int64, np.int32) for v in span)
                            or not previous <= span[0] < span[1] <= end
                            or not mask[i, span[0]:span[1]].all()):
                        raise ValueError('Invalid, overlapping or observation call span')
                    previous = span[1]
            elif spans is not None:
                raise ValueError('Ineligible turn must not provide trainable call spans')
            groups.setdefault((str(uids[i]), k), []).append(i)

    advantages = baseline.copy()
    replaced = np.zeros(mask.shape, dtype=bool)
    reasons = Counter()
    for (uid, k), members in groups.items():
        values = np.asarray([details['turn_returns'][i][k] for i in members])
        immediate = np.asarray([turn_rewards[i][k] for i in members])
        mean_r, mean_h = float(immediate.mean()), float((values - immediate).mean())
        std = float(values.std(ddof=0))
        call_members = [i for i in members if len(call_rewards[i][k])]
        b_call = (float(np.mean([np.mean(call_rewards[i][k]) for i in call_members]))
                  if call_members else None)
        for i in members:
            q = call_rewards[i][k]
            reason = ('no_calls' if not len(q) else
                      'attribution_ineligible' if not eligible_turns[i][k] else
                      'insufficient_turn_support' if len(members) < min_group_size else
                      'insufficient_call_support' if len(call_members) < min_group_size else
                      'zero_return_variance' if std == 0 else None)
            if residual and reason is None:
                reason = ('single_call' if len(q) == 1 else
                          'uniform_call_rewards' if all(x == q[0] for x in q) else None)
            row = dict(uid=uid, turn_index=k, immediate=float(turn_rewards[i][k]),
                       future=float(details['turn_returns'][i][k] - turn_rewards[i][k]),
                       mean_immediate=mean_r, mean_future=mean_h, std_return=std,
                       call_baseline=b_call, turn_support=len(members), call_support=len(call_members),
                       call_rewards=list(map(float, q)), call_spans=call_spans[i][k],
                       baseline_advantage=details['turn_advantages'][i][k],
                       fallback_reason=reason, call_advantages=[])
            if residual:
                row.update(call_baseline=float(np.mean(q)) if len(q) else None,
                           baseline_scope='within_turn', residual_beta=1.0)
            if reason:
                reasons[reason] += 1
            else:
                if residual:
                    # Keep the complete turn advantage; redistribute only within
                    # the same turn. Equal-call mean is conserved, not PPO gradient.
                    corrected = row['baseline_advantage'] + (np.asarray(q) - row['call_baseline']) / (std + eps)
                else:
                    # Algebraically replaces ONLY the centered immediate term.
                    corrected = (row['baseline_advantage']
                                 + ((np.asarray(q) - b_call) - (turn_rewards[i][k] - mean_r))
                                 / (std + eps))
                if not np.isfinite(corrected).all():
                    raise ValueError('Non-finite call credit correction')
                for span, value in zip(call_spans[i][k], corrected, strict=True):
                    a, b = span
                    advantages[i, a:b] = value
                    replaced[i, a:b] = True
                row['call_advantages'] = corrected.tolist()
            records[i].append(row)
    selected = mask.astype(bool)
    values, old = advantages[selected], baseline[selected]
    stats = {
        'candidate_turns': sum(len(x) for x in records),
        'applied_turns': sum(r['fallback_reason'] is None for row in records for r in row),
        'applied_calls': sum(len(r['call_advantages']) for row in records for r in row),
        'applied_tokens': int(replaced.sum()),
        'changed_tokens': int(np.count_nonzero(advantages != baseline)),
        'token_coverage': float(replaced.sum() / max(selected.sum(), 1)),
        'rms': float(np.sqrt(np.mean(values**2))) if len(values) else 0.,
        'baseline_rms': float(np.sqrt(np.mean(old**2))) if len(old) else 0.,
        'correction_rms': float(np.sqrt(np.mean((values-old)**2))) if len(old) else 0.,
        'abs_max': float(np.max(np.abs(values))) if len(values) else 0.,
        'abs_p95': float(np.quantile(np.abs(values), .95)) if len(values) else 0.,
        **{f'fallback/{name}': count for name, count in reasons.items()},
    }
    if residual:
        applied = [r for row in records for r in row if r['fallback_reason'] is None]
        stats.update(
            correction_abs_max=float(np.max(np.abs(values-old))) if len(values) else 0.,
            token_weighted_correction_sum=float((values-old).sum()),
            call_mean_drift_abs_max=max((abs(float(np.mean(r['call_advantages'])) - r['baseline_advantage'])
                                         for r in applied), default=0.),
            applied_return_std_min=min((r['std_return'] for r in applied), default=0.),
            near_zero_scale_turns=sum(r['std_return'] <= eps for r in applied),
        )
    details.update(baseline_nonzero_turns=details['nonzero_turns'],
                   nonzero_turns=sum(bool(np.any(np.abs(advantages[i, a:b]) > 1e-12))
                                    for i in np.flatnonzero(mask.any(axis=1))
                                    for a, b in turn_spans[i]),
                   credit_mode=credit_mode, call_credit=records, call_credit_stats=stats)
    return advantages, returns, details
