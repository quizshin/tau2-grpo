"""Turn-position GTPO hybrid, independent of anchors and the training runtime.

Terminal reward is at virtual turn K: G[k] = r[k] + gamma * G[k+1].
Population standard deviation (ddof=0) is an explicit implementation choice.
Missing turns never enter group statistics; zero-mask rows are excluded entirely.
"""

from __future__ import annotations

import numpy as np


def group_normalize(values, keys, *, eps=1e-6, min_group_size=2):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(keys) or not np.isfinite(values).all():
        raise ValueError("invalid group values/keys")
    result = np.zeros_like(values)
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        if len(indices) < min_group_size:
            continue
        subset = values[indices]
        std = subset.std(ddof=0)
        if std > 0:
            result[indices] = (subset - subset.mean()) / (std + eps)
    return result


def compute_mt_gtpo(
    outcomes,
    uids,
    turn_rewards,
    turn_spans,
    response_mask,
    *,
    gamma=0.9,
    lambda_outcome=0.3,
    eps=1e-6,
    min_group_size=2,
):
    """Return token advantages, token returns and fully replayable diagnostics.

    All generated tokens of an active row must be covered exactly once. Unknown
    metadata is an error, not a fallback to trajectory-level credit.
    """
    if (
        not np.isfinite([gamma, lambda_outcome, eps]).all()
        or not 0 <= gamma <= 1
        or lambda_outcome < 0
        or eps <= 0
        or type(min_group_size) is not int
        or min_group_size < 2
    ):
        raise ValueError("invalid mt_gtpo settings")
    mask = np.asarray(response_mask)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("response_mask must be a binary matrix")
    n, length = mask.shape
    if any(len(x) != n for x in (outcomes, uids, turn_rewards, turn_spans)):
        raise ValueError("mt_gtpo batch length mismatch")
    if not np.isfinite(outcomes).all():
        raise ValueError("non-finite outcome")
    active = np.flatnonzero(mask.any(axis=1))
    episode = np.zeros(n)
    episode[active] = group_normalize(
        outcomes[active], [str(uids[i]) for i in active], eps=eps, min_group_size=min_group_size
    )
    positions, keys, values = [], [], []
    for i in active:
        rewards, spans = turn_rewards[i], turn_spans[i]
        if len(rewards) != len(spans) or not spans or not np.isfinite(rewards).all():
            raise ValueError("invalid or missing turn metadata")
        covered = np.zeros(length, dtype=np.int8)
        end_previous = 0
        for span in spans:
            if len(span) != 2 or any(type(v) not in (int, np.int64, np.int32) for v in span):
                raise ValueError("invalid turn span")
            start, end = span
            if not end_previous <= start < end <= length or not mask[i, start:end].all():
                raise ValueError("overlapping, observation or out-of-bounds turn span")
            covered[start:end] += 1
            end_previous = end
        if not np.array_equal(covered, mask[i]):
            raise ValueError("turn spans must cover all generated tokens exactly once")
        returns = [0.0] * len(rewards)
        running = float(outcomes[i])
        for k in reversed(range(len(rewards))):
            running = float(rewards[k]) + gamma * running
            returns[k] = running
        for k, value in enumerate(returns):
            positions.append((i, k))
            keys.append((str(uids[i]), k))
            values.append(value)
    normalized = group_normalize(values, keys, eps=eps, min_group_size=min_group_size)
    token_adv = np.zeros(mask.shape, dtype=np.float64)
    token_returns = np.zeros_like(token_adv)
    turn_adv = [[] for _ in range(n)]
    turn_returns = [[] for _ in range(n)]
    for p, (i, k) in enumerate(positions):
        a = normalized[p] + lambda_outcome * episode[i]
        start, end = turn_spans[i][k]
        token_adv[i, start:end] = a
        token_returns[i, start:end] = values[p]
        turn_adv[i].append(float(a))
        turn_returns[i].append(float(values[p]))
    return (
        token_adv,
        token_returns,
        {
            "episode_advantages": episode.tolist(),
            "turn_advantages": turn_adv,
            "turn_returns": turn_returns,
            "active_rows": len(active),
            "turns": len(positions),
            "nonzero_turns": sum(abs(a) > 1e-12 for row in turn_adv for a in row),
        },
    )
