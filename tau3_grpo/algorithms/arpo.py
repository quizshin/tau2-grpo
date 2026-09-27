"""ARPO τ v1: explicit partial-entropy proxy and deterministic group budgeting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping


@dataclass(frozen=True)
class ARPOConfig:
    version: str = 'arpo_tau_v1'
    initial_rollouts: int = 4
    entropy_window: int = 20
    entropy_top_k: int = 10
    base_probability: float = 0.5
    entropy_weight: float = 0.5
    entropy_mode: str = 'topk_partial'
    advantage_mode: str = 'soft'

    def __post_init__(self):
        if (self.version != 'arpo_tau_v1' or self.entropy_mode != 'topk_partial'
                or self.advantage_mode != 'soft'):
            raise ValueError('Unsupported ARPO protocol/entropy/advantage mode')
        for name in ('initial_rollouts', 'entropy_window', 'entropy_top_k'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'ARPO {name} must be a positive integer')
        if self.entropy_top_k > 20:
            raise ValueError('ARPO entropy_top_k exceeds the supported vLLM limit (20)')
        if (not math.isfinite(self.base_probability) or not 0 <= self.base_probability <= 1
                or not math.isfinite(self.entropy_weight) or self.entropy_weight < 0):
            raise ValueError('Invalid ARPO branching probability/weight')

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def settings(algorithm: Mapping) -> ARPOConfig:
    if algorithm.get('adv_estimator') != 'arpo':
        raise ValueError('ARPO requires its explicit estimator')
    for key in ('dynamic_filter', 'filter_groups'):
        if (algorithm.get(key) or {}).get('enable', False):
            raise ValueError('ARPO v1 does not support group filtering')
    if algorithm.get('use_kl_in_reward', False):
        raise ValueError('ARPO requires separate actor KL loss')
    if not algorithm.get('norm_adv_by_std_in_grpo', True):
        raise ValueError('ARPO v1 requires reward standardization')
    try:
        return ARPOConfig(**dict(algorithm.get('arpo') or {}))
    except TypeError as exc:
        raise ValueError('Unknown ARPO setting') from exc


def partial_entropy(logprob_rows, *, vocab_size: int, window: int) -> dict:
    """Sum unrenormalized top-k entropy contributions / log(V).

    This is a truncated proxy, not full vocabulary entropy or sampled surprisal.
    Store per-position contributions and count so the scalar can be recomputed.
    """
    if vocab_size <= 1 or window <= 0 or not logprob_rows:
        raise ValueError('ARPO entropy requires actual token distributions')
    contributions = []
    for row in logprob_rows[:window]:
        values = list(row)
        if not values or any(not math.isfinite(v) or v > 1e-6 for v in values):
            raise ValueError('Invalid or missing ARPO logprob distribution')
        if sum(math.exp(v) for v in values) > 1.0001:
            raise ValueError('ARPO logprob probability mass exceeds one')
        contributions.append(-sum(math.exp(v) * v for v in values))
    return dict(value=sum(contributions) / math.log(vocab_size),
                contributions=contributions, tokens=len(contributions), vocab_size=vocab_size,
                mode='topk_partial')


def branch_probability(current: float, initial: float, config: ARPOConfig) -> float:
    if not math.isfinite(current) or not math.isfinite(initial):
        raise ValueError('Non-finite ARPO entropy')
    return max(0., min(1., config.base_probability + config.entropy_weight * (current - initial)))


def stable_seed(*parts) -> int:
    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big') % (2**31 - 1)


def group_rows(uids, size: int) -> list[list[int]]:
    groups = {}
    for index, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(index)
    if not groups or any(len(rows) != size for rows in groups.values()):
        raise ValueError(f'ARPO requires complete groups of {size} rows')
    return list(groups.values())


def worker_rows(uids, size: int, workers: int) -> list[list[int]]:
    groups = group_rows(uids, size)
    bins = [[] for _ in range(min(workers, len(groups)))]
    if not bins:
        raise ValueError('ARPO requires a worker')
    for i, rows in enumerate(groups):
        bins[i % len(bins)].extend(rows)
    return bins
