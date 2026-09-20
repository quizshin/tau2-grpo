"""CPU-only diagnostic candidates, deliberately outside the production trainer.

These helpers do not change existing checkpoints, advantage definitions or
anchor protocols. ScopedConfirmation assumes an externally supplied structured
proposal: automatic extraction of that proposal from live dialogue is NOT
implemented here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch

from tau3_grpo.algorithms.anchors.encoder import AnchorState, encode_anchor
from tau3_grpo.algorithms.anchors.features import (
    KNOWN_INFO_TOOLS, confirmation_flags, known_info_mask, last_observation_type, policy_precondition_flags,
)
from tau3_grpo.algorithms.dynamic_filtering import apply_dynamic_filter
from tau3_grpo.algorithms.tau_gigpo import (
    StepRecord, combine_advantages, compute_tau_gigpo_advantage, step_advantages, step_returns,
)
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def stock_grpo(rewards, uids, mask, *, normalize=True, dtype=torch.float64):
    mask = torch.as_tensor(mask, dtype=dtype, device='cpu')
    tensor = torch.zeros(mask.shape, dtype=dtype, device='cpu')
    tensor[:, -1] = torch.as_tensor(rewards, dtype=dtype)
    advantage, _ = compute_grpo_outcome_advantage(
        tensor, mask, np.asarray(uids), norm_adv_by_std_in_grpo=normalize)
    return advantage


def aligned_candidate(rewards, uids, mask, steps, *, omega=1., gamma=.95, normalize=True,
                      dtype=torch.float64):
    """Preserve the exact stock episode term; only the step term is additional."""
    baseline = stock_grpo(rewards, uids, mask, normalize=normalize, dtype=dtype)
    step, _ = step_advantages(steps, rewards, gamma=gamma)
    step_tokens = combine_advantages(np.zeros(len(rewards)), step, steps,
                                     num_trajectories=len(rewards), response_length=baseline.shape[1],
                                     response_mask=np.asarray(mask), omega=omega)
    return baseline + torch.as_tensor(step_tokens, dtype=dtype, device='cpu')


def message(role, content='', **kwargs):
    return SimpleNamespace(role=role, content=content, tool_calls=None, **kwargs)


def read_messages(reservation='A', price=100):
    call = SimpleNamespace(name='get_reservation_details', id='call-1',
                           arguments=json.dumps({'reservation_id': reservation}))
    return [SimpleNamespace(role='assistant', content='', tool_calls=[call]),
            message('tool', json.dumps({'reservation_id': reservation, 'price': price}),
                    id='call-1', error=False)]


def current_anchor(messages, *, task='task', db='unchanged-db'):
    return encode_anchor(AnchorState(task_id=task, db_hash=db,
                                    known_info_mask=known_info_mask(messages),
                                    confirmation_flags=confirmation_flags(messages),
                                    policy_precondition_flags=policy_precondition_flags(messages),
                                    last_observation=last_observation_type(messages)))


def observation_fingerprint(messages):
    """Conservative ordered successful-read ledger, not a semantic state model."""
    pending, reads = {}, []
    for msg in messages:
        for call in getattr(msg, 'tool_calls', None) or []:
            if call.name not in KNOWN_INFO_TOOLS:
                continue
            args = call.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            pending[str(call.id)] = (call.name, args)
        if msg.role == 'tool':
            call = pending.pop(str(getattr(msg, 'id', '')), None)
            if call is not None and not getattr(msg, 'error', False):
                content = msg.content
                try:
                    content = json.loads(content)
                except (TypeError, json.JSONDecodeError):
                    pass
                reads.append({'tool': call[0], 'arguments': call[1], 'result': content})
    return stable_hash(reads)


@dataclass
class ScopedConfirmation:
    """Fixture-level protocol candidate; caller must supply the action scope."""
    pending: str | None = None
    confirmed: str | None = None
    status: str = 'unknown'

    def propose(self, action):
        for field in ('operation', 'reservation_id', 'amount', 'currency'):
            if field not in action:
                raise ValueError(f'missing proposal scope: {field}')
        signature = stable_hash(action)
        if self.pending != signature:
            self.confirmed = None
            self.status = 'unknown'
        self.pending = signature

    def reply(self, text):
        text = text.strip().lower()
        self.confirmed = None
        if re.search(r"\b(no|not|don't|cannot|can't|withdraw|revoke|stop|wait)\b", text):
            self.status = 'denied'
        elif self.pending and re.fullmatch(
                r'(?:yes(?:,? please)?(?:,? (?:go ahead|proceed))?|i confirm|go ahead|proceed)[.!]?', text):
            self.status = 'confirmed'
            self.confirmed = self.pending
        else:
            self.status = 'unknown'
        return self.status

    def authorizes(self, action):
        return self.status == 'confirmed' and self.confirmed == stable_hash(action)


def cross_trajectory_steps(steps, rewards, *, gamma=.95):
    values = step_returns(steps, rewards, gamma=gamma)
    out = np.zeros(len(steps)); groups = defaultdict(list)
    for i, step in enumerate(steps):
        if step.anchor_id is not None:
            groups[step.anchor_id].append(i)
    for positions in groups.values():
        if len({steps[i].trajectory_index for i in positions}) >= 2:
            out[positions] = values[positions] - values[positions].mean()
    return out


def signal_stats(steps, step_values, before, after):
    groups = defaultdict(set)
    for step in steps:
        if step.anchor_id is not None:
            groups[step.anchor_id].add(step.trajectory_index)
    stats = dict(nonzero_steps_before=0, nonzero_steps_after=0,
                 cross_trajectory_noninitial_nonzero_steps=0,
                 nonzero_tokens_before=0, nonzero_tokens_after=0,
                 abs_step_advantage_mass_before=0., abs_step_advantage_mass_after=0.)
    for step, value in zip(steps, step_values, strict=True):
        if step.span is None or abs(value) <= 1e-12:
            continue
        start, end = max(0, step.span[0]), min(before.shape[1], step.span[1])
        n = int(np.count_nonzero(before[step.trajectory_index, start:end]))
        m = int(np.count_nonzero(after[step.trajectory_index, start:end]))
        stats['nonzero_steps_before'] += int(n > 0)
        stats['nonzero_steps_after'] += int(m > 0)
        stats['nonzero_tokens_before'] += n
        stats['nonzero_tokens_after'] += m
        stats['abs_step_advantage_mass_before'] += n * abs(float(value))
        stats['abs_step_advantage_mass_after'] += m * abs(float(value))
        if n and step.step_index > 0 and len(groups[step.anchor_id]) >= 2:
            stats['cross_trajectory_noninitial_nonzero_steps'] += 1
    return stats


def all_success_fixture():
    lengths = list(range(2, 10))
    steps = [StepRecord(i, j, 'initial' if j == 0 else f'unique-{i}-{j}', (j, j + 1))
             for i, n in enumerate(lengths) for j in range(n)]
    mask = np.array([[float(j < n) for j in range(9)] for n in lengths])
    return steps, mask


def run_validation(snapshot):
    torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('This validation must run with CUDA_VISIBLE_DEVICES empty and CUDA uninitialized')
    normalizations = []
    for arm, data in snapshot['arms'].items():
        for update in data['updates']:
            rows = update['rows']; rewards = [r['score'] for r in rows];uids = [r['task_id'] for r in rows]
            # Real reward/group data; these token masks/spans are SYNTHETIC.
            mask = np.array([[int(j < 2 + i % 5 and j % 3 != 1) for j in range(8)]
                             for i in range(len(rows))], dtype=float)
            steps = [StepRecord(i, 0, 'probe-' + uid, (0, 8)) for i, uid in enumerate(uids)]
            current, _ = compute_tau_gigpo_advantage(rewards, uids, steps, response_length=8,
                                                    response_mask=mask, omega=0)
            baseline = stock_grpo(rewards, uids, mask)
            aligned = aligned_candidate(rewards, uids, mask, steps, omega=0)
            mean_base = stock_grpo(rewards, uids, mask, normalize=False)
            np.testing.assert_allclose(current, mean_base.numpy(), atol=1e-12, rtol=0)
            assert torch.equal(aligned, baseline)
            normalizations.append({'arm': arm, 'step': update['step'],
                                   'current_vs_std_max_abs': float(np.max(np.abs(current - baseline.numpy()))),
                                   'aligned_vs_std_max_abs': float(torch.max(torch.abs(aligned - baseline))),
                                   'current_vs_mean_max_abs': float(np.max(np.abs(current - mean_base.numpy())))})
    cases = [('negative', 'I do not confirm this cancellation.', False),
             ('identity_only', 'Yes, that is my name.', False),
             ('hypothetical', 'What happens if I confirm?', False),
             ('wait', 'Please wait, do not proceed.', False),
             ('unambiguous_yes', 'Yes, please go ahead.', True)]
    current_cases = []
    for name, text, expected in cases:
        actual = 'user_confirmed' in confirmation_flags([message('user', text)])
        current_cases.append({'case': name, 'text': text, 'expected_action_confirmation': expected,
                              'actual_global_confirmed': actual, 'meets_expectation': actual == expected})
    withdrawal = [message('user', 'Yes, please go ahead.'), message('user', 'Wait, I withdraw my consent.')]
    current_cases.append({'case': 'withdrawal', 'expected_action_confirmation': False,
                          'actual_global_confirmed': 'user_confirmed' in confirmation_flags(withdrawal),
                          'meets_expectation': 'user_confirmed' not in confirmation_flags(withdrawal)})
    action = {'operation': 'cancel_reservation', 'reservation_id': 'A', 'amount': 100, 'currency': 'USD'}
    tracker = ScopedConfirmation();tracker.propose(action)
    candidate_cases=[]
    for name, text, expected in cases:
        tracker.reply(text);actual=tracker.authorizes(action)
        assert actual == expected
        candidate_cases.append({'case': name, 'authorizes': actual})
    for changed in [dict(action, amount=200), dict(action, reservation_id='B')]:
        tracker.propose(action);tracker.reply('Yes.');assert tracker.authorizes(action)
        tracker.propose(changed);assert not tracker.authorizes(changed)
    tracker.propose(action);tracker.reply('Yes.');tracker.reply('Wait, I withdraw my consent.')
    assert not tracker.authorizes(action)
    other = ScopedConfirmation();other.reply('Yes.');assert not other.authorizes(action)
    read_a, read_b, read_changed = read_messages('A'), read_messages('B'), read_messages('A', 200)
    state_cases = {'different_reservation_old_collides': current_anchor(read_a) == current_anchor(read_b),
                   'different_observation_old_collides': current_anchor(read_a) == current_anchor(read_changed),
                   'read_ledger_separates_reservation': observation_fingerprint(read_a) != observation_fingerprint(read_b),
                   'read_ledger_separates_observation': observation_fingerprint(read_a) != observation_fingerprint(read_changed),
                   'read_ledger_deterministic': observation_fingerprint(read_a) == observation_fingerprint(read_messages('A'))}
    assert all(state_cases.values())
    steps, mask = all_success_fixture();rewards=[1.] * 8
    filtered, _ = apply_dynamic_filter(mask, rewards, ['task'] * 8)
    raw, _ = step_advantages(steps, rewards)
    gamma1, _ = step_advantages(steps, rewards, gamma=1.)
    zero, zero_stats = step_advantages(steps, [0.] * 8)
    self_steps = [StepRecord(0, j, 'self', (j, j + 1)) for j in range(2)]
    self_raw, _ = step_advantages(self_steps, [1.]); self_cross = cross_trajectory_steps(self_steps, [1.])
    mixed = [StepRecord(i, j, 'shared-initial' if j == 0 else 'shared-later', (j * 2, j * 2 + 2))
             for i in range(8) for j in range(2)]
    mixed_rewards=[1.] * 4 + [0.] * 4; mixed_mask=np.ones((8,4))
    mixed_values,_=step_advantages(mixed,mixed_rewards)
    mixed_filtered,_=apply_dynamic_filter(mixed_mask,mixed_rewards,['task']*8)
    result = {'device': {'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                         'cuda_device_count': torch.cuda.device_count(),
                         'cuda_initialized': torch.cuda.is_initialized(), 'tensor_device': 'cpu'},
              'normalization': {'reward_updates_checked': len(normalizations),
                                'mask_scope': 'synthetic masks; historical rewards and task groups',
                                'updates': normalizations},
              'anchors': {'current_confirmation_cases': current_cases,
                          'candidate_confirmation_cases': candidate_cases, 'state_cases': state_cases,
                          'candidate_scope': 'structured proposals supplied by fixtures; no live dialogue proposal extractor',
                          'read_ledger_tradeoff': 'exact ordered history can split semantically equivalent states; coverage not measured'},
              'filtering': {'all_success_gamma095': signal_stats(steps,raw,mask,filtered),
                            'all_success_gamma1': signal_stats(steps,gamma1,mask,filtered),
                            'all_zero': signal_stats(steps,zero,mask,mask),
                            'all_zero_repeated_coverage': zero_stats.usable_step_coverage,
                            'single_trajectory_old_step_values': self_raw.tolist(),
                            'single_trajectory_cross_only_values': self_cross.tolist(),
                            'mixed': signal_stats(mixed,mixed_values,mixed_mask,mixed_filtered),
                            'scope': 'controlled fixtures; not historical lost-gradient or performance measurements'},
              'production_algorithm_modified': False, 'rl_training_started': False}
    assert result['device']['cuda_device_count'] == 0 and not result['device']['cuda_initialized']
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run_validation(json.loads(args.snapshot.read_text()))
    result['snapshot_sha256']=hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='normalization'},ensure_ascii=False))
