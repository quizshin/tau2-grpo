"""CPU integration: native tool loop, real τ DB, scripted policy/user transports."""
import asyncio
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import test_token_harness as harness
import torch
from omegaconf import OmegaConf

from tau3_grpo.algorithms.arpo import ARPOConfig
from tau3_grpo.integrations.verl.arpo import (
    compute_arpo_verl,
    resume_identity,
    validate_training_config,
)
from tau3_grpo.integrations.verl.arpo_rollout import GroupRollout

# Reuse pinned fixtures without downloading model assets.
tokenizer = harness.tokenizer
entry = harness.entry
xml = harness.xml


def test_soft_advantage_uses_real_group_std_and_masks():
    from verl.trainer.ppo.ray_trainer import compute_advantage

    from verl import DataProto
    c = ARPOConfig(initial_rollouts=1)
    config = OmegaConf.create(dict(adv_estimator='arpo', arpo={'initial_rollouts': 1}))
    rewards = torch.tensor([[0., 0., 1.], [0., 0., 0.]])
    mask = torch.tensor([[1., 0., 1.], [1., 0., 0.]])
    records = [json.dumps(dict(schema='arpo_tau_rollout_v1', config_sha256=c.identity,
        group_uid='g', group_size=2, node=i, parent=None if i == 0 else 0,
        new_tokens=2, shared_response_tokens=i)) for i in range(2)]
    batch = DataProto.from_dict(tensors={'token_level_rewards': rewards, 'response_mask': mask},
        non_tensors={'uid': np.array(['g', 'g']), 'arpo_rollout_json': np.array(records, dtype=object)})
    compute_advantage(batch, 'arpo', config=config)
    expected = .5 / (2**-.5 + 1e-6)
    assert batch.batch['advantages'][0].tolist() == pytest.approx([expected, 0., expected])
    assert batch.batch['advantages'][1].tolist() == pytest.approx([-expected, 0., 0.])
    assert len(batch.non_tensor_batch['arpo_replay_json']) == 2
    from tau3_grpo.algorithms.arpo import replay_soft
    replayed = replay_soft([json.loads(raw) for raw in batch.non_tensor_batch['arpo_replay_json']])
    assert torch.tensor(replayed).tolist() == batch.batch['advantages'].tolist()
    from tau3_grpo.tracking.trainer_telemetry import collect_rollout_metrics
    metrics = collect_rollout_metrics(batch.non_tensor_batch, adv_estimator='arpo',
                                     estimator_diagnostics=batch.meta_info['tau3_estimator_diagnostics'])
    assert metrics['arpo/branches'] == 1
    assert batch.meta_info['tau3_estimator_diagnostics']['stats']['branches'] == 1
    for zero in (torch.zeros_like(mask), mask):
        a, _ = compute_arpo_verl(torch.zeros_like(rewards), zero, index=np.array(['g', 'g']),
                                config=config, non_tensor_batch=batch.non_tensor_batch)
        assert torch.isfinite(a).all() and not a.any()
    with pytest.raises(ValueError, match='lineage'):
        compute_arpo_verl(rewards, mask, index=['g', 'g'], config=config)


async def native_group(tokenizer, entry, directory, monkeypatch, *, probability=1., fail=False, multilevel=False, truncate=False):
    from tau2.data_model.message import AssistantMessage
    from tau2.user import user_simulator
    from verl.workers.rollout.replica import TokenOutput

    from tau3_grpo.data.parquet_builder import build_row
    from tau3_grpo.evaluation.runtime import Endpoint
    from tau3_grpo.evaluation.token_runtime import make_loop
    from tau3_grpo.paths import TAU2_BENCH_ROOT

    monkeypatch.setattr(user_simulator, 'generate', lambda **kwargs:
                        AssistantMessage(role='assistant', content='###STOP###'))
    c = GroupRollout(ARPOConfig(initial_rollouts=1 if multilevel else 2, base_probability=probability, entropy_weight=0),
                    4, 'native-g', 42, 1)

    class Manager:
        def __init__(self):
            self.requests = []
            self.counts = {}

        async def generate(self, *, request_id, prompt_ids, sampling_params, **kwargs):
            self.requests.append((list(prompt_ids), dict(sampling_params)))
            slot, parent = c.current
            if fail == 'cancel' and parent is not None:
                raise asyncio.CancelledError()
            if fail and parent is not None:
                raise RuntimeError('injected branch generation failure')
            count = self.counts.get(request_id, 0)
            self.counts[request_id] = count + 1
            text = xml() + xml('2+2') if (parent is None or multilevel) and count == 0 else 'Done.'
            emitted = xml()[:-12] if truncate else text + '<|im_end|>'
            ids = tokenizer.encode(emitted, add_special_tokens=False)
            return TokenOutput(token_ids=ids, log_probs=[-.2] * len(ids), stop_reason='completed',
                extra_fields={'finish_reason': 'length' if truncate else 'stop', 'sampling_seed': sampling_params['seed'],
                    'arpo_entropy': dict(value=.1 if count == 0 else .8,
                        contributions=[.1], tokens=1, vocab_size=len(tokenizer),
                        mode='topk_partial', generated_tokens=len(ids), top_k=10)})
    manager = Manager()
    loop, _ = await make_loop(tokenizer=tokenizer, manager=manager,
        user=Endpoint('user', 'http://unused'), directory=directory, estimator='arpo', evaluation=False)
    row = build_row(entry, policy=(TAU2_BENCH_ROOT / 'data/tau2/domains/airline/policy.md').read_text(),
                    split='selection', seed=42)

    async def run_one(slot, resume, control):
        return await loop.run({'temperature': .7, 'top_p': 1., 'top_k': -1, 'logprobs': True},
            raw_prompt=row['prompt'], extra_info=row['extra_info'],
            tau3_sampling_identity={'sample_group_uid': 'native-g', 'trial': slot, 'seed': 42},
            _tau3_arpo=control, _tau3_resume=resume)
    return await c.run(run_one), manager


@pytest.mark.parametrize('probability', [0., 1.])
def test_native_forks_budget_multicall_and_exact_prefix(tokenizer, entry, tmp_path, monkeypatch, probability):
    from tau3_grpo.envs.registry import SESSIONS
    outputs, manager = asyncio.run(native_group(tokenizer, entry, tmp_path, monkeypatch, probability=probability))
    assert len(outputs) == 4 and SESSIONS.active_count() == 0
    rows = [json.loads(o.extra_fields['arpo_rollout_json']) for o in outputs]
    assert sum(r['parent'] is not None for r in rows) == (2 if probability else 0)
    for out, r in zip(outputs, rows):
        facts = json.loads(out.extra_fields['trajectory_facts_json'])
        assert len(facts['turns'][0]['tool_calls']) == 2
        assert len(out.response_ids) == len(out.response_mask) == len(out.response_logprobs)
        if r['parent'] is not None:
            parent = outputs[r['parent']]
            n = r['shared_response_tokens']
            assert n > 0 and r['prefix_tool_calls'] == 2
            assert out.response_ids[:n] == parent.response_ids[:n]
            assert out.response_logprobs[:n] == parent.response_logprobs[:n]
    assert len(manager.requests) == (6 if probability else 8)
    assert len({p['seed'] for _, p in manager.requests}) == len(manager.requests)
    group = json.loads(outputs[0].extra_fields['arpo_group_json'])
    assert all(o.extra_fields['arpo_group_json'] == outputs[0].extra_fields['arpo_group_json'] for o in outputs)
    assert len(group['nodes']) == 4


def test_branch_failure_cleans_queued_sessions(tokenizer, entry, tmp_path, monkeypatch):
    from tau3_grpo.envs.registry import SESSIONS
    with pytest.raises(RuntimeError, match='injected'):
        asyncio.run(native_group(tokenizer, entry, tmp_path, monkeypatch, fail=True))
    assert SESSIONS.active_count() == 0


def test_arpo_profile_reaches_hydra_and_resume_identity(tmp_path, monkeypatch):
    from tau3_grpo.paths import CODE_ROOT
    from tau3_grpo.training.rl.runner import resolve
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + ':' + __import__('os').environ['PATH'])
    monkeypatch.setenv('TAU3_RUN_ROOT', str(tmp_path / 'results'))
    monkeypatch.setenv('TAU3_MODEL_ROOT', str(tmp_path / 'models'))
    command, env, _ = resolve(tmp_path / 'run', estimator='arpo', updates=20)
    dry = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN='1'), text=True)
    args = shlex.split(dry.splitlines()[-1])
    output = subprocess.check_output(args + ['--cfg', 'job'], env=env, cwd=CODE_ROOT, text=True)
    import yaml
    cfg = yaml.safe_load(output)
    assert cfg['algorithm']['adv_estimator'] == 'arpo'
    assert cfg['actor_rollout_ref']['rollout']['n'] == 8
    assert cfg['data']['train_batch_size'] == 4
    assert cfg['trainer']['save_freq'] == cfg['trainer']['test_freq'] == 10
    assert cfg['trainer']['val_before_train'] is True
    assert validate_training_config(cfg).initial_rollouts == 4
    identity = resume_identity(cfg)
    changed = deepcopy(cfg)
    changed['algorithm']['arpo']['base_probability'] = .2
    assert resume_identity(changed) != identity
    changed['algorithm']['dynamic_filter']['enable'] = True
    with pytest.raises(ValueError, match='filter'):
        validate_training_config(changed)
    with pytest.raises(ValueError, match='filter'):
        resolve(tmp_path / 'run2', estimator='arpo', dynamic_filter=True)


def test_nested_branch_budget_and_inherited_root(tokenizer, entry, tmp_path, monkeypatch):
    from tau3_grpo.envs.registry import SESSIONS
    outputs, _ = asyncio.run(native_group(tokenizer, entry, tmp_path, monkeypatch, multilevel=True))
    rows = [json.loads(o.extra_fields['arpo_rollout_json']) for o in outputs]
    assert [r['parent'] for r in rows] == [None, 0, 1, 2]
    assert {r['root'] for r in rows} == {0}
    assert all(rows[i]['shared_response_tokens'] < rows[i+1]['shared_response_tokens'] for i in range(3))
    assert sum(r['new_tool_calls'] for r in rows) == 8
    assert SESSIONS.active_count() == 0


def test_cancellation_cleans_all_child_sessions(tokenizer, entry, tmp_path, monkeypatch):
    from tau3_grpo.envs.registry import SESSIONS
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(native_group(tokenizer, entry, tmp_path, monkeypatch, fail='cancel'))
    assert SESSIONS.active_count() == 0


def test_truncated_roots_still_fill_budget(tokenizer, entry, tmp_path, monkeypatch):
    from tau3_grpo.envs.registry import SESSIONS
    outputs, _ = asyncio.run(native_group(tokenizer, entry, tmp_path, monkeypatch, truncate=True))
    assert len(outputs) == 4
    assert all(json.loads(o.extra_fields['arpo_rollout_json'])['parent'] is None for o in outputs)
    assert SESSIONS.active_count() == 0


def test_manager_keeps_whole_groups_and_restores_trainer_order(monkeypatch):
    from types import SimpleNamespace

    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    from verl import DataProto

    monkeypatch.setenv('TAU3_RECORD_TRAJECTORY_FACTS', '0')
    monkeypatch.setenv('TAU3_RECORD_CALL_ATTRIBUTION', '0')
    observed = []
    async def remote(chunk):
        observed.append(list(chunk.non_tensor_batch['uid']))
        return DataProto.from_dict(tensors={'marker': chunk.batch['marker']},
            non_tensors={'uid': chunk.non_tensor_batch['uid']}, meta_info={'metrics': [{}] * len(chunk)})
    worker = SimpleNamespace(generate_sequences=SimpleNamespace(remote=remote))
    manager = SimpleNamespace(config=OmegaConf.create({'algorithm': {'adv_estimator': 'arpo'},
        'actor_rollout_ref': {'rollout': {'n': 2}}}), agent_loop_workers=[worker] * 4,
        _performance_metrics=lambda metrics, output: {})
    batch = DataProto.from_dict(tensors={'marker': torch.arange(6).reshape(6, 1)},
        non_tensors={'uid': np.array(['a','b','c','a','b','c'])})
    output = AgentLoopManager.generate_sequences(manager, batch)
    assert observed == [['a','a'], ['b','b'], ['c','c']]
    assert output.batch['marker'].flatten().tolist() == list(range(6))
    assert output.non_tensor_batch['uid'].tolist() == ['a','b','c','a','b','c']
