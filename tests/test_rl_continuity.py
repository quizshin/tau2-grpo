import copy
import json
from types import SimpleNamespace

import pytest

from tau3_grpo.tracking.rl_continuity import start_continuous_run
from tau3_grpo.integrations.boundary_checkpoint import complete_boundary


class FakeSwanlab:
    def __init__(self):
        self.inits, self.points = [], []

    def init(self, **kwargs):
        self.inits.append(kwargs)
        self.current = kwargs['id']
        return SimpleNamespace(id=self.current, name=kwargs['name'],
            url=f'https://swanlab.cn/@tester/project/runs/{self.current}', dir='/logs')

    def log(self, data, step):
        self.points.append((self.current, step, data))

    def finish(self):
        pass

    def Api(self):
        return SimpleNamespace(run=lambda path: SimpleNamespace(metrics=lambda **kw: {
            'list': [{'key': 'trainer/global_step', 'metrics': [
                {'step': step, 'value': step} for run_id, step, _ in self.points if run_id == path.split('/')[-1]]}]}))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv('TAU3_GRPO_ARM', 'e0')
    monkeypatch.setenv('TAU3_RESTORED_STEP', '0')
    return {'trainer': {'default_local_dir': str(tmp_path), 'total_training_steps': 30},
            'algorithm': {'adv_estimator': 'grpo'}, 'data': {'seed': 42, 'train_batch_size': 8},
            'actor_rollout_ref': {'model': {'path': '/sft'}, 'rollout': {'n': 8},
                                 'actor': {'optim': {'lr': 1e-6}, 'kl_loss_coef': 0.01}}}


def start(sdk, config):
    return start_continuous_run(sdk, project='project', name='E0', config=config,
                               options={}, log_dir='/logs', mode='online')


def test_same_run_continues_from_30_with_real_step_31(config, monkeypatch):
    sdk = FakeSwanlab()
    first = start(sdk, config)
    for step in range(1, 31):
        first.log({'actor/loss': 1 / step}, step)
    first.finish()
    original = copy.deepcopy(sdk.points)
    monkeypatch.setenv('TAU3_RESTORED_STEP', '30')
    config['trainer']['total_training_steps'] = 50
    second = start(sdk, config)
    assert sdk.inits[0]['id'] == sdk.inits[1]['id']
    assert sdk.inits[1]['resume'] == 'must'
    second.log({'actor/loss': 1 / 31}, 31)
    assert sdk.points[:30] == original
    assert [step for _, step, _ in sdk.points] == list(range(1, 32))
    assert second.state['run_path'].startswith('tester/project/')
    with pytest.raises(ValueError, match='Non-increasing'):
        second.log({'actor/loss': 9}, 30)


def test_different_arm_cannot_append_to_original_curve(config, monkeypatch):
    sdk = FakeSwanlab()
    start(sdk, config).finish()
    monkeypatch.setenv('TAU3_GRPO_ARM', 'e1')
    with pytest.raises(ValueError, match='original arm'):
        start(sdk, config)


def test_string_metadata_stays_local_without_breaking_numeric_or_media_logging(config):
    import numpy as np
    class StrictSwanlab(FakeSwanlab):
        def log(self, data, step):
            assert not any(isinstance(value, str) for value in data.values())
            super().log(data, step)
    sdk = StrictSwanlab()
    config['algorithm']['dynamic_filter'] = {'mode': 'fixed_rollout'}
    logger = start(sdk, config)
    media = object()
    payload = {'dynamic_filter/mode': 'fixed_rollout',
               'dynamic_filter/effective_groups': np.int64(2),
               'actor/grad_norm': np.float64(0.81), 'examples': media}
    logger.log(payload, 1)
    logger.log({'dynamic_filter/mode': 'fixed_rollout', 'actor/grad_norm': 0.5}, 2)
    assert payload['dynamic_filter/mode'] == 'fixed_rollout'
    assert sdk.inits[0]['config']['algorithm']['dynamic_filter']['mode'] == 'fixed_rollout'
    assert sdk.points[0][2]['examples'] is media
    assert sdk.points[0][2]['dynamic_filter/effective_groups'] == 2
    assert sdk.points[0][2]['actor/grad_norm'] == 0.81
    assert [point[2]['trainer/global_step'] for point in sdk.points] == [1, 2]
    rows = [json.loads(line) for line in (logger.path.parent / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['metrics']['dynamic_filter/mode'] == 'fixed_rollout' for row in rows)


def test_abnormal_older_checkpoint_is_not_silently_mixed(config, monkeypatch):
    sdk = FakeSwanlab()
    logger = start(sdk, config)
    logger.log({'actor/loss': 1}, 29)
    logger.finish()
    monkeypatch.setenv('TAU3_RESTORED_STEP', '20')
    with pytest.raises(ValueError, match='ahead of checkpoint'):
        start(sdk, config)
    assert len(sdk.inits) == 1


def make_shards(root, step, data=True):
    current = root / f'global_step_{step}'
    (current / 'actor').mkdir(parents=True)
    for prefix in ('model', 'optim', 'extra_state'):
        for rank in range(4):
            (current / 'actor' / f'{prefix}_world_size_4_rank_{rank}.pt').write_bytes(b'saved')
    if data:
        (current / 'data.pt').write_bytes(b'dataloader')


def test_old_checkpoint_removed_only_after_whole_new_checkpoint_is_complete(tmp_path):
    (tmp_path / 'validation').mkdir()
    (tmp_path / 'validation/20.jsonl').write_text('evaluation')
    (tmp_path / 'swanlab-run.json').write_text(json.dumps({'run_id': 'stable'}))
    make_shards(tmp_path, 20)
    complete_boundary(tmp_path, 20, 4)
    make_shards(tmp_path, 30, data=False)
    with pytest.raises(ValueError, match='previous checkpoint retained'):
        complete_boundary(tmp_path, 30, 4)
    assert (tmp_path / 'global_step_20/actor').is_dir()
    assert (tmp_path / 'latest_checkpointed_iteration.txt').read_text() == '20'
    (tmp_path / 'global_step_30/data.pt').write_bytes(b'dataloader')
    complete_boundary(tmp_path, 30, 4)
    assert not (tmp_path / 'global_step_20').exists()
    assert (tmp_path / 'latest_checkpointed_iteration.txt').read_text() == '30'
    assert json.loads((tmp_path / 'global_step_30/swanlab-run.json').read_text())['run_id'] == 'stable'
    assert (tmp_path / 'validation/20.jsonl').read_text() == 'evaluation'


def test_real_verl_checkpoint_control_restores_cpu_model_optimizer_rng_and_loader(tmp_path, monkeypatch):
    """Run actual trainer save/load control with a CPU worker, no FSDP/GPU claim."""
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    monkeypatch.setenv('TAU3_KEEP_COMPLETE_BOUNDARY', '1')
    class Worker:
        def __init__(self):
            self.model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.2))
            self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
        def update(self):
            self.optim.zero_grad()
            self.model(torch.ones(2, 3)).square().mean().backward()
            self.optim.step()
        def save_checkpoint(self, path, remote, step, max_ckpt_to_keep):
            assert max_ckpt_to_keep == 2
            from pathlib import Path
            p = Path(path); p.mkdir(parents=True)
            torch.save(self.model.state_dict(), p / 'model_world_size_1_rank_0.pt')
            torch.save(self.optim.state_dict(), p / 'optim_world_size_1_rank_0.pt')
            torch.save(torch.get_rng_state(), p / 'extra_state_world_size_1_rank_0.pt')
        def load_checkpoint(self, path, del_local_after_load):
            from pathlib import Path
            p = Path(path)
            self.model.load_state_dict(torch.load(p / 'model_world_size_1_rank_0.pt', weights_only=True))
            self.optim.load_state_dict(torch.load(p / 'optim_world_size_1_rank_0.pt', weights_only=True))
            torch.set_rng_state(torch.load(p / 'extra_state_world_size_1_rank_0.pt', weights_only=True))
    class Loader:
        position = 0
        def state_dict(self): return {'position': self.position}
        def load_state_dict(self, state): self.position = state['position']
    def trainer():
        tr = RayPPOTrainer.__new__(RayPPOTrainer)
        tr.config = OmegaConf.create({'trainer': {'default_local_dir': str(tmp_path), 'default_hdfs_dir': None,
            'max_actor_ckpt_to_keep': 1, 'max_critic_ckpt_to_keep': 1, 'n_gpus_per_node': 1, 'nnodes': 1,
            'resume_mode': 'auto', 'del_local_ckpt_after_load': False},
            'actor_rollout_ref': {'actor': {'checkpoint': {'async_save': False}}}})
        tr.use_critic = False
        tr.actor_rollout_wg, tr.train_dataloader = Worker(), Loader()
        tr.global_steps = 0
        return tr
    torch.manual_seed(42)
    first = trainer()
    for step in range(1, 31):
        first.actor_rollout_wg.update()
        first.global_steps = first.train_dataloader.position = step
        if step in (20, 30): first._save_checkpoint()
    assert not (tmp_path / 'global_step_20').exists()
    first.actor_rollout_wg.update()
    expected = copy.deepcopy(first.actor_rollout_wg.model.state_dict())
    torch.rand(20)
    second = trainer()
    second._load_checkpoint()
    assert second.global_steps == second.train_dataloader.position == 30
    assert all(int(s['step']) == 30 for s in second.actor_rollout_wg.optim.state.values())
    second.global_steps += 1
    second.actor_rollout_wg.update()
    assert second.global_steps == 31
    assert all(torch.equal(value, expected[key]) for key, value in second.actor_rollout_wg.model.state_dict().items())
