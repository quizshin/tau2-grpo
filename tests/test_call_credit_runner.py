import json

import pytest
import yaml
from test_formal_components import resolved

from tau3_grpo.integrations.matched_budget import MatchedBudget
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.training.rl import runner

PROFILE = CODE_ROOT / 'configs/train/rl/formal50_5090_a45_mt_gtpo_call_local_v1.yaml'


@pytest.fixture
def local_paths(tmp_path, monkeypatch):
    for key, value in {'TAU3_ROOT': tmp_path, 'TAU3_RUN_ROOT': tmp_path / 'runs',
                       'TAU3_DATA_ROOT': tmp_path / 'data', 'TAU3_MODEL_ROOT': tmp_path / 'models',
                       'TAU3_ENV_FILE': tmp_path / 'absent'}.items():
        monkeypatch.setenv(key, str(value))
    return tmp_path / 'run'


@pytest.mark.parametrize('mode', ['call_local_v1', 'call_residual_v1'])
def test_two_step_candidate_reaches_hydra_and_worker_environment(local_paths, monkeypatch, mode):
    profile = PROFILE.with_name(f'formal50_5090_a45_mt_gtpo_{mode}.yaml')
    command, env, snapshot = runner.resolve(local_paths, updates=2, estimator='mt_gtpo',
        reward_version='paper_env_split_v4', profile_override=profile,
        token_protocol='tau3_token_budget_v1', engineering_smoke=True)
    config = resolved(command, env)
    assert config['algorithm']['mt_gtpo']['credit_mode'] == mode
    assert config['algorithm']['adv_estimator'] == 'mt_gtpo'
    assert config['algorithm']['dynamic_filter']['enable'] is False
    assert config['critic']['enable'] is False
    assert config['trainer']['n_gpus_per_node'] == 8
    assert config['trainer']['total_training_steps'] == 2
    assert config['trainer']['save_freq'] == 2
    assert config['trainer']['test_freq'] == -1
    assert not config['trainer']['val_before_train']
    worker = config['ray_kwargs']['ray_init']['runtime_env']['env_vars']
    assert worker['TAU3_RECORD_CALL_ATTRIBUTION'] == env['TAU3_RECORD_CALL_ATTRIBUTION'] == '1'
    assert snapshot['engineering_smoke'] and snapshot['uncalibrated_initializer']
    assert snapshot['credit_mode'] == mode
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert MatchedBudget.from_environment(save_freq=2, test_freq=-1) is None


def test_formal_recipe_gate_stays_and_smoke_is_bounded(local_paths):
    with pytest.raises(ValueError, match='passed --reward-recipe'):
        runner.resolve(local_paths, reward_version='paper_env_split_v4', profile_override=PROFILE)
    for updates in [0, 3, 10, 30]:
        with pytest.raises(ValueError, match='1 or 2'):
            runner.resolve(local_paths, updates=updates, engineering_smoke=True)
    with pytest.raises(ValueError, match='fresh run'):
        runner.resolve(local_paths, updates=2, engineering_smoke=True, resume_from=local_paths / 'global_step_2')
    with pytest.raises(ValueError, match='requires mt_gtpo'):
        runner.resolve(local_paths, estimator='grpo', credit_mode='call_local_v1')


@pytest.mark.parametrize('previous,requested', [('turn_v1', 'call_local_v1'),
                                              ('call_local_v1', 'call_residual_v1'),
                                              ('call_residual_v1', 'turn_v1')])
def test_resume_cannot_switch_credit_mode(local_paths, previous, requested):
    local_paths.mkdir()
    (local_paths / 'resolved-hydra.yaml').write_text(yaml.safe_dump({
        'algorithm': {'adv_estimator': 'mt_gtpo', 'mt_gtpo': {'credit_mode': previous},
                      'process_reward': {'version': 'v3'}, 'dynamic_filter': {'enable': False}}}))
    with pytest.raises(ValueError, match='credit mode'):
        runner.resolve(local_paths, credit_mode=requested, resume_from=local_paths / 'global_step_10')


def test_smoke_completion_requires_checkpoint_and_cloud_steps_not_selection(tmp_path, monkeypatch):
    import swanlab

    (tmp_path / 'latest_checkpointed_iteration.txt').write_text('2')
    (tmp_path / 'swanlab-run.json').write_text(json.dumps({'run_path': 'test', 'url': 'http://unused'}))
    calls = []
    monkeypatch.setattr(runner, 'validate_checkpoint', lambda *a, **kw: calls.append((a, kw)))

    class Run:
        def metrics(self, **kw):
            return {'list': [{'metrics': [{'step': i, 'value': i} for i in [1, 2]]}]}

    class Api:
        def run(self, path):
            return Run()

    monkeypatch.setattr(swanlab, 'Api', Api)
    result = runner.verify_completion(tmp_path, 2, world_size=8, engineering_smoke=True)
    assert result['completed_step'] == 2 and result['evaluation_scores'] == {}
    assert calls[0][1]['world_size'] == 8
