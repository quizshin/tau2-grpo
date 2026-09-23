"""Explicit exploration gate, real Hydra composition and resume identity."""
import copy
import json

import pytest
import yaml
from test_formal_components import resolved

from tau3_grpo.evaluation.process_reward import reward_settings
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.training.rl import runner

PROFILE = CODE_ROOT / 'configs/train/rl/formal50_5090_a45_mt_gtpo_v4.yaml'


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for key, value in {'TAU3_ROOT': tmp_path, 'TAU3_RUN_ROOT': tmp_path / 'runs',
                       'TAU3_DATA_ROOT': tmp_path / 'data', 'TAU3_MODEL_ROOT': tmp_path / 'models',
                       'TAU3_ENV_FILE': tmp_path / 'absent'}.items():
        monkeypatch.setenv(key, str(value))
    return tmp_path / 'run', dict(updates=30, estimator='mt_gtpo',
        reward_version='paper_env_split_v4', profile_override=PROFILE,
        credit_mode='turn_v1', token_protocol='tau3_token_budget_v1',
        uncalibrated_exploration=True)


def test_exploration_preserves_training_math_and_ten_step_lifecycle(setup):
    result, args = setup
    command, env, snapshot = runner.resolve(result, **args)
    config = resolved(command, env)
    baseline_args = dict(args, uncalibrated_exploration=False, allow_uncalibrated=True)
    baseline_command, baseline_env, _ = runner.resolve(result, **baseline_args)
    baseline = resolved(baseline_command, baseline_env)
    assert config.pop('tau3_experiment_kind') == 'uncalibrated_exploration'
    assert config.pop('tau3_irc_calibrated') is False
    name = config['trainer'].pop('experiment_name')
    old_name = baseline['trainer'].pop('experiment_name')
    assert name.endswith('-uncalibrated-exploration-df-0-s42')
    assert config['actor_rollout_ref']['rollout']['trace'].pop('experiment_name') == name
    assert baseline['actor_rollout_ref']['rollout']['trace'].pop('experiment_name') == old_name
    assert config == baseline
    assert env['MODEL_PATH'].endswith('/A/merged')
    assert snapshot['uncalibrated_exploration'] and snapshot['uncalibrated_initializer']
    assert snapshot['irc_recipe'] is None and not snapshot['engineering_smoke']
    assert config['trainer']['total_training_steps'] == 30
    assert config['trainer']['save_freq'] == config['trainer']['test_freq'] == 10
    assert config['trainer']['max_actor_ckpt_to_keep'] == 1
    assert config['trainer']['logger'] == ['console', 'swanlab']
    assert config['trainer']['n_gpus_per_node'] == 8
    assert config['data']['train_batch_size'] == config['actor_rollout_ref']['rollout']['n'] == 8
    assert config['algorithm']['dynamic_filter']['enable'] is False
    assert reward_settings(config['algorithm']['process_reward'])['weights']['gold_write'] == 1
    assert env['TAU3_BUDGET_INTERVAL'] == '10'
    assert env['TAU3_STOP_REQUEST_PATH'].endswith('/STOP_AFTER_BOUNDARY')
    assert 'uncalibrated-exploration' in env['SWANLAB_EXPERIMENT_NAME']
    assert baseline_env['SWANLAB_EXPERIMENT_NAME'] != env['SWANLAB_EXPERIMENT_NAME']


@pytest.mark.parametrize('override,match', [
    ({'uncalibrated_exploration': False}, 'passed --reward-recipe'),
    ({'engineering_smoke': True}, 'cannot use'),
    ({'reward_recipe': 'not-a-real-recipe.json'}, 'cannot use'),
    ({'credit_mode': 'call_local_v1'}, 'requires mt_gtpo'),
    ({'credit_mode': 'call_residual_v1'}, 'requires mt_gtpo'),
    ({'reward_version': 'paper_env_split_v3'}, 'requires mt_gtpo'),
    ({'estimator': 'grpo'}, 'requires mt_gtpo'),
    ({'estimator': 'tau_gigpo'}, 'requires mt_gtpo'),
    ({'updates': 2}, 'multiples of ten'),
])
def test_exploration_cannot_silently_bypass_other_modes(setup, override, match):
    result, args = setup
    with pytest.raises(ValueError, match=match):
        runner.resolve(result, **(args | override))


def test_exploration_resume_requires_same_run_identity(setup, monkeypatch):
    result, args = setup
    result.mkdir()
    checkpoint = result / 'global_step_10'
    with pytest.raises(ValueError, match='exploration identity'):
        runner.resolve(result, **(args | {'resume_from': checkpoint}))
    command, env, snapshot = runner.resolve(result, **args)
    config = resolved(command, env)
    (result / 'launch.json').write_text(json.dumps(snapshot))
    (result / 'resolved-hydra.yaml').write_text(yaml.safe_dump(config))
    (result / 'swanlab-run.json').write_text('{}')
    calls = []
    monkeypatch.setattr(runner, 'validate_checkpoint', lambda *a, **kw: calls.append((a, kw)))
    command, env, _ = runner.resolve(result, **(args | {'resume_from': checkpoint}))
    resumed = resolved(command, env)
    assert resumed['trainer']['resume_mode'] == 'resume_path'
    assert resumed['trainer']['resume_from_path'] == str(checkpoint)
    assert runner.validate_exploration_config(resumed, config)
    assert calls[0][1]['world_size'] == 8
    with pytest.raises(ValueError, match='exploration identity'):
        runner.resolve(result, **(args | {'resume_from': checkpoint, 'uncalibrated_exploration': False}))


@pytest.mark.parametrize('change', ['reward', 'algorithm', 'identity'])
def test_exploration_resume_rejects_changed_effective_settings(setup, change):
    result, args = setup
    command, env, _ = runner.resolve(result, **args)
    previous = resolved(command, env)
    config = copy.deepcopy(previous)
    if change == 'reward':
        config['algorithm']['process_reward']['weights'] = {'gold_write': 0.5}
    elif change == 'algorithm':
        config['algorithm']['mt_gtpo']['gamma'] = 0.5
    else:
        config['tau3_irc_calibrated'] = True
    with pytest.raises(ValueError, match='exploration'):
        runner.validate_exploration_config(config, previous)


def test_cli_dry_run_records_exploration_without_starting_services(setup, monkeypatch, capsys):
    result, args = setup
    command, env, _ = runner.resolve(result, **args)
    config = resolved(command, env)
    # Model/data checks and remote Hydra subprocess are mocked; composition above is real.
    monkeypatch.setattr(runner, 'load_tracking_env', lambda: None)
    monkeypatch.setattr(runner, 'validate_inputs', lambda *a: {'candidate_trajectories': 1920})
    outputs = iter(['python ignored\n', yaml.safe_dump(config)])
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: next(outputs))
    monkeypatch.setattr(runner, 'active_gpu_pids', lambda: pytest.fail('Dry run queried GPU'))
    monkeypatch.setattr(runner, 'launch_process', lambda *a, **kw: pytest.fail('Started service'))
    runner.main(['--result-dir', str(result), '--profile', str(PROFILE), '--updates', '30',
                 '--reward-version', 'paper_env_split_v4', '--credit-mode', 'turn_v1',
                 '--token-protocol', 'tau3_token_budget_v1', '--uncalibrated-exploration', '--dry-run'])
    launch = json.loads((result / 'launch.json').read_text())
    preflight = json.loads((result / 'preflight.json').read_text())
    assert launch['uncalibrated_exploration'] and launch['uncalibrated_initializer']
    assert launch['exploration_settings']['reward']['weights']['gold_write'] == 1
    assert preflight['uncalibrated_exploration'] and preflight['irc_calibrated'] is False
    assert not (result / 'frozen-recipe.json').exists()
    assert not (result / 'controller-state.json').exists()
    assert 'uncalibrated_exploration' in capsys.readouterr().out
