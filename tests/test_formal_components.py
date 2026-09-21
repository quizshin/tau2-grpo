"""Compare composed current profiles with retained historical protocol inputs."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tau3_grpo.configuration import load_config_with_sources
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.training.rl import runner

HISTORICAL = {
    "v2": "formal_20260916", "v3": "v3_20260917", "paper_v1": "paper_20260917",
    "paper_env_v2": "env_20260918", "paper_env_split_v3": "split_20260918",
}


def test_5090_profile_preserves_formal_budget_and_tracking(tmp_path, monkeypatch):
    for key, value in {'TAU3_ROOT': tmp_path, 'TAU3_RUN_ROOT': tmp_path / 'runs',
                       'TAU3_DATA_ROOT': tmp_path / 'data', 'TAU3_MODEL_ROOT': tmp_path / 'models',
                       'TAU3_ENV_FILE': tmp_path / 'absent'}.items():
        monkeypatch.setenv(key, str(value))
    command, env, _ = runner.resolve(tmp_path / 'run', updates=30, estimator='grpo',
        token_protocol='tau3_token_budget_v1',
        profile_override=CODE_ROOT / 'configs/train/rl/formal50_5090_a45.yaml')
    config = resolved(command, env)
    actor = config['actor_rollout_ref']
    assert config['algorithm']['adv_estimator'] == 'grpo'
    assert not config['algorithm']['dynamic_filter']['enable']
    assert config['data']['train_batch_size'] == actor['rollout']['n'] == 8
    assert actor['actor']['ppo_mini_batch_size'] == 4
    assert actor['model']['lora_rank'] == 0
    assert actor['model']['enable_activation_offload']
    assert config['trainer']['total_training_steps'] == 30
    assert config['trainer']['save_freq'] == config['trainer']['test_freq'] == 10
    assert config['trainer']['max_actor_ckpt_to_keep'] == 1
    assert config['trainer']['logger'] == ['console', 'swanlab']
    assert actor['actor']['checkpoint']['save_contents'] == ['model', 'optimizer', 'extra']
    assert config['tau3_token_protocol'] == 'tau3_token_budget_v1'
    assert env['SWANLAB_MODE'] == 'online'
    simulator = runner.simulator_environment(env)
    assert simulator['TAU3_USER_CUDA_DEVICES'] == '4,5,6,7'
    assert simulator['TAU3_USER_TP'] == '4'
    with pytest.raises(ValueError, match='distinct'):
        runner.simulator_environment(dict(env, TAU3_USER_CUDA_DEVICES='3,4,5,6'))
    with pytest.raises(ValueError, match='endpoint'):
        runner.simulator_environment(dict(env, TAU3_USER_PORT='9999'))


def test_deployed_source_snapshot_excludes_secrets(tmp_path, monkeypatch):
    root = tmp_path / 'code'
    (root / 'tau3_grpo').mkdir(parents=True)
    (root / 'tau3_grpo/example.py').write_text('VALUE = 1\n')
    (root / '.env').write_text('SECRET=private\n')
    (root / 'results').mkdir()
    (root / 'results/private.json').write_text('{}')
    monkeypatch.setattr(runner, 'CODE_ROOT', root)
    destination = tmp_path / 'snapshot'
    runner.snapshot_source(destination)
    import json
    hashes = json.loads((destination / 'source-sha256.json').read_text())
    assert set(hashes) == {'tau3_grpo/example.py'}
    identity = json.loads((destination / 'source-identity.json').read_text())
    assert identity['git_revision'] is None


@pytest.mark.parametrize('profile', ['formal50_5090_a45_shared8.yaml', 'formal50_5090_a45_fsdp2.yaml'])
def test_shared8_profile_preserves_minibatch_and_has_phase_manager(tmp_path, monkeypatch, profile):
    for key, value in {'TAU3_ROOT': tmp_path, 'TAU3_RUN_ROOT': tmp_path / 'runs',
                       'TAU3_DATA_ROOT': tmp_path / 'data', 'TAU3_MODEL_ROOT': tmp_path / 'models',
                       'TAU3_ENV_FILE': tmp_path / 'absent'}.items():
        monkeypatch.setenv(key, str(value))
    command, env, _ = runner.resolve(tmp_path / 'run', updates=30, estimator='grpo',
        token_protocol='tau3_token_budget_v1',
        profile_override=CODE_ROOT / 'configs/train/rl' / profile)
    config = resolved(command, env)
    if profile.endswith('_fsdp2.yaml'):
        assert config['actor_rollout_ref']['actor']['strategy'] == 'fsdp2'
        assert config['actor_rollout_ref']['actor']['fsdp_config']['offload_policy']
        for role in ['actor', 'ref']:
            assert config['actor_rollout_ref'][role]['fsdp_config']['wrap_policy']['transformer_layer_cls_to_wrap'] == ['Qwen3_5DecoderLayer', 'Qwen3_5VisionBlock']
    assert config['trainer']['n_gpus_per_node'] == 8
    assert config['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == 4
    assert config['actor_rollout_ref']['rollout']['n'] == 8
    assert not config['actor_rollout_ref']['model']['enable_activation_offload']
    assert config['actor_rollout_ref']['model']['external_lib'].endswith('cpu_saved_tensors')
    assert config['actor_rollout_ref']['rollout']['agent']['agent_loop_manager_class'].endswith('SleepingSimulatorAgentLoopManager')
    assert runner.simulator_environment(env)['TAU3_USER_CUDA_DEVICES'] == '4,5,6,7'
    with pytest.raises(ValueError, match='distinct'):
        runner.simulator_environment(dict(env, TAU3_SIMULATOR_COLOCATED_SLEEP='0'))


def resolved(command, env):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    output = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN="1"), text=True)
    args = shlex.split(output.splitlines()[-1])
    overrides = args[args.index("tau3_grpo.training.rl.train") + 1:]
    with initialize_config_dir(config_dir=str(CODE_ROOT / "verl/verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    return OmegaConf.to_container(config, resolve=True)


@pytest.mark.parametrize("df", [False, True])
@pytest.mark.parametrize("estimator,version", [("grpo", "v3"), ("tau_gigpo", "v3")]
                         + [("mt_gtpo", v) for v in HISTORICAL])
def test_current_formal_hydra_matches_historical(estimator, version, df, tmp_path, monkeypatch):
    for key, value in {"TAU3_ROOT": tmp_path, "TAU3_RUN_ROOT": tmp_path / "runs",
                       "TAU3_MODEL_ROOT": tmp_path / "models", "TAU3_ENV_FILE": tmp_path / "absent"}.items():
        monkeypatch.setenv(key, str(value))
    args = dict(estimator=estimator, reward_version=version, dynamic_filter=df, allow_uncalibrated=True)
    command, env, snapshot = runner.resolve(tmp_path / "same-run", **args)
    current = resolved(command, env)
    assert all("202609" not in Path(s["path"]).name for s in snapshot["configuration_sources"])
    if estimator == "mt_gtpo":
        monkeypatch.setitem(runner.PROFILES, version, CODE_ROOT / (
            f"configs/train/rl/qwen35_4b_full_a800_mt_gtpo_{HISTORICAL[version]}.yaml"))
    else:
        arm = ("e1" if df else "e0") if estimator == "grpo" else ("e3" if df else "e2")
        monkeypatch.setattr(runner, "BASE_PROFILE", CODE_ROOT / (
            f"configs/train/rl/qwen35_4b_full_a800_c50_matched6h_{arm}_20260912.yaml"))
    old_command, old_env, _ = runner.resolve(tmp_path / "same-run", **args)
    historical = resolved(old_command, old_env)
    # Only sampling defaults intentionally changed on 2026-09-20.
    assert current["actor_rollout_ref"]["rollout"]["temperature"] == 0.7
    assert current["actor_rollout_ref"]["rollout"]["val_kwargs"]["temperature"] == 0.7
    assert historical["actor_rollout_ref"]["rollout"]["temperature"] == 1.0
    assert historical["actor_rollout_ref"]["rollout"]["val_kwargs"]["temperature"] == 0.4
    historical["actor_rollout_ref"]["rollout"]["temperature"] = 0.7
    historical["actor_rollout_ref"]["rollout"]["val_kwargs"]["temperature"] = 0.7
    assert current == historical
    for key in env:
        if key not in os.environ and key not in {"PYTHONPATH", "TRAIN_TEMP"}:
            assert env[key] == old_env[key], key


def test_active_include_graph_has_no_historical_profiles():
    for profile in [runner.BASE_PROFILE, *runner.PROFILES.values()]:
        config, sources = load_config_with_sources(profile)
        assert config["launch"]["environment"]["TOTAL_UPDATES"] == "20"
        assert not any("pilot" in s["path"] or "202609" in Path(s["path"]).name for s in sources)


@pytest.mark.parametrize('df', [False, True])
def test_v4_profile_reaches_hydra_and_retains_recipe_gate(df, tmp_path, monkeypatch):
    from tau3_grpo.evaluation.process_reward import reward_settings

    for key, value in {'TAU3_ROOT': tmp_path, 'TAU3_RUN_ROOT': tmp_path / 'runs',
                       'TAU3_MODEL_ROOT': tmp_path / 'models', 'TAU3_ENV_FILE': tmp_path / 'absent'}.items():
        monkeypatch.setenv(key, str(value))
    with pytest.raises(ValueError, match='passed --reward-recipe'):
        runner.resolve(tmp_path / 'run', reward_version='paper_env_split_v4')
    command, env, _ = runner.resolve(tmp_path / 'run', reward_version='paper_env_split_v4',
                                     dynamic_filter=df, allow_uncalibrated=True)
    config = resolved(command, env)
    settings = reward_settings(config['algorithm']['process_reward'])
    assert settings['version'] == 'paper_env_split_v4'
    assert settings['weights']['generic'] == 0
    assert settings['paper_options']['soft_scoring'] == 'constant'
