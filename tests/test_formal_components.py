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
    assert current == resolved(old_command, old_env)
    for key in env:
        if key not in os.environ and key != "PYTHONPATH":
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
