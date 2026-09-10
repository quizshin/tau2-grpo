"""Exercise the new entrypoint without launching GPU work or a tracking run."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tau3_grpo.launch import load_config, prepare
from tau3_grpo.paths import CODE_ROOT


@pytest.mark.parametrize("family", ["qwen25", "qwen35"])
@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("arm", ["e0", "e1", "e2", "e3"])
def test_rl_profiles_reach_real_trainer_and_preserve_algorithm(family, method, arm, tmp_path):
    path = CODE_ROOT / f"configs/train/rl/{family}_{method}.yaml"
    inherited = {**os.environ, "TAU3_ENV_FILE": str(tmp_path / "absent.env"),
                 "TAU3_DRY_RUN": "1", "TAU3_RUN_ROOT": str(tmp_path / "runs"),
                 "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    command, env, snapshot = prepare("rl", path, arm, 43,
                                     ["trainer.save_freq=7"], inherited)
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True,
                            text=True, check=True, timeout=20)
    assert "tau3_grpo.training.rl.train" in result.stdout
    estimator = "tau_gigpo" if arm in {"e2", "e3"} else "grpo"
    assert f"algorithm.adv_estimator={estimator}" in result.stdout
    rendered_command = " ".join(shlex.split(result.stdout.splitlines()[-1]))
    assert f"enable:{str(arm in {'e1', 'e3'}).lower()},mode:fixed_rollout,group_size:4" in rendered_command
    assert "trainer.save_freq=7" in result.stdout
    assert f"actor_rollout_ref.model.lora_rank={16 if method == 'lora' else 0}" in result.stdout
    assert snapshot["experiment"]["dynamic_filter"]["enable"] is (arm in {"e1", "e3"})
    assert not (tmp_path / "runs").exists()


def test_sft_config_and_external_model_path_are_forwarded(tmp_path):
    config = CODE_ROOT / "configs/train/sft/qwen35_full.yaml"
    command, env, _ = prepare("sft", config, "e0", None, [],
                              {"TAU3_MODEL_ROOT": str(tmp_path / "models")})
    assert command[-2:] == ["--config", str(config)]
    assert env["SFT_MODEL_NAME_OR_PATH"] == str(tmp_path / "models/Qwen3.5-0.8B")
    assert load_config(config)["train"]["method"] == "full"


def test_config_inheritance_preserves_explicit_environment_and_overrides(tmp_path):
    config = CODE_ROOT / "configs/train/rl/qwen35_lora.yaml"
    _, env, _ = prepare("rl", config, "e0", None, [],
                         {"QWEN35_SIZE": "9B", "LR": "2e-6", "MODEL_PATH": "/my/sft"})
    assert env["LR"] == "2e-6"
    assert env["MODEL_PATH"] == "/my/sft"
    assert env["QWEN35_MODEL_PATH"].endswith("Qwen3.5-9B")


def test_include_cycle_is_rejected(tmp_path):
    config = tmp_path / "loop.yaml"
    config.write_text("includes: [loop.yaml]\n")
    with pytest.raises(ValueError, match="cyclic"):
        load_config(config)


def test_unified_dry_run_from_other_directory_does_not_expose_credentials(tmp_path):
    env = {**os.environ, "TAU3_ENV_FILE": str(tmp_path / "absent.env"),
           "SWANLAB_API_KEY": "test-secret-never-print", "PYTHONPATH": str(CODE_ROOT)}
    result = subprocess.run([sys.executable, "-m", "tau3_grpo.launch", "simulator", "--config",
                             "configs/simulator/qwen35_4b.yaml", "--dry-run"],
                            cwd=tmp_path, env=env, capture_output=True, text=True,
                            check=True, timeout=20)
    output = json.loads(result.stdout)
    assert output["stage"] == "simulator"
    assert "test-secret-never-print" not in result.stdout
    assert not list(tmp_path.iterdir())
