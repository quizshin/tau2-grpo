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


def test_formal_input_paths_follow_external_data_root_without_changing_pool(tmp_path):
    from tau3_grpo.data.manifest import read_manifest
    from tau3_grpo.paths import RL_CURRICULUM50_MANIFEST_ROOT
    from tau3_grpo.utils.hashing import sha256_file

    profile = CODE_ROOT / 'configs/train/rl/formal50_a800.yaml'
    _, env, _ = prepare('rl', profile, 'e0', 42, [], {'TAU3_DATA_ROOT': str(tmp_path / 'assets'), 'TAU3_ROOT': str(tmp_path / 'runtime')})
    assert env['TRAIN_MANIFEST_DIR'] == str(tmp_path / 'assets/manifests/rl_curriculum50_seed42')
    assert env['VAL_PARQUET'] == str(tmp_path / 'assets/parquet/airline_selection_seed42.parquet')
    path = RL_CURRICULUM50_MANIFEST_ROOT / 'areal_airline_train_seed42.jsonl'
    assert sha256_file(path) == '641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae'
    assert len(read_manifest(path)) == 50


def test_qwen38_simulator_dry_run_uses_current_roots_and_preserves_explicit_runtime(tmp_path):
    script = CODE_ROOT / 'scripts/serve/simulator_qwen38.sh'
    inherited = {k: v for k, v in os.environ.items() if not k.startswith('TAU3_')}
    env = dict(inherited, TAU3_DRY_RUN='1', TAU3_ROOT=str(tmp_path / 'runtime'),
               TAU3_MODEL_ROOT=str(tmp_path / 'models'), TAU3_CACHE_ROOT=str(tmp_path / 'cache'))
    command = shlex.split(subprocess.check_output(['bash', str(script)], env=env, text=True))
    assert command[0] == str(tmp_path / 'runtime/environment/venvs/qwen38-sim/bin/python')
    assert command[command.index('serve') + 1] == str(tmp_path / 'models/Qwen3.8-27B-AWQ-INT4')
    env.update(TAU3_SIM_PYTHON='/custom/python', TAU3_USER_MODEL='/custom/model')
    command = shlex.split(subprocess.check_output(['bash', str(script)], env=env, text=True))
    assert command[0] == '/custom/python' and command[command.index('serve') + 1] == '/custom/model'
