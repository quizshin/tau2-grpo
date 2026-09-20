"""Protect direct-shell compatibility, catalog routing and override precedence."""
import os
import shlex
import subprocess
import sys

import pytest

from tau3_grpo.configuration import load_config, resolve_arm
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.training.rl.runtime_defaults import ARMS, arm_environment, defaults


@pytest.mark.parametrize("arm", ["e0", "e1", "e2", "e3", "mt_gtpo",
                                 "e2_db_hash_only", "e2_similarity"])
def test_direct_shell_uses_catalog_and_cli_last(arm, tmp_path):
    env = dict(os.environ, TAU3_DRY_RUN="1", TAU3_ENV_FILE=str(tmp_path / "absent"),
               GROUP_SIZE="2", GROUPS_PER_UPDATE="3", POLICY_GPUS="1", ROLLOUT_TP="1",
               PPO_MINI_GROUPS="3", DATA_SPLIT_SEED="47", RESULTS_DIR=str(tmp_path / "run"),
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(CODE_ROOT / "scripts/train/rl/run_base.sh"),
                             arm, "43", "trainer.save_freq=7"], env=env, cwd=tmp_path,
                            text=True, capture_output=True, check=True, timeout=30)
    command = shlex.split(result.stdout.splitlines()[-1])
    catalog = resolve_arm(arm, load_config(ARMS))
    assert f"algorithm.adv_estimator={catalog['adv_estimator']}" in command
    assert "data.train_batch_size=3" in command
    assert "actor_rollout_ref.rollout.n=2" in command
    assert "data.seed=43" in command
    assert any("airline_selection_seed47.parquet" in x for x in command)
    assert command[-1] == "trainer.save_freq=7"
    assert not (tmp_path / "run").exists()


def test_empty_values_match_shell_fallback_and_inherited_values_win():
    values = defaults("base", {"GROUP_SIZE": "", "LR": "2e-6", "TOTAL_UPDATES": "0"})
    assert values["GROUP_SIZE"] == "8"
    assert values["LR"] == "2e-6"
    assert values["TOTAL_UPDATES"] == "0"
    assert defaults("qwen35", {}, size="9B")["UPDATE_WEIGHTS_BUCKET_MEGABYTES"] == "4096"
    assert defaults("qwen35", {"UPDATE_WEIGHTS_BUCKET_MEGABYTES": "8192"}, size="9B")[
        "UPDATE_WEIGHTS_BUCKET_MEGABYTES"] == "8192"


def test_legacy_algorithm_override_and_invalid_arm():
    assert arm_environment("e2", {"TAU3_GRPO_CONFIG_DF_ENABLE": "true"})["DF_ENABLE"] == "true"
    with pytest.raises(ValueError, match="unknown arm"):
        arm_environment("typo", {})


def test_shell_output_quotes_inherited_text_without_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    value = f"$(touch {marker}); `touch {marker}` ' \"\n"
    output = subprocess.check_output(
        [sys.executable, "-m", "tau3_grpo.training.rl.runtime_defaults", "base"],
        env=dict(os.environ, LR=value), text=True)
    result = subprocess.check_output(["bash", "-c", output + '\nprintf "%s" "$LR"'], text=True)
    assert result == value
    assert not marker.exists()
