"""Verify the reduced-budget profile reaches trainer with its own task pool."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tau3_grpo.launch import prepare
from tau3_grpo.paths import CODE_ROOT


@pytest.mark.parametrize(
    "size,phase,updates",
    [(40, "pilot", 10), (40, "extend", 50), (50, "pilot", 10), (50, "extend", 100)],
)
def test_curriculum_budget_and_resume_profile(size, phase, updates, tmp_path):
    config = (
        CODE_ROOT / f"configs/train/rl/qwen35_4b_full_a800_curriculum{size}_{phase}_20260912.yaml"
    )
    command, env, _ = prepare(
        "rl",
        config,
        "e0",
        42,
        [],
        {
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "TAU3_DRY_RUN": "1",
            "TAU3_ENV_FILE": str(tmp_path / "absent.env"),
        },
    )
    assert env["TRAIN_MANIFEST_DIR"].endswith(
        f"rl_curriculum{'50' if size == 50 else ''}_20260912/manifests"
    )
    assert (
        int(env["GROUPS_PER_UPDATE"]) * int(env["GROUP_SIZE"]) * int(env["TOTAL_UPDATES"])
        == 64 * updates
    )
    result = subprocess.run(
        command, env=env, cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert "data.train_batch_size=8" in result.stdout
    assert f"trainer.total_training_steps={updates}" in result.stdout
    assert result.stdout.rfind("trainer.resume_mode=auto") > result.stdout.rfind(
        "trainer.resume_mode=disable"
    )
    assert "actor_rollout_ref.model.lora_rank=0" in result.stdout
