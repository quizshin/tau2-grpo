import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tau3_grpo.configuration import load_config_with_sources
from tau3_grpo.data.messages import visible_message
from tau3_grpo.launch import prepare
from tau3_grpo.paths import CODE_ROOT


def test_config_source_order_and_environment_priority(tmp_path):
    parent = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("launch:\n  environment:\n    LR: 1e-6\n  overrides: [one]\n")
    child.write_text("includes: [base.yaml]\nlaunch:\n  overrides: [two]\n")
    data, sources = load_config_with_sources(child)
    assert data["launch"]["overrides"] == ["one", "two"]
    assert [Path(s["path"]).name for s in sources] == ["base.yaml", "child.yaml"]
    assert all(len(s["sha256"]) == 64 for s in sources)
    _, env, snapshot = prepare("rl", CODE_ROOT / "configs/train/rl/qwen35_full.yaml",
                               "e0", 42, [], {"LR": "2e-6"})
    assert env["LR"] == "2e-6"
    assert snapshot["environment_sources"]["LR"] == "inherited_environment"
    assert snapshot["snapshot_scope"] == "launch_inputs_not_final_hydra"


@pytest.mark.parametrize("text", ["[]", "includes: x", "includes: [12]"])
def test_bad_include_shape_is_rejected(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config_with_sources(path)


def test_visible_projection_does_not_leak_or_alias():
    raw = {"role": "assistant", "tool_calls": [{"arguments": {"x": [1]}}],
           "reward": 1, "gold": {"hidden": True}, "usage": {"secret": 1}}
    projected = visible_message(raw)
    assert set(projected) == {"role", "tool_calls"}
    projected["tool_calls"][0]["arguments"]["x"].append(2)
    assert raw["tool_calls"][0]["arguments"]["x"] == [1]


@pytest.mark.parametrize("old,new", [("verl_estimator", "gigpo"), ("mt_gtpo_verl", "mt_gtpo")])
def test_legacy_estimator_is_same_module_object(old, new):
    legacy = importlib.import_module(f"tau3_grpo.algorithms.{old}")
    canonical = importlib.import_module(f"tau3_grpo.integrations.verl.{new}")
    assert legacy is canonical


@pytest.mark.parametrize("estimator", ["grpo", "tau_gigpo", "mt_gtpo"])
@pytest.mark.parametrize("df", [False, True])
def test_formal_runner_six_arms_resolve_without_launch(tmp_path, monkeypatch, estimator, df):
    from tau3_grpo.training.rl.runner import resolve

    for key, value in {"TAU3_ROOT": str(tmp_path), "TAU3_RUN_ROOT": str(tmp_path / "runs"),
                       "TAU3_MODEL_ROOT": str(tmp_path / "models"),
                       "TAU3_ENV_FILE": str(tmp_path / "absent.env"),
                       "MODEL_PATH": "/must/not/override/sft", "TOTAL_UPDATES": "999",
                       "TAU3_E0_DISCOVERY_SECONDS": "1"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    destination = tmp_path / "new"
    command, env, snapshot = resolve(destination, estimator=estimator, dynamic_filter=df)
    assert env["MODEL_PATH"] == str(tmp_path / "code/checkpoints/sft-merged/new-off")
    assert env["TOTAL_UPDATES"] == "20" and "TAU3_E0_DISCOVERY_SECONDS" not in env
    assert env["GROUP_SIZE"] == env["GROUPS_PER_UPDATE"] == "8"
    output = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN="1"), text=True, timeout=30)
    settings = dict(a.lstrip("+").split("=", 1) for a in shlex.split(output.splitlines()[-1]) if "=" in a)
    assert settings["algorithm.adv_estimator"] == estimator
    assert settings["algorithm.dynamic_filter.enable"] == str(df).lower()
    assert settings["trainer.total_training_steps"] == "20"
    assert settings["trainer.save_freq"] == settings["trainer.test_freq"] == "10"
    assert settings["actor_rollout_ref.rollout.val_kwargs.n"] == "4"
    assert settings["actor_rollout_ref.rollout.calculate_log_probs"] == "true"
    assert settings["actor_rollout_ref.rollout.logprobs_mode"] == "processed_logprobs"
    assert snapshot["estimator"] == estimator
    assert snapshot["environment_sources"]["MODEL_PATH"] == "formal_profile"
    assert snapshot["environment_sources"]["TOTAL_UPDATES"] == "formal_controller"
    assert not destination.exists()


def test_formal_runner_guards_algorithm_and_recipe(tmp_path):
    from tau3_grpo.training.rl.runner import resolve

    with pytest.raises(ValueError, match="Unknown estimator"):
        resolve(tmp_path, estimator="other")
    with pytest.raises(ValueError, match="requires mt_gtpo"):
        resolve(tmp_path, estimator="grpo", reward_recipe=tmp_path / "recipe.json")


@pytest.mark.parametrize("previous", ["grpo", ""])
def test_resume_rejects_cross_algorithm_or_missing_identity(tmp_path, monkeypatch, previous):
    from tau3_grpo.training.rl.runner import resolve

    monkeypatch.setenv("TAU3_ROOT", str(tmp_path))
    monkeypatch.setenv("TAU3_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("TAU3_MODEL_ROOT", str(tmp_path / "models"))
    (tmp_path / "resolved-hydra.yaml").write_text(
        f"algorithm:\n  adv_estimator: '{previous}'\n  dynamic_filter:\n    enable: false\n")
    with pytest.raises(ValueError, match="estimator"):
        resolve(tmp_path, estimator="tau_gigpo", resume_from=tmp_path / "global_step_10")
