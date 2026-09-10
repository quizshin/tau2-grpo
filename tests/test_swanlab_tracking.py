"""Logging must preserve step semantics, real trajectories and credential boundaries."""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from tau3_grpo.tracking import swanlab as tracking
from tau3_grpo.tracking.upload_sft import historical_metrics


@pytest.fixture(autouse=True)
def isolated_tracking_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU3_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def test_historical_metrics_preserve_baseline_steps_and_validation():
    summary = {"baseline_validation": {"eval_loss": 0.8}}
    state = {"log_history": [
        {"step": 1, "loss": 0.7, "learning_rate": 0.01},
        {"step": 1, "eval_loss": 0.75, "epoch": 1},
        {"step": 2, "loss": 0.6},
    ]}
    values = historical_metrics(summary, state)
    assert list(values) == [0, 1, 2]
    assert values[0] == {"val/loss": 0.8, "train/global_step": 0}
    assert values[1]["train/loss"] == 0.7
    assert values[1]["val/loss"] == 0.75
    assert values[1]["train/lr"] == 0.01


def test_config_redaction_and_source_archive_exclude_runtime_secrets(tmp_path):
    config = tracking.redact_config({"SWANLAB_API_KEY": "secret", "model": {"api_key": "key", "total_tokens": 42}})
    assert config["SWANLAB_API_KEY"] == "[REDACTED]"
    assert config["model"] == {"api_key": "[REDACTED]", "total_tokens": 42}
    root = tmp_path / "code"
    source = root / "tau3_grpo"
    source.mkdir(parents=True)
    (source / "train.py").write_text("pass\n")
    (source / ".env").write_text("PRIVATE_KEY=secret")
    (source / "credentials.py").symlink_to(source / ".env")
    (root / ".netrc").write_text("secret")
    (root / "model.safetensors").write_bytes(b"weights")
    archive = tracking.create_source_archive(tmp_path / "out", root)
    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ["tau3_grpo/train.py"]


def test_conversation_preserves_tool_arguments_outputs_and_roles():
    text = tracking.conversation_text({"messages": [
        {"role": "assistant", "tool_calls": [{"name": "lookup", "arguments": {"id": "0012"}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "result"},
    ]})
    assert "0012" in text and "lookup" in text and "result" in text
    assert "assistant" in text and "tool_call_id=call1" in text


def test_rl_logging_uses_real_unpadded_trajectories(monkeypatch, tmp_path):
    import swanlab
    import torch

    run = SimpleNamespace(dir=tmp_path / "run")
    monkeypatch.setattr(swanlab, "get_run", lambda: run)
    monkeypatch.setattr(tracking, "attach_sources", lambda run, out: out.mkdir(parents=True))
    captured = []
    monkeypatch.setattr(tracking, "log_examples", lambda run, rows, **kw: captured.extend(rows))
    batch = SimpleNamespace(
        batch={"token_level_scores": torch.tensor([[1.0], [0.0], [1.0]])},
        non_tensor_batch={
            "tau3_is_padding": np.array([False, False, True]),
            "task_id": np.array(["correct", "wrong", "padding"]),
            "trajectory_json": np.array([json.dumps({"messages": [{"role": "user", "content": "hi"}]})] * 3),
        },
    )
    tracking.log_rl_examples(batch=batch, tokenizer=None, update_index=3,
                             loggers=["console", "swanlab"], output_dir=str(tmp_path))
    assert [row["task_id"] for row in captured] == ["correct", "wrong"]
    assert captured[0]["trajectory"]["messages"][0]["content"] == "hi"


def test_metadata_only_trajectory_decodes_all_roles_without_padding(monkeypatch, tmp_path):
    import swanlab
    import torch

    run = SimpleNamespace(dir=tmp_path / "run")
    monkeypatch.setattr(swanlab, "get_run", lambda: run)
    monkeypatch.setattr(tracking, "attach_sources", lambda run, out: out.mkdir(parents=True))
    captured = []
    monkeypatch.setattr(tracking, "log_examples", lambda run, rows, **kw: captured.extend(rows))
    batch = SimpleNamespace(
        batch={
            "token_level_scores": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "prompts": torch.tensor([[0, 0, 1, 2]]),
            "responses": torch.tensor([[3, 4, 5, 0]]),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 0, 1, 0]]),
        },
        non_tensor_batch={"task_id": np.array(["t1"]), "trajectory_json": np.array([
            json.dumps({"session_id": "session1", "num_turns": 3})])},
    )
    words = {0: "PAD", 1: "system", 2: "user", 3: "assistant", 4: "tool-result", 5: "answer"}
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: "\n".join(words[int(i)] for i in ids))
    tracking.log_rl_examples(batch=batch, tokenizer=tokenizer, update_index=1,
                             loggers=["swanlab"], output_dir=str(tmp_path))
    row = captured[0]
    assert row["trajectory_metadata"]["session_id"] == "session1"
    text = tracking.conversation_text(row["trajectory"])
    assert "PAD" not in text and "tool-result" in text and "answer" in text
    assert "system\nuser" in text and "assistant\ntool-result" in text


def test_example_sampling_preserves_reward_balance_and_task_diversity():
    rows = [{"task_id": task, "reward": reward} for task, reward in
            [("a", 1), ("b", 1), ("c", 0), ("d", 0)] for _ in range(4)]
    indices = tracking.sample_indices(rows, count=4, seed=43)
    assert len({rows[i]["task_id"] for i in indices}) == 4
    assert sum(rows[i]["reward"] for i in indices) == 2
    assert indices == tracking.sample_indices(rows, count=4, seed=43)


def test_recovery_checks_recorded_points_instead_of_tied_summary_locations():
    from copy import deepcopy

    from tau3_grpo.tracking.upload_rl import scalar_records

    original = {"list": [{"key": "actor/grad_norm", "metrics": [
        {"step": 1, "value": 0}, {"step": 2, "value": 0}],
        "min": {"index": 1, "data": 0}}]}
    reread = deepcopy(original)
    reread["list"][0]["min"]["index"] = 2
    assert scalar_records(original) == scalar_records(reread)
    reread["list"][0]["metrics"][1]["value"] = 0.5
    assert scalar_records(original) != scalar_records(reread)


def test_recovery_preserves_original_wrapped_training_config():
    from tau3_grpo.tracking.upload_rl import config_values

    wrapped = {"trainer": {"desc": "", "sort": 1, "value": {"phase": "rl"}},
               "actor": {"desc": "", "sort": 0, "value": {"model": "Qwen3.5-0.8B"}}}
    restored = config_values(wrapped)
    assert restored["trainer"] == {"phase": "rl"}
    assert restored["actor"]["model"] == "Qwen3.5-0.8B"


def test_real_swanlab_offline_metrics_table_and_artifacts(monkeypatch, tmp_path):
    import swanlab

    monkeypatch.setenv("SWANLAB_MODE", "offline")
    monkeypatch.setenv("SWANLAB_LOG_DIR", str(tmp_path / "logs"))
    source = tmp_path / "source"
    (source / "tau3_grpo").mkdir(parents=True)
    (source / "tau3_grpo/train.py").write_text("pass\n")
    create_archive = tracking.create_source_archive
    monkeypatch.setattr(tracking, "create_source_archive", lambda out: create_archive(out, source))
    run = tracking.start_run(name="offline-regression", config={"method": "full"}, output=tmp_path / "output")
    try:
        assert run.config["trainer"]["phase"] == "sft"
        run.log({"train/loss": 0.5}, step=1)
        tracking.log_examples(run, [{"task_id": "t1", "reward": 1.0,
            "simulation": {"messages": [{"role": "assistant", "content": "done"}]}}],
            key="selection/examples", step=1, output=tmp_path / "examples")
        assert tracking.run_url(run) is None
        assert (tmp_path / "examples/selection-examples-step000001.jsonl").exists()
    finally:
        run.finish()
    with pytest.raises(RuntimeError, match="No active Run"):
        swanlab.get_run()


def test_sft_callback_logs_real_step_only_on_primary_process():
    received = []
    run = SimpleNamespace(log=lambda metrics, step: received.append((metrics, step)))
    callback = tracking.sft_callback(run)
    state = SimpleNamespace(is_world_process_zero=True, global_step=12)
    callback.on_log(None, state, None, logs={"eval_loss": 0.56})
    state.is_world_process_zero = False
    callback.on_log(None, state, None, logs={"eval_loss": 99.0})
    assert received == [({"val/loss": 0.56, "train/global_step": 12}, 12)]


def test_native_verl_swanlab_backend_accepts_training_samples_offline(monkeypatch, tmp_path):
    import swanlab
    import torch
    from verl.utils.tracking import Tracking

    monkeypatch.setenv("SWANLAB_MODE", "offline")
    monkeypatch.setenv("SWANLAB_LOG_DIR", str(tmp_path / "logs"))
    source = tmp_path / "source"
    source.mkdir()
    create_archive = tracking.create_source_archive
    monkeypatch.setattr(tracking, "create_source_archive", lambda out: create_archive(out, source))
    logger = Tracking(project_name="offline-rl", experiment_name="rl-compat",
                      default_backend=["swanlab"], config={"algorithm": "grpo"})
    batch = SimpleNamespace(batch={"token_level_scores": torch.tensor([[1.0]])},
        non_tensor_batch={"task_id": np.array(["t1"]), "trajectory_json": np.array([
            json.dumps({"messages": [{"role": "tool", "content": "actual tool result"}]})])})
    try:
        run = swanlab.get_run()
        logged = []
        sdk_log = run.log

        def capture_log(data, **kwargs):
            logged.append((data, kwargs))
            return sdk_log(data, **kwargs)

        monkeypatch.setattr(run, "log", capture_log)
        logger.log({"actor/loss": 0.3}, step=1)
        tracking.log_rl_examples(batch=batch, tokenizer=None, update_index=1,
                                 loggers=["swanlab"], output_dir=str(tmp_path / "results"))
        row = json.loads((tmp_path / "results/swanlab-artifacts/train-rollout_examples-step000001.jsonl").read_text())
        assert row["task_id"] == "t1" and row["reward"] == 1.0
        media, options = next(item for item in logged if "cases/batch" in item[0])
        assert options["step"] == 1
        assert {"train/rollout_examples", "cases/batch", "cases/preview_1"} <= media.keys()
    finally:
        logger.logger.pop("swanlab").finish()


def test_reference_naming_and_resolved_rl_overrides(monkeypatch):
    from omegaconf import OmegaConf

    assert tracking.experiment_name(stage="sft", model="Qwen/Qwen3.5-0.8B", method="full",
                                    lr=2e-5, seed=42) == "sft-airline-Qwen3.5-0.8B-full-lr2e-5-ep5-s42"
    monkeypatch.setenv("TAU3_POLICY_DISPLAY_MODEL", "Qwen/Qwen3.5-0.8B")
    monkeypatch.setenv("TAU3_GRPO_ARM", "e3")
    config = OmegaConf.create({"trainer": {"experiment_name": "tau3-auto", "total_training_steps": 3},
        "data": {"seed": 42, "train_batch_size": 2}, "actor_rollout_ref": {
            "model": {"path": "/checkpoints/merged", "lora_rank": 0},
            "actor": {"optim": {"lr": 1e-6}}, "rollout": {"n": 4}}})
    assert tracking.rl_experiment_name(config) == "rl-E3-DF-GiGPO-Qwen3.5-0.8B-full-lr1e-6-g2x4-u3-s42"
    config.trainer.experiment_name = "custom"
    assert tracking.rl_experiment_name(config) == "custom"


def test_rl_trainer_initializes_validation_logger_with_resolved_name(monkeypatch):
    """Exercise the constructor: fit-local imports cannot name validation runs."""
    from omegaconf import OmegaConf
    from verl.trainer.ppo import ray_trainer

    monkeypatch.setattr(ray_trainer, "need_reference_policy", lambda config: True)
    monkeypatch.setattr(ray_trainer, "need_reward_model", lambda config: False)
    monkeypatch.setattr(ray_trainer, "need_critic", lambda config: False)
    monkeypatch.setattr(ray_trainer.RayPPOTrainer, "_create_dataloader", lambda *args: None)
    monkeypatch.setenv("TAU3_POLICY_DISPLAY_MODEL", "Qwen/Qwen3.5-0.8B")
    monkeypatch.setenv("TAU3_GRPO_ARM", "e0")
    config = OmegaConf.create({
        "actor_rollout_ref": {"hybrid_engine": True,
            "model": {"path": "/merged", "lora_rank": 16},
            "actor": {"optim": {"lr": 1e-6}}, "rollout": {"n": 4}},
        "trainer": {"project_name": "test", "experiment_name": "tau3-auto",
                    "device": "cpu", "total_training_steps": 3},
        "data": {"seed": 42, "train_batch_size": 2},
        "algorithm": {"use_kl_in_reward": False},
    })
    trainer = ray_trainer.RayPPOTrainer(config, None,
        {ray_trainer.Role.ActorRollout: object}, None)
    assert trainer.validation_generations_logger.experiment_name == (
        "rl-E0-base-Qwen3.5-0.8B-lora16-lr1e-6-g2x4-u3-s42")
    assert trainer.ref_in_actor is True


def test_dotenv_preserves_explicit_environment(monkeypatch, tmp_path):
    path = tmp_path / "tracking.env"
    path.write_text("TAU3_SWANLAB_PROJECT=fixture-project\nSWANLAB_MODE=online\n")
    monkeypatch.setenv("TAU3_ENV_FILE", str(path))
    monkeypatch.delenv("TAU3_SWANLAB_PROJECT", raising=False)
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    tracking.load_tracking_env()
    import os
    assert os.environ["TAU3_SWANLAB_PROJECT"] == "fixture-project"
    assert os.environ["SWANLAB_MODE"] == "offline"
def test_rl_metadata_matches_existing_phase_filter_and_preserves_source(monkeypatch):
    from tau3_grpo.tracking.swanlab import rl_tracking_metadata

    monkeypatch.setenv("TAU3_GRPO_ARM", "e0")
    config = {"trainer": {"total_training_steps": 3}, "actor_rollout_ref": {
        "model": {"path": "Qwen/Qwen3.5-0.8B", "lora_rank": 16},
        "actor": {"optim": {"lr": 1e-6}},
    }}
    actual, options = rl_tracking_metadata(config)
    assert actual["trainer"]["phase"] == "rl"
    assert actual["trainer"]["total_training_steps"] == 3
    assert "phase" not in config["trainer"]
    assert options["group"] == "RL-E0"
    assert options["job_type"] == "rl"
    assert "RL" in options["tags"]
