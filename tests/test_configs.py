"""Config files must carry the frozen v2-2 values and the flat project paths."""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML not installed")

from tau3_grpo.algorithms.tau_gigpo import (  # noqa: E402
    DEFAULT_OMEGA,
    FNORM,
    GAMMA,
    MIN_ANCHOR_GROUP_SIZE,
)
from tau3_grpo.data.parquet_builder import INTERACTION_NAME  # noqa: E402
from tau3_grpo.paths import CONFIG_ROOT, PROJECT_ROOT  # noqa: E402
from tau3_grpo.training.rl.runtime_defaults import defaults  # noqa: E402


def _load(relative: str) -> dict:
    path = CONFIG_ROOT / relative
    assert path.is_file(), f"missing config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_base_config_has_v2_2_values():
    config = _load("train/rl/base.yaml")
    assert config["model"]["path"] == "Qwen/Qwen2.5-7B-Instruct"
    assert config["rollout"]["group_size"] == 8
    assert config["rollout"]["groups_per_update"] == 16
    assert config["rollout"]["temperature_train"] == 1.0
    assert config["rollout"]["temperature_eval"] == 0.4
    assert config["rollout"]["max_user_turns"] == 15
    assert config["rollout"]["max_assistant_turns"] == 15
    assert config["rollout"]["tensor_model_parallel_size"] == 2
    assert config["optim"]["lr"] == 1.0e-6
    assert config["optim"]["kl_coef"] == 0.01
    assert config["optim"]["ppo_mini_groups"] == 6
    assert config["optim"]["policy_batch_divisor"] == 48


def test_base_config_schedule_and_resources():
    config = _load("train/rl/base.yaml")
    assert config["schedule"]["screening_updates"] == 40
    assert config["schedule"]["confirmatory_updates"] == 60
    assert config["schedule"]["seeds"] == [42, 43]
    assert config["resources"]["policy_gpus"] == 6
    assert config["resources"]["simulator_gpus"] == 2


def test_base_config_split_sizes_match_data_boundary():
    config = _load("train/rl/base.yaml")
    assert config["data"]["split_seed"] == 42
    assert config["data"]["train_size"] == 200
    assert config["data"]["selection_size"] == 60
    assert config["data"]["reserve_size"] == 888
    assert config["data"]["source_revision"] == (
        "86971dc03da6e7c1a7933295e05b84aab8215386"
    )


def test_base_config_gigpo_matches_frozen_constants():
    gigpo = _load("train/rl/base.yaml")["gigpo"]
    assert gigpo["omega"] == DEFAULT_OMEGA == 1.0
    assert gigpo["gamma"] == GAMMA
    assert gigpo["fnorm"] == FNORM
    assert gigpo["min_anchor_group_size"] == MIN_ANCHOR_GROUP_SIZE


def test_vllm_is_the_main_backend():
    assert _load("train/rl/base.yaml")["rollout"]["backend"] == "vllm"


def test_dynamic_filter_is_off_in_the_base_config():
    assert _load("train/rl/base.yaml")["dynamic_filter"]["enable"] is False


def test_original_arms_and_mt_gtpo_are_defined():
    arms = _load("experiments/arms.yaml")["arms"]
    assert set(arms) == {"e0", "e1", "e2", "e3", "mt_gtpo"}
    assert arms["mt_gtpo"]["adv_estimator"] == "mt_gtpo"
    assert arms["mt_gtpo"]["dynamic_filter"]["enable"] is False


def test_arm_algorithm_matrix():
    arms = _load("experiments/arms.yaml")["arms"]
    assert arms["e0"]["adv_estimator"] == "grpo"
    assert arms["e0"]["dynamic_filter"]["enable"] is False
    assert arms["e1"]["adv_estimator"] == "grpo"
    assert arms["e1"]["dynamic_filter"]["enable"] is True
    assert arms["e2"]["adv_estimator"] == "tau_gigpo"
    assert arms["e2"]["dynamic_filter"]["enable"] is False
    assert arms["e3"]["adv_estimator"] == "tau_gigpo"
    assert arms["e3"]["dynamic_filter"]["enable"] is True


def test_dynamic_filter_uses_fixed_rollout():
    arms = _load("experiments/arms.yaml")["arms"]
    for name in ("e1", "e3"):
        assert arms[name]["dynamic_filter"]["mode"] == "fixed_rollout"
        assert arms[name]["dynamic_filter"]["group_size"] == 8


def test_anchor_ablation_covers_structured_and_db_hash():
    ablations = _load("experiments/arms.yaml")["ablations"]
    assert ablations["e2_db_hash_only"]["anchors"]["mode"] == "db_hash_only"
    assert ablations["e2_similarity"]["anchors"]["similarity_threshold"] == 0.9


def test_interaction_config_names_match_parquet_literal():
    config = _load("envs/interaction_config.yaml")
    entry = config["interaction"][0]
    assert entry["name"] == INTERACTION_NAME
    assert entry["class_name"].endswith("Tau3AirlineInteraction")


def test_interaction_config_turn_caps():
    config = _load("envs/interaction_config.yaml")["interaction"][0]["config"]
    assert config["max_user_turns"] == 15
    assert config["max_assistant_turns"] == 15


def test_training_user_simulator_is_deterministic():
    base = _load("train/rl/base.yaml")["user_simulator"]
    interaction = _load("envs/interaction_config.yaml")["interaction"][0]["config"]
    assert base["temperature"] == 0.0
    assert interaction["user_temperature"] == 0.0


def test_tool_config_uses_the_live_environment_tool():
    tools = _load("envs/tool_config.yaml")["tools"]
    assert tools, "tool config is empty"
    for tool in tools:
        assert tool["class_name"].endswith("Tau3AirlineTool")
        assert tool["config"]["type"] == "native"
        assert tool["tool_schema"]["type"] == "function"
        assert tool["tool_schema"]["function"]["name"]
        assert isinstance(tool["tool_schema"]["function"]["parameters"]["required"], list)


def test_tool_config_covers_read_and_write_tools():
    names = {
        tool["tool_schema"]["function"]["name"]
        for tool in _load("envs/tool_config.yaml")["tools"]
    }
    for required in (
        "get_user_details",
        "get_reservation_details",
        "search_direct_flight",
        "book_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "cancel_reservation",
    ):
        assert required in names, f"tool config missing {required}"


def test_tool_config_covers_every_live_airline_tool():
    from tau3_grpo.envs.adapter import airline_tool_schemas

    names = {
        tool["tool_schema"]["function"]["name"]
        for tool in _load("envs/tool_config.yaml")["tools"]
    }
    live_names = {schema["function"]["name"] for schema in airline_tool_schemas()}
    assert names == live_names


def test_tool_names_match_anchor_feature_lists():
    from tau3_grpo.algorithms.anchors.features import CONFIRMATION_TOOLS, KNOWN_INFO_TOOLS

    names = {
        tool["tool_schema"]["function"]["name"]
        for tool in _load("envs/tool_config.yaml")["tools"]
    }
    for tool in KNOWN_INFO_TOOLS:
        assert tool in names, f"known-info tool {tool} absent from tool config"
    for tool in CONFIRMATION_TOOLS:
        assert tool in names, f"confirmation tool {tool} absent from tool config"


# ---- scripts -----------------------------------------------------------


def _script(name: str) -> str:
    path = PROJECT_ROOT / "scripts" / name
    assert path.is_file(), f"missing script: {path}"
    return path.read_text(encoding="utf-8")


def test_train_script_calls_same_process_trainer_wrapper():
    text = _script("train/rl/run_base.sh")
    assert "python -m tau3_grpo.training.rl.train" in text


def test_train_script_verifies_patches_first():
    assert "tau3_grpo.integrations.verify_patches" in _script("train/rl/run_base.sh")


def test_train_script_registers_the_estimator():
    wrapper = (PROJECT_ROOT / "tau3_grpo/training/rl/train.py").read_text(encoding="utf-8")
    assert "tau3_grpo.integrations.verl.gigpo" in wrapper
    assert 'runpy.run_module("verl.trainer.main_ppo"' in wrapper


def test_train_script_generates_live_tool_config():
    text = _script("train/rl/run_base.sh")
    assert "tau3_grpo.envs.generate_tool_config" in text
    assert "TAU3_GRPO_ANCHOR_HOOK" in text
    assert "TAU3_GRPO_ANCHOR_MODE" in text


def test_train_script_uses_four_root_relative_paths():
    text = _script("train/rl/run_base.sh")
    assert 'CODE_ROOT="${PROJECT_ROOT}"' in text
    assert "tau2-bench/src" in text


def test_training_seed_does_not_change_the_frozen_data_split():
    text = _script("train/rl/run_base.sh")
    assert defaults("base", {})["DATA_SPLIT_SEED"] == "42"
    assert '--data-seed "${DATA_SPLIT_SEED}"' in text
    assert "TAU3_GRPO_TRAIN_SEED" in text
    assert 'data.seed="${SEED}"' in text

    prepare = _script("data/prepare.sh")
    assert "DATA_SPLIT_SEED" in prepare
    assert "for SEED in 42 43" not in prepare


def test_train_script_carries_v2_2_constants():
    text = _script("train/rl/run_base.sh")
    values = defaults("base", {})
    assert {key: values[key] for key in ("GROUP_SIZE", "GROUPS_PER_UPDATE", "LR",
                                        "KL_COEF", "MAX_USER_TURNS")} == {
        "GROUP_SIZE": "8", "GROUPS_PER_UPDATE": "16", "LR": "1e-6",
        "KL_COEF": "0.01", "MAX_USER_TURNS": "15"}
    assert "actor_rollout_ref.actor.use_kl_loss=true" in text
    assert 'actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}"' in text


def test_train_script_selects_real_multiturn_tool_agent():
    text = _script("train/rl/run_base.sh")
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent" in text
    assert "actor_rollout_ref.rollout.multi_turn.enable=true" in text
    assert "tool_config_path" in text
    assert "interaction_config_path" in text


def test_train_script_has_long_context_and_per_turn_cap():
    text = _script("train/rl/run_base.sh")
    values = defaults("base", {})
    assert [values[key] for key in ("MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH",
                                    "MAX_MODEL_LENGTH", "MAX_TOKENS_PER_TURN")] == [
        "8192", "16384", "24576", "1024"]
    assert 'TAU3_GRPO_MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN}"' in text
    assert 'actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH}"' in text


def test_train_script_has_memory_safe_micro_batches_and_telemetry():
    text = _script("train/rl/run_base.sh")
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" in text
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" in text
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" in text
    assert "actor_rollout_ref.model.enable_gradient_checkpointing=true" in text
    assert "actor_rollout_ref.model.use_remove_padding=true" in text
    assert 'TAU3_GRPO_TELEMETRY_PATH="${RESULTS_DIR}/telemetry.jsonl"' in text
    assert 'TAU3_GRPO_ARM="${ARM}"' in text
    assert 'ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${RESULTS_DIR}/rollouts}"' in text
    assert 'trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"' in text


def test_train_script_does_not_use_unsupported_rollout_seed():
    assert "actor_rollout_ref.rollout.seed" not in _script("train/rl/run_base.sh")


def test_train_script_preserves_128_real_rollouts_with_explicit_zero_loss_padding():
    text = _script("train/rl/run_base.sh")
    assert "REAL_ROLLOUTS=$((GROUP_SIZE * GROUPS_PER_UPDATE))" in text
    assert defaults("base", {})["PPO_MINI_GROUPS"] == "6"
    assert 'POLICY_BATCH_DIVISOR="${POLICY_BATCH_DIVISOR:-$((PPO_MINI_GROUPS * GROUP_SIZE))}"' in text
    assert 'TAU3_GRPO_POLICY_BATCH_DIVISOR="${POLICY_BATCH_DIVISOR}"' in text
    assert 'actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_GROUPS}"' in text


def test_train_defaults_to_merged_sft_checkpoint_and_disables_shuffle():
    text = _script("train/rl/run_base.sh")
    assert "${TAU3_RUN_ROOT}/sft_airline_merged_seed42" in text
    assert "data.shuffle=false" in text


def test_train_materialises_the_experiment_schedule():
    text = _script("train/rl/run_base.sh")
    assert "tau3_grpo.experiments.prepare" in text
    assert 'TRAIN_PARQUET="${CUSTOM_TRAIN_PARQUET:-${RESULTS_DIR}/train_schedule.parquet}"' in text
    assert '--total-updates "${TOTAL_UPDATES}"' in text


def test_train_consumes_wrapper_positionals_before_hydra_overrides():
    import os
    import shlex
    import subprocess

    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/train/rl/run_base.sh"), "e2", "43", "trainer.save_freq=2"],
        env=dict(os.environ, TAU3_DRY_RUN="1"), capture_output=True, text=True, check=True,
    )
    command = shlex.split(result.stdout.strip().splitlines()[-1])
    assert command[:3] == ["python", "-m", "tau3_grpo.training.rl.train"]
    assert "e2" not in command and "43" not in command
    assert "algorithm.adv_estimator=tau_gigpo" in command
    assert "data.seed=43" in command
    assert "actor_rollout_ref.rollout.multi_turn.tool_execution_mode=sequential" in command
    assert command[-1] == "trainer.save_freq=2"


def test_train_appends_local_algorithm_blocks_to_upstream_hydra_config():
    text = _script("train/rl/run_base.sh")
    assert '"+algorithm.dynamic_filter=' in text
    assert '"+algorithm.gigpo=' in text


def test_train_uses_vllm_only():
    text = _script("train/rl/run_base.sh")
    assert "ROLLOUT_BACKEND=vllm" in text
    assert "sglang" not in text.lower()


def test_eval_script_uses_eval_temperature():
    assert "EVAL_TEMP=0.4" in _script("eval/run.sh")


def test_eval_script_goes_through_the_guard():
    assert "tau3_grpo.evaluation.run" in _script("eval/run.sh")


def test_user_simulator_service_enables_prefix_caching():
    assert "--enable-prefix-caching" in _script("serve/simulator_base.sh")


def test_policy_serve_script_attests_checkpoint_content_and_pid():
    text = _script("serve/policy.sh")
    assert "tau3_grpo.evaluation.attest_service" in text
    assert '--pid "$$"' in text
    assert "policy_service_attestation.json" in text
    assert 'TAU3_POLICY_MAX_MODEL_LEN:-24576' in text


def test_a800_preflight_script_exists_and_checks_cuda_contract():
    path = PROJECT_ROOT / "scripts/maintenance/check_installation.sh"
    text = path.read_text(encoding="utf-8")
    assert "torch.__version__.startswith(\"2.8\")" in text
    assert 'torch.version.cuda == "12.8"' in text
    assert 'vllm.__version__ == "0.10.2"' in text
    assert 'transformers.__version__ == "4.56.1"' in text
    assert "tau3_grpo.integrations.verify_patches" in text


def test_one_trajectory_smoke_is_isolated_from_formal_defaults():
    path = PROJECT_ROOT / "scripts/maintenance/run_one_trajectory.sh"
    text = path.read_text(encoding="utf-8")
    assert "GROUP_SIZE=1" in text
    assert "GROUPS_PER_UPDATE=1" in text
    assert "POLICY_GPUS=1" in text
    assert "ROLLOUT_TP=1" in text
    assert "TOTAL_UPDATES=1" in text
    assert 'MAX_USER_TURNS="${MAX_USER_TURNS:-3}"' in text
    assert 'MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-3}"' in text
    assert "${TAU3_RUN_ROOT}/smoke/e0_seed42" in text
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in text
    assert "data.dataloader_num_workers=0" in text
    assert "actor_rollout_ref.model.use_remove_padding=false" in text
    assert 'actor_rollout_ref.actor.fsdp_config.model_dtype="${SMOKE_FSDP_MODEL_DTYPE:-bfloat16}"' in text
    assert 'actor_rollout_ref.ref.fsdp_config.model_dtype="${SMOKE_FSDP_MODEL_DTYPE:-bfloat16}"' in text
    assert 'actor_rollout_ref.actor.fsdp_config.param_offload="${SMOKE_PARAM_OFFLOAD:-true}"' in text
    assert 'actor_rollout_ref.actor.fsdp_config.optimizer_offload="${SMOKE_OPTIMIZER_OFFLOAD:-true}"' in text
    assert 'SMOKE_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20' in text
    assert "actor_rollout_ref.rollout.agent.num_workers=1" in text
    assert "SMOKE_USER_SIMULATOR_MODELS_URL" in text
    assert "curl --fail --silent --show-error --max-time 5" in text
    assert 'TAU3_USER_API_KEY="${TAU3_USER_API_KEY:-EMPTY}"' in text
    assert '"$@"' in text


def test_setup_requires_python_312_for_both_environments():
    text = (PROJECT_ROOT / "setup.sh").read_text(encoding="utf-8")
    assert "require_python_312" in text
    assert "sys.version_info[:2] != (3, 12)" in text
    assert 'VENV_DIR="${ROOT_DIR}/.venv-cpu"' in text
    assert 'TAU3_VENV_DIR:-${ROOT_DIR}/.venv-a800' in text
    assert "tau3_grpo_local_sources.pth" in text
    assert "a800-constraints.txt" in text
    assert 'verl[vllm]' in text
    assert 'PIP_CACHE_DIR=' in text
    assert 'TMPDIR=' in text
    assert 'TAU3_PIP_INDEX_URL' in text
    assert 'mirrors.aliyun.com/pypi/simple' in text


def test_a800_constraints_keep_vllm_on_torch28():
    root = PROJECT_ROOT
    constraints = (root / "env_info/a800-constraints.txt").read_text(encoding="utf-8")
    assert "torch==2.8.0" in constraints
    assert "vllm==0.10.2" in constraints
    assert "transformers==4.56.1" in constraints
    assert "opencv-python-headless==4.11.0.86" in constraints
    assert "cupy-cuda12x==13.6.0" in constraints
    assert "scipy==1.14.1" in constraints
    assert "litellm==1.82.6" in constraints
    assert "openai==3.3.1" in constraints

    setup = (root / "setup.sh").read_text(encoding="utf-8")
    assert 'verl[vllm]' in setup
    assert "sglang" not in setup.lower()
    assert "flashinfer" not in setup.lower()


def test_sft_config_matches_frozen_45_plus_5_budget():
    config = _load("train/sft/qwen25_lora.yaml")
    assert config["model"]["name_or_path"] == "Qwen/Qwen2.5-7B-Instruct"
    assert config["model"]["attn_implementation"] == "flash_attention_2"
    assert config["data"]["max_length"] == 16384
    assert config["lora"]["r"] == 16
    assert config["lora"]["alpha"] == 32
    assert config["train"]["num_epochs"] == 5
    assert config["train"]["per_device_batch_size"] == 1
    assert config["train"]["gradient_accumulation_steps"] == 8
    assert (
        config["train"]["per_device_batch_size"]
        * config["train"]["gradient_accumulation_steps"]
        == 8
    )


def test_qwen35_sft_comparison_keeps_data_and_budget_equal():
    full = _load("train/sft/qwen35_full.yaml")
    lora = _load("train/sft/qwen35_lora.yaml")
    assert full["model"] == lora["model"]
    assert full["data"] == lora["data"]
    assert full["train"]["method"] == "full"
    assert lora["train"]["method"] == "lora"
    for key in ("num_epochs", "per_device_batch_size", "gradient_accumulation_steps", "seed"):
        assert full["train"][key] == lora["train"][key]
    for config in (full, lora):
        assert config["train"]["load_best_model_at_end"]
        assert config["train"]["evaluate_before_train"]


def test_sft_scripts_require_python312_torch28_cuda128():
    from types import SimpleNamespace

    from tau3_grpo.models.compat import require_training_runtime

    text = _script("train/sft/run_base.sh")
    assert "tau3_grpo.training.sft.train" in text
    assert "SFT_MODEL_NAME_OR_PATH" in text
    runtime = SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"),
                              cuda=SimpleNamespace(is_available=lambda: True))
    require_training_runtime(runtime, "legacy")
    runtime.__version__ = "2.11.0"
    with pytest.raises(RuntimeError, match="legacy Qwen2.5"):
        require_training_runtime(runtime, "legacy")
