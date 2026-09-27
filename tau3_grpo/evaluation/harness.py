"""Versioned independent-evaluation control flow, separate from official scoring.

v2 matches current training's turn/error/observation rules, not its tokenizer,
context budget, generation service, or initial prompt construction.
"""
from __future__ import annotations

from tau3_grpo.data.opening import OPENING_VERSION

LEGACY = "tau3_eval_legacy_v1"
CONTROL_V2 = "tau3_eval_train_control_v2"
INPUTS_V3 = "tau3_eval_train_inputs_v3"
TOKENS_V4 = "tau3_eval_token_v4"
PROTOCOLS = (LEGACY, CONTROL_V2, INPUTS_V3, TOKENS_V4)


def validate_target(name, target):
    if name in (INPUTS_V3, TOKENS_V4) and target != "selection":
        raise ValueError(f"{name} supports AReaL selection only; official final requires its own protocol")


def validate_opening(task, opening):
    if not isinstance(opening, str) or not opening.strip():
        raise ValueError("train_inputs_v3 requires a nonempty pinned initial user message")
    if task.initial_state and task.initial_state.message_history:
        raise ValueError("train_inputs_v3 requires a fresh message history")


def request_args(name, endpoint, *, role):
    """Resolve effective requests without changing legacy/v2 server defaults."""
    args = endpoint.llm_args()
    if name in (INPUTS_V3, TOKENS_V4):
        if role == "policy":
            args["max_tokens"] = 1024
        elif role != "user":
            raise ValueError(f"Unknown endpoint role: {role}")
        args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if name == TOKENS_V4:
        from tau3_grpo.models.token_budget import default_budget

        budget = default_budget()
        args["max_tokens"] = budget.per_turn if role == "policy" else budget.user_per_turn
        args["top_p"] = 1.0
    return args


def protocol_metadata(name, *, max_steps=30, max_errors=10):
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown harness protocol: {name}")
    if name == LEGACY:
        return {"version": name, "budget_unit": "orchestrator_steps",
                "max_steps": max_steps, "max_errors": max_errors,
                "stop_priority": "completed_stop_before_budget_v1", "tool_observation": "raw",
                "generation_limits": "server_defaults_no_request_override",
                "execution_errors": "preserve_observed_state_and_continue_other_trials_v1",
                "context_limit_handling": "typed_generation_limit_preserve_trajectory_v1"}
    # This named profile deliberately fixes the envelope. New envelopes need
    # a new version, rather than silently ignoring caller overrides.
    if (max_steps, max_errors) != (30, 10):
        raise ValueError("v2/v3/v4 use fixed 15-turn budgets/no error cap; do not override legacy max_steps/max_errors")
    metadata = {"version": name, "max_assistant_turns": 15, "max_user_messages": 15,
            "max_observation_batches": 15, "initial_user_counts_as_observation": False,
            "last_assistant_tool_batch": "execute_all", "last_text_user_reply": "allowed_within_user_budget",
            "max_errors": None, "stop_priority": "completed_terminal_first",
            "tool_observation": {"version": "tau3_tool_chars_v1", "limit": 65536,
                                 "side": "middle", "marker_outside_limit": True},
            "token_context_equivalence": "not_established"}
    if name in (CONTROL_V2, INPUTS_V3):
        metadata["context_limit_handling"] = "typed_generation_limit_preserve_trajectory_v1"
        metadata["execution_errors"] = "preserve_observed_state_and_continue_other_trials_v1"
    if name in (INPUTS_V3, TOKENS_V4):
        metadata.update(
            initial_user_message=OPENING_VERSION,
            initial_assistant_greeting=False,
            supported_target="selection",
            policy_max_tokens_per_request=1024,
            policy_enable_thinking=False,
            user_enable_thinking=False,
            default_policy_temperature=0.7,
            default_user_temperature=0.7,
            sampling_temperatures="explicit_endpoint_parameters",
            context_budget="server_enforced_not_training_cumulative_response_budget",
        )
    if name == TOKENS_V4:
        from dataclasses import asdict

        from tau3_grpo.models.token_budget import default_budget

        metadata.update(
            version=TOKENS_V4, execution_path="shared_native_tool_agent_loop",
            token_context_equivalence="shared_implementation_live_service_pending",
            context_budget=asdict(default_budget()),
            token_history="append_original_generated_ids_and_native_observation_suffixes",
            policy_transport="vllm_completions_token_ids_required",
            tool_parser="native_qwen3_coder_complete_envelopes_only",
            tool_schema="tau3_full_schema_v2", official_final_supported=False,
        )
    return metadata
