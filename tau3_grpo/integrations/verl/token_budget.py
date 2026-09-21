"""Opt-in shared-loop policy: raw token history, bounded observations and receipts."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from tau3_grpo.models.token_budget import (
    VERSION,
    complete_tool_envelopes,
    default_budget,
    token_digest,
)


def configure(loop):
    name = loop.config.get("tau3_token_protocol")
    if name is None:
        return None
    if name != VERSION:
        raise ValueError(f"Unknown token protocol: {name}")
    budget = default_budget()
    if (loop.prompt_length, loop.response_length, loop.rollout_config.max_model_len,
            int(os.getenv("TAU3_GRPO_MAX_TOKENS_PER_TURN", "1024"))) != (
            budget.prompt, budget.response, budget.context, budget.per_turn):
        raise ValueError("Rollout limits differ from the declared token protocol")
    if (loop.max_assistant_turns, loop.max_user_turns, loop.max_tool_response_length,
            loop.tool_response_truncate_side) != (15, 15, 65536, "middle"):
        raise ValueError("Turn/observation limits differ from the declared token protocol")
    if loop.tool_parser_name != "qwen3_coder" or loop.tool_execution_mode != "sequential":
        raise ValueError("Token protocol requires the sequential qwen3_coder parser")
    if loop.apply_chat_template_kwargs.get("enable_thinking") is not False:
        raise ValueError("Token protocol requires explicit enable_thinking=false")
    from tau3_grpo.envs.adapter import airline_tool_schemas

    if loop.tool_schemas != airline_tool_schemas():
        raise ValueError("Token protocol requires the complete official tool schema")
    for interaction in loop.interaction_map.values():
        interaction.token_budget = budget
        interaction._user_config.max_tokens = budget.user_per_turn
        # Exactly the same native user simulator path in both callers.
        interaction._user_config.extra_llm_args.update(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            top_p=1.0,
        )
    return budget


def start(loop, data):
    budget = loop.token_budget
    if data.image_data or data.video_data:
        raise ValueError("Token protocol is text-only")
    if len(data.prompt_ids) > budget.prompt:
        raise ValueError("Initial prompt exceeds the token protocol")
    data.token_receipts = [{"kind": "initial", "token_ids": list(data.prompt_ids),
                            "sha256": token_digest(data.prompt_ids)}]


def generation_request(loop, data):
    allowed = loop.token_budget.allowance(len(data.prompt_ids), len(data.response_mask))
    data.token_receipts.append({"kind": "request", "input_sha256": token_digest(data.prompt_ids),
                               "input_tokens": len(data.prompt_ids),
                               "appended_tokens": len(data.response_mask), "max_tokens": allowed})
    return allowed


def generation_result(loop, data, output):
    finish = output.extra_fields.get("finish_reason")
    if finish not in {"stop", "length"}:
        raise ValueError(f"Missing/invalid raw generation finish_reason: {finish}")
    allowed = data.token_receipts[-1]["max_tokens"]
    if not output.token_ids or len(output.token_ids) > allowed:
        raise ValueError("Generation violates the requested token allowance")
    text = loop.tokenizer.decode(output.token_ids, skip_special_tokens=True)
    complete = complete_tool_envelopes(text)
    data.token_receipts.append({"kind": "generation", "token_ids": list(output.token_ids),
                               "finish_reason": finish, "complete_tool_envelopes": complete})
    # A length-ended text is not sent to the user as a complete turn. A complete
    # XML batch may execute even at the exact limit; a partial batch never does.
    if not complete or (finish == "length" and "<tool_call>" not in text):
        return "context_window_exceeded" if finish == "length" else "agent_error"
    return None


def observation(loop, data, ids, *, kind, terminal=False):
    fits = loop.token_budget.fits(len(data.prompt_ids), len(data.response_mask), len(ids))
    data.token_receipts.append({"kind": kind, "token_ids": list(ids), "retained": fits,
                               "terminal": terminal, "before_tokens": len(data.prompt_ids),
                               "before_appended_tokens": len(data.response_mask)})
    return fits


def publish(loop, data):
    if not getattr(loop, "token_budget", None):
        return
    payload = {"budget": asdict(loop.token_budget), "receipts": data.token_receipts,
               "termination_reason": data.termination_reason,
               "final_input_sha256": token_digest(data.prompt_ids),
               "appended_tokens": len(data.response_mask),
               "full_schema": "tau3_full_schema_v2"}
    data.extra_fields["token_protocol_json"] = json.dumps(payload, sort_keys=True, allow_nan=False)
