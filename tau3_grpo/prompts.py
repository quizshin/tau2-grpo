"""Versioned Airline agent prompt shared by SFT, RL and project evaluation.

The pinned benchmark policy stays on disk unchanged. This project explicitly
replaces its single-call restriction with sequential multi-call semantics.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from tau3_grpo.paths import TAU2_BENCH_ROOT

TOOL_PROTOCOL_VERSION = "airline_sequential_multicall_v1"
SINGLE_CALL_RULE = (
    "You should only make one tool call at a time, and if you make a tool call, "
    "you should not respond to the user simultaneously. If you respond to the "
    "user, you should not make a tool call at the same time."
)
MULTI_CALL_RULES = """In each assistant turn, either send a message to the user or submit one or more tool calls. Do not send a user-facing reply and tool calls in the same turn.
All tool calls in a turn execute sequentially in the order listed. You receive all their results before your next turn.
You may batch calls only when all arguments and required user confirmations are already available. If a call requires observing or evaluating another call's result, wait for that result and issue the dependent call in a later turn. Never invent missing arguments or assume a previous call succeeded.
Each database-changing action, including every action in a batch, must satisfy the policy's explicit user-confirmation requirements.
A tool error produces an error result; later calls in the same batch still execute. Inspect every result before deciding your next action.
Use the provided tool schemas and valid arguments. Never fabricate tool results."""


def build_system_prompt(policy: str | None = None) -> str:
    """Use the official business rules with the project's multi-call protocol."""

    if policy is None:
        policy = (TAU2_BENCH_ROOT / "data/tau2/domains/airline/policy.md").read_text(
            encoding="utf-8"
        )
    if not policy.strip():
        raise ValueError("Airline policy text is empty; refusing to build a prompt without it")
    domain_policy = policy.replace(SINGLE_CALL_RULE, "").strip()
    return (
        "<instructions>\nYou are a customer service agent for an airline. "
        "Follow the policy below.\n"
        f"<tool_call_protocol version=\"{TOOL_PROTOCOL_VERSION}\">\n"
        f"{MULTI_CALL_RULES}\n</tool_call_protocol>\n</instructions>\n"
        f"<policy>\n{domain_policy}\n</policy>"
    )


def prepare_agent_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace cached/source system instructions without editing input records.

    In particular, AReaL SFT contains both single-call instructions and relaxed
    confirmation rules. Use one canonical prompt rather than appending a
    contradictory override to those source instructions.
    """

    prepared = [dict(message) for message in messages]
    system = {"role": "system", "content": build_system_prompt()}
    if prepared and prepared[0].get("role") == "system":
        prepared[0] = system
    else:
        prepared.insert(0, system)
    return prepared


def prompt_provenance(system_prompt: str | None = None) -> dict[str, Any]:
    return {
        "tool_protocol": TOOL_PROTOCOL_VERSION,
        "agent_system_prompt_sha256": hashlib.sha256(
            (system_prompt if system_prompt is not None else build_system_prompt()).encode("utf-8")
        ).hexdigest(),
        "benchmark_policy_modified": True,
    }
