"""The public task opening used by training and explicitly matched evaluation."""

from collections.abc import Mapping

OPENING_VERSION = "areal_reason_for_call_v1"
DEFAULT_OPENING = "Hello, I need help with my reservation."


def initial_user_message(task: Mapping) -> str:
    """Preserve the parquet builder's existing rule, without exposing hidden goals."""
    instructions = task.get("user_scenario", {}).get("instructions", {})
    opening = ""
    if isinstance(instructions, dict):
        opening = str(instructions.get("reason_for_call", "") or "")
    return opening or DEFAULT_OPENING
