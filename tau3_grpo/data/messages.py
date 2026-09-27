"""Model-visible message projection shared by runtime clients and offline analysis."""

from copy import deepcopy


def visible_message(raw: dict) -> dict:
    """Copy only visible fields; never pass reward, reference or audit metadata to models."""
    fields = ("role", "content", "tool_calls", "id", "tool_call_id", "error", "requestor")
    return {key: deepcopy(raw[key]) for key in fields if key in raw}
