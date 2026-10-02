"""Model-visible message projection shared by runtime clients and offline analysis."""

from copy import deepcopy


def visible_message(raw: dict) -> dict:
    """Copy only visible fields; never pass reward, reference or audit metadata to models."""
    fields = ("role", "content", "tool_calls", "id", "tool_call_id", "error", "requestor")
    return {key: deepcopy(raw[key]) for key in fields if key in raw}


def visible_events(messages):
    """An allowlist excludes private thinking, source labels and dataset metadata."""
    fields = ("role", "content", "tool_calls", "tool_call_id", "name")
    return [
        dict(event_id=f"m{i:03d}", **{k: m[k] for k in fields if k in m})
        for i, m in enumerate(messages)
    ]
