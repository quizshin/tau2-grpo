"""Policy-visible tool projection; raw benchmark messages stay untouched."""
from __future__ import annotations

import hashlib

OBSERVATION_VERSION = "tau3_tool_chars_v1"


def project_tool_text(text: str | None, limit: int, side: str) -> tuple[str | None, dict]:
    """Preserve historical veRL character limits (marker is extra characters)."""
    if limit < 2 or side not in {"left", "right", "middle"}:
        raise ValueError("tool observation requires limit >= 2 and left/right/middle")
    visible = text
    truncated = text is not None and len(text) > limit
    if truncated:
        if side == "left":
            visible = text[:limit] + "...(truncated)"
        elif side == "right":
            visible = "(truncated)..." + text[-limit:]
        else:
            half = limit // 2
            visible = text[:half] + "...(truncated)..." + text[-half:]
    return visible, {
        "version": OBSERVATION_VERSION, "unit": "unicode_characters",
        "limit": limit, "side": side, "marker_outside_limit": True,
        "truncated": truncated,
        "raw_chars": len(text) if text is not None else None,
        "visible_chars": len(visible) if visible is not None else None,
        "raw_sha256": hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
        "visible_sha256": hashlib.sha256(visible.encode()).hexdigest() if visible is not None else None,
    }
