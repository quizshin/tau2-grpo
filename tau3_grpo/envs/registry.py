"""Process-local registry binding a veRL request id to one tau2 session.

Milestone D3. veRL creates the interaction and the tools independently, but both
must act on the *same* `TrajectorySession` for a given rollout. veRL's
`ToolAgentLoop` threads `request_id` through both paths (`start_interaction`
receives it, and `agent_data.request_id` is available inside `tool.execute`), so
the request id is the join key.

The registry is per worker process. Sessions are removed in
`finalize_interaction`, so a long run does not accumulate FlightDB copies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from tau3_grpo.algorithms.anchors.encoder import AnchorMode
from tau3_grpo.envs.session import TrajectorySession


@dataclass
class SessionEntry:
    """A live rollout: its session plus the anchor trail collected so far."""

    session: TrajectorySession
    anchor_mode: AnchorMode = AnchorMode.STRUCTURED
    similarity_threshold: float = 0.9
    anchor_ids: list[Optional[str]] = field(default_factory=list)
    anchor_spans: list[Optional[tuple[int, int]]] = field(default_factory=list)
    terminated: bool = False
    termination_reason: Optional[str] = None
    tool_error_count: int = 0

    def record_segment(
        self, anchor_id: Optional[str], span: Optional[tuple[int, int]]
    ) -> None:
        self.anchor_ids.append(anchor_id)
        self.anchor_spans.append(span)


class SessionRegistry:
    """Thread-safe request_id -> SessionEntry map."""

    def __init__(self) -> None:
        self._entries: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str, entry: SessionEntry) -> SessionEntry:
        with self._lock:
            self._entries[request_id] = entry
        return entry

    def get(self, request_id: str) -> Optional[SessionEntry]:
        with self._lock:
            return self._entries.get(request_id)

    def require(self, request_id: str) -> SessionEntry:
        entry = self.get(request_id)
        if entry is None:
            raise KeyError(
                f"no tau2 session registered for request_id={request_id!r}; "
                "the interaction must call start_interaction before tools execute"
            )
        return entry

    def pop(self, request_id: str) -> Optional[SessionEntry]:
        with self._lock:
            return self._entries.pop(request_id, None)

    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


#: Shared per-process registry.
SESSIONS = SessionRegistry()


def session_for(agent_data: Any) -> SessionEntry:
    """Resolve the session for a veRL `agent_data` handed to `tool.execute`."""

    request_id = getattr(agent_data, "request_id", None)
    if request_id is None:
        raise KeyError("agent_data has no request_id; cannot resolve tau2 session")
    return SESSIONS.require(str(request_id))
