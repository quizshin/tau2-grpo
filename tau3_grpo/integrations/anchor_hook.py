"""Bridge the veRL ToolAgentLoop anchor hook to the tau2 session registry.

Milestone D6. The patched `ToolAgentLoop` calls a registered hook once per token
segment and stores whatever it returns. veRL has no notion of an anchor, so this
module supplies the resolver: it looks up the rollout's `SessionEntry` by
`request_id` and encodes the current environment state.

Import order matters. The patched ToolAgentLoop lazily imports `current_anchor`
inside every rollout worker from the `TAU3_GRPO_ANCHOR_HOOK` environment value.
`install()` remains available for direct integration tests and custom launchers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from tau3_grpo.algorithms.anchors.encoder import (
    AnchorMode,
    AnchorState,
    encode_anchor,
    encode_similarity_candidate,
)
from tau3_grpo.algorithms.anchors.features import (
    confirmation_flags,
    known_info_mask,
    last_observation_type,
    policy_precondition_flags,
)
from tau3_grpo.envs.registry import SESSIONS
from tau3_grpo.algorithms.anchors.evidence import decision_evidence, validate_version

logger = logging.getLogger(__name__)

_INSTALLED = False


def current_anchor(agent_data: Any, segment_kind: str) -> Optional[str]:
    """Resolve the anchor for the segment veRL just appended.

    Returns None when this rollout has no tau2 session (for example a non-Airline
    agent loop sharing the same worker), which makes the hook inert rather than
    fatal.
    """

    request_id = getattr(agent_data, "request_id", None)
    if request_id is None:
        return None
    entry = SESSIONS.get(str(request_id))
    if entry is None:
        return None

    session = entry.session
    messages = session.messages
    version = validate_version(entry.anchor_version)
    evidence = decision_evidence(messages, version=version) if version in ("v2", "v3") else None
    state = AnchorState(
        task_id=session.task_id,
        db_hash=session.db_hash(),
        known_info_mask=known_info_mask(messages),
        confirmation_flags=(tuple(f"committed:{name}" for name in evidence.committed_tools)
                            if evidence else confirmation_flags(messages)),
        policy_precondition_flags=policy_precondition_flags(messages),
        last_observation=last_observation_type(messages),
        anchor_version=version,
        decision_evidence_hash=evidence.digest if evidence else None,
    )
    mode = entry.anchor_mode
    if mode is AnchorMode.SIMILARITY:
        return encode_similarity_candidate(state)
    return encode_anchor(state, mode)


def install(force: bool = False) -> bool:
    """Register the hook with the patched ToolAgentLoop. Idempotent."""

    global _INSTALLED
    if _INSTALLED and not force:
        return False
    from verl.experimental.agent_loop.tool_agent_loop import set_tau3_anchor_hook

    set_tau3_anchor_hook(current_anchor)
    _INSTALLED = True
    return True


def uninstall() -> None:
    """Remove the hook, restoring veRL's stock behaviour."""

    global _INSTALLED
    from verl.experimental.agent_loop.tool_agent_loop import set_tau3_anchor_hook

    set_tau3_anchor_hook(None)
    _INSTALLED = False


def is_installed() -> bool:
    from verl.experimental.agent_loop.tool_agent_loop import get_tau3_anchor_hook

    return get_tau3_anchor_hook() is not None
