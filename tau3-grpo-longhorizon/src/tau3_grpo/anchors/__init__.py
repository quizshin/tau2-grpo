"""Structured / DB-hash / similarity anchors for Tau-GiGPO step grouping."""

from tau3_grpo.anchors.encoder import (
    DEFAULT_SIMILARITY_THRESHOLD,
    AnchorMode,
    AnchorState,
    AnchorTelemetry,
    ObservationType,
    anchor_state_from_session,
    encode_anchor,
    encode_anchors,
    encode_similarity_candidate,
    jaccard,
    resolve_similarity_candidates,
)
from tau3_grpo.anchors.features import (
    CONFIRMATION_TOOLS,
    KNOWN_INFO_TOOLS,
    confirmation_flags,
    known_info_mask,
    last_observation_type,
    policy_precondition_flags,
    tool_error_count,
)

__all__ = [
    "CONFIRMATION_TOOLS",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "KNOWN_INFO_TOOLS",
    "AnchorMode",
    "AnchorState",
    "AnchorTelemetry",
    "ObservationType",
    "anchor_state_from_session",
    "confirmation_flags",
    "encode_anchor",
    "encode_anchors",
    "encode_similarity_candidate",
    "jaccard",
    "known_info_mask",
    "last_observation_type",
    "policy_precondition_flags",
    "resolve_similarity_candidates",
    "tool_error_count",
]
