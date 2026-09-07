"""Structured anchor encoder for Tau-GiGPO step grouping.

Milestone D6–D7, D12. Three modes:

- ``structured``  (default): task_id + canonical mutable DB hash +
  known-information mask + confirmation flags + policy precondition flags +
  last observation type.
- ``db_hash_only``: the D12 ablation. task_id + DB hash only.
- ``similarity``: structured features grouped by Jaccard similarity >= 0.9.

Encoding is deterministic: the same state yields the same anchor id in any
process, because every component is either a hash of canonically serialized JSON
or a sorted flag tuple. No RNG, no dict iteration order, no `id()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from tau3_grpo.utils.hashing import canonical_json, sha256_text

ANCHOR_ID_LENGTH = 16
DEFAULT_SIMILARITY_THRESHOLD = 0.9
SIMILARITY_CANDIDATE_PREFIX = "similarity_features:"


class AnchorMode(str, Enum):
    STRUCTURED = "structured"
    DB_HASH_ONLY = "db_hash_only"
    SIMILARITY = "similarity"


class ObservationType(str, Enum):
    """Type of the most recent observation the policy conditioned on."""

    NONE = "none"
    USER = "user"
    TOOL_OK = "tool_ok"
    TOOL_ERROR = "tool_error"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class AnchorState:
    """The state features an anchor is computed from."""

    task_id: str
    db_hash: Optional[str]
    known_info_mask: tuple[bool, ...] = ()
    confirmation_flags: tuple[str, ...] = ()
    policy_precondition_flags: tuple[str, ...] = ()
    last_observation: ObservationType = ObservationType.NONE

    def structured_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "db_hash": self.db_hash,
            "known_info_mask": [bool(bit) for bit in self.known_info_mask],
            "confirmation_flags": sorted(self.confirmation_flags),
            "policy_precondition_flags": sorted(self.policy_precondition_flags),
            "last_observation": self.last_observation.value,
        }

    def db_hash_payload(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "db_hash": self.db_hash}

    def feature_set(self) -> frozenset[str]:
        """Flat feature set used by similarity grouping."""

        features = {
            f"task:{self.task_id}",
            f"db:{self.db_hash}",
            f"obs:{self.last_observation.value}",
        }
        for index, bit in enumerate(self.known_info_mask):
            features.add(f"known:{index}:{int(bool(bit))}")
        for flag in self.confirmation_flags:
            features.add(f"confirm:{flag}")
        for flag in self.policy_precondition_flags:
            features.add(f"precond:{flag}")
        return frozenset(features)


@dataclass
class AnchorTelemetry:
    """Collision and coverage counters reported with every run."""

    total: int = 0
    unique: int = 0
    mode: str = AnchorMode.STRUCTURED.value
    group_sizes: dict[str, int] = field(default_factory=dict)
    distinct_states_per_anchor: dict[str, int] = field(default_factory=dict)

    @property
    def collision_count(self) -> int:
        """Anchors that were reached from more than one distinct state.

        With ``structured`` this must be zero by construction; with
        ``similarity`` it is expected and is the knob being ablated.
        """

        return sum(1 for count in self.distinct_states_per_anchor.values() if count > 1)

    @property
    def collision_rate(self) -> float:
        if not self.distinct_states_per_anchor:
            return 0.0
        return self.collision_count / len(self.distinct_states_per_anchor)

    @property
    def coverage(self) -> float:
        """Share of anchors that sit in a group large enough to give a step signal."""

        if self.total == 0:
            return 0.0
        grouped = sum(size for size in self.group_sizes.values() if size >= 2)
        return grouped / self.total

    @property
    def groups_with_signal(self) -> int:
        return sum(1 for size in self.group_sizes.values() if size >= 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_anchors": self.total,
            "unique_anchors": self.unique,
            "groups_with_signal": self.groups_with_signal,
            "collision_count": self.collision_count,
            "collision_rate": self.collision_rate,
            "coverage": self.coverage,
        }


def encode_anchor(state: AnchorState, mode: AnchorMode | str = AnchorMode.STRUCTURED) -> str:
    """Return the deterministic anchor id for one state."""

    resolved = AnchorMode(mode)
    if resolved is AnchorMode.DB_HASH_ONLY:
        payload = state.db_hash_payload()
    else:
        # SIMILARITY starts from the structured id; grouping merges afterwards.
        payload = state.structured_payload()
    digest = sha256_text(canonical_json(payload))
    return f"{resolved.value}:{digest[:ANCHOR_ID_LENGTH]}"


def encode_similarity_candidate(state: AnchorState) -> str:
    """Serialize the feature set needed for batch-level similarity grouping.

    A rollout worker sees only one trajectory, whereas similarity grouping must
    compare states across the whole GRPO batch. The worker therefore emits this
    deterministic candidate; the estimator resolves candidates after gathering.
    """

    return SIMILARITY_CANDIDATE_PREFIX + canonical_json(sorted(state.feature_set()))


def resolve_similarity_candidates(
    candidates: list[str],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[str]:
    """Greedily cluster serialized candidates, independent of batch order."""

    import json

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity threshold must be in [0, 1]")
    parsed: list[frozenset[str] | None] = []
    for candidate in candidates:
        if not candidate.startswith(SIMILARITY_CANDIDATE_PREFIX):
            parsed.append(None)
            continue
        payload = json.loads(candidate[len(SIMILARITY_CANDIDATE_PREFIX) :])
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("invalid similarity anchor candidate payload")
        parsed.append(frozenset(payload))

    candidate_positions = [index for index, value in enumerate(parsed) if value is not None]
    order = sorted(candidate_positions, key=lambda index: candidates[index])
    representatives: list[int] = []
    assignment: dict[int, int] = {}
    for index in order:
        features = parsed[index]
        assert features is not None
        task_features = {item for item in features if item.startswith("task:")}
        for rep in representatives:
            rep_features = parsed[rep]
            assert rep_features is not None
            rep_tasks = {item for item in rep_features if item.startswith("task:")}
            if task_features != rep_tasks:
                continue
            if jaccard(rep_features, features) >= threshold:
                assignment[index] = rep
                break
        else:
            representatives.append(index)
            assignment[index] = index

    resolved = list(candidates)
    for index in candidate_positions:
        representative = candidates[assignment[index]]
        digest = sha256_text(representative)
        resolved[index] = f"similarity:{digest[:ANCHOR_ID_LENGTH]}"
    return resolved


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 1.0
    return len(left & right) / union


def encode_anchors(
    states: list[AnchorState],
    mode: AnchorMode | str = AnchorMode.STRUCTURED,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[list[str], AnchorTelemetry]:
    """Encode a batch of states and collect collision/coverage telemetry."""

    resolved = AnchorMode(mode)
    if resolved is AnchorMode.SIMILARITY:
        anchor_ids = _similarity_anchors(states, threshold=similarity_threshold)
    else:
        anchor_ids = [encode_anchor(state, resolved) for state in states]

    group_sizes: dict[str, int] = {}
    distinct: dict[str, set[str]] = {}
    for anchor_id, state in zip(anchor_ids, states, strict=True):
        group_sizes[anchor_id] = group_sizes.get(anchor_id, 0) + 1
        distinct.setdefault(anchor_id, set()).add(canonical_json(state.structured_payload()))

    telemetry = AnchorTelemetry(
        total=len(states),
        unique=len(group_sizes),
        mode=resolved.value,
        group_sizes=group_sizes,
        distinct_states_per_anchor={key: len(value) for key, value in distinct.items()},
    )
    return anchor_ids, telemetry


def _similarity_anchors(states: list[AnchorState], *, threshold: float) -> list[str]:
    """Greedy deterministic clustering by Jaccard similarity.

    Cluster representatives are chosen in sorted structured-id order rather than
    input order, so the assignment does not depend on how the batch was shuffled.
    """

    features = [state.feature_set() for state in states]
    structured_ids = [encode_anchor(state, AnchorMode.STRUCTURED) for state in states]
    order = sorted(range(len(states)), key=lambda i: (structured_ids[i], i))

    representatives: list[int] = []
    assignment: dict[int, int] = {}
    for index in order:
        for rep in representatives:
            # Same task only: merging across tasks would mix unrelated states.
            if states[rep].task_id != states[index].task_id:
                continue
            if jaccard(features[rep], features[index]) >= threshold:
                assignment[index] = rep
                break
        else:
            representatives.append(index)
            assignment[index] = index

    return [
        f"{AnchorMode.SIMILARITY.value}:{structured_ids[assignment[i]].split(':', 1)[1]}"
        for i in range(len(states))
    ]


def anchor_state_from_session(
    session: Any,
    *,
    last_observation: ObservationType = ObservationType.NONE,
    known_info_mask: tuple[bool, ...] = (),
    confirmation_flags: tuple[str, ...] = (),
    policy_precondition_flags: tuple[str, ...] = (),
) -> AnchorState:
    """Build an `AnchorState` from a live `TrajectorySession`."""

    return AnchorState(
        task_id=session.task_id,
        db_hash=session.db_hash(),
        known_info_mask=known_info_mask,
        confirmation_flags=confirmation_flags,
        policy_precondition_flags=policy_precondition_flags,
        last_observation=last_observation,
    )
