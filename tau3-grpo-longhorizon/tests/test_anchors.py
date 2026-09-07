"""Anchor encoding: determinism, mode separation, collision/coverage telemetry."""

from __future__ import annotations

import pytest

from tau3_grpo.anchors.encoder import (
    DEFAULT_SIMILARITY_THRESHOLD,
    AnchorMode,
    AnchorState,
    ObservationType,
    encode_anchor,
    encode_anchors,
    encode_similarity_candidate,
    jaccard,
    resolve_similarity_candidates,
)
from tau3_grpo.anchors.features import (
    KNOWN_INFO_TOOLS,
    confirmation_flags,
    known_info_mask,
    last_observation_type,
    policy_precondition_flags,
    tool_error_count,
)


def _state(**overrides) -> AnchorState:
    base = {
        "task_id": "airline_1",
        "db_hash": "db-aaa",
        "known_info_mask": (True, False, False, False, False),
        "confirmation_flags": ("user_confirmed",),
        "policy_precondition_flags": ("user_identified",),
        "last_observation": ObservationType.TOOL_OK,
    }
    base.update(overrides)
    return AnchorState(**base)


def test_encoding_is_deterministic():
    assert encode_anchor(_state()) == encode_anchor(_state())


def test_flag_order_does_not_change_anchor():
    left = _state(confirmation_flags=("a", "b"))
    right = _state(confirmation_flags=("b", "a"))
    assert encode_anchor(left) == encode_anchor(right)


def test_db_hash_change_changes_anchor():
    assert encode_anchor(_state()) != encode_anchor(_state(db_hash="db-bbb"))


def test_known_info_change_changes_structured_anchor():
    other = _state(known_info_mask=(True, True, False, False, False))
    assert encode_anchor(_state()) != encode_anchor(other)


def test_last_observation_change_changes_structured_anchor():
    other = _state(last_observation=ObservationType.TOOL_ERROR)
    assert encode_anchor(_state()) != encode_anchor(other)


def test_db_hash_only_ignores_structured_features():
    left = _state()
    right = _state(
        known_info_mask=(False, True, True, False, False),
        confirmation_flags=(),
        last_observation=ObservationType.USER,
    )
    assert encode_anchor(left, AnchorMode.DB_HASH_ONLY) == encode_anchor(
        right, AnchorMode.DB_HASH_ONLY
    )
    assert encode_anchor(left) != encode_anchor(right)


def test_db_hash_only_still_separates_tasks():
    left = _state()
    right = _state(task_id="airline_2")
    assert encode_anchor(left, AnchorMode.DB_HASH_ONLY) != encode_anchor(
        right, AnchorMode.DB_HASH_ONLY
    )


def test_mode_prefix_prevents_cross_mode_collision():
    assert encode_anchor(_state()).startswith("structured:")
    assert encode_anchor(_state(), AnchorMode.DB_HASH_ONLY).startswith("db_hash_only:")


def test_structured_mode_has_no_collisions():
    states = [_state(db_hash=f"db-{i}") for i in range(5)]
    _, telemetry = encode_anchors(states, AnchorMode.STRUCTURED)
    assert telemetry.collision_count == 0
    assert telemetry.unique == 5


def test_coverage_counts_only_groups_of_two_or_more():
    states = [_state(), _state(), _state(db_hash="db-unique")]
    _, telemetry = encode_anchors(states, AnchorMode.STRUCTURED)
    assert telemetry.total == 3
    assert telemetry.groups_with_signal == 1
    assert telemetry.coverage == pytest.approx(2 / 3)


def test_coverage_is_zero_when_every_anchor_is_unique():
    states = [_state(db_hash=f"db-{i}") for i in range(4)]
    _, telemetry = encode_anchors(states, AnchorMode.STRUCTURED)
    assert telemetry.coverage == 0.0


def test_db_hash_only_increases_coverage_by_merging():
    states = [
        _state(last_observation=ObservationType.TOOL_OK),
        _state(last_observation=ObservationType.USER),
    ]
    _, structured = encode_anchors(states, AnchorMode.STRUCTURED)
    _, db_only = encode_anchors(states, AnchorMode.DB_HASH_ONLY)
    assert structured.coverage == 0.0
    assert db_only.coverage == pytest.approx(1.0)
    assert db_only.collision_count == 1


def test_jaccard_bounds():
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset({"a"}), frozenset({"a"})) == 1.0
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"a"})) == pytest.approx(0.5)


def test_similarity_merges_near_identical_states():
    left = _state()
    right = _state(confirmation_flags=("user_confirmed", "committed:book_reservation"))
    ids, telemetry = encode_anchors(
        [left, right], AnchorMode.SIMILARITY, similarity_threshold=0.5
    )
    assert ids[0] == ids[1]
    assert telemetry.groups_with_signal == 1


def test_similarity_keeps_distant_states_apart():
    left = _state()
    right = _state(task_id="airline_2", db_hash="db-zzz")
    ids, _ = encode_anchors(
        [left, right], AnchorMode.SIMILARITY, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD
    )
    assert ids[0] != ids[1]


def test_similarity_never_merges_across_tasks():
    left = _state()
    right = _state(task_id="airline_2")
    ids, _ = encode_anchors([left, right], AnchorMode.SIMILARITY, similarity_threshold=0.0)
    assert ids[0] != ids[1]


def test_similarity_is_order_independent():
    states = [_state(db_hash=f"db-{i}") for i in range(4)]
    forward, _ = encode_anchors(states, AnchorMode.SIMILARITY)
    backward, _ = encode_anchors(list(reversed(states)), AnchorMode.SIMILARITY)
    assert set(forward) == set(backward)


def test_rollout_similarity_candidates_are_resolved_at_batch_scope():
    left = encode_similarity_candidate(_state())
    right = encode_similarity_candidate(
        _state(confirmation_flags=("user_confirmed", "committed:book_reservation"))
    )
    resolved = resolve_similarity_candidates([left, right], threshold=0.5)
    assert resolved[0] == resolved[1]
    assert resolved[0].startswith("similarity:")


def test_similarity_candidate_resolver_never_merges_tasks():
    left = encode_similarity_candidate(_state())
    right = encode_similarity_candidate(_state(task_id="airline_2"))
    resolved = resolve_similarity_candidates([left, right], threshold=0.0)
    assert resolved[0] != resolved[1]


# ---- feature extraction ------------------------------------------------


class _Msg:
    def __init__(self, role, content=None, tool_calls=None, id="", error=False):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.id = id
        self.error = error


class _Call:
    def __init__(self, name, id):
        self.name = name
        self.id = id


def test_known_info_mask_tracks_successful_reads():
    messages = [
        _Msg("assistant", tool_calls=[_Call("get_user_details", "c1")]),
        _Msg("tool", content="{}", id="c1"),
    ]
    mask = known_info_mask(messages)
    assert mask[KNOWN_INFO_TOOLS.index("get_user_details")] is True
    assert mask[KNOWN_INFO_TOOLS.index("get_reservation_details")] is False


def test_known_info_mask_includes_flight_status():
    messages = [
        _Msg("assistant", tool_calls=[_Call("get_flight_status", "c-status")]),
        _Msg("tool", content="on_time", id="c-status"),
    ]
    assert known_info_mask(messages)[KNOWN_INFO_TOOLS.index("get_flight_status")] is True


def test_failed_read_does_not_set_known_info():
    messages = [
        _Msg("assistant", tool_calls=[_Call("get_user_details", "c1")]),
        _Msg("tool", content="Error: nope", id="c1", error=True),
    ]
    assert not any(known_info_mask(messages))


def test_confirmation_flags_from_write_and_user():
    messages = [
        _Msg("assistant", tool_calls=[_Call("book_reservation", "c1")]),
        _Msg("tool", content="{}", id="c1"),
        _Msg("user", content="Yes, please go ahead"),
    ]
    flags = confirmation_flags(messages)
    assert "committed:book_reservation" in flags
    assert "user_confirmed" in flags


def test_policy_preconditions_track_identification():
    messages = [
        _Msg("assistant", tool_calls=[_Call("get_user_details", "c1")]),
        _Msg("tool", content="{}", id="c1"),
    ]
    assert "user_identified" in policy_precondition_flags(messages)


def test_tool_error_precondition_flag():
    messages = [
        _Msg("assistant", tool_calls=[_Call("get_user_details", "c1")]),
        _Msg("tool", content="Error", id="c1", error=True),
    ]
    assert "recovering_from_tool_error" in policy_precondition_flags(messages)


def test_last_observation_type_variants():
    assert last_observation_type([]) is ObservationType.NONE
    assert last_observation_type([_Msg("user", "hi")]) is ObservationType.USER
    assert last_observation_type([_Msg("tool", "{}", id="c")]) is ObservationType.TOOL_OK
    assert (
        last_observation_type([_Msg("tool", "e", id="c", error=True)])
        is ObservationType.TOOL_ERROR
    )


def test_tool_error_count():
    messages = [
        _Msg("tool", "e", id="a", error=True),
        _Msg("tool", "{}", id="b"),
        _Msg("tool", "e", id="c", error=True),
    ]
    assert tool_error_count(messages) == 2
