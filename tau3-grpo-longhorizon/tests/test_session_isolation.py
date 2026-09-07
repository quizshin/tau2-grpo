"""DB and session isolation.

Two rollouts of the same task must not share a FlightDB. The expensive user
simulator is stubbed here (that is the only permitted mock, and only in tests);
the DB and Environment are the real tau2 objects when tau2 is importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau3_grpo.env.adapter import AdaptedTask
from tau3_grpo.env.session import (
    SessionFactory,
    SessionLimits,
    TrajectorySession,
    UserSimulatorConfig,
)
from tau3_grpo.integration.registry import SessionEntry, SessionRegistry

USER_CONFIG = UserSimulatorConfig(model="stub-model", temperature=1.0)


# ---- stub tau2 surface -------------------------------------------------


class _StubDB:
    """Minimal DB double: mutable state plus a content-derived hash."""

    def __init__(self, payload: dict):
        self.payload = dict(payload)

    def get_hash(self) -> str:
        return json.dumps(self.payload, sort_keys=True)


class _StubToolMessage:
    def __init__(self, id, content, error=False):
        self.id = id
        self.content = content
        self.error = error
        self.role = "tool"
        self.requestor = "assistant"


class _StubEnvironment:
    def __init__(self, db: _StubDB):
        self.db = db

    def get_db_hash(self) -> str:
        return self.db.get_hash()

    def get_policy(self) -> str:
        return "STUB POLICY"

    def get_response(self, tool_call):
        # Mutating tool writes into this session's own DB only.
        if tool_call.name == "book_reservation":
            self.db.payload["reservations"] = self.db.payload.get("reservations", 0) + 1
        return _StubToolMessage(tool_call.id, json.dumps(self.db.payload))


class _StubToolCall:
    def __init__(self, id, name, arguments, requestor="assistant"):
        self.id = id
        self.name = name
        self.arguments = arguments
        self.requestor = requestor


class _StubAssistantMessage:
    def __init__(self, role="assistant", content=None, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _StubUserMessage:
    def __init__(self, role="user", content=None, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _StubUserSimulator:
    """Stands in for the paid LLM user simulator."""

    def __init__(
        self,
        replies=None,
        *,
        llm=None,
        instructions=None,
        llm_args=None,
        **kwargs,
    ):
        self.llm = llm
        self.instructions = instructions
        self.llm_args = dict(llm_args or {})
        self.replies = list(replies or ["ok"])
        self.calls = 0
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def get_init_state(self, message_history=None):
        return {"messages": list(message_history or [])}

    def generate_next_message(self, message, state):
        self.calls += 1
        content = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return _StubUserMessage(content=content), state

    @classmethod
    def is_stop(cls, message):
        return "###STOP###" in (message.content or "")


@pytest.fixture
def stub_session(monkeypatch, tmp_path):
    """Build a TrajectorySession whose tau2 surface is stubbed."""

    db_file = tmp_path / "db.json"
    db_file.write_text(json.dumps({"reservations": 0}), encoding="utf-8")

    monkeypatch.setattr(
        "tau3_grpo.env.session.load_flight_db",
        lambda path: _StubDB(json.loads(Path(path).read_text(encoding="utf-8"))),
    )
    monkeypatch.setattr(
        "tau3_grpo.env.session.build_environment", lambda db: _StubEnvironment(db)
    )
    monkeypatch.setattr(
        "tau3_grpo.env.session.message_models",
        lambda: {
            "AssistantMessage": _StubAssistantMessage,
            "UserMessage": _StubUserMessage,
            "ToolMessage": _StubToolMessage,
            "SystemMessage": dict,
            "ToolCall": _StubToolCall,
        },
    )
    monkeypatch.setattr(
        "tau3_grpo.env.session.user_simulator_cls", lambda: _StubUserSimulator
    )

    adapted = AdaptedTask(
        task_id="airline_1",
        task=object(),
        db_path=db_file,
        db_file_hash="filehash",
        user_instructions="do the thing",
        known_info=None,
    )
    return adapted


def _make(adapted, **kwargs):
    return TrajectorySession(adapted, user_config=USER_CONFIG, **kwargs)


def test_two_sessions_have_distinct_db_objects(stub_session):
    a = _make(stub_session)
    b = _make(stub_session)
    assert a.db is not b.db
    assert a.environment is not b.environment


def test_write_in_one_session_does_not_leak(stub_session):
    a = _make(stub_session)
    b = _make(stub_session)
    assert a.db_hash() == b.db_hash()

    call = a.make_tool_call("book_reservation", {}, "c1")
    a.record_assistant_tool_calls([call])
    a.execute_tool_call(call)

    assert a.db_hash() != b.db_hash()
    assert json.loads(b.db_hash())["reservations"] == 0
    assert json.loads(a.db_hash())["reservations"] == 1


def test_eight_rollouts_of_same_task_are_independent(stub_session):
    sessions = [_make(stub_session) for _ in range(8)]
    call = sessions[0].make_tool_call("book_reservation", {}, "c1")
    sessions[0].record_assistant_tool_calls([call])
    sessions[0].execute_tool_call(call)

    mutated = json.loads(sessions[0].db_hash())["reservations"]
    others = {json.loads(s.db_hash())["reservations"] for s in sessions[1:]}
    assert mutated == 1
    assert others == {0}


def test_session_ids_are_unique(stub_session):
    ids = {_make(stub_session).session_id for _ in range(8)}
    assert len(ids) == 8


def test_messages_are_recorded_in_order(stub_session):
    session = _make(stub_session)
    call = session.make_tool_call("get_user_details", {"user_id": "u"}, "c1")
    session.record_assistant_tool_calls([call])
    session.execute_tool_call(call)
    roles = [m.role for m in session.messages]
    assert roles == ["assistant", "tool"]


def test_initial_db_hash_is_captured(stub_session):
    session = _make(stub_session)
    initial = session.initial_db_hash
    call = session.make_tool_call("book_reservation", {}, "c1")
    session.record_assistant_tool_calls([call])
    session.execute_tool_call(call)
    assert session.initial_db_hash == initial
    assert session.db_hash() != initial


def test_turn_limit_reached(stub_session):
    session = _make(stub_session, limits=SessionLimits(max_user_turns=2, max_assistant_turns=2))
    assert not session.turn_limit_reached()
    session.record_assistant_text("one")
    session.record_assistant_text("two")
    assert session.turn_limit_reached()


def test_user_simulator_is_only_stubbed_in_tests(stub_session):
    session = _make(stub_session)
    session.set_user_simulator(_StubUserSimulator(["hello"]))
    message = session.record_assistant_text("hi")
    reply = session.user_respond(message)
    assert reply.content == "hello"
    assert session.user_turns == 1


def test_stop_sentinel_detected(stub_session):
    session = _make(stub_session)
    session.set_user_simulator(_StubUserSimulator(["done ###STOP###"]))
    message = session.record_assistant_text("hi")
    reply = session.user_respond(message)
    assert session.user_is_stop(reply)


def test_metadata_carries_hashes_and_counts(stub_session):
    session = _make(stub_session, seed=42)
    call = session.make_tool_call("book_reservation", {}, "c1")
    session.record_assistant_tool_calls([call])
    session.execute_tool_call(call)
    meta = session.metadata()
    assert meta["task_id"] == "airline_1"
    assert meta["seed"] == 42
    assert meta["tool_calls"] == 1
    assert meta["db_file_hash"] == "filehash"
    assert meta["initial_db_hash"] != meta["final_db_hash"]


def test_session_threads_seed_into_user_simulator(stub_session):
    session = _make(stub_session, seed=42)
    assert session._user.seed == 42
    session.set_seed(43)
    assert session.seed == 43
    assert session._user.seed == 43


def test_factory_creates_isolated_sessions(stub_session):
    factory = SessionFactory({"airline_1": stub_session}, user_config=USER_CONFIG)
    a = factory.create("airline_1")
    b = factory.create("airline_1")
    assert a.db is not b.db
    assert factory.task_ids() == ["airline_1"]


def test_factory_rejects_unknown_task(stub_session):
    factory = SessionFactory({"airline_1": stub_session}, user_config=USER_CONFIG)
    with pytest.raises(KeyError, match="unknown task_id"):
        factory.create("airline_999")


# ---- registry ---------------------------------------------------------


def test_registry_isolates_by_request_id(stub_session):
    registry = SessionRegistry()
    a = _make(stub_session)
    b = _make(stub_session)
    registry.register("req-a", SessionEntry(session=a))
    registry.register("req-b", SessionEntry(session=b))
    assert registry.require("req-a").session is a
    assert registry.require("req-b").session is b
    assert registry.active_count() == 2


def test_registry_raises_for_unknown_request():
    registry = SessionRegistry()
    with pytest.raises(KeyError, match="no tau2 session registered"):
        registry.require("missing")


def test_registry_pop_releases_session(stub_session):
    registry = SessionRegistry()
    registry.register("req", SessionEntry(session=_make(stub_session)))
    assert registry.pop("req") is not None
    assert registry.active_count() == 0
    assert registry.pop("req") is None


def test_registry_records_anchor_segments(stub_session):
    entry = SessionEntry(session=_make(stub_session))
    entry.record_segment("anchor-1", (0, 5))
    entry.record_segment(None, None)
    assert entry.anchor_ids == ["anchor-1", None]
    assert entry.anchor_spans == [(0, 5), None]


# ---- real tau2 ---------------------------------------------------------


@pytest.mark.tau3
def test_real_flight_db_instances_are_distinct(requires_tau2):
    from tau3_grpo.env.adapter import load_default_flight_db

    a = load_default_flight_db()
    b = load_default_flight_db()
    assert a is not b
    assert a.get_hash() == b.get_hash()


@pytest.mark.tau3
def test_real_environment_reports_policy_and_hash(requires_tau2):
    from tau3_grpo.env.adapter import build_environment, load_default_flight_db

    env = build_environment(load_default_flight_db())
    assert env.get_db_hash()
    assert "airline" in env.get_domain_name()
    assert env.get_policy().strip()
