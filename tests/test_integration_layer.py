"""veRL interaction and tool layer.

These need veRL importable (for `BaseInteraction` / `BaseTool`) and skip
otherwise. The tau2 surface and the paid user simulator are stubbed; the session,
registry and anchor plumbing are real.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

pytest.importorskip("verl", reason="veRL not installed")

pytestmark = pytest.mark.verl

from tau3_grpo.data.schema import DataSource  # noqa: E402
from tau3_grpo.envs.adapter import AdaptedTask  # noqa: E402
from tau3_grpo.envs.session import SessionFactory, UserSimulatorConfig  # noqa: E402

USER_CONFIG = UserSimulatorConfig(model="stub", temperature=1.0)


class _StubDB:
    def __init__(self, payload):
        self.payload = dict(payload)

    def get_hash(self):
        return json.dumps(self.payload, sort_keys=True)


class _StubToolMessage:
    def __init__(self, id, content, error=False):
        self.id = id
        self.content = content
        self.error = error
        self.role = "tool"


class _StubEnvironment:
    def __init__(self, db):
        self.db = db

    def get_db_hash(self):
        return self.db.get_hash()

    def get_policy(self):
        return "STUB POLICY"

    def get_response(self, tool_call):
        if tool_call.name == "unknown_tool":
            return _StubToolMessage(tool_call.id, "Error: no such tool", error=True)
        if any(str(key).startswith("__") for key in tool_call.arguments):
            return _StubToolMessage(tool_call.id, "Error: invalid arguments", error=True)
        if tool_call.name == "book_reservation":
            self.db.payload["n"] = self.db.payload.get("n", 0) + 1
        return _StubToolMessage(tool_call.id, json.dumps(self.db.payload))


class _StubToolCall:
    def __init__(self, id, name, arguments, requestor="assistant"):
        self.id = id
        self.name = name
        self.arguments = arguments
        self.requestor = requestor


class _StubMessage:
    def __init__(self, role="assistant", content=None, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _StubUser:
    def __init__(self, replies=None, **_official_user_kwargs):
        # Production constructs tau2's UserSimulator with llm/instructions/
        # llm_args. Accept those official keyword arguments while preserving
        # the positional replies seam used by individual tests below.
        self.replies = list(replies or ["and then?"])
        self.calls = 0

    def get_init_state(self, message_history=None):
        return {"history": list(message_history or [])}

    def generate_next_message(self, message, state):
        self.calls += 1
        return _StubMessage("user", self.replies[min(self.calls - 1, len(self.replies) - 1)]), state

    @classmethod
    def is_stop(cls, message):
        return "###STOP###" in (message.content or "")


@pytest.fixture
def factory(monkeypatch, tmp_path):
    db_file = tmp_path / "db.json"
    db_file.write_text(json.dumps({"n": 0}), encoding="utf-8")

    monkeypatch.setattr(
        "tau3_grpo.envs.session.load_flight_db",
        lambda path: _StubDB(json.loads(open(path).read())),
    )
    monkeypatch.setattr("tau3_grpo.envs.session.build_environment", _StubEnvironment)
    monkeypatch.setattr(
        "tau3_grpo.envs.session.message_models",
        lambda: {
            "AssistantMessage": _StubMessage,
            "UserMessage": _StubMessage,
            "ToolMessage": _StubToolMessage,
            "SystemMessage": dict,
            "ToolCall": _StubToolCall,
        },
    )
    monkeypatch.setattr("tau3_grpo.envs.session.user_simulator_cls", lambda: _StubUser)

    adapted = AdaptedTask(
        task_id="airline_1",
        task=object(),
        db_path=db_file,
        db_file_hash="fh",
        user_instructions="do it",
        known_info=None,
    )
    return SessionFactory({"airline_1": adapted}, user_config=USER_CONFIG)


@pytest.fixture
def interaction(factory):
    from tau3_grpo.envs.interaction import Tau3AirlineInteraction
    from tau3_grpo.envs.registry import SESSIONS

    SESSIONS.clear()
    handler = Tau3AirlineInteraction({"name": "tau3_airline", "user_model": "stub"})
    handler.set_session_factory(factory)
    yield handler
    SESSIONS.clear()


def test_start_interaction_creates_isolated_session(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    asyncio.run(interaction.start_interaction("req-2", task_id="airline_1"))
    a = SESSIONS.require("req-1").session
    b = SESSIONS.require("req-2").session
    assert a.db is not b.db
    assert SESSIONS.active_count() == 2


def test_initial_prompt_user_is_seeded_into_tau2_session(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(
        interaction.start_interaction(
            "req-initial",
            task_id="airline_1",
            initial_user_message="I need to change my reservation.",
        )
    )
    session = SESSIONS.require("req-initial").session
    assert [message.role for message in session.messages] == ["user"]
    assert session.messages[0].content == "I need to change my reservation."
    assert session.user_turns == 1
    assert session._user_state["history"][0].content == "I need to change my reservation."


def test_start_interaction_refuses_tau3_official(interaction):
    from tau3_grpo.data.official import SourceIsolationError

    with pytest.raises(SourceIsolationError):
        asyncio.run(
            interaction.start_interaction(
                "req-x",
                task_id="airline_1",
                source=DataSource.TAU3_OFFICIAL_AIRLINE.value,
            )
        )


def test_start_interaction_requires_task_id(interaction):
    with pytest.raises(ValueError, match="task_id"):
        asyncio.run(interaction.start_interaction("req-1"))


def test_missing_factory_raises():
    from tau3_grpo.envs.interaction import Tau3AirlineInteraction

    handler = Tau3AirlineInteraction({"name": "tau3_airline"})
    with pytest.raises(RuntimeError, match="either a SessionFactory"):
        asyncio.run(handler.start_interaction("req", task_id="airline_1"))


def test_invalid_record_json_is_rejected():
    from tau3_grpo.envs.interaction import Tau3AirlineInteraction

    handler = Tau3AirlineInteraction({"name": "tau3_airline"})
    with pytest.raises(ValueError, match="record_json is invalid"):
        asyncio.run(
            handler.start_interaction(
                "req-json",
                task_id="airline_1",
                record_json="{not-json",
            )
        )


def test_generate_response_advances_user(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    SESSIONS.require("req-1").session.set_user_simulator(_StubUser(["what next?"]))

    terminate, content, score, info = asyncio.run(
        interaction.generate_response("req-1", [{"role": "assistant", "content": "hi"}])
    )
    assert terminate is False
    assert content == "what next?"
    assert score == 0.0


def test_generate_response_offloads_blocking_simulator(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-thread", task_id="airline_1"))
    session = SESSIONS.require("req-thread").session
    caller_thread = threading.get_ident()
    worker_threads = []
    original = session.user_respond

    def tracked(message):
        worker_threads.append(threading.get_ident())
        return original(message)

    session.user_respond = tracked
    asyncio.run(
        interaction.generate_response(
            "req-thread", [{"role": "assistant", "content": "hi"}]
        )
    )
    assert worker_threads and worker_threads[0] != caller_thread


def test_user_stop_terminates(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    SESSIONS.require("req-1").session.set_user_simulator(_StubUser(["bye ###STOP###"]))

    terminate, _, _, info = asyncio.run(
        interaction.generate_response("req-1", [{"role": "assistant", "content": "hi"}])
    )
    assert terminate is True
    assert info["termination_reason"] == "user_stop"


def test_user_turn_limit_terminates_before_another_simulator_call(interaction, factory):
    from tau3_grpo.envs.registry import SESSIONS, SessionEntry
    from tau3_grpo.envs.session import SessionLimits

    session = factory.create("airline_1")
    session.limits = SessionLimits(max_user_turns=1, max_assistant_turns=1)
    session.record_initial_user_text("Hello")
    SESSIONS.register("req-lim", SessionEntry(session=session))

    terminate, _, _, info = asyncio.run(
        interaction.generate_response("req-lim", [{"role": "assistant", "content": "hi"}])
    )
    assert terminate is True
    assert info["termination_reason"] == "max_steps"
    assert session._user.calls == 0


def test_last_allowed_assistant_turn_can_receive_user_stop(interaction, factory):
    from tau3_grpo.envs.registry import SESSIONS, SessionEntry
    from tau3_grpo.envs.session import SessionLimits

    session = factory.create("airline_1")
    session.limits = SessionLimits(max_user_turns=2, max_assistant_turns=1)
    session.record_initial_user_text("Hello")
    session.set_user_simulator(_StubUser(["###STOP###"]))
    SESSIONS.register("req-last", SessionEntry(session=session))
    terminate, _, _, info = asyncio.run(
        interaction.generate_response("req-last", [{"role": "assistant", "content": "Done"}])
    )
    assert terminate is True
    assert info["termination_reason"] == "user_stop"
    assert session.assistant_turns == 1
    assert session.user_turns == 2
    assert session._user.calls == 1


def test_finalize_releases_session(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    asyncio.run(interaction.finalize_interaction("req-1"))
    assert SESSIONS.active_count() == 0


def test_turn_score_is_always_zero(interaction):
    assert asyncio.run(interaction.calculate_score()) == 0.0


# ---- tools -------------------------------------------------------------


def _tool(name: str):
    from verl.tools.schemas import OpenAIFunctionToolSchema

    from tau3_grpo.envs.tools import Tau3AirlineTool

    schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "stub",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    return Tau3AirlineTool({}, schema)


class _AgentData:
    def __init__(self, request_id):
        self.request_id = request_id


def test_tool_hits_the_live_environment(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    session = SESSIONS.require("req-1").session
    before = session.db_hash()

    tool = _tool("book_reservation")
    instance_id, _ = asyncio.run(tool.create())
    response, reward, metrics = asyncio.run(
        tool.execute(instance_id, {}, agent_data=_AgentData("req-1"))
    )

    assert reward == 0.0
    assert metrics["tool"] == "book_reservation"
    assert session.db_hash() != before
    assert json.loads(response.text)["n"] == 1


def test_tool_records_replayable_message_pair(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    tool = _tool("get_user_details")
    instance_id, _ = asyncio.run(tool.create())
    asyncio.run(tool.execute(instance_id, {"user_id": "u"}, agent_data=_AgentData("req-1")))

    messages = SESSIONS.require("req-1").session.messages
    assert [m.role for m in messages] == ["assistant", "tool"]
    assert messages[0].tool_calls[0].name == "get_user_details"
    assert messages[0].tool_calls[0].id == messages[1].id


def test_tool_error_is_counted_not_raised(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    tool = _tool("unknown_tool")
    instance_id, _ = asyncio.run(tool.create())
    response, _, metrics = asyncio.run(
        tool.execute(instance_id, {}, agent_data=_AgentData("req-1"))
    )
    assert metrics["error"] is True
    assert "Error" in response.text
    assert SESSIONS.require("req-1").tool_error_count == 1


@pytest.mark.parametrize(
    ("tool_name", "raw_arguments", "expected_marker"),
    [
        ("unknown_tool", "{}", None),
        ("get_user_details", "{broken", "__malformed_json__"),
        ("get_user_details", "[]", "__invalid_arguments__"),
    ],
)
def test_dispatch_failure_is_recorded_for_official_replay(
    interaction, tool_name, raw_arguments, expected_marker
):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-failed", task_id="airline_1"))
    content = interaction.record_tool_failure(
        "req-failed",
        tool_name=tool_name,
        raw_arguments=raw_arguments,
        error="parser or dispatch failed",
    )

    entry = SESSIONS.require("req-failed")
    messages = entry.session.messages
    assert [message.role for message in messages] == ["assistant", "tool"]
    assert messages[0].tool_calls[0].name == tool_name
    assert messages[0].tool_calls[0].id == messages[1].id
    if expected_marker is not None:
        assert expected_marker in messages[0].tool_calls[0].arguments
    assert messages[1].error is True
    assert content.startswith("Error:")
    assert entry.tool_error_count == 1


def test_tool_without_agent_data_raises(interaction):
    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    tool = _tool("get_user_details")
    instance_id, _ = asyncio.run(tool.create())
    with pytest.raises(RuntimeError, match="without agent_data"):
        asyncio.run(tool.execute(instance_id, {}))


def test_tool_with_unknown_request_raises(interaction):
    tool = _tool("get_user_details")
    instance_id, _ = asyncio.run(tool.create())
    with pytest.raises(KeyError, match="no tau2 session registered"):
        asyncio.run(tool.execute(instance_id, {}, agent_data=_AgentData("nope")))


def test_two_sessions_do_not_share_tool_writes(interaction):
    from tau3_grpo.envs.registry import SESSIONS

    asyncio.run(interaction.start_interaction("req-1", task_id="airline_1"))
    asyncio.run(interaction.start_interaction("req-2", task_id="airline_1"))

    tool = _tool("book_reservation")
    instance_id, _ = asyncio.run(tool.create())
    asyncio.run(tool.execute(instance_id, {}, agent_data=_AgentData("req-1")))

    assert json.loads(SESSIONS.require("req-1").session.db_hash())["n"] == 1
    assert json.loads(SESSIONS.require("req-2").session.db_hash())["n"] == 0
