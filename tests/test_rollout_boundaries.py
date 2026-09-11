"""Exercise native loop/tool/replay code with deterministic model and user replies."""

# ruff: noqa: E402 -- optional runtime dependencies are checked before imports.

import asyncio
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verl.experimental.agent_loop.tool_agent_loop", reason="veRL runtime not installed")
pytest.importorskip("tau2.domains.airline.environment", reason="tau2 runtime not installed")

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema

from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.adapter import adapt_record, airline_tool_schemas
from tau3_grpo.envs.interaction import Tau3AirlineInteraction
from tau3_grpo.envs.registry import SESSIONS
from tau3_grpo.envs.session import SessionFactory, SessionLimits, UserSimulatorConfig
from tau3_grpo.envs.tools import Tau3AirlineTool
from tau3_grpo.evaluation.verifier import verify_trajectory
from tau3_grpo.launch import prepare
from tau3_grpo.paths import CODE_ROOT

NEEDLE = "Your refund is 123.45"


class TextTokenizer:
    """Token identity is irrelevant to boundary/recording tests; keep text lossless."""

    @staticmethod
    def encode(text, **kwargs):
        return list(text.encode())

    @staticmethod
    def decode(ids, **kwargs):
        return bytes(ids).decode()


class UserReply:
    def __init__(self, text="###STOP###"):
        self.text = text
        self.calls = 0

    def get_init_state(self, **kwargs):
        return None

    def generate_next_message(self, message, state):
        from tau2.data_model.message import UserMessage

        self.calls += 1
        return UserMessage(role="user", content=self.text), state

    @staticmethod
    def is_stop(message):
        return "###STOP###" in message.content


@pytest.fixture
def live_session(requires_tau2, areal_jsonl):
    from tau2.data_model.tasks import EvaluationCriteria, RewardType

    record = next(json.loads(line) for line in areal_jsonl.read_text().splitlines()
                  if json.loads(line)["id"] == "airline_709")
    adapted = adapt_record(ArealTaskRecord.model_validate(record), dataset_root=areal_jsonl.parent)
    task = adapted.task.model_copy(update={"evaluation_criteria": EvaluationCriteria(
        communicate_info=[NEEDLE], reward_basis=[RewardType.COMMUNICATE],
    )})
    factory = SessionFactory(
        {adapted.task_id: replace(adapted, task=task)},
        user_config=UserSimulatorConfig(model="openai/not-called"), limits=SessionLimits(15, 15),
    )
    handler = Tau3AirlineInteraction({"name": "tau3_airline", "user_model": "openai/not-called"})
    handler.set_session_factory(factory)
    SESSIONS.clear()
    asyncio.run(handler.start_interaction("boundary", task_id=adapted.task_id,
                                         initial_user_message="Please help"))
    session = SESSIONS.require("boundary").session
    session.set_user_simulator(UserReply())
    yield handler, session
    SESSIONS.clear()


def make_loop(handler, text, parser="qwen3_coder"):
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.tokenizer = TextTokenizer()
    loop.response_length = 100000
    loop.prompt_length = 100000
    loop.max_assistant_turns = loop.max_user_turns = 15
    loop.max_parallel_calls = 1
    loop.tool_execution_mode = "sequential"
    loop.max_tool_response_length = 100000
    loop.tool_response_truncate_side = "right"
    schema = next(s for s in airline_tool_schemas() if s["function"]["name"] == "calculate")
    loop.tools = {"calculate": Tau3AirlineTool({}, OpenAIFunctionToolSchema.model_validate(schema))}
    loop.tool_parser = ToolParser.get_tool_parser(parser, loop.tokenizer)
    loop.tool_parser_name = parser
    loop.interaction_config_file = "configured"
    loop.loop = asyncio.get_running_loop()
    generated = []

    async def generate(**kwargs):
        generated.append(text)
        return SimpleNamespace(token_ids=loop.tokenizer.encode(text), num_preempted=0,
                               extra_fields={}, log_probs=None, routed_experts=None)

    async def apply_chat_template(messages, **kwargs):
        return loop.tokenizer.encode("\n".join(m["content"] for m in messages))

    loop.server_manager = SimpleNamespace(generate=generate)
    loop.apply_chat_template = apply_chat_template
    data = AgentData(messages=[{"role": "user", "content": "Please help"}],
                     image_data=[], video_data=[], metrics={}, request_id="boundary",
                     tools_kwargs={}, interaction=handler)
    data.prompt_ids = [42]
    return loop, data, generated


def tool_text(parser, expression="1+1"):
    if parser == "hermes":
        call = json.dumps({"name": "calculate", "arguments": {"expression": expression}})
    else:
        call = f"<function=calculate><parameter=expression>{expression}</parameter></function>"
    return f"Let me calculate.\n<tool_call>{call}</tool_call>\n{NEEDLE}"


@pytest.mark.parametrize("profile,expected_limit", [("qwen35_lora", 65536), ("qwen35_full", 65536)])
def test_real_airline_observation_survives_production_launcher(live_session, tmp_path, profile, expected_limit):
    """Use the actual launcher limit and actual tool; old 256-char defaults lose JSON fields."""
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    inherited = {**os.environ, "TAU3_ENV_FILE": str(tmp_path / "absent.env"),
                 "TAU3_DRY_RUN": "1", "TAU3_RUN_ROOT": str(tmp_path / "runs"),
                 "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    inherited.pop("MAX_TOOL_RESPONSE_CHARS", None)
    command, env, _ = prepare("rl", CODE_ROOT / f"configs/train/rl/{profile}.yaml", "e0", 42, [], inherited)
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=True, timeout=20)
    arguments = shlex.split(completed.stdout.splitlines()[-1])
    limits = [int(arg.split("=", 1)[1]) for arg in arguments
              if arg.startswith("actor_rollout_ref.rollout.multi_turn.max_tool_response_length=")]
    assert limits[-1] == expected_limit
    handler, session = live_session

    async def run():
        loop, data, _ = make_loop(handler, "")
        loop.max_tool_response_length = limits[-1]
        schema = next(s for s in airline_tool_schemas() if s["function"]["name"] == "get_reservation_details")
        loop.tools = {"get_reservation_details": Tau3AirlineTool({}, OpenAIFunctionToolSchema.model_validate(schema))}
        response, _, metrics = await loop._call_tool(
            FunctionCall(name="get_reservation_details", arguments=json.dumps({"reservation_id": "4FDFNE"})), {}, data)
        recorded = session.messages[-1].content
        assert len(recorded) > 256
        assert not metrics["error"]
        assert response.text == recorded
        reservation = json.loads(response.text)
        assert reservation["flights"][0]["flight_number"] == "HAT237"
        assert reservation["payment_history"][0]["payment_id"] == "credit_card_5476036"

    asyncio.run(run())


@pytest.mark.parametrize("parser", ["hermes", "qwen3_coder"])
def test_tool_prose_reaches_official_reward_without_scoring_arguments(live_session, parser):
    handler, session = live_session

    async def run():
        loop, data, _ = make_loop(handler, tool_text(parser), parser)
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        assert await loop._handle_processing_tools_state(data) == AgentState.GENERATING
        message = session.messages[-2]
        assert message.content == f"Let me calculate.\n\n{NEEDLE}"
        assert "1+1" not in message.content
        assert session.messages[-1].content == "2.0"
        reward = verify_trajectory(session, termination_reason="user_stop")
        assert reward.reward == 1.0

    asyncio.run(run())


@pytest.mark.parametrize("parser", ["hermes", "qwen3_coder"])
def test_tool_argument_is_not_counted_as_communicated_text(live_session, parser):
    handler, session = live_session

    async def run():
        text = tool_text(parser, expression=NEEDLE).removesuffix(f"\n{NEEDLE}")
        loop, data, _ = make_loop(handler, text, parser)
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        await loop._handle_processing_tools_state(data)
        assert NEEDLE not in session.messages[-2].content
        assert verify_trajectory(session, termination_reason="user_stop").reward == 0.0

    asyncio.run(run())


def test_last_tool_turn_applies_real_database_write(live_session):
    handler, session = live_session
    session.assistant_turns = 14
    reservation_id = next(iter(session.db.reservations))
    before = session.db_hash()

    async def run():
        call = {"name": "cancel_reservation", "arguments": {"reservation_id": reservation_id}}
        loop, data, _ = make_loop(handler, f"<tool_call>{json.dumps(call)}</tool_call>", "hermes")
        schema = next(s for s in airline_tool_schemas() if s["function"]["name"] == "cancel_reservation")
        loop.tools["cancel_reservation"] = Tau3AirlineTool({}, OpenAIFunctionToolSchema.model_validate(schema))
        data.assistant_turns = 14
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        await loop._handle_processing_tools_state(data)
        assert session.messages[-1].error is False
        # The native environment copies its input DB to isolate replay state.
        assert session.environment.tools.db.reservations[reservation_id].status == "cancelled"
        assert session.db.reservations[reservation_id].status is None
        assert session.db_hash() != before

    asyncio.run(run())


@pytest.mark.parametrize("parser", ["hermes", "qwen3_coder"])
def test_final_tool_turn_executes_once_and_no_extra_generation(live_session, parser):
    handler, session = live_session
    session.assistant_turns = 14

    async def run():
        loop, data, generated = make_loop(handler, tool_text(parser), parser)
        data.assistant_turns = 14
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        assert await loop._handle_processing_tools_state(data) == AgentState.GENERATING
        assert session.assistant_turns == 15
        assert session.tool_calls == 1
        assert session.messages[-1].content == "2.0"
        before = list(data.prompt_ids)
        assert await loop._handle_generating_state(data, {}) == AgentState.TERMINATED
        assert len(generated) == 1
        assert data.prompt_ids == before
        assert data.termination_reason == "max_steps"
        # Executing a last tool is not fabricated into a normal completion.
        assert verify_trajectory(session, termination_reason=data.termination_reason).reward == 0

    asyncio.run(run())


@pytest.mark.parametrize("reply,expected_reason", [("###STOP###", "user_stop"), ("Continue", "max_steps")])
def test_final_text_turn_is_recorded_and_user_can_stop(live_session, reply, expected_reason):
    handler, session = live_session
    session.assistant_turns = 14
    session.set_user_simulator(UserReply(reply))

    async def run():
        loop, data, generated = make_loop(handler, NEEDLE)
        data.assistant_turns = 14
        assert await loop._handle_generating_state(data, {}) == AgentState.INTERACTING
        state = await loop._handle_interacting_state(data)
        if state != AgentState.TERMINATED:
            state = await loop._handle_generating_state(data, {})
        assert state == AgentState.TERMINATED
        assert data.termination_reason == expected_reason
        assert session.assistant_turns == 15
        assert session._user.calls == 1
        assert session.messages[-2].content == NEEDLE
        assert len(generated) == 1
        assert verify_trajectory(session, termination_reason=expected_reason).reward == float(reply == "###STOP###")

    asyncio.run(run())


def test_exhausted_budget_does_not_generate_or_mutate_session(live_session):
    handler, session = live_session

    async def run():
        loop, data, generated = make_loop(handler, tool_text("hermes"), "hermes")
        data.assistant_turns = 15
        before = session.messages
        assert await loop._handle_generating_state(data, {}) == AgentState.TERMINATED
        assert generated == []
        assert session.messages == before
        assert data.response_mask == []

    asyncio.run(run())


def test_context_limit_still_prevents_tool_execution(live_session):
    handler, session = live_session

    async def run():
        loop, data, _ = make_loop(handler, tool_text("hermes"), "hermes")
        loop.response_length = len(loop.tokenizer.encode(tool_text("hermes")))
        assert await loop._handle_generating_state(data, {}) == AgentState.TERMINATED
        assert data.termination_reason == "context_window_exceeded"
        assert session.tool_calls == 0

    asyncio.run(run())


def test_dispatch_error_preserves_prose_in_replay(live_session):
    handler, session = live_session

    async def run():
        text = NEEDLE + '\n<tool_call>{"name":"unknown_tool","arguments":{}}</tool_call>'
        loop, data, _ = make_loop(handler, text, "hermes")
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        await loop._handle_processing_tools_state(data)
        assert session.messages[-2].content.strip() == NEEDLE
        assert session.messages[-1].error is True
        assert SESSIONS.require("boundary").tool_error_count == 1

    asyncio.run(run())


@pytest.mark.parametrize("parser", ["hermes", "qwen3_coder"])
@pytest.mark.parametrize("starting_turn", [0, 14])
def test_full_batch_is_sequential_one_turn_and_token_aligned(live_session, parser, starting_turn):
    """Yield inside each tool so a gather implementation would expose overlap."""
    handler, session = live_session
    session.assistant_turns = starting_turn

    async def run():
        text = "\n".join(tool_text(parser, expression=str(i)) for i in (1, 2, 3))
        loop, data, generated = make_loop(handler, text, parser)
        data.assistant_turns = starting_turn
        tool = loop.tools["calculate"]
        original_execute = tool.execute
        events = []

        async def execute(instance_id, parameters, **kwargs):
            value = parameters["expression"]
            events.append(("start", value))
            await asyncio.sleep(0)
            result = await original_execute(instance_id, parameters, **kwargs)
            events.append(("end", value))
            return result

        tool.execute = execute
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        generated_ids = list(data.response_ids)
        data.response_logprobs = [-0.25] * len(generated_ids)
        assert await loop._handle_processing_tools_state(data) == AgentState.GENERATING
        assert events == [(kind, str(i)) for i in (1, 2, 3) for kind in ("start", "end")]
        assert len(generated) == 1
        assert session.assistant_turns == starting_turn + 1
        assert session.tool_calls == 3
        messages = session.messages[1:]
        assert [m.role for m in messages] == ["assistant", "tool", "tool", "tool"]
        ids = [call.id for call in messages[0].tool_calls]
        assert len(set(ids)) == 3
        assert ids == [m.id for m in messages[1:]]
        assert ids == [m["tool_call_id"] for m in data.messages[-3:]]
        assert [m.content for m in messages[1:]] == ["1.0", "2.0", "3.0"]
        assert data.prompt_ids[1:1 + len(generated_ids)] == generated_ids
        n = len(generated_ids)
        assert data.response_mask[:n] == [1] * n
        assert set(data.response_mask[n:]) == {0}
        assert data.response_logprobs[:n] == [-0.25] * n
        assert set(data.response_logprobs[n:]) == {0.0}
        assert len(data.response_mask) == len(data.response_logprobs) == len(data.prompt_ids) - 1
        assert len(data.anchor_spans) == 2  # one assistant segment, one observation segment
        if starting_turn == 14:
            assert await loop._handle_generating_state(data, {}) == AgentState.TERMINATED
            assert len(generated) == 1

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["unknown", "bad_json", "bad_arguments", "tool_error"])
def test_middle_error_has_own_result_and_does_not_drop_later_calls(live_session, failure):
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    handler, session = live_session

    async def run():
        loop, data, _ = make_loop(handler, "")
        middle = {
            "unknown": FunctionCall(name="unknown_tool", arguments="{}"),
            "bad_json": FunctionCall(name="calculate", arguments="{broken"),
            "bad_arguments": FunctionCall(name="calculate", arguments="[]"),
            "tool_error": FunctionCall(name="calculate", arguments='{"expression":"1/0"}'),
        }[failure]
        data.tool_calls = [
            FunctionCall(name="calculate", arguments='{"expression":"1+1"}'),
            middle,
            FunctionCall(name="calculate", arguments='{"expression":"2+2"}'),
        ]
        await loop._handle_processing_tools_state(data)
        assert session.assistant_turns == 1
        assert session.tool_calls == 3
        assistant, first, second, third = session.messages[1:]
        assert [call.id for call in assistant.tool_calls] == [m.id for m in (first, second, third)]
        assert first.content == "2.0"
        assert second.error is True
        assert third.content == "4.0"
        assert SESSIONS.require("boundary").tool_error_count == 1

    asyncio.run(run())


def test_write_then_read_batch_matches_strict_official_replay(live_session):
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    from tau3_grpo.envs.adapter import build_environment, load_flight_db

    handler, session = live_session
    reservation_id = next(iter(session.db.reservations))

    async def run():
        loop, data, _ = make_loop(handler, "")
        names = ("cancel_reservation", "get_reservation_details")
        for schema in airline_tool_schemas():
            name = schema["function"]["name"]
            if name in names:
                loop.tools[name] = Tau3AirlineTool({}, OpenAIFunctionToolSchema.model_validate(schema))
        data.tool_calls = [FunctionCall(name=name, arguments=json.dumps({
            "reservation_id": reservation_id,
        })) for name in names]
        await loop._handle_processing_tools_state(data)
        assert json.loads(session.messages[-1].content)["status"] == "cancelled"
        replay = build_environment(load_flight_db(session.adapted.db_path))
        replay.set_state(initialization_data=None, initialization_actions=None,
                         message_history=session.messages, strict=True)
        assert replay.get_db_hash() == session.db_hash()

    asyncio.run(run())


def test_unexpected_execution_exception_is_not_retried_as_tool_error(live_session):
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    handler, session = live_session

    async def run():
        loop, data, _ = make_loop(handler, "")
        tool = loop.tools["calculate"]
        execute = tool.execute

        async def fail_after_execution(*args, **kwargs):
            await execute(*args, **kwargs)
            raise RuntimeError("wrapper failed after execution")

        tool.execute = fail_after_execution
        data.tool_calls = [FunctionCall(name="calculate", arguments='{"expression":"1"}')]
        with pytest.raises(RuntimeError, match="wrapper failed"):
            await loop._handle_processing_tools_state(data)
        assert session.tool_calls == 1
        assert len(session.messages) == 3  # initial user + one assistant + one result

    asyncio.run(run())


def test_pending_updates_legacy_parquet_prompt_before_tokenization(live_session):
    from tau3_grpo.prompts import build_system_prompt

    handler, _ = live_session

    async def run():
        loop, data, generated = make_loop(handler, "")
        data.messages.insert(0, {"role": "system", "content": "only ONE tool call"})
        loop.tool_schemas = []
        assert await loop._handle_pending_state(data, {}) == AgentState.GENERATING
        assert data.messages[0]["content"] == build_system_prompt()
        assert loop.tokenizer.decode(data.prompt_ids).startswith(build_system_prompt())
        assert data.response_mask == []
        assert generated == []
        loop.tool_execution_mode = "parallel"
        with pytest.raises(ValueError, match="requires tool_execution_mode=sequential"):
            await loop._handle_pending_state(data, {})

    asyncio.run(run())
