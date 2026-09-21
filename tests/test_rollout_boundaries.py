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


@pytest.mark.parametrize("reason", ["max_steps", "timeout", "context_window_exceeded"])
def test_capped_real_session_keeps_zero_outcome_and_explicit_eligibility(live_session, reason):
    handler, _ = live_session
    payload = asyncio.run(handler.finalize_rollout("boundary", termination_reason=reason))
    assert payload["reward"] == 0 and payload["scored"] is False
    eligibility = payload["execution_eligibility"]
    assert eligibility["category"] == "budget_limit"
    assert eligibility["training_candidate_eligible"] and eligibility["automatic_retries"] == 0
    assert SESSIONS.get("boundary") is None


@pytest.mark.parametrize("reason", ["infrastructure_error", "unexpected_error"])
def test_infrastructure_finalization_aborts_and_releases_private_session(live_session, reason):
    handler, _ = live_session
    with pytest.raises(RuntimeError, match="Unresolved execution"):
        asyncio.run(handler.finalize_rollout("boundary", termination_reason=reason))
    assert SESSIONS.get("boundary") is None


@pytest.mark.parametrize("recipe", [
    {"mode": "conservative"}, {"mode": "paper", "version": "paper_v1"},
])
def test_mt_gtpo_turn_records_multi_call_and_terminal_payload(live_session, recipe):
    """Native parser, real calculator, isolated session and official verifier."""
    from tau3_grpo.evaluation.process_reward import reward_settings

    handler, session = live_session

    async def run():
        text = "\n".join(tool_text("qwen3_coder", e) for e in ("1+1", "1/0", "2+2"))
        loop, data, _ = make_loop(handler, text)
        loop.record_process_turns = True
        loop.process_reward_config = reward_settings(recipe)
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        assert data.anchor_ids == []  # no semantic/state extraction
        span = [0, len(data.response_ids)]
        await loop._handle_processing_tools_state(data)
        assert len(data.turn_records) == 1
        assert data.turn_records[0]["token_span"] == span
        events = data.turn_records[0]["tool_calls"]
        assert [e["arguments"]["expression"] for e in events] == ["1+1", "1/0", "2+2"]
        assert [e["error"] for e in events] == [False, True, False]
        assert len({e["id"] for e in events}) == 3
        payload = await handler.finalize_rollout("boundary", termination_reason="agent_stop",
            turn_records=data.turn_records, process_reward_config=loop.process_reward_config)
        process = json.loads(payload["process_reward_json"])
        assert process["turn_rewards"] == pytest.approx([-.1])
        assert process["turn_spans"] == [span]
        assert process["termination_reason"] == "agent_stop"
        assert SESSIONS.get("boundary") is None

    asyncio.run(run())


def test_mt_gtpo_context_truncation_keeps_unparsed_turn(live_session):
    handler, _ = live_session

    async def run():
        text = tool_text("qwen3_coder")
        loop, data, _ = make_loop(handler, text)
        loop.record_process_turns = True
        loop.response_length = len(loop.tokenizer.encode(text))
        assert await loop._handle_generating_state(data, {}) == AgentState.TERMINATED
        assert data.turn_records[0]["truncated"] is True
        assert data.turn_records[0]["parsed"] is False
        assert data.turn_records[0]["tool_calls"] == []

    asyncio.run(run())


@pytest.mark.parametrize("estimator", ["grpo", "tau_gigpo", "mt_gtpo"])
@pytest.mark.parametrize("record_logprobs", [False, True])
def test_full_loop_publishes_shared_facts_without_changing_reward(live_session, estimator, record_logprobs):
    from tau3_grpo.evaluation.process_reward import reward_settings

    handler, _ = live_session

    async def run():
        loop, _, _ = make_loop(handler, tool_text("qwen3_coder"))
        loop.record_process_turns = estimator == "mt_gtpo"
        loop.record_turn_facts = True
        loop.process_reward_config = reward_settings()
        loop.interaction_map = {"tau3_airline": handler}
        original_start = handler.start_interaction

        async def start(*args, **kwargs):
            request = await original_start(*args, **kwargs)
            SESSIONS.require(request).session.set_user_simulator(UserReply())
            return request

        async def vision(messages):
            return {}

        async def pending(data, params):
            data.prompt_ids = [42]
            return AgentState.GENERATING

        count = 0

        async def generate(**kwargs):
            nonlocal count
            text = tool_text("qwen3_coder") if count == 0 else NEEDLE
            count += 1
            return SimpleNamespace(token_ids=loop.tokenizer.encode(text), num_preempted=0,
                                   extra_fields={"finish_reason": "length" if count == 1 else "stop",
                                                 "native_stop_reason": None if count == 1 else "eos"},
                                   log_probs=[-.25] * len(loop.tokenizer.encode(text)) if record_logprobs else None,
                                   routed_experts=None)

        handler.start_interaction = start
        loop.process_vision_info = vision
        loop._handle_pending_state = pending
        loop.server_manager.generate = generate
        output = await loop.run({}, raw_prompt=[{"role": "user", "content": "help"}],
            tau3_sampling_identity={"sample_group_uid": "actual-group", "trial": 3, "seed": 42,
                                    "seed_semantics": "configured_data_seed"},
            extra_info={"interaction_kwargs": {"name": "tau3_airline", "task_id": "airline_709",
                                                "initial_user_message": "help"}})
        raw = output.extra_fields["trajectory_facts_json"]
        assert output.extra_fields["reward_extra_info"]["trajectory_facts_json"] == raw
        facts = json.loads(raw)
        assert facts["tokens"]["response_ids"] == output.response_ids
        assert facts["tokens"]["response_mask"] == output.response_mask
        assert [t["finish_reason"] for t in facts["turns"]] == ["length", "stop"]
        assert facts["identity"]["sample_group_uid"] == "actual-group"
        assert facts["identity"]["trial"] == 3 and facts["identity"]["seed"] == 42
        assert facts["terminal"]["scored"] is True
        assert facts["capabilities"]["complete_response_logprobs"] is record_logprobs
        if record_logprobs:
            assert len(facts["tokens"]["response_logprobs"]) == len(output.response_ids)
            assert all(p == (-.25 if m else 0.) for p, m in
                       zip(facts["tokens"]["response_logprobs"], output.response_mask))
        assert facts["terminal"]["execution_eligibility"]["training_candidate_eligible"]
        if estimator == "mt_gtpo":
            payload = json.loads(output.extra_fields["process_reward_json"])
            assert payload["turn_rewards"] == [0, 0]
        else:
            assert "process_reward_json" not in output.extra_fields
            assert output.extra_fields["anchor_ids"]  # fact recording retains the anchor hook
        assert output.reward_score == 1.

    asyncio.run(run())


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
        loop.record_turn_facts = True
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        await loop._handle_processing_tools_state(data)
        assert session.messages[-2].content.strip() == NEEDLE
        assert session.messages[-1].error is True
        assert SESSIONS.require("boundary").tool_error_count == 1
        assert all(event["db_hash_after"] == session.db_hash() for event in data.turn_records[0]["tool_calls"])

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
        loop.record_turn_facts = True
        data.turn_records = [{"tool_calls": []}]
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
        assert all(event["db_hash_after"] == session.db_hash() for event in data.turn_records[0]["tool_calls"])

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


@pytest.mark.parametrize("correct_text", [True, False])
def test_training_and_independent_harness_agree_on_scripted_tool_execution(live_session, monkeypatch, correct_text):
    """Run both real harnesses with scripted participants, no endpoint calls."""
    import tau2.runner.build as build
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

    import tau3_grpo.envs.agent as agent_module
    from tau3_grpo.evaluation import runtime

    handler, session = live_session
    reply = NEEDLE if correct_text else "Done."
    reservation_id = next(iter(session.db.reservations))
    calls = [ToolCall(id=str(i), requestor="assistant", name=name, arguments=args) for i, (name, args) in enumerate([
        ("cancel_reservation", {"reservation_id": reservation_id}),
        ("get_reservation_details", {"reservation_id": reservation_id}),
        ("calculate", {"expression": "1/0"}),
    ])]

    class Scripted:
        def __init__(self, messages):
            self.messages = messages

        def set_seed(self, seed):
            pass

        def get_init_state(self, **kwargs):
            return 0

        def generate_next_message(self, message, state):
            return self.messages[state].model_copy(deep=True), state + 1

        def stop(self, *args):
            pass

        @staticmethod
        def is_stop(message):
            return message.content == "###STOP###"

    monkeypatch.setattr(agent_module, "MultiCallAirlineAgent", lambda **kw: Scripted([
        AssistantMessage(role="assistant", tool_calls=calls),
        AssistantMessage(role="assistant", content=reply),
    ]))
    monkeypatch.setattr(build, "build_user", lambda *args, **kw: Scripted([
        UserMessage(role="user", content="Please help"), UserMessage(role="user", content="###STOP###"),
    ]))
    environments = []
    original_build = runtime.build_environment

    def capture(db):
        env = original_build(db)
        environments.append(env)
        return env

    monkeypatch.setattr(runtime, "build_environment", capture)
    simulation = runtime._run_one(task=session.adapted.task, db_path=session.db_path,
        policy=runtime.Endpoint("scripted", "http://not-called"),
        user=runtime.Endpoint("scripted", "http://not-called"), seed=42, max_steps=30, max_errors=10)

    async def training():
        text = "\n".join('<tool_call>' + json.dumps({"name": c.name, "arguments": c.arguments}) + '</tool_call>' for c in calls)
        loop, data, _ = make_loop(handler, text, parser="hermes")
        loop.record_turn_facts = True
        for name in ("cancel_reservation", "get_reservation_details"):
            schema = next(s for s in airline_tool_schemas() if s["function"]["name"] == name)
            loop.tools[name] = Tau3AirlineTool({}, OpenAIFunctionToolSchema.model_validate(schema))
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        await loop._handle_processing_tools_state(data)
        session.record_assistant_text(reply)
        session.record_user_message(UserMessage(role="user", content="###STOP###"))
        return await handler.finalize_rollout("boundary", termination_reason="user_stop")

    result = asyncio.run(training())
    assert result["reward"] == simulation.reward_info.reward == float(correct_text)
    assert result["termination_reason"] == simulation.termination_reason.value
    assert result["db_hash"] == environments[0].get_db_hash()
    independent_tools = [(m.content, m.error) for m in simulation.messages if m.role == "tool"]
    training_tools = [(m.content, m.error) for m in session.messages if m.role == "tool"]
    assert training_tools == independent_tools
    assert [error for _, error in training_tools] == [False, False, True]


def test_manager_preserves_group_identity_across_real_dataproto_chunks(monkeypatch):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    from verl import DataProto

    monkeypatch.setenv("TAU3_RECORD_TRAJECTORY_FACTS", "1")
    observed = []

    async def remote(chunk):
        from copy import deepcopy

        chunk = deepcopy(chunk)  # Ray transport gives each worker its own metadata.
        observed.extend(chunk.non_tensor_batch["tau3_sampling_identity"].tolist())
        chunk.meta_info["metrics"] = []
        return chunk

    manager = object.__new__(AgentLoopManager)
    manager.config = OmegaConf.create({"data": {"seed": 42}})
    manager.agent_loop_workers = [SimpleNamespace(generate_sequences=SimpleNamespace(remote=remote)) for _ in range(3)]
    manager._performance_metrics = lambda metrics, output: {}
    batch = DataProto.from_dict(tensors={"input_ids": torch.zeros(12, 1, dtype=torch.long)},
                                non_tensors={"uid": np.array(["a"] * 6 + ["b"] * 6, dtype=object)},
                                meta_info={"global_steps": 2})
    output = manager.generate_sequences(batch)
    assert len(output) == 12
    assert [row["trial"] for row in observed] == list(range(6)) * 2
    assert [row["sample_group_uid"] for row in observed] == ["a"] * 6 + ["b"] * 6
    assert all(row["global_step"] == 2 for row in observed)


def test_probability_diagnostics_do_not_enable_importance_or_rejection_sampling():
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

    from tau3_grpo.paths import CODE_ROOT
    from verl import DataProto

    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    old = torch.tensor([[-2., -1., 0.], [-3., 0., 0.]])
    batch = DataProto.from_dict(tensors={"old_log_probs": old.clone(),
                                         "rollout_log_probs": old * 2,
                                         "response_mask": mask.clone()})
    output, metrics = compute_rollout_correction_and_add_to_batch(
        batch, OmegaConf.load(CODE_ROOT / "verl/verl/trainer/config/algorithm/rollout_correction.yaml"))
    assert torch.equal(output.batch["response_mask"], mask)
    assert torch.equal(output.batch["old_log_probs"], old)
    assert "rollout_is_weights" not in output.batch
    assert metrics


@pytest.mark.parametrize("scenario", ["last_text", "ten_errors", "last_tool_batch", "user_cap", "last_user_stop", "stop_at_observation_cap"])
@pytest.mark.parametrize("protocol", ["tau3_eval_legacy_v1", "tau3_eval_train_control_v2"])
def test_versioned_native_harness_boundaries(live_session, monkeypatch, scenario, protocol):
    import tau2.agent.llm_agent as agent_module
    import tau2.user.user_simulator as user_module
    from tau2.data_model.message import AssistantMessage, ToolCall

    from tau3_grpo.evaluation.runtime import Endpoint, _run_one

    _, session = live_session
    calls = 0
    user_calls = 0
    if scenario in {"last_text", "last_tool_batch", "ten_errors"}:
        rounds = {"last_text": 14, "last_tool_batch": 15, "ten_errors": 10}[scenario]
        responses = [AssistantMessage(role="assistant", tool_calls=[ToolCall(
            id=f"call-{i}-{j}", name="calculate", requestor="assistant",
            arguments={"expression": "1/0" if scenario == "ten_errors" else "1+1"},
        ) for j in range(2 if scenario == "last_tool_batch" else 1)]) for i in range(rounds)]
        responses.append(AssistantMessage(role="assistant", content=NEEDLE))
        users = ["Please help", "###STOP###"]
    else:
        responses = [AssistantMessage(role="assistant", content=NEEDLE)] * 15
        users = ["Please help"] + ["Continue"] * 13 + [
            "###STOP###" if scenario in {"last_user_stop", "stop_at_observation_cap"} else "Continue"]
        if scenario == "stop_at_observation_cap":
            responses = [AssistantMessage(role="assistant", tool_calls=[ToolCall(
                id="initial-calc", name="calculate", requestor="assistant", arguments={"expression": "1+1"},
            )])] + responses[:14]

    def agent_generate(**kwargs):
        nonlocal calls
        message = responses[calls].model_copy(deep=True)
        calls += 1
        return message

    def user_generate(**kwargs):
        nonlocal user_calls
        message = AssistantMessage(role="assistant", content=users[user_calls])
        user_calls += 1
        return message

    monkeypatch.setattr(agent_module, "generate", agent_generate)
    monkeypatch.setattr(user_module, "generate", user_generate)
    result = _run_one(task=session.adapted.task, db_path=session.db_path,
                      policy=Endpoint("scripted", "http://never-called"),
                      user=Endpoint("scripted", "http://never-called"), seed=42,
                      max_steps=30, max_errors=10, harness_protocol=protocol)
    corrected = protocol.endswith("v2")
    expected = "user_stop" if (corrected and scenario in {"last_text", "ten_errors", "stop_at_observation_cap"}) or scenario == "last_user_stop" else (
        "too_many_errors" if scenario == "ten_errors" else "max_steps")
    assert result.termination_reason.value == expected
    assert result.reward_info.reward == float(expected == "user_stop")
    if scenario == "last_tool_batch":
        assert calls == 15
        assert len([m for m in result.messages if m.role == "tool"]) == 30
    if scenario == "user_cap":
        assert calls == 15 and user_calls == 15
    if corrected:
        assert result.info["harness_protocol"]["token_context_equivalence"] == "not_established"


def test_dispatch_failure_receipt_is_between_two_real_writes(live_session):
    """A failure receipt must describe that call, not the initial/final batch DB."""
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    handler, session = live_session
    reservations = list(session.db.reservations)[:2]
    initial_hash = session.db_hash()

    async def run():
        loop, data, _ = make_loop(handler, "")
        schema = next(s for s in airline_tool_schemas() if s["function"]["name"] == "cancel_reservation")
        loop.tools["cancel_reservation"] = Tau3AirlineTool({}, OpenAIFunctionToolSchema.model_validate(schema))
        loop.record_turn_facts = True
        data.turn_records = [{"tool_calls": []}]
        data.tool_calls = [
            FunctionCall(name="cancel_reservation", arguments=json.dumps({"reservation_id": reservations[0]})),
            FunctionCall(name="unknown_tool", arguments="{}"),
            FunctionCall(name="cancel_reservation", arguments=json.dumps({"reservation_id": reservations[1]})),
        ]
        await loop._handle_processing_tools_state(data)
        first, failed, last = data.turn_records[0]["tool_calls"]
        assert first["db_hash_after"] != initial_hash
        assert failed["error"] and failed["db_hash_after"] == first["db_hash_after"]
        assert last["db_hash_after"] != failed["db_hash_after"]
        assert last["db_hash_after"] == session.db_hash()

    asyncio.run(run())


@pytest.mark.parametrize('estimator', ['grpo', 'tau_gigpo', 'mt_gtpo'])
def test_call_attribution_preserves_native_execution_and_credit(live_session, estimator):
    """Same real output IDs, native parser, private DB, verifier and MT rewards."""
    import numpy as np
    from test_call_attribution import RealTokenizer

    from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
    from tau3_grpo.data.trajectory import trajectory_facts
    from tau3_grpo.evaluation.process_reward import reward_settings

    handler, session = live_session
    task_id = session.adapted.task_id
    reservations = list(session.db.reservations)[:2]
    assert len(reservations) == 2
    tokenizer = RealTokenizer()
    text = (NEEDLE + '\n' + ''.join(
        f'<tool_call><function={name}><parameter=reservation_id>{rid}</parameter></function></tool_call>'
        for name, rid in [('cancel_reservation', reservations[0]), ('unknown_tool', 'bad'),
                          ('cancel_reservation', reservations[1])]))

    async def run(enabled):
        if SESSIONS.get('boundary') is None:
            await handler.start_interaction('boundary', task_id=task_id, initial_user_message='Please help')
            SESSIONS.require('boundary').session.set_user_simulator(UserReply())
        initial_hash = SESSIONS.require('boundary').session.db_hash()
        loop, data, _ = make_loop(handler, text)
        loop.tokenizer = tokenizer
        loop.tool_parser = ToolParser.get_tool_parser('qwen3_coder', tokenizer)
        loop.record_call_attribution = enabled
        loop.record_turn_facts = True
        loop.record_process_turns = estimator == 'mt_gtpo'
        loop.process_reward_config = reward_settings({'mode': 'paper', 'version': 'paper_env_split_v4'})
        schema = next(s for s in airline_tool_schemas() if s['function']['name'] == 'cancel_reservation')
        loop.tools['cancel_reservation'] = Tau3AirlineTool({}, OpenAIFunctionToolSchema.model_validate(schema))
        assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
        await loop._handle_processing_tools_state(data)
        events = data.turn_records[0]['tool_calls']
        assert [e['error'] for e in events] == [False, True, False]
        kwargs = {'turn_records': data.turn_records, 'process_reward_config': loop.process_reward_config} if loop.record_process_turns else {}
        payload = await handler.finalize_rollout('boundary', termination_reason='agent_stop', **kwargs)
        facts = trajectory_facts(request_id='boundary', task_id=task_id, turns=data.turn_records,
            response_ids=data.prompt_ids[1:], response_mask=data.response_mask,
            call_attributions=data.call_attributions if enabled else None)
        if enabled:
            attr = facts['turns'][0].pop('call_attribution')
            assert attr['eligible_for_call_credit'] and len(attr['calls']) == 3
            assert [c['call_id'] for c in attr['calls']] == [e['id'] for e in events]
            first, failed, last = attr['calls']
            assert first['db_hash_before'] == initial_hash
            assert first['db_hash_after'] == failed['db_hash_before'] == failed['db_hash_after']
            assert last['db_hash_before'] == failed['db_hash_after']
            assert first['db_hash_before'] != first['db_hash_after'] != last['db_hash_after']
            for call, event in zip(attr['calls'], events, strict=True):
                assert call['error'] == event['error'] and call['db_hash_after'] == event['db_hash_after']
        evidence = {'facts': facts, 'turns': data.turn_records, 'ids': data.prompt_ids,
                    'mask': data.response_mask, 'reward': payload['reward'],
                    'db_hash': payload['db_hash'], 'process_json': payload.get('process_reward_json')}
        if loop.record_process_turns:
            process = json.loads(payload['process_reward_json'])
            # Complete two-sample synthetic group only for equality of estimator inputs/outputs.
            rewards = process['turn_rewards']
            advantage, returns, detail = compute_mt_gtpo([payload['reward'], 0], ['g', 'g'],
                [rewards, [r-0.2 for r in rewards]], [process['turn_spans']]*2,
                np.array([data.response_mask]*2))
            evidence['advantage'] = advantage.tolist()
            evidence['returns'] = returns.tolist()
            evidence['detail'] = detail
        return evidence

    baseline = asyncio.run(run(False))
    observed = asyncio.run(run(True))
    assert baseline == observed
