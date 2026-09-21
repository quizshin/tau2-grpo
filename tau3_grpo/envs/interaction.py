"""veRL `BaseInteraction` driving the official tau2 user simulator.

Milestone D3. `start_interaction` builds the isolated `TrajectorySession` for the
rollout, `generate_response` advances the real `tau2.user.UserSimulator` by one
turn, and `finalize_interaction` scores the trajectory with the official verifier
and releases the FlightDB.

Production code always talks to the official simulator through an
OpenAI-compatible endpoint. Only tests substitute a stub, via
`TrajectorySession.set_user_simulator`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from verl.interactions.base import BaseInteraction

from tau3_grpo.algorithms.anchors.encoder import AnchorMode
from tau3_grpo.algorithms.anchors.evidence import validate_version
from tau3_grpo.data.official import assert_trainable_source
from tau3_grpo.data.parquet_builder import INTERACTION_NAME as _PARQUET_INTERACTION_NAME
from tau3_grpo.data.schema import ArealTaskRecord, DataSource
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.envs.registry import SESSIONS, SessionEntry
from tau3_grpo.envs.session import (
    SessionFactory,
    SessionLimits,
    TrajectorySession,
    UserSimulatorConfig,
)
from tau3_grpo.evaluation.verifier import verify_trajectory
from tau3_grpo.prompts import prepare_agent_messages, prompt_provenance

logger = logging.getLogger(__name__)

INTERACTION_NAME = "tau3_airline"

# The parquet builder writes this name into interaction_kwargs without importing
# veRL, so the two literals must agree or rollouts would fail to find the handler.
assert INTERACTION_NAME == _PARQUET_INTERACTION_NAME, (
    "interaction name drifted from the parquet builder's literal"
)


class Tau3AirlineInteraction(BaseInteraction):
    """One rollout of the Airline task against a private tau2 environment."""

    required_tool_execution_mode = "sequential"
    prepare_agent_messages = staticmethod(prepare_agent_messages)

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.name = config.get("name", INTERACTION_NAME)
        self._factory: Optional[SessionFactory] = None
        self._limits = SessionLimits(
            max_user_turns=int(config.get("max_user_turns", 15)),
            max_assistant_turns=int(config.get("max_assistant_turns", 15)),
        )
        self._user_config = UserSimulatorConfig(
            model=config.get("user_model", "gpt-4o-mini"),
            base_url=config.get("user_base_url") or os.environ.get("TAU3_USER_BASE_URL"),
            api_key_env=config.get("user_api_key_env", "OPENAI_API_KEY"),
            temperature=float(config.get("user_temperature", 0.7)),
            max_tokens=config.get("user_max_tokens"),
            extra_llm_args=dict(config.get("user_llm_args", {}) or {}),
        )
        self._anchor_mode = AnchorMode(config.get("anchor_mode", AnchorMode.STRUCTURED.value))
        self._similarity_threshold = float(config.get("anchor_similarity_threshold", 0.9))
        self._strict_replay = bool(config.get("strict_replay", True))

    # ---- wiring -----------------------------------------------------

    def set_session_factory(self, factory: SessionFactory) -> None:
        """Attach the factory holding the adapted training tasks."""

        self._factory = factory

    def _require_factory(self) -> SessionFactory:
        if self._factory is None:
            raise RuntimeError(
                "Tau3AirlineInteraction has no SessionFactory; call "
                "set_session_factory() during rollout setup"
            )
        return self._factory

    def _create_session(
        self,
        task_id: str,
        interaction_kwargs: dict[str, Any],
        *,
        session_id: str,
    ) -> TrajectorySession:
        """Create a private session inside the rollout worker.

        Tests and custom launchers may inject a SessionFactory. Normal veRL
        execution reconstructs the pinned AReaL record carried by the parquet
        row, because a driver-process factory cannot be shared with Ray workers.
        """

        training_seed = os.environ.get("TAU3_GRPO_TRAIN_SEED")
        seed = int(training_seed) if training_seed is not None else interaction_kwargs.get("seed")

        if self._factory is not None:
            return self._factory.create(
                task_id,
                session_id=session_id,
                seed=seed,
            )

        raw_record = interaction_kwargs.get("record")
        if raw_record is None and interaction_kwargs.get("record_json"):
            try:
                raw_record = json.loads(interaction_kwargs["record_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("interaction record_json is invalid") from exc
        if not raw_record:
            raise RuntimeError(
                "Tau3AirlineInteraction needs either a SessionFactory or the "
                "pinned AReaL 'record_json' in interaction_kwargs"
            )
        record = ArealTaskRecord.model_validate(raw_record)
        if record.id != task_id:
            raise ValueError(
                f"interaction task_id {task_id!r} does not match record id {record.id!r}"
            )
        adapted = adapt_record(record)
        return TrajectorySession(
            adapted,
            user_config=self._user_config,
            limits=self._limits,
            session_id=session_id,
            seed=seed,
        )

    # ---- BaseInteraction -------------------------------------------

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs: Any
    ) -> str:
        request_id = instance_id or kwargs.get("request_id")
        if request_id is None:
            raise ValueError("start_interaction requires the veRL request id")
        task_id = kwargs.get("task_id")
        if task_id is None:
            raise ValueError("interaction_kwargs must carry 'task_id'")

        # A τ³ official task must never reach a rollout that feeds the optimizer.
        source = kwargs.get("source", DataSource.AREAL_TAU2_AIRLINE.value)
        assert_trainable_source(source, context="Tau3AirlineInteraction.start_interaction")

        anchor_version = validate_version(
            os.environ.get("TAU3_GRPO_ANCHOR_VERSION") or kwargs.get("anchor_version", "v1")
        )

        session = self._create_session(
            str(task_id),
            kwargs,
            session_id=str(request_id),
        )
        initial_user_message = kwargs.get("initial_user_message")
        if initial_user_message:
            session.record_initial_user_text(str(initial_user_message))
        anchor_mode = os.environ.get("TAU3_GRPO_ANCHOR_MODE") or kwargs.get(
            "anchor_mode", self._anchor_mode.value
        )
        similarity_threshold = os.environ.get(
            "TAU3_GRPO_ANCHOR_SIMILARITY_THRESHOLD"
        ) or kwargs.get("anchor_similarity_threshold", self._similarity_threshold)
        SESSIONS.register(
            str(request_id),
            SessionEntry(
                session=session,
                anchor_mode=AnchorMode(anchor_mode),
                similarity_threshold=float(similarity_threshold),
                anchor_version=anchor_version,
            ),
        )
        return str(request_id)

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[bool, str, float, dict[str, Any]]:
        entry = SESSIONS.require(str(instance_id))
        session = entry.session

        assistant_text = _last_assistant_text(messages)
        assistant_message = session.record_assistant_text(assistant_text)

        # The last allowed assistant turn can still receive the user's STOP.
        # ToolAgentLoop blocks a further assistant generation. Enforce the
        # independent user budget before calling the simulator.
        if (
            session.user_turns >= session.limits.max_user_turns
            or session.assistant_turns > session.limits.max_assistant_turns
        ):
            entry.terminated = True
            entry.termination_reason = "max_steps"
            return True, "", 0.0, {"termination_reason": "max_steps"}

        # tau2's UserSimulator uses a synchronous LiteLLM client. Running it on
        # the rollout event loop would serialize every concurrent trajectory.
        try:
            user_message = await asyncio.to_thread(session.user_respond, assistant_message)
        except Exception as exc:
            from litellm import ContextWindowExceededError

            if not getattr(self, "token_budget", None) or not isinstance(exc, ContextWindowExceededError):
                raise
            entry.terminated = True
            entry.termination_reason = "context_window_exceeded"
            return True, "", 0.0, {"termination_reason": "context_window_exceeded", "budget_role": "user"}
        content = user_message.content or ""

        if session.user_is_stop(user_message):
            entry.terminated = True
            entry.termination_reason = "user_stop"
            return True, content, 0.0, {"termination_reason": "user_stop"}

        # Turn-level score stays 0: the reward is terminal and verifiable only.
        return False, content, 0.0, {"turn": session.user_turns}

    async def calculate_score(self) -> float:
        """No intermediate reward: Tau-GiGPO derives step credit from anchors."""

        return 0.0

    def record_tool_batch(
        self, instance_id: str, *, tool_calls: list[Any],
        assistant_content: Optional[str] = None,
    ) -> list[Any]:
        """Record one assistant turn with stable IDs before executing its calls."""

        session = SESSIONS.require(str(instance_id)).session
        recorded = [session.make_tool_call(
            str(call.name), _replay_tool_arguments(call.arguments),
            f"{instance_id}-turn-{session.assistant_turns}-call-{index}",
        ) for index, call in enumerate(tool_calls)]
        if recorded:
            session.record_assistant_tool_calls(recorded, content=assistant_content)
        return recorded

    def tool_state_receipt(self, instance_id: str) -> dict[str, Any]:
        """Snapshot the live state after a recorded call, including dispatch errors."""
        return {"db_hash": SESSIONS.require(str(instance_id)).session.db_hash()}

    def record_tool_failure(
        self,
        instance_id: str,
        *,
        tool_name: str,
        raw_arguments: Any,
        error: str,
        assistant_content: Optional[str] = None,
        recorded_tool_call: Any = None,
    ) -> str:
        """Record parser/dispatch failures in the official tau2 trajectory.

        veRL can parse a function-call envelope while the JSON arguments or the
        tool name are invalid.  Those failures happen before ``BaseTool`` runs,
        so without this callback the policy sees an error observation that the
        terminal verifier never sees.  Represent malformed arguments as a
        stable dictionary, then let the live Environment create the official
        error ToolMessage.  Replaying the pair is a deterministic no-op.
        """

        entry = SESSIONS.require(str(instance_id))
        session = entry.session

        tool_call = recorded_tool_call
        if tool_call is None:
            call_id = f"{instance_id}-failed-{session.tool_calls}"
            tool_call = session.make_tool_call(
                str(tool_name), _replay_tool_arguments(raw_arguments), call_id
            )
            session.record_assistant_tool_calls([tool_call], content=assistant_content)
        tool_message = session.execute_tool_call(tool_call)
        entry.tool_error_count += 1

        official_error = tool_message.content or ""
        if not getattr(tool_message, "error", False):
            logger.warning(
                "tool failure replay unexpectedly succeeded: request=%s tool=%s error=%s",
                instance_id,
                tool_name,
                error,
            )
        return official_error or f"Error when executing tool: {error}"

    async def finalize_interaction(self, instance_id: Optional[str] = None, **kwargs: Any) -> None:
        if instance_id is None:
            return
        SESSIONS.pop(str(instance_id))

    async def finalize_rollout(
        self,
        instance_id: str,
        *,
        termination_reason: Optional[str] = None,
        duration: float = 0.0,
        anchor_ids: Optional[list[Optional[str]]] = None,
        anchor_spans: Optional[list[Optional[tuple[int, int]]]] = None,
        turn_records: Optional[list[dict[str, Any]]] = None,
        process_reward_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Score the terminal trajectory, then release its private session."""

        try:
            # Official replay is also synchronous and may touch the record DB.
            # Each rollout owns its session, so offloading preserves isolation
            # while allowing other simulator/generation coroutines to progress.
            return await asyncio.to_thread(
                self.score_trajectory,
                instance_id,
                termination_reason=termination_reason,
                duration=duration,
                anchor_ids=anchor_ids,
                anchor_spans=anchor_spans,
                turn_records=turn_records,
                process_reward_config=process_reward_config,
            )
        finally:
            SESSIONS.pop(str(instance_id))

    # ---- terminal reward -------------------------------------------

    def score_trajectory(
        self,
        instance_id: str,
        *,
        termination_reason: Optional[str] = None,
        duration: float = 0.0,
        anchor_ids: Optional[list[Optional[str]]] = None,
        anchor_spans: Optional[list[Optional[tuple[int, int]]]] = None,
        turn_records: Optional[list[dict[str, Any]]] = None,
        process_reward_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run the official verifier for a finished rollout.

        Kept synchronous and separate from `finalize_interaction` so the reward
        manager can call it and still read the anchor trail afterwards.
        """

        entry = SESSIONS.require(str(instance_id))
        reason = termination_reason or entry.termination_reason or "agent_stop"
        result = verify_trajectory(
            entry.session,
            termination_reason=reason,
            duration=duration,
            strict_replay=self._strict_replay,
            tool_error_count=entry.tool_error_count,
        )
        payload = result.to_dict()
        if not payload["execution_eligibility"]["training_candidate_eligible"]:
            raise RuntimeError(f"Unresolved execution cannot enter a training batch: {reason}")
        payload.update(prompt_provenance())
        payload["anchor_ids"] = list(anchor_ids if anchor_ids is not None else entry.anchor_ids)
        payload["anchor_spans"] = list(
            anchor_spans if anchor_spans is not None else entry.anchor_spans
        )
        if turn_records is not None:
            from tau3_grpo.evaluation.process_reward import payload_json, score_turns

            criteria = entry.session.adapted.task.evaluation_criteria
            gold = [action.model_dump(mode="json") for action in (criteria.actions or [])] if criteria else []
            basis = [getattr(b, "value", str(b)) for b in (criteria.reward_basis or [])] if criteria else []
            process = score_turns(
                turn_records, gold, basis, process_reward_config, official_outcome=result.reward
            )
            process["termination_reason"] = reason
            process["trajectory_id"] = str(instance_id)
            payload["process_reward_json"] = payload_json(process)
        return payload


def _replay_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    arguments = raw_arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"__malformed_json__": raw_arguments}
    if not isinstance(arguments, dict):
        arguments = {"__invalid_arguments__": arguments}
    return arguments


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
    return ""
