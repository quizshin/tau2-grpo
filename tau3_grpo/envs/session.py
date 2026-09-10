"""Per-trajectory session: one Environment, one FlightDB, one user state.

Milestone D2–D3: every rollout gets its own `TrajectorySession`. The session owns
the loaded FlightDB, the official Airline `Environment`, the tau2
`UserSimulator` state and the recorded message list. Nothing is shared between
sessions, so `group_size=8` rollouts of the same task cannot see each other's DB
writes.

The recorded messages are official tau2 message objects, so the official
verifier can replay them without any translation layer.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tau3_grpo.envs.adapter import AdaptedTask, build_environment, load_flight_db
from tau3_grpo.envs.tau2_bridge import message_models, user_simulator_cls


@dataclass
class UserSimulatorConfig:
    """OpenAI-compatible endpoint configuration for the tau2 user simulator."""

    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    extra_llm_args: dict[str, Any] = field(default_factory=dict)

    def to_llm_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            args["max_tokens"] = self.max_tokens
        if self.base_url:
            # tau2 forwards these kwargs to LiteLLM. `api_base` is the
            # OpenAI-compatible endpoint parameter understood by LiteLLM.
            args["api_base"] = self.base_url
            args["api_key"] = os.environ.get(self.api_key_env, "EMPTY")
        args.update(self.extra_llm_args)
        return args

    def resolved_model(self) -> str:
        """Return a LiteLLM model name for an OpenAI-compatible local server."""

        if self.base_url and not self.model.startswith(("openai/", "hosted_vllm/")):
            return f"openai/{self.model}"
        return self.model


@dataclass
class SessionLimits:
    """Turn caps from the v2-2 configuration."""

    max_user_turns: int = 15
    max_assistant_turns: int = 15


class TrajectorySession:
    """Isolated tau2 runtime for a single rollout."""

    def __init__(
        self,
        adapted: AdaptedTask,
        *,
        user_config: UserSimulatorConfig,
        limits: Optional[SessionLimits] = None,
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.adapted = adapted
        self.user_config = user_config
        self.limits = limits or SessionLimits()
        self.session_id = session_id or uuid.uuid4().hex
        self.seed = seed

        # Isolation: a fresh DB instance and a fresh Environment per session.
        self.db = load_flight_db(adapted.db_path)
        self.environment = build_environment(self.db)

        self._messages: list[Any] = []
        self._models = message_models()
        self._lock = threading.Lock()
        self.assistant_turns = 0
        self.user_turns = 0
        self.tool_calls = 0
        self.initial_db_hash = self.environment.get_db_hash()

        self._user = user_simulator_cls()(
            llm=user_config.resolved_model(),
            instructions=adapted.user_instructions,
            llm_args=user_config.to_llm_args(),
        )
        if seed is not None:
            setter = getattr(self._user, "set_seed", None)
            if callable(setter):
                setter(seed)
        self._user_state = self._user.get_init_state()

    # ---- properties -------------------------------------------------

    @property
    def task_id(self) -> str:
        return self.adapted.task_id

    @property
    def db_path(self) -> Path:
        return self.adapted.db_path

    @property
    def messages(self) -> list[Any]:
        """Official tau2 messages recorded so far, in order."""

        return list(self._messages)

    def db_hash(self) -> Optional[str]:
        """Canonical hash of the mutable DB, used by the anchor encoder."""

        return self.environment.get_db_hash()

    def policy(self) -> str:
        return self.environment.get_policy()

    # ---- recording --------------------------------------------------

    def record_initial_user_text(self, content: str) -> Any:
        """Seed both replay history and the simulator with the prompt's user turn.

        veRL builds the initial policy prompt before ``BaseInteraction`` starts.
        Without mirroring that user turn here, the policy and tau2 simulator see
        different conversations and the first structured anchor incorrectly has
        ``last_observation=none``.
        """

        if not content or not content.strip():
            raise ValueError("initial user message cannot be empty")
        with self._lock:
            if self._messages:
                raise RuntimeError("initial user message must be the first session message")
            message = self._models["UserMessage"](role="user", content=content)
            self._messages.append(message)
            self.user_turns = 1
            # UserSimulator.get_init_state accepts the already-observed dialogue
            # history.  The next call can therefore respond to the policy's first
            # assistant turn without forgetting what the user initially said.
            self._user_state = self._user.get_init_state(message_history=[message])
        return message

    def record_assistant_text(self, content: str) -> Any:
        message = self._models["AssistantMessage"](role="assistant", content=content)
        with self._lock:
            self._messages.append(message)
            self.assistant_turns += 1
        return message

    def record_assistant_tool_calls(self, tool_calls: list[Any], content: str | None = None) -> Any:
        message = self._models["AssistantMessage"](
            role="assistant", content=content, tool_calls=list(tool_calls)
        )
        with self._lock:
            self._messages.append(message)
            self.assistant_turns += 1
        return message

    def make_tool_call(self, name: str, arguments: dict[str, Any], call_id: str) -> Any:
        return self._models["ToolCall"](
            id=call_id, name=name, arguments=arguments, requestor="assistant"
        )

    def execute_tool_call(self, tool_call: Any) -> Any:
        """Run a tool against *this* session's Environment and record both sides.

        `Environment.get_response` never raises: a failing tool returns a
        `ToolMessage(error=True)`, matching official behaviour and keeping the
        trajectory replayable.
        """

        tool_message = self.environment.get_response(tool_call)
        with self._lock:
            self._messages.append(tool_message)
            self.tool_calls += 1
        return tool_message

    def record_user_message(self, message: Any) -> None:
        with self._lock:
            self._messages.append(message)
            self.user_turns += 1

    # ---- user simulator --------------------------------------------

    def user_respond(self, message: Any) -> Any:
        """Advance the real tau2 user simulator by one turn.

        Production code always uses the official simulator; only tests replace
        it via `set_user_simulator`.
        """

        user_message, self._user_state = self._user.generate_next_message(
            message, self._user_state
        )
        self.record_user_message(user_message)
        return user_message

    def user_is_stop(self, message: Any) -> bool:
        return type(self._user).is_stop(message)

    def set_user_simulator(self, simulator: Any, state: Any = None) -> None:
        """Test seam: inject a stub simulator to avoid paid LLM calls."""

        self._user = simulator
        self._user_state = state if state is not None else simulator.get_init_state()

    def set_seed(self, seed: int) -> None:
        """Update the simulator sampling seed for an independent training run."""

        self.seed = int(seed)
        setter = getattr(self._user, "set_seed", None)
        if callable(setter):
            setter(self.seed)

    # ---- limits -----------------------------------------------------

    def turn_limit_reached(self) -> bool:
        return (
            self.assistant_turns >= self.limits.max_assistant_turns
            or self.user_turns >= self.limits.max_user_turns
        )

    def metadata(self) -> dict[str, Any]:
        """Trajectory metadata carried into telemetry and reports."""

        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "db_path": str(self.db_path),
            "db_file_hash": self.adapted.db_file_hash,
            "initial_db_hash": self.initial_db_hash,
            "final_db_hash": self.db_hash(),
            "assistant_turns": self.assistant_turns,
            "user_turns": self.user_turns,
            "tool_calls": self.tool_calls,
            "seed": self.seed,
            "user_model": self.user_config.model,
        }


class SessionFactory:
    """Builds one isolated session per rollout from adapted tasks."""

    def __init__(
        self,
        adapted_tasks: dict[str, AdaptedTask],
        *,
        user_config: UserSimulatorConfig,
        limits: Optional[SessionLimits] = None,
    ):
        self._tasks = dict(adapted_tasks)
        self.user_config = user_config
        self.limits = limits or SessionLimits()

    def task_ids(self) -> list[str]:
        return sorted(self._tasks)

    def create(
        self,
        task_id: str,
        *,
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> TrajectorySession:
        if task_id not in self._tasks:
            raise KeyError(f"unknown task_id: {task_id}")
        return TrajectorySession(
            self._tasks[task_id],
            user_config=self.user_config,
            limits=self.limits,
            session_id=session_id,
            seed=seed,
        )
