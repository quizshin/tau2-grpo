"""Independent evaluation adapter; official tool execution and scoring are unchanged."""
import traceback

from litellm.exceptions import ContextWindowExceededError
from tau2.data_model.simulation import TerminationReason
from tau2.orchestrator.orchestrator import Orchestrator, Role

from tau3_grpo.evaluation.harness import validate_opening


class EvaluationOrchestrator(Orchestrator):
    """Keep native behavior, finalizing typed generation context limits with evidence."""

    def step(self):
        recipient = self.to_role
        try:
            return super().step()
        except ContextWindowExceededError as error:
            if recipient not in (Role.AGENT, Role.USER):
                raise
            self.done = True
            self.termination_reason = TerminationReason.CONTEXT_WINDOW_EXCEEDED
            self.context_limit_receipt = {
                "role": recipient.value,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "retry_count": 0,
            }

    def _check_termination(self):
        # A completed STOP remains eligible for official scoring even when a
        # step/error/time budget is reached in the same native step.
        if self.done and self.termination_reason in (
            TerminationReason.USER_STOP, TerminationReason.AGENT_STOP,
        ):
            return
        if hasattr(self, "context_limit_receipt"):
            return
        super()._check_termination()

    def _finalize(self):
        result = super()._finalize()
        if hasattr(self, "context_limit_receipt"):
            result.info = {**(result.info or {}), "context_limit": self.context_limit_receipt}
        return result

    def failure_snapshot(self):
        """Capture available state without scoring or inventing a completed simulation."""
        evidence = {"scope": "observed_state_at_exception", "automatic_retries": 0}
        readers = {
            "messages": lambda: [m.model_dump(mode="json") for m in self.get_trajectory()],
            "pending_message": lambda: self.message.model_dump(mode="json"),
            "agent_state": lambda: self.agent_state.model_dump(mode="json"),
            "user_state": lambda: self.user_state.model_dump(mode="json"),
            "task": lambda: self.task.model_dump(mode="json"),
            "policy": self.environment.get_policy,
            "database": lambda: self.environment.tools.db.model_dump(mode="json"),
            "database_hash": self.environment.get_db_hash,
            "from_role": lambda: self.from_role.value,
            "to_role": lambda: self.to_role.value,
            "step_count": lambda: self.step_count,
        }
        for name, read in readers.items():
            try:
                evidence[name] = read()
            except Exception as error:
                # A missing/unserializable component must not erase the original failure.
                evidence.setdefault("capture_errors", {})[name] = {
                    "type": type(error).__name__, "message": str(error),
                }
        return evidence


class TrainingControlOrchestrator(EvaluationOrchestrator):
    """Airline fresh-session adapter for the fixed training control v2 profile."""

    def initialize(self):
        # Counters describe generated messages, excluding the stock greeting.
        if self.task.initial_state and self.task.initial_state.message_history:
            raise ValueError("train_control_v2 requires a fresh message history")
        super().initialize()
        if self.solo_mode:
            raise ValueError("train_control_v2 requires a user simulator")
        self.assistant_generations = 0
        self.real_user_messages = 0
        self.observation_batches = 0

    def step(self):
        recipient = self.to_role
        super().step()
        if recipient == Role.AGENT:
            self.assistant_generations += 1
        elif recipient == Role.USER:
            if self.real_user_messages:
                self.observation_batches += 1
            self.real_user_messages += 1
        elif recipient == Role.ENV and self.to_role == Role.AGENT:
            self.observation_batches += 1

    def _check_termination(self):
        if self.done or self.to_role == Role.ENV:
            return
        # Training permits a reply to the final assistant text, but blocks the
        # next policy generation. Tool batches always complete before checking.
        blocked = (
            self.to_role == Role.AGENT
            and (self.assistant_generations >= 15 or self.observation_batches >= 15)
        ) or (
            self.to_role == Role.USER and self.real_user_messages >= 15
        )
        if blocked:
            self.done = True
            self.termination_reason = TerminationReason.MAX_STEPS
            return
        self._check_timeout()


class TrainingInputOrchestrator(TrainingControlOrchestrator):
    """v3 starts both histories at the training row's pinned user opening."""

    def __init__(self, *args, initial_user_message, **kwargs):
        super().__init__(*args, **kwargs)
        validate_opening(self.task, initial_user_message)
        self.initial_user_message = initial_user_message

    def initialize(self):
        from tau2.data_model.message import UserMessage

        super().initialize()  # Keeps native DB/action initialization and seeding.
        # Native fresh initialization does not generate a user reply. Replace its
        # stock greeting before the first step, without mutating the pinned task.
        opening = UserMessage(role="user", content=self.initial_user_message)
        self.agent_state = self.agent.get_init_state()
        self.user_state = self.user.get_init_state(message_history=[opening.model_copy(deep=True)])
        self.trajectory = [opening]
        self.message = opening
        self.from_role = Role.USER
        self.to_role = Role.AGENT
        self.real_user_messages = 1
