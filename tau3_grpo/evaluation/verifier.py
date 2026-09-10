"""Official verifier bridge: terminal verifiable reward from tau2.

Milestone D2: the terminal reward is whatever
`tau2.evaluator.evaluator.evaluate_simulation(..., EvaluationType.ALL)` returns
for a `SimulationRun` built from the session's recorded messages. We do not
reimplement scoring.

Two upstream semantics drive everything downstream:

1. The ALL reward is the **product** of the components named in the task's
   `reward_basis`. Any zero component zeroes the trajectory.
2. If `termination_reason` is not AGENT_STOP or USER_STOP, upstream returns
   reward 0.0 **without evaluating**. So an unfinished rollout is indistinguishable
   from a wrong one by reward alone, and telemetry has to keep the category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from tau3_grpo.envs.adapter import AIRLINE_DOMAIN, load_flight_db
from tau3_grpo.envs.tau2_bridge import evaluator, simulation_models

#: Terminations upstream is willing to score.
SCORABLE_TERMINATIONS = frozenset({"agent_stop", "user_stop"})


class FailureCategory(str, Enum):
    """Why a trajectory earned no reward."""

    NONE = "none"
    WRONG_OUTCOME = "wrong_outcome"
    TURN_LIMIT = "turn_limit"
    USER_TRANSFER = "user_transfer"
    TOOL_ERRORS = "tool_errors"
    AGENT_ERROR = "agent_error"
    INFRASTRUCTURE = "infrastructure"
    UNFINISHED = "unfinished"


@dataclass
class TerminalReward:
    """Terminal reward plus everything telemetry and reports need."""

    task_id: str
    reward: float
    termination_reason: str
    failure_category: FailureCategory
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    reward_basis: list[str] = field(default_factory=list)
    db_hash: Optional[str] = None
    initial_db_hash: Optional[str] = None
    scored: bool = True
    info: dict[str, Any] = field(default_factory=dict)
    trajectory: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.reward > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reward": self.reward,
            "termination_reason": self.termination_reason,
            "failure_category": self.failure_category.value,
            "reward_breakdown": dict(self.reward_breakdown),
            "reward_basis": list(self.reward_basis),
            "db_hash": self.db_hash,
            "initial_db_hash": self.initial_db_hash,
            "scored": self.scored,
            "info": self.info,
            "trajectory": self.trajectory,
        }


def classify_failure(
    reward: float,
    termination_reason: str,
    *,
    tool_error_count: int = 0,
) -> FailureCategory:
    """Map reward and termination onto an actionable failure category.

    Kept separate from the reward so telemetry can distinguish "the agent
    finished and got it wrong" from "the agent never finished", which the
    upstream reward of 0.0 conflates.
    """

    reason = str(termination_reason)
    if reward > 0.0:
        return FailureCategory.NONE
    if reason in ("max_steps", "timeout"):
        return FailureCategory.TURN_LIMIT
    if reason == "too_many_errors":
        return FailureCategory.TOOL_ERRORS
    if reason in ("agent_error", "context_window_exceeded"):
        return FailureCategory.AGENT_ERROR
    if reason in ("infrastructure_error", "unexpected_error"):
        return FailureCategory.INFRASTRUCTURE
    if reason == "user_error":
        return FailureCategory.USER_TRANSFER
    if reason not in SCORABLE_TERMINATIONS:
        return FailureCategory.UNFINISHED
    if tool_error_count > 0:
        return FailureCategory.TOOL_ERRORS
    return FailureCategory.WRONG_OUTCOME


def build_simulation_run(
    *,
    task_id: str,
    messages: list[Any],
    termination_reason: Any,
    session_id: str,
    duration: float = 0.0,
    seed: Optional[int] = None,
    trial: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Any:
    """Assemble the official `SimulationRun` the verifier consumes."""

    models = simulation_models()
    now = datetime.now(timezone.utc).isoformat()
    return models["SimulationRun"](
        id=session_id,
        task_id=task_id,
        start_time=start_time or now,
        end_time=end_time or now,
        duration=duration,
        termination_reason=termination_reason,
        messages=list(messages),
        seed=seed,
        trial=trial,
    )


def _termination_value(reason: Any) -> str:
    return getattr(reason, "value", str(reason))


def verify_trajectory(
    session: Any,
    *,
    termination_reason: Any,
    duration: float = 0.0,
    trial: Optional[int] = None,
    strict_replay: bool = True,
    tool_error_count: int = 0,
) -> TerminalReward:
    """Score one finished session with the official evaluator.

    Args:
        session: a `TrajectorySession` whose messages are official tau2 objects.
        termination_reason: a `tau2` `TerminationReason` (or its string value).
        strict_replay: True for live scoring. Pass False only when re-grading
            stored trajectories whose recorded tool output predates current tool
            code.
    """

    ev = evaluator()
    simulation = build_simulation_run(
        task_id=session.task_id,
        messages=session.messages,
        termination_reason=termination_reason,
        session_id=session.session_id,
        duration=duration,
        seed=session.seed,
        trial=trial,
    )
    reward_info = ev.evaluate_simulation(
        simulation=simulation,
        task=session.adapted.task,
        evaluation_type=ev.EvaluationType.ALL,
        solo_mode=False,
        domain=AIRLINE_DOMAIN,
        # AReaL tasks carry record-specific initial databases. Replay against a
        # fresh copy of that exact DB; falling back to tau2's stock Airline DB
        # would silently assign the wrong terminal reward.
        env_kwargs={"db": load_flight_db(session.adapted.db_path)},
        strict_replay=strict_replay,
    )
    reason = _termination_value(termination_reason)
    reward = float(reward_info.reward)
    breakdown = {
        getattr(key, "value", str(key)): float(value)
        for key, value in (reward_info.reward_breakdown or {}).items()
    }
    basis = [getattr(item, "value", str(item)) for item in (reward_info.reward_basis or [])]
    return TerminalReward(
        task_id=session.task_id,
        reward=reward,
        termination_reason=reason,
        failure_category=classify_failure(
            reward, reason, tool_error_count=tool_error_count
        ),
        reward_breakdown=breakdown,
        reward_basis=basis,
        db_hash=session.db_hash(),
        initial_db_hash=session.initial_db_hash,
        scored=reason in SCORABLE_TERMINATIONS,
        info=dict(reward_info.info or {}),
        trajectory=session.metadata(),
    )
