"""tau2 runtime adapters: task conversion, per-trajectory sessions, verifier."""

from tau3_grpo.envs.adapter import (
    AIRLINE_DOMAIN,
    AdaptedTask,
    adapt_record,
    airline_policy,
    build_environment,
    environment_tool_names,
    load_default_flight_db,
    load_flight_db,
)
from tau3_grpo.envs.session import (
    SessionFactory,
    SessionLimits,
    TrajectorySession,
    UserSimulatorConfig,
)
from tau3_grpo.evaluation.verifier import (
    FailureCategory,
    TerminalReward,
    classify_failure,
    verify_trajectory,
)

__all__ = [
    "AIRLINE_DOMAIN",
    "AdaptedTask",
    "FailureCategory",
    "SessionFactory",
    "SessionLimits",
    "TerminalReward",
    "TrajectorySession",
    "UserSimulatorConfig",
    "adapt_record",
    "airline_policy",
    "build_environment",
    "classify_failure",
    "environment_tool_names",
    "load_default_flight_db",
    "load_flight_db",
    "verify_trajectory",
]
