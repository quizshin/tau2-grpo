"""Import bridge to the vendored Sierra tau2-bench v1.0.1 (commit fc0055d).

Milestone D1: the release is the τ³-bench release but its Python package and CLI
are still named `tau2`. Nothing in this repository re-implements tau2 models; we
import the official ones so the official verifier can replay our trajectories.

Imports are lazy so the pure-function test suite (splits, anchors, Dynamic
Filtering, Tau-GiGPO) runs on a machine without tau2 installed.
"""

from __future__ import annotations

import functools
import importlib
import sys
from types import ModuleType
from typing import Any

from tau3_grpo.paths import tau2_src_root

TAU2_COMMIT = "fc0055d"
TAU2_VERSION = "1.0.1"


class Tau2Unavailable(RuntimeError):
    """Raised when tau2-bench cannot be imported."""


def ensure_tau2_importable() -> None:
    """Put the vendored `tau2-bench/src` on `sys.path` exactly once."""

    try:
        src = tau2_src_root()
    except FileNotFoundError as exc:  # pragma: no cover - layout error
        raise Tau2Unavailable(str(exc)) from exc
    text = str(src)
    if text not in sys.path:
        sys.path.insert(0, text)


def import_tau2(module: str) -> ModuleType:
    """Import a `tau2.*` module, raising `Tau2Unavailable` with context."""

    ensure_tau2_importable()
    try:
        return importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise Tau2Unavailable(f"cannot import {module}: {exc}") from exc


@functools.lru_cache(maxsize=1)
def tau2_available() -> bool:
    """Return True when `tau2` imports cleanly. Cached; safe in test collection."""

    try:
        import_tau2("tau2.data_model.tasks")
    except Tau2Unavailable:
        return False
    return True


def message_models() -> dict[str, Any]:
    """Return the official message models used to build replayable trajectories."""

    mod = import_tau2("tau2.data_model.message")
    return {
        "AssistantMessage": mod.AssistantMessage,
        "UserMessage": mod.UserMessage,
        "ToolMessage": mod.ToolMessage,
        "SystemMessage": mod.SystemMessage,
        "ToolCall": mod.ToolCall,
    }


def simulation_models() -> dict[str, Any]:
    """Return `SimulationRun`, `RewardInfo` and `TerminationReason`."""

    mod = import_tau2("tau2.data_model.simulation")
    return {
        "SimulationRun": mod.SimulationRun,
        "RewardInfo": mod.RewardInfo,
        "TerminationReason": mod.TerminationReason,
    }


def task_model() -> Any:
    """Return the official `Task` model."""

    return import_tau2("tau2.data_model.tasks").Task


def airline_domain() -> ModuleType:
    """Return `tau2.domains.airline.environment`."""

    return import_tau2("tau2.domains.airline.environment")


def flight_db_model() -> Any:
    """Return the Airline `FlightDB` model."""

    return import_tau2("tau2.domains.airline.data_model").FlightDB


def evaluator() -> ModuleType:
    """Return `tau2.evaluator.evaluator`."""

    return import_tau2("tau2.evaluator.evaluator")


def user_simulator_cls() -> Any:
    """Return the official `UserSimulator`."""

    return import_tau2("tau2.user.user_simulator").UserSimulator
