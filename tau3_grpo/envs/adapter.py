"""AReaL task + record-specific FlightDB -> tau2 v1.0.1 Task/Environment.

Milestone D1–D2: each AReaL record carries its own `db_path`, so the adapter
loads that record's FlightDB and injects it into
`tau2.domains.airline.environment.get_environment(db=...)`. Sharing one DB
instance across trajectories would leak mutations between rollouts, so the
adapter never caches a loaded DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.tau2_bridge import airline_domain, flight_db_model, task_model
from tau3_grpo.paths import AREAL_DB_ROOT
from tau3_grpo.utils.hashing import sha256_file

AIRLINE_DOMAIN = "airline"


@dataclass(frozen=True)
class AdaptedTask:
    """An AReaL record materialized as an official tau2 `Task`."""

    task_id: str
    task: Any
    db_path: Path
    db_file_hash: str
    user_instructions: str
    known_info: str | None

    @property
    def domain(self) -> str:
        return AIRLINE_DOMAIN


def adapt_record(
    record: ArealTaskRecord,
    *,
    dataset_root: str | Path = AREAL_DB_ROOT,
) -> AdaptedTask:
    """Convert one AReaL record into a tau2 `Task` with its own DB path."""

    task_cls = task_model()
    db_path = record.resolve_db_path(dataset_root)
    if not db_path.is_file():
        raise FileNotFoundError(f"missing FlightDB for task {record.id}: {db_path}")
    task = task_cls.model_validate(record.to_tau2_task_dict())
    instructions = record.user_scenario.get("instructions", {})
    if isinstance(instructions, dict):
        known = instructions.get("known_info")
    else:  # pragma: no cover - upstream always uses a dict
        known = None
    return AdaptedTask(
        task_id=record.id,
        task=task,
        db_path=db_path,
        db_file_hash=sha256_file(db_path),
        # Use the native scenario renderer: task_instructions alone omits user
        # IDs, reservation details, persona, and known/unknown information.
        user_instructions=str(task.user_scenario),
        known_info=str(known) if known is not None else None,
    )


def load_flight_db(db_path: str | Path) -> Any:
    """Load a fresh `FlightDB` instance from disk.

    A new instance per call is the isolation guarantee: two trajectories on the
    same task never observe each other's writes.
    """

    return flight_db_model().load(str(db_path))


def build_environment(db: Any) -> Any:
    """Build the official Airline `Environment` around an injected DB.

    Airline does not support solo mode, so `solo_mode` stays False.
    """

    return airline_domain().get_environment(db=db, solo_mode=False)


def airline_policy() -> str:
    """Return the official Airline policy text for the agent system prompt."""

    return build_environment(load_default_flight_db()).get_policy()


def load_default_flight_db() -> Any:
    """Load the stock Airline DB, used only to read the policy text."""

    from tau3_grpo.envs.tau2_bridge import import_tau2

    utils = import_tau2("tau2.domains.airline.utils")
    return flight_db_model().load(utils.AIRLINE_DB_PATH)


def environment_tool_names(environment: Any) -> list[str]:
    """Return the assistant tool names the environment exposes.

    Used to check `configs/envs/tool_config.yaml` against the live tau2
    signatures. The OpenAI-shaped schemas come from
    `tau3_grpo.envs.tools.airline_tool_schemas`.
    """

    return sorted(tool.name for tool in environment.get_tools())


def airline_tool_schemas() -> list[dict[str, Any]]:
    """Return the exact OpenAI schemas exposed by tau2-bench v1.0.1.

    Keeping this helper outside the veRL integration package lets data/config
    preparation run on a CPU machine without importing Torch or veRL.
    """

    environment = build_environment(load_default_flight_db())
    return [tool.openai_schema for tool in environment.get_tools()]
