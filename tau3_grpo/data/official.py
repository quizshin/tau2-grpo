"""Official τ³ Airline final loader and training-path source isolation.

Milestone D13: the official final set is the tau2-bench v1.0.1 Airline `base`
split (50 tasks). It may only be run after the winner lock is frozen. Any
training or model-selection entrypoint must call `assert_trainable_source` so a
τ³ official task can never reach an optimizer or a selection decision.
"""

from __future__ import annotations

from typing import Any

from tau3_grpo.data.manifest import TAU3_REVISION, ManifestEntry, official_tau3_manifest
from tau3_grpo.data.schema import DataSource
from tau3_grpo.envs.tau2_bridge import airline_domain

OFFICIAL_AIRLINE_SPLIT = "base"
OFFICIAL_AIRLINE_TASK_COUNT = 50

#: Sources that an optimizer or an internal-selection decision may consume.
TRAINABLE_SOURCES = frozenset({DataSource.AREAL_TAU2_AIRLINE})

#: Sources reserved for the frozen-winner final report only.
FINAL_ONLY_SOURCES = frozenset({DataSource.TAU3_OFFICIAL_AIRLINE})


class SourceIsolationError(RuntimeError):
    """Raised when τ³ official data reaches a training or selection path."""


def load_official_airline_tasks(split: str = OFFICIAL_AIRLINE_SPLIT) -> list[Any]:
    """Load the official τ³ Airline tasks straight from tau2-bench.

    No conversion happens here: these are Sierra's own `Task` objects.
    """

    tasks = airline_domain().get_tasks(task_split_name=split)
    if split == OFFICIAL_AIRLINE_SPLIT and len(tasks) != OFFICIAL_AIRLINE_TASK_COUNT:
        raise ValueError(
            f"official τ³ Airline '{split}' split must contain "
            f"{OFFICIAL_AIRLINE_TASK_COUNT} tasks, found {len(tasks)}"
        )
    return tasks


def official_airline_task_ids(split: str = OFFICIAL_AIRLINE_SPLIT) -> list[str]:
    """Return the official task IDs in the order tau2-bench reports them."""

    return [task.id for task in load_official_airline_tasks(split)]


def build_official_manifest(split: str = OFFICIAL_AIRLINE_SPLIT) -> list[ManifestEntry]:
    """Build the τ³ final manifest, enforcing the 50-unique-task invariant."""

    return official_tau3_manifest(official_airline_task_ids(split))


def assert_trainable_source(source: DataSource | str, *, context: str) -> None:
    """Reject τ³ official data on any training or model-selection path.

    Args:
        source: the data source under consideration.
        context: human-readable caller name used in the error message.
    """

    value = DataSource(source) if not isinstance(source, DataSource) else source
    if value in FINAL_ONLY_SOURCES:
        raise SourceIsolationError(
            f"{context} refuses source '{value.value}': τ³ official final data is "
            "reserved for the frozen-winner report and must never be used for "
            "training or internal selection"
        )
    if value not in TRAINABLE_SOURCES:
        raise SourceIsolationError(f"{context} received unknown source '{value.value}'")


def assert_trainable_entries(entries: list[ManifestEntry], *, context: str) -> None:
    """Apply `assert_trainable_source` to every manifest entry."""

    for entry in entries:
        assert_trainable_source(entry.source, context=context)
        if entry.split == "tau3_final":
            raise SourceIsolationError(
                f"{context} refuses split 'tau3_final' (task {entry.task_id})"
            )


def official_revision() -> str:
    """Return the pinned tau2-bench revision backing the official final set."""

    return TAU3_REVISION
