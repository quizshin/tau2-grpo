"""Experiment manifest: fixed seed and task schedule, resumable.

Milestone D8–D11. The manifest is written before training starts and pins the
exact task order for every update, so a resumed run replays the same schedule
rather than reshuffling. `resume_from` returns the schedule tail for a given
completed update count, which is what makes checkpoint resume reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from tau3_grpo.data.manifest import AREAL_REVISION
from tau3_grpo.paths import RESULTS_ROOT
from tau3_grpo.utils.hashing import canonical_json, sha256_text

MANIFEST_FILENAME = "experiment_manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class UpdateSchedule:
    """The task ids consumed by one optimizer update."""

    update_index: int
    task_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"update_index": self.update_index, "task_ids": list(self.task_ids)}


@dataclass
class ExperimentManifest:
    """Everything needed to reproduce or resume a run."""

    experiment: str
    arm: str
    seed: int
    group_size: int
    groups_per_update: int
    total_updates: int
    train_split_hash: str
    source_revision: str = AREAL_REVISION
    anchor_mode: str = "structured"
    schedule: list[UpdateSchedule] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schedule_hash: str = ""
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "experiment": self.experiment,
            "arm": self.arm,
            "seed": self.seed,
            "group_size": self.group_size,
            "groups_per_update": self.groups_per_update,
            "total_updates": self.total_updates,
            "train_split_hash": self.train_split_hash,
            "source_revision": self.source_revision,
            "anchor_mode": self.anchor_mode,
            "created_at": self.created_at,
            "schedule_hash": self.schedule_hash,
            "notes": self.notes,
            "schedule": [item.to_dict() for item in self.schedule],
        }


def build_schedule(
    task_ids: Sequence[str],
    *,
    seed: int,
    group_size: int,
    groups_per_update: int,
    total_updates: int,
) -> list[UpdateSchedule]:
    """Deterministic task schedule.

    Ordering is a stable hash rank of ``seed:task_id`` rather than an RNG shuffle,
    matching how the data splits are frozen, so the schedule is identical across
    Python and numpy versions. Tasks cycle when the pool is exhausted.
    """

    if group_size <= 0 or groups_per_update <= 0 or total_updates <= 0:
        raise ValueError("group_size, groups_per_update and total_updates must be positive")
    pool = sorted(task_ids, key=lambda task_id: sha256_text(f"{seed}:{task_id}"))
    if not pool:
        raise ValueError("cannot build a schedule from an empty task pool")

    schedule: list[UpdateSchedule] = []
    cursor = 0
    for update_index in range(total_updates):
        chosen: list[str] = []
        for _ in range(groups_per_update):
            chosen.append(pool[cursor % len(pool)])
            cursor += 1
        schedule.append(UpdateSchedule(update_index=update_index, task_ids=tuple(chosen)))
    return schedule


def flatten_schedule(schedule: Sequence[UpdateSchedule]) -> list[str]:
    """Return the exact pre-repeat task order consumed by veRL's dataloader."""

    return [task_id for update in schedule for task_id in update.task_ids]


def create_manifest(
    *,
    experiment: str,
    arm: str,
    seed: int,
    task_ids: Sequence[str],
    train_split_hash: str,
    group_size: int = 8,
    groups_per_update: int = 16,
    total_updates: int = 40,
    anchor_mode: str = "structured",
    notes: Optional[str] = None,
) -> ExperimentManifest:
    """Build a manifest with its schedule and schedule hash."""

    schedule = build_schedule(
        task_ids,
        seed=seed,
        group_size=group_size,
        groups_per_update=groups_per_update,
        total_updates=total_updates,
    )
    manifest = ExperimentManifest(
        experiment=experiment,
        arm=arm,
        seed=seed,
        group_size=group_size,
        groups_per_update=groups_per_update,
        total_updates=total_updates,
        train_split_hash=train_split_hash,
        anchor_mode=anchor_mode,
        schedule=schedule,
        notes=notes,
    )
    manifest.schedule_hash = compute_schedule_hash(manifest)
    return manifest


def compute_schedule_hash(manifest: ExperimentManifest) -> str:
    """Hash every field that determines the dataloader's task sequence."""

    return sha256_text(
        canonical_json(
            {
                "seed": manifest.seed,
                "arm": manifest.arm,
                "group_size": manifest.group_size,
                "groups_per_update": manifest.groups_per_update,
                "train_split_hash": manifest.train_split_hash,
                "schedule": [item.to_dict() for item in manifest.schedule],
            }
        )
    )


def write_manifest(
    manifest: ExperimentManifest, directory: str | Path = RESULTS_ROOT
) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_manifest(directory: str | Path = RESULTS_ROOT) -> ExperimentManifest:
    path = Path(directory) / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no experiment manifest at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported experiment manifest version: {data.get('version')}")
    manifest = ExperimentManifest(
        experiment=data["experiment"],
        arm=data["arm"],
        seed=int(data["seed"]),
        group_size=int(data["group_size"]),
        groups_per_update=int(data["groups_per_update"]),
        total_updates=int(data["total_updates"]),
        train_split_hash=data["train_split_hash"],
        source_revision=data.get("source_revision", AREAL_REVISION),
        anchor_mode=data.get("anchor_mode", "structured"),
        schedule=[
            UpdateSchedule(
                update_index=int(item["update_index"]), task_ids=tuple(item["task_ids"])
            )
            for item in data.get("schedule", [])
        ],
        created_at=data.get("created_at", ""),
        schedule_hash=data.get("schedule_hash", ""),
        notes=data.get("notes"),
    )
    actual_hash = compute_schedule_hash(manifest)
    if manifest.schedule_hash != actual_hash:
        raise ValueError(
            "experiment manifest schedule_hash does not match its schedule content"
        )
    return manifest


def resume_from(manifest: ExperimentManifest, completed_updates: int) -> list[UpdateSchedule]:
    """Return the schedule tail after `completed_updates` finished updates."""

    if completed_updates < 0:
        raise ValueError("completed_updates cannot be negative")
    if completed_updates > len(manifest.schedule):
        raise ValueError(
            f"completed_updates={completed_updates} exceeds scheduled "
            f"{len(manifest.schedule)} updates"
        )
    return manifest.schedule[completed_updates:]


def verify_resume(
    manifest: ExperimentManifest, other: ExperimentManifest
) -> None:
    """Confirm a resumed run uses the same schedule as the original."""

    if manifest.schedule_hash != other.schedule_hash:
        raise ValueError(
            "experiment schedule hash mismatch: the resumed run would not replay the "
            "original task order"
        )
    if manifest.train_split_hash != other.train_split_hash:
        raise ValueError("train split hash mismatch between original and resumed run")


def verify_extension(
    original: ExperimentManifest, extended: ExperimentManifest
) -> None:
    """Allow 40→60 update continuation only when the old schedule is a prefix."""

    invariant_fields = (
        "experiment",
        "arm",
        "seed",
        "group_size",
        "groups_per_update",
        "train_split_hash",
        "source_revision",
        "anchor_mode",
    )
    for field_name in invariant_fields:
        if getattr(original, field_name) != getattr(extended, field_name):
            raise ValueError(f"experiment extension changes {field_name}")
    if original.total_updates >= extended.total_updates:
        raise ValueError(
            "experiment extension must increase total_updates "
            f"({original.total_updates} -> {extended.total_updates})"
        )
    if original.schedule != extended.schedule[: len(original.schedule)]:
        raise ValueError("extended experiment does not preserve the original schedule prefix")
