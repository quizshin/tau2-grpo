"""Deterministic train/selection/final manifests.

Milestone D2: freeze exactly 200 train and 60 internal-selection tasks from the
real 1,148-record Airline pool. Official τ³ tasks use a separate constructor.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from tau3_grpo.data.schema import ArealTaskRecord, DataSource
from tau3_grpo.utils.hashing import canonical_json, sha256_file, sha256_json, sha256_text

AREAL_REVISION = "86971dc03da6e7c1a7933295e05b84aab8215386"
TAU3_REVISION = "fc0055dc4e0a316c3f83133267fbd6faaa770992"


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    split: Literal["train", "selection", "reserve", "tau3_final"]
    source: DataSource
    source_revision: str
    task_hash: str
    db_path: str | None = None
    db_hash: str | None = None
    task: dict[str, Any] | None = None


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    source_revision: str
    source_file_hash: str
    train: list[ManifestEntry]
    selection: list[ManifestEntry]
    reserve: list[ManifestEntry]
    split_hash: str


def load_areal_records(path: str | Path) -> list[ArealTaskRecord]:
    records: list[ArealTaskRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ArealTaskRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid AReaL record on line {line_number}") from exc
    return records


def _entry(
    record: ArealTaskRecord,
    split: Literal["train", "selection", "reserve"],
    dataset_root: Path | None,
) -> ManifestEntry:
    db_hash = None
    if dataset_root is not None:
        db_file = record.resolve_db_path(dataset_root)
        if not db_file.is_file():
            raise FileNotFoundError(f"missing DB for {record.id}: {db_file}")
        db_hash = sha256_file(db_file)
    return ManifestEntry(
        task_id=record.id,
        split=split,
        source=DataSource.AREAL_TAU2_AIRLINE,
        source_revision=AREAL_REVISION,
        task_hash=record.fingerprint,
        db_path=record.db_path,
        db_hash=db_hash,
        task=record.model_dump(mode="json"),
    )


def build_airline_splits(
    records: Iterable[ArealTaskRecord],
    *,
    seed: int,
    train_size: int = 200,
    selection_size: int = 60,
    expected_airline_count: int = 1148,
    dataset_root: str | Path | None = None,
    source_file_hash: str = "unknown",
) -> SplitManifest:
    records = list(records)
    airline = [record for record in records if record.domain == "airline"]
    ids = [record.id for record in airline]
    if len(ids) != len(set(ids)):
        raise ValueError("AReaL Airline task IDs are not unique")
    if len(airline) != expected_airline_count:
        raise ValueError(
            f"expected {expected_airline_count} Airline records, found {len(airline)}"
        )
    if train_size + selection_size > len(airline):
        raise ValueError("requested split exceeds Airline pool")

    # Hash ranking is stable across Python/numpy versions, unlike RNG shuffling.
    ordered = sorted(
        airline,
        key=lambda record: sha256_text(f"{seed}:{record.id}:{record.fingerprint}"),
    )
    train_records = ordered[:train_size]
    selection_records = ordered[train_size : train_size + selection_size]
    reserve_records = ordered[train_size + selection_size :]
    root = Path(dataset_root) if dataset_root is not None else None

    train = [_entry(record, "train", root) for record in train_records]
    selection = [_entry(record, "selection", root) for record in selection_records]
    reserve = [_entry(record, "reserve", root) for record in reserve_records]
    split_payload = {
        "seed": seed,
        "source_revision": AREAL_REVISION,
        "source_file_hash": source_file_hash,
        "train": [entry.task_id for entry in train],
        "selection": [entry.task_id for entry in selection],
        "reserve": [entry.task_id for entry in reserve],
    }
    return SplitManifest(
        seed=seed,
        source_revision=AREAL_REVISION,
        source_file_hash=source_file_hash,
        train=train,
        selection=selection,
        reserve=reserve,
        split_hash=sha256_json(split_payload),
    )


def official_tau3_manifest(task_ids: Iterable[str]) -> list[ManifestEntry]:
    ids = list(task_ids)
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError(f"τ³ v1.0.1 Airline final must contain 50 unique tasks, got {len(ids)}")
    return [
        ManifestEntry(
            task_id=task_id,
            split="tau3_final",
            source=DataSource.TAU3_OFFICIAL_AIRLINE,
            source_revision=TAU3_REVISION,
            task_hash=sha256_text(f"{TAU3_REVISION}:airline:{task_id}"),
        )
        for task_id in ids
    ]


def write_jsonl(entries: Iterable[ManifestEntry], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(canonical_json(entry.model_dump(mode="json")) + "\n")


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    with Path(path).open(encoding="utf-8") as handle:
        return [ManifestEntry.model_validate_json(line) for line in handle if line.strip()]

