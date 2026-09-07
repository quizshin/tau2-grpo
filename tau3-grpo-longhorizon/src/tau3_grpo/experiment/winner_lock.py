"""Winner lock and τ³ final guard.

Milestone D13. The τ³ official Airline final (50 tasks) may only be run once a
single winner has been frozen from *internal selection* results. The lock records
which checkpoint won, on what evidence, and its hash; the guard refuses to run
the final set unless a matching lock exists on disk.

This is the mechanism that keeps the headline number honest: without a frozen
lock there is no way to run τ³ official, so the final set cannot be used to pick
a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tau3_grpo.data.official import OFFICIAL_AIRLINE_TASK_COUNT, official_revision
from tau3_grpo.paths import RESULTS_ROOT
from tau3_grpo.utils.hashing import canonical_json, sha256_path, sha256_text

WINNER_LOCK_FILENAME = "winner_lock.json"
LOCK_VERSION = 2


class WinnerLockError(RuntimeError):
    """Raised when the τ³ final guard refuses to proceed."""


@dataclass(frozen=True)
class WinnerLock:
    """An immutable record of the frozen winner."""

    experiment: str
    checkpoint_path: str
    checkpoint_hash: str
    selection_metric: str
    selection_score: float
    selection_task_count: int
    seeds: tuple[int, ...]
    frozen_at: str
    tau3_revision: str
    lock_hash: str
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": LOCK_VERSION,
            "experiment": self.experiment,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "selection_metric": self.selection_metric,
            "selection_score": self.selection_score,
            "selection_task_count": self.selection_task_count,
            "seeds": list(self.seeds),
            "frozen_at": self.frozen_at,
            "tau3_revision": self.tau3_revision,
            "lock_hash": self.lock_hash,
            "notes": self.notes,
        }


def _lock_payload(
    experiment: str,
    checkpoint_path: str,
    checkpoint_hash: str,
    selection_metric: str,
    selection_score: float,
    selection_task_count: int,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "version": LOCK_VERSION,
        "experiment": experiment,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "selection_metric": selection_metric,
        "selection_score": round(float(selection_score), 12),
        "selection_task_count": selection_task_count,
        "seeds": sorted(seeds),
        "tau3_revision": official_revision(),
    }


def freeze_winner(
    *,
    experiment: str,
    checkpoint_path: str,
    selection_metric: str,
    selection_score: float,
    selection_task_count: int,
    seeds: tuple[int, ...] | list[int],
    notes: Optional[str] = None,
    output_dir: str | Path = RESULTS_ROOT,
    overwrite: bool = False,
) -> tuple[WinnerLock, Path]:
    """Write the winner lock. Refuses to silently replace an existing lock."""

    seed_tuple = tuple(int(seed) for seed in seeds)
    if selection_task_count <= 0:
        raise ValueError("selection_task_count must be positive")
    checkpoint_hash = sha256_path(checkpoint_path)
    payload = _lock_payload(
        experiment,
        checkpoint_path,
        checkpoint_hash,
        selection_metric,
        selection_score,
        selection_task_count,
        seed_tuple,
    )
    lock = WinnerLock(
        experiment=experiment,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        selection_metric=selection_metric,
        selection_score=float(selection_score),
        selection_task_count=selection_task_count,
        seeds=seed_tuple,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        tau3_revision=official_revision(),
        lock_hash=sha256_text(canonical_json(payload)),
        notes=notes,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / WINNER_LOCK_FILENAME
    if path.exists() and not overwrite:
        raise WinnerLockError(
            f"winner lock already exists at {path}; refusing to overwrite a frozen "
            "winner (pass overwrite=True only to correct a mistake before any τ³ run)"
        )
    path.write_text(json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock, path


def load_winner_lock(directory: str | Path = RESULTS_ROOT) -> WinnerLock:
    """Load and verify the lock, recomputing its hash."""

    path = Path(directory) / WINNER_LOCK_FILENAME
    if not path.is_file():
        raise WinnerLockError(
            f"no winner lock at {path}; the τ³ official final cannot run before a "
            "winner is frozen from internal selection"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != LOCK_VERSION:
        raise WinnerLockError(f"unsupported winner lock version: {data.get('version')}")
    seeds = tuple(int(seed) for seed in data["seeds"])
    expected = sha256_text(
        canonical_json(
            _lock_payload(
                data["experiment"],
                data["checkpoint_path"],
                data["checkpoint_hash"],
                data["selection_metric"],
                data["selection_score"],
                int(data["selection_task_count"]),
                seeds,
            )
        )
    )
    if expected != data["lock_hash"]:
        raise WinnerLockError(
            f"winner lock hash mismatch at {path}: the lock was edited after freezing"
        )
    return WinnerLock(
        experiment=data["experiment"],
        checkpoint_path=data["checkpoint_path"],
        checkpoint_hash=data["checkpoint_hash"],
        selection_metric=data["selection_metric"],
        selection_score=float(data["selection_score"]),
        selection_task_count=int(data["selection_task_count"]),
        seeds=seeds,
        frozen_at=data["frozen_at"],
        tau3_revision=data["tau3_revision"],
        lock_hash=data["lock_hash"],
        notes=data.get("notes"),
    )


def assert_final_run_allowed(
    *,
    checkpoint_path: str,
    task_count: int,
    directory: str | Path = RESULTS_ROOT,
) -> WinnerLock:
    """Gate the τ³ official final run.

    Requires a valid lock, the exact checkpoint named by that lock, and the
    official 50-task count.
    """

    lock = load_winner_lock(directory)
    if Path(checkpoint_path).as_posix() != Path(lock.checkpoint_path).as_posix():
        raise WinnerLockError(
            f"τ³ final refused: checkpoint {checkpoint_path!r} is not the frozen "
            f"winner {lock.checkpoint_path!r}"
        )
    try:
        current_hash = sha256_path(checkpoint_path)
    except (FileNotFoundError, ValueError) as exc:
        raise WinnerLockError(f"τ³ final refused: cannot verify frozen checkpoint: {exc}") from exc
    if current_hash != lock.checkpoint_hash:
        raise WinnerLockError(
            "τ³ final refused: frozen checkpoint content hash changed after winner lock"
        )
    if task_count != OFFICIAL_AIRLINE_TASK_COUNT:
        raise WinnerLockError(
            f"τ³ final refused: expected {OFFICIAL_AIRLINE_TASK_COUNT} official "
            f"Airline tasks, got {task_count}"
        )
    return lock


def lock_exists(directory: str | Path = RESULTS_ROOT) -> bool:
    return (Path(directory) / WINNER_LOCK_FILENAME).is_file()
