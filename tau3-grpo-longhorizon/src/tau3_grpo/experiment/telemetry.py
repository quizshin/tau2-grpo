"""Run telemetry: per-update records and the aggregated report payload.

Milestone D14. Every record is JSON-serialisable and written as JSONL so a run can
be inspected while it is still going. The aggregate deliberately separates
"failed" from "never finished" (see `env.verifier.classify_failure`), because the
upstream reward of 0.0 conflates them and `d_bar` is uninterpretable otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from tau3_grpo.experiment.metrics import aggregate_pass_at_k, mean_reward, solve_rate
from tau3_grpo.paths import RESULTS_ROOT

TELEMETRY_FILENAME = "telemetry.jsonl"
REPORT_FILENAME = "report.json"


@dataclass
class UpdateRecord:
    """One optimizer update's telemetry."""

    update_index: int
    arm: str
    seed: int
    mean_reward: float
    solve_rate: float
    dynamic_filter: dict[str, Any] = field(default_factory=dict)
    gigpo: dict[str, Any] = field(default_factory=dict)
    anchors: dict[str, Any] = field(default_factory=dict)
    failure_categories: dict[str, int] = field(default_factory=dict)
    termination_reasons: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_index": self.update_index,
            "arm": self.arm,
            "seed": self.seed,
            "mean_reward": self.mean_reward,
            "solve_rate": self.solve_rate,
            "dynamic_filter": self.dynamic_filter,
            "gigpo": self.gigpo,
            "anchors": self.anchors,
            "failure_categories": self.failure_categories,
            "termination_reasons": self.termination_reasons,
            "timestamp": self.timestamp,
        }


def count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = getattr(value, "value", str(value))
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_update_record(
    *,
    update_index: int,
    arm: str,
    seed: int,
    rewards: Sequence[float],
    filter_stats: Optional[Any] = None,
    gigpo_stats: Optional[Any] = None,
    anchor_telemetry: Optional[Any] = None,
    failure_categories: Iterable[Any] = (),
    termination_reasons: Iterable[Any] = (),
) -> UpdateRecord:
    """Assemble one record from the per-update objects the trainer produced."""

    return UpdateRecord(
        update_index=update_index,
        arm=arm,
        seed=seed,
        mean_reward=mean_reward(rewards),
        solve_rate=solve_rate(rewards),
        dynamic_filter=filter_stats.to_dict() if filter_stats is not None else {},
        gigpo=gigpo_stats.to_dict() if gigpo_stats is not None else {},
        anchors=anchor_telemetry.to_dict() if anchor_telemetry is not None else {},
        failure_categories=count_values(failure_categories),
        termination_reasons=count_values(termination_reasons),
    )


class TelemetryWriter:
    """Appends `UpdateRecord`s to a JSONL file."""

    def __init__(self, directory: str | Path = RESULTS_ROOT, filename: str = TELEMETRY_FILENAME):
        self.path = Path(directory) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: UpdateRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def summarise_telemetry(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a run's telemetry for the report."""

    if not records:
        return {"updates": 0}
    rewards = [float(record["mean_reward"]) for record in records]
    d_bars = [
        float(record.get("dynamic_filter", {}).get("d_bar", 0.0))
        for record in records
        if record.get("dynamic_filter")
    ]
    coverage = [
        float(record.get("anchors", {}).get("coverage", 0.0))
        for record in records
        if record.get("anchors")
    ]
    failures: dict[str, int] = {}
    terminations: dict[str, int] = {}
    for record in records:
        for key, value in (record.get("failure_categories") or {}).items():
            failures[key] = failures.get(key, 0) + int(value)
        for key, value in (record.get("termination_reasons") or {}).items():
            terminations[key] = terminations.get(key, 0) + int(value)

    summary: dict[str, Any] = {
        "updates": len(records),
        "mean_reward_first": rewards[0],
        "mean_reward_last": rewards[-1],
        "mean_reward_max": max(rewards),
        "mean_reward_mean": float(np.mean(rewards)),
        "failure_categories": failures,
        "termination_reasons": terminations,
    }
    if d_bars:
        summary["d_bar_mean"] = float(np.mean(d_bars))
        summary["d_bar_max"] = float(np.max(d_bars))
        summary["inverse_one_minus_d_bar_mean"] = float(
            np.mean([1.0 / max(1e-9, 1.0 - value) for value in d_bars])
        )
    if coverage:
        summary["anchor_coverage_mean"] = float(np.mean(coverage))
    return summary


def build_report(
    *,
    experiment: str,
    arms: dict[str, Sequence[dict[str, Any]]],
    per_task_rewards: Optional[dict[str, dict[str, Sequence[float]]]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the deterministic report payload."""

    report: dict[str, Any] = {
        "experiment": experiment,
        "arms": {name: summarise_telemetry(records) for name, records in arms.items()},
    }
    if per_task_rewards:
        report["pass_at_k"] = {
            arm: aggregate_pass_at_k(tasks) for arm, tasks in per_task_rewards.items()
        }
    if extra:
        report["extra"] = extra
    return report


def write_report(
    report: dict[str, Any], directory: str | Path = RESULTS_ROOT, filename: str = REPORT_FILENAME
) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
