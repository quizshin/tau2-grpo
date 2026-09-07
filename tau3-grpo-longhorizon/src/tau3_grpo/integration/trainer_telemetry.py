"""Bridge rollout/GiGPO telemetry into veRL's logger and project JSONL.

The rollout worker publishes anchors and official verifier categories in
``non_tensor_batch``.  This module turns them into scalar metrics after the full
GRPO batch has been gathered on the trainer driver.  Keeping the aggregation here
avoids process-local counters and makes every update independently auditable.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_SAFE_METRIC_COMPONENT = re.compile(r"[^a-zA-Z0-9_.-]+")


def _python_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _flatten_anchor_ids(value: Any) -> list[str]:
    anchors: list[str] = []
    for row in _python_rows(value):
        if row is None:
            continue
        if isinstance(row, np.ndarray):
            row = row.tolist()
        for anchor_id in row:
            if anchor_id is not None:
                anchors.append(str(anchor_id))
    return anchors


def categorical_counts(value: Any) -> dict[str, int]:
    """Count scalar categorical values from a gathered non-tensor column."""

    counts: Counter[str] = Counter()
    for item in _python_rows(value):
        if item is None:
            continue
        counts[str(getattr(item, "value", item))] += 1
    return dict(counts)


def _metric_name(value: str) -> str:
    return _SAFE_METRIC_COMPONENT.sub("_", value).strip("_") or "unknown"


def collect_rollout_metrics(
    non_tensor_batch: dict[str, Any],
    *,
    adv_estimator: Any = None,
) -> dict[str, float | int]:
    """Return logger-safe scalar metrics for one gathered training update."""

    padding = non_tensor_batch.get("tau3_is_padding")
    if padding is not None:
        keep = np.flatnonzero(~np.asarray(padding, dtype=bool))
        real_batch: dict[str, Any] = {}
        for key, value in non_tensor_batch.items():
            try:
                if len(value) == len(padding):
                    real_batch[key] = np.asarray(value, dtype=object)[keep]
                else:
                    real_batch[key] = value
            except TypeError:
                real_batch[key] = value
        non_tensor_batch = real_batch

    metrics: dict[str, float | int] = {}
    anchors = _flatten_anchor_ids(non_tensor_batch.get("anchor_ids"))
    group_sizes = Counter(anchors)
    grouped_steps = sum(size for size in group_sizes.values() if size >= 2)
    metrics.update(
        {
            "anchors/anchored_steps": len(anchors),
            "anchors/unique": len(group_sizes),
            "anchors/groups_with_signal": sum(size >= 2 for size in group_sizes.values()),
            "anchors/usable_step_coverage": (
                grouped_steps / len(anchors) if anchors else 0.0
            ),
        }
    )

    if str(adv_estimator) == "tau_gigpo":
        from tau3_grpo.algo.verl_estimator import last_stats

        for key, value in last_stats().items():
            if isinstance(value, (bool, int, float, np.number)):
                metrics[f"gigpo/{key}"] = float(value) if isinstance(value, np.floating) else value

    for column, prefix in (
        ("failure_category", "rollout/failure_category"),
        ("termination_reason", "rollout/termination_reason"),
    ):
        for category, count in categorical_counts(non_tensor_batch.get(column)).items():
            metrics[f"{prefix}/{_metric_name(category)}"] = count

    turns = _python_rows(non_tensor_batch.get("__num_turns__"))
    if turns:
        values = np.asarray(turns, dtype=np.float64)
        metrics["rollout/trajectories"] = int(values.size)
        metrics["rollout/mean_num_turns"] = float(values.mean())
        metrics["rollout/max_num_turns"] = float(values.max())
    return metrics


def _subtree(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    marker = prefix + "/"
    return {key[len(marker) :]: value for key, value in metrics.items() if key.startswith(marker)}


def write_training_update(
    *,
    batch: Any,
    metrics: dict[str, Any],
    update_index: int,
) -> Path | None:
    """Append one real trainer update when ``TAU3_GRPO_TELEMETRY_PATH`` is set."""

    configured = os.environ.get("TAU3_GRPO_TELEMETRY_PATH")
    if not configured:
        return None

    from tau3_grpo.experiment.telemetry import TelemetryWriter, UpdateRecord

    path = Path(configured)
    scores = batch.batch["token_level_scores"].sum(dim=1).detach().to("cpu").numpy()
    rewards = [float(value) for value in scores]
    failures = categorical_counts(batch.non_tensor_batch.get("failure_category"))
    terminations = categorical_counts(batch.non_tensor_batch.get("termination_reason"))

    record = UpdateRecord(
        update_index=int(update_index),
        arm=os.environ.get("TAU3_GRPO_ARM", "unknown"),
        seed=int(os.environ.get("TAU3_GRPO_TRAIN_SEED", "0")),
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        solve_rate=float(np.mean([value > 0.0 for value in rewards])) if rewards else 0.0,
        dynamic_filter=_subtree(metrics, "dynamic_filter"),
        gigpo=_subtree(metrics, "gigpo"),
        anchors=_subtree(metrics, "anchors"),
        failure_categories=failures,
        termination_reasons=terminations,
    )
    TelemetryWriter(path.parent, filename=path.name).append(record)
    return path
