"""Trainer-driver telemetry bridge, independent of torch/Ray."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tau3_grpo.tracking.trainer_telemetry import (
    categorical_counts,
    collect_rollout_metrics,
    write_training_update,
)


class _FakeScores:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float64)

    def sum(self, dim):
        assert dim == 1
        return _FakeScores(self._values.sum(axis=1))

    def detach(self):
        return self

    def to(self, device):
        assert device == "cpu"
        return self

    def numpy(self):
        return self._values


def test_categorical_counts_handles_numpy_and_enum_like_values():
    enum_like = SimpleNamespace(value="wrong_outcome")
    values = np.array(["turn_limit", "turn_limit", enum_like, None], dtype=object)
    assert categorical_counts(values) == {"turn_limit": 2, "wrong_outcome": 1}


def test_collect_rollout_metrics_counts_only_non_singleton_anchor_signal():
    metrics = collect_rollout_metrics(
        {
            "anchor_ids": np.array(
                [["shared", None], ["shared", "singleton"], []], dtype=object
            ),
            "failure_category": np.array(
                ["wrong_outcome", "wrong_outcome", "turn_limit"], dtype=object
            ),
            "termination_reason": np.array(
                ["agent_stop", "agent_stop", "max_steps"], dtype=object
            ),
            "__num_turns__": np.array([3, 5, 7]),
        }
    )
    assert metrics["anchors/anchored_steps"] == 3
    assert metrics["anchors/unique"] == 2
    assert metrics["anchors/groups_with_signal"] == 1
    assert metrics["anchors/usable_step_coverage"] == pytest.approx(2 / 3)
    assert metrics["rollout/failure_category/wrong_outcome"] == 2
    assert metrics["rollout/termination_reason/max_steps"] == 1
    assert metrics["rollout/mean_num_turns"] == pytest.approx(5.0)


def test_collect_rollout_metrics_is_safe_for_e0_without_gigpo_state():
    metrics = collect_rollout_metrics({}, adv_estimator="grpo")
    assert metrics["anchors/anchored_steps"] == 0
    assert metrics["anchors/usable_step_coverage"] == 0.0
    assert not any(key.startswith("gigpo/") for key in metrics)


def test_write_training_update_appends_configured_jsonl(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "telemetry.jsonl"
    monkeypatch.setenv("TAU3_GRPO_TELEMETRY_PATH", str(output))
    monkeypatch.setenv("TAU3_GRPO_ARM", "e3")
    monkeypatch.setenv("TAU3_GRPO_TRAIN_SEED", "43")
    batch = SimpleNamespace(
        batch={"token_level_scores": _FakeScores([[0.0, 1.0], [0.0, 0.0]])},
        non_tensor_batch={
            "failure_category": np.array(["success", "wrong_outcome"], dtype=object),
            "termination_reason": np.array(["user_stop", "agent_stop"], dtype=object),
        },
    )

    written = write_training_update(
        batch=batch,
        metrics={
            "dynamic_filter/d_bar": 0.25,
            "gigpo/usable_step_coverage": 0.5,
            "anchors/unique": 3,
        },
        update_index=7,
    )

    assert written == output
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["update_index"] == 7
    assert record["arm"] == "e3"
    assert record["seed"] == 43
    assert record["mean_reward"] == pytest.approx(0.5)
    assert record["solve_rate"] == pytest.approx(0.5)
    assert record["dynamic_filter"] == {"d_bar": 0.25}
    assert record["gigpo"] == {"usable_step_coverage": 0.5}
    assert record["anchors"] == {"unique": 3}
    assert record["failure_categories"] == {"success": 1, "wrong_outcome": 1}


def test_write_training_update_is_disabled_without_path(monkeypatch):
    monkeypatch.delenv("TAU3_GRPO_TELEMETRY_PATH", raising=False)
    assert write_training_update(batch=object(), metrics={}, update_index=0) is None
