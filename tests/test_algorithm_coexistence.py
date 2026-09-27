"""Exercise all three estimators in one trainer process after source integration."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import compute_advantage, tau3_dynamic_filter

from tau3_grpo.evaluation.process_reward import payload_json, score_turns
from tau3_grpo.tracking.signal_audit import audit_update, capture_mask
from tau3_grpo.tracking.trainer_telemetry import collect_rollout_metrics
from verl import DataProto


def _objects(rows):
    result = np.empty(len(rows), dtype=object)
    result[:] = rows
    return result


def _batch():
    payloads = []
    for error in (False, True, False, False):
        turns = [{
            "schema": "tau3_turn_v1", "turn_index": 0, "token_span": [0, 2],
            "tool_calls": [{
                "name": "calculate", "arguments": {"expression": "1+1"}, "error": error,
            }],
        }]
        payloads.append(payload_json(score_turns(turns, [], [], {"mode": "conservative"})))
    return DataProto.from_dict(
        tensors={"token_level_rewards": torch.zeros(4, 2), "response_mask": torch.ones(4, 2)},
        non_tensors={
            "uid": np.array(["u", "u", "v", "v"]),
            "anchor_ids": _objects([["u"], ["u"], ["v"], ["v"]]),
            "anchor_spans": _objects([[[0, 2]]] * 4),
            "process_reward_json": np.array(payloads, dtype=object),
        },
    )


@pytest.mark.parametrize("order", [
    ("tau_gigpo", "mt_gtpo", "grpo", "tau_gigpo"),
    ("mt_gtpo", "tau_gigpo", "grpo", "mt_gtpo"),
])
def test_switching_estimators_preserves_filter_and_audit_boundaries(order, tmp_path, monkeypatch):
    # A persistent GiGPO audit setting must not capture GTPO or ordinary GRPO.
    monkeypatch.setenv("TAU3_GRPO_SIGNAL_AUDIT_DIR", str(tmp_path))
    audit_count = 0
    for update, estimator in enumerate(order, 1):
        data = _batch()
        config = OmegaConf.create({
            "adv_estimator": estimator,
            "gigpo": {"episode_normalization": "grpo"},
            "process_reward": {"mode": "conservative"},
            "dynamic_filter": {
                "enable": estimator != "grpo", "group_size": 2, "metric": "advantage",
            },
        })
        before = capture_mask(data, config)
        filter_metrics = tau3_dynamic_filter(data, config)
        compute_advantage(data, estimator, config=config)
        audit_metrics = audit_update(data, before, config, update)
        metrics = collect_rollout_metrics(data.non_tensor_batch, adv_estimator=estimator)

        if estimator == "mt_gtpo":
            # The all-failure u group retains real process contrast; v has none.
            assert filter_metrics == {}
            expected = 0.05 / (0.05 + 1e-6)
            assert data.batch["advantages"][0].tolist() == pytest.approx([expected] * 2)
            assert data.batch["advantages"][1].tolist() == pytest.approx([-expected] * 2)
            assert not data.batch["advantages"][2:].any()
            assert data.batch["response_mask"].sum() == 4
            assert "mt_gtpo_replay_json" in data.non_tensor_batch
            assert "mt_gtpo/nonzero_turns" in metrics
            assert not any(key.startswith("gigpo/") for key in metrics)
            assert before is None and audit_metrics == {}
        elif estimator == "tau_gigpo":
            # Historical terminal-reward filtering still drops both uniform groups.
            assert filter_metrics
            assert not data.batch["response_mask"].any()
            assert not data.batch["advantages"].any()
            assert audit_metrics and before is not None
            assert "gigpo/episode_grpo_normalization" in metrics
            assert not any(key.startswith("mt_gtpo/") for key in metrics)
            audit_count += 1
        else:
            assert filter_metrics == {}
            assert data.batch["response_mask"].sum() == 8
            assert not data.batch["advantages"].any()
            assert before is None and audit_metrics == {}
            assert not any(key.startswith(("mt_gtpo/", "gigpo/")) for key in metrics)
        assert len(list(tmp_path.glob("*.json"))) == audit_count
