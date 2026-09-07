"""Regression coverage for rollout JSONL metadata serialization."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl.trainer.ppo.ray_trainer import (
    _json_default,
    tau3_restore_candidate_mask_for_metrics,
)


def test_numpy_scalar_is_converted_to_native_python_value():
    assert _json_default(np.bool_(True)) is True
    assert _json_default(np.int64(7)) == 7


def test_unknown_value_still_raises_type_error():
    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_default(object())


def test_all_dropped_df_restores_mask_only_for_post_update_metrics():
    batch = SimpleNamespace(
        batch={
            "responses": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.long
            ),
            "response_mask": torch.zeros((2, 3), dtype=torch.long),
        }
    )

    assert tau3_restore_candidate_mask_for_metrics(batch, skipped=True)
    assert torch.equal(
        batch.batch["response_mask"],
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    )


def test_non_skipped_df_keeps_effective_mask_unchanged():
    mask = torch.tensor([[1, 0]], dtype=torch.long)
    batch = SimpleNamespace(
        batch={
            "responses": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "response_mask": mask.clone(),
        }
    )

    assert not tau3_restore_candidate_mask_for_metrics(batch, skipped=False)
    assert torch.equal(batch.batch["response_mask"], mask)
