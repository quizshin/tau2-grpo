"""A supplemental attempt must match one failed, non-final task/trial/seed."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "single_selection_retry", Path(__file__).resolve().parents[1]
    / "env_info/a800_20260912/retry_one_selection_trial.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
IDENTITY = {"task_id": "airline_802", "trial": 3, "seed": 45}


def test_preserves_original_trial_and_seed():
    run = {"spec": {"target": "selection"}, "planned": [IDENTITY]}
    error = {**IDENTITY, "error": "context overflow"}
    identity, original = MODULE.select_failed_identity(run, [error], "airline_802", 3)
    assert identity == IDENTITY and original == error


@pytest.mark.parametrize("target,planned,errors", [
    ("selection", [IDENTITY], []),
    ("selection", [IDENTITY], [{**IDENTITY, "seed": 44}]),
    ("selection", [IDENTITY, IDENTITY], [IDENTITY]),
    ("tau3-final", [IDENTITY], [IDENTITY]),
])
def test_rejects_unmatched_or_official_attempt(target, planned, errors):
    with pytest.raises(ValueError):
        MODULE.select_failed_identity({"spec": {"target": target}, "planned": planned},
                                      errors, "airline_802", 3)
