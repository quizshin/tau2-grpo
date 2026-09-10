"""Winner lock and τ³ final guard."""

from __future__ import annotations

import json

import pytest

from tau3_grpo.experiments.winner_lock import (
    WINNER_LOCK_FILENAME,
    WinnerLockError,
    assert_final_run_allowed,
    freeze_winner,
    load_winner_lock,
    lock_exists,
)


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    model = checkpoint / "model.safetensors"
    if not model.exists():
        model.write_bytes(b"frozen weights")
    return str(checkpoint)


def _freeze(tmp_path, **overrides):
    kwargs = {
        "experiment": "e3",
        "checkpoint_path": _checkpoint(tmp_path),
        "selection_metric": "selection_solve_rate",
        "selection_score": 0.42,
        "selection_task_count": 60,
        "seeds": (42, 43),
        "output_dir": tmp_path,
    }
    kwargs.update(overrides)
    return freeze_winner(**kwargs)


def test_final_run_refused_without_lock(tmp_path):
    assert not lock_exists(tmp_path)
    with pytest.raises(WinnerLockError, match="no winner lock"):
        assert_final_run_allowed(
            checkpoint_path=_checkpoint(tmp_path), task_count=50, directory=tmp_path
        )


def test_final_run_allowed_after_freeze(tmp_path):
    _freeze(tmp_path)
    lock = assert_final_run_allowed(
        checkpoint_path=_checkpoint(tmp_path), task_count=50, directory=tmp_path
    )
    assert lock.checkpoint_path == _checkpoint(tmp_path)
    assert len(lock.checkpoint_hash) == 64
    assert lock.seeds == (42, 43)


def test_final_run_refuses_other_checkpoint(tmp_path):
    _freeze(tmp_path)
    with pytest.raises(WinnerLockError, match="is not the frozen winner"):
        assert_final_run_allowed(
            checkpoint_path="results/e0_seed42/global_step_40",
            task_count=50,
            directory=tmp_path,
        )


@pytest.mark.parametrize("count", [49, 51, 0, 60])
def test_final_run_requires_exactly_50_tasks(tmp_path, count):
    _freeze(tmp_path)
    with pytest.raises(WinnerLockError, match="50 official"):
        assert_final_run_allowed(
            checkpoint_path=_checkpoint(tmp_path), task_count=count, directory=tmp_path
        )


def test_freeze_refuses_silent_overwrite(tmp_path):
    _freeze(tmp_path)
    with pytest.raises(WinnerLockError, match="refusing to overwrite"):
        _freeze(tmp_path)


def test_freeze_allows_explicit_overwrite(tmp_path):
    _freeze(tmp_path)
    lock, _ = _freeze(tmp_path, overwrite=True, selection_score=0.5)
    assert lock.selection_score == pytest.approx(0.5)


def test_edited_lock_is_detected(tmp_path):
    _freeze(tmp_path)
    path = tmp_path / WINNER_LOCK_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["selection_score"] = 0.99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(WinnerLockError, match="hash mismatch"):
        load_winner_lock(tmp_path)


def test_edited_checkpoint_is_detected(tmp_path):
    _freeze(tmp_path)
    path = tmp_path / WINNER_LOCK_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["checkpoint_path"] = "results/other/step_1"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(WinnerLockError, match="hash mismatch"):
        load_winner_lock(tmp_path)


def test_changed_checkpoint_content_is_detected(tmp_path):
    _freeze(tmp_path)
    (tmp_path / "checkpoint/model.safetensors").write_bytes(b"different weights")
    with pytest.raises(WinnerLockError, match="content hash changed"):
        assert_final_run_allowed(
            checkpoint_path=_checkpoint(tmp_path), task_count=50, directory=tmp_path
        )


def test_lock_hash_is_deterministic(tmp_path):
    lock_a, _ = _freeze(tmp_path)
    lock_b, _ = _freeze(tmp_path, overwrite=True)
    assert lock_a.lock_hash == lock_b.lock_hash


def test_seed_order_does_not_change_hash(tmp_path):
    lock_a, _ = _freeze(tmp_path, seeds=(42, 43))
    lock_b, _ = _freeze(tmp_path, seeds=(43, 42), overwrite=True)
    assert lock_a.lock_hash == lock_b.lock_hash


def test_unsupported_version_is_rejected(tmp_path):
    _freeze(tmp_path)
    path = tmp_path / WINNER_LOCK_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(WinnerLockError, match="unsupported winner lock version"):
        load_winner_lock(tmp_path)


def test_zero_selection_tasks_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        _freeze(tmp_path, selection_task_count=0)
