"""Policy endpoint attestation binds evaluation to checkpoint bytes."""

from __future__ import annotations

import json
import os

import pytest

from tau3_grpo.evaluation.service_attestation import (
    ServiceAttestationError,
    assert_service_matches_checkpoint,
    load_service_attestation,
    write_service_attestation,
)


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    model = checkpoint / "model.safetensors"
    if not model.exists():
        model.write_bytes(b"weights")
    return checkpoint


def _write(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "attestation.json"
    write_service_attestation(
        checkpoint_path=str(checkpoint),
        served_model_name="policy",
        base_url="http://127.0.0.1:8000/v1",
        pid=os.getpid(),
        output=path,
    )
    return checkpoint, path


def test_live_service_attestation_matches_checkpoint(tmp_path):
    checkpoint, path = _write(tmp_path)
    result = assert_service_matches_checkpoint(
        attestation_path=path,
        checkpoint_path=str(checkpoint),
        served_model_name="policy",
        base_url="http://127.0.0.1:8000/v1/",
    )
    assert result.pid == os.getpid()
    assert len(result.checkpoint_hash) == 64


def test_attestation_refuses_changed_checkpoint_content(tmp_path):
    checkpoint, path = _write(tmp_path)
    (checkpoint / "model.safetensors").write_bytes(b"different")
    with pytest.raises(ServiceAttestationError, match="content changed"):
        assert_service_matches_checkpoint(
            attestation_path=path,
            checkpoint_path=str(checkpoint),
            served_model_name="policy",
            base_url="http://127.0.0.1:8000/v1",
            require_live_pid=False,
        )


def test_attestation_refuses_other_served_name(tmp_path):
    checkpoint, path = _write(tmp_path)
    with pytest.raises(ServiceAttestationError, match="model name"):
        assert_service_matches_checkpoint(
            attestation_path=path,
            checkpoint_path=str(checkpoint),
            served_model_name="other",
            base_url="http://127.0.0.1:8000/v1",
            require_live_pid=False,
        )


def test_edited_attestation_is_detected(tmp_path):
    _, path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ServiceAttestationError, match="hash mismatch"):
        load_service_attestation(path)
