"""Bind a live vLLM process to the checkpoint selected for evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau3_grpo.utils.hashing import canonical_json, sha256_path, sha256_text

ATTESTATION_VERSION = 1


class ServiceAttestationError(RuntimeError):
    """Raised when evaluation cannot prove which checkpoint vLLM serves."""


@dataclass(frozen=True)
class ServiceAttestation:
    checkpoint_path: str
    checkpoint_hash: str
    served_model_name: str
    base_url: str
    pid: int
    created_at: str
    attestation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": ATTESTATION_VERSION,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "served_model_name": self.served_model_name,
            "base_url": self.base_url,
            "pid": self.pid,
            "created_at": self.created_at,
            "attestation_hash": self.attestation_hash,
        }


def _payload(
    checkpoint_path: str,
    checkpoint_hash: str,
    served_model_name: str,
    base_url: str,
    pid: int,
) -> dict[str, Any]:
    return {
        "version": ATTESTATION_VERSION,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "served_model_name": served_model_name,
        "base_url": base_url.rstrip("/"),
        "pid": int(pid),
    }


def write_service_attestation(
    *,
    checkpoint_path: str,
    served_model_name: str,
    base_url: str,
    pid: int,
    output: str | Path,
) -> ServiceAttestation:
    if pid <= 0:
        raise ValueError("service pid must be positive")
    checkpoint_hash = sha256_path(checkpoint_path)
    payload = _payload(
        checkpoint_path, checkpoint_hash, served_model_name, base_url, pid
    )
    attestation = ServiceAttestation(
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        served_model_name=served_model_name,
        base_url=base_url.rstrip("/"),
        pid=int(pid),
        created_at=datetime.now(timezone.utc).isoformat(),
        attestation_hash=sha256_text(canonical_json(payload)),
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(attestation.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return attestation


def load_service_attestation(path: str | Path) -> ServiceAttestation:
    source = Path(path)
    if not source.is_file():
        raise ServiceAttestationError(f"policy service attestation not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("version") != ATTESTATION_VERSION:
        raise ServiceAttestationError(
            f"unsupported service attestation version: {data.get('version')}"
        )
    expected = sha256_text(
        canonical_json(
            _payload(
                data["checkpoint_path"],
                data["checkpoint_hash"],
                data["served_model_name"],
                data["base_url"],
                int(data["pid"]),
            )
        )
    )
    if expected != data["attestation_hash"]:
        raise ServiceAttestationError("policy service attestation hash mismatch")
    return ServiceAttestation(
        checkpoint_path=data["checkpoint_path"],
        checkpoint_hash=data["checkpoint_hash"],
        served_model_name=data["served_model_name"],
        base_url=data["base_url"],
        pid=int(data["pid"]),
        created_at=data["created_at"],
        attestation_hash=data["attestation_hash"],
    )


def assert_service_matches_checkpoint(
    *,
    attestation_path: str | Path,
    checkpoint_path: str,
    served_model_name: str,
    base_url: str,
    require_live_pid: bool = True,
) -> ServiceAttestation:
    attestation = load_service_attestation(attestation_path)
    if Path(attestation.checkpoint_path).as_posix() != Path(checkpoint_path).as_posix():
        raise ServiceAttestationError("served checkpoint path does not match evaluation checkpoint")
    if attestation.served_model_name != served_model_name:
        raise ServiceAttestationError("served model name does not match evaluation model name")
    if attestation.base_url.rstrip("/") != base_url.rstrip("/"):
        raise ServiceAttestationError("served base URL does not match evaluation endpoint")
    try:
        current_hash = sha256_path(checkpoint_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ServiceAttestationError(f"cannot hash evaluation checkpoint: {exc}") from exc
    if current_hash != attestation.checkpoint_hash:
        raise ServiceAttestationError("served checkpoint content changed after vLLM launch")
    if require_live_pid:
        try:
            os.kill(attestation.pid, 0)
        except OSError as exc:
            raise ServiceAttestationError(
                f"attested vLLM process {attestation.pid} is not running"
            ) from exc
    return attestation
