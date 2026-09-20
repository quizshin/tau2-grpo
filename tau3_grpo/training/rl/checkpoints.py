"""Shared structural validation of full veRL recovery checkpoints.

Size receipts establish completeness, not loadability or numerical correctness.
GPU restore and cloud continuity remain separate acceptance checks.
"""
from __future__ import annotations

import json
from pathlib import Path


def required_files(world_size: int) -> set[str]:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("Checkpoint world size must be a positive integer")
    return {"data.pt"} | {
        f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state") for rank in range(world_size)
    }


def validate_checkpoint(run, step=20, *, world_size=4, require_latest=True) -> Path:
    if type(step) is not int or step <= 0:
        raise ValueError("Checkpoint step must be a positive integer")
    required = required_files(world_size)
    run = Path(run).resolve()
    cp = run / f"global_step_{step}"
    if cp.is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    receipt_path = cp / "checkpoint-complete.json"
    if receipt_path.is_symlink():
        raise ValueError("Checkpoint receipt must not be a symlink")
    receipt = json.loads(receipt_path.read_text())
    if (type(receipt.get("step")) is not int or receipt["step"] != step
            or type(receipt.get("world_size")) is not int or receipt["world_size"] != world_size):
        raise ValueError("Unexpected checkpoint step/world size")
    files = receipt.get("files")
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("Checkpoint receipt lacks a complete recovery state")
    for name, size in files.items():
        relative = Path(name)
        path = cp / relative
        if (relative.is_absolute() or ".." in relative.parts or type(size) is not int or size <= 0
                or cp not in path.resolve().parents or not path.is_file()
                or path.stat().st_size != size):
            raise ValueError(f"Incomplete checkpoint file: {name}")
    if require_latest and int((run / "latest_checkpointed_iteration.txt").read_text()) != step:
        raise ValueError("Latest checkpoint marker differs from requested step")
    return cp
