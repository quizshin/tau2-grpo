"""Stable hashing helpers.

Milestone D1: source, task, DB and split hashes are explicit experiment inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data identically across processes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash one file or a complete directory tree including relative names.

    Checkpoint locks use this instead of hashing the path string. Symlinks are
    rejected so the same lock cannot be redirected to different model content.
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"path to hash does not exist: {root}")
    if root.is_symlink():
        raise ValueError(f"refusing to hash symlink: {root}")
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise ValueError(f"path is neither a regular file nor directory: {root}")

    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise ValueError(f"refusing to hash empty checkpoint directory: {root}")
    for item in files:
        if item.is_symlink():
            raise ValueError(f"checkpoint tree contains symlink: {item}")
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
