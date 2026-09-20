"""Shared YAML composition with provenance; no services or environment mutation.

The legacy merge order is retained: includes in order, then the child mapping;
``overrides`` lists concatenate. Source hashes describe launch inputs, not the
fully resolved Hydra configuration produced later by the training framework.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        elif key == "overrides" and isinstance(value, list):
            result[key] = result.get(key, []) + value
        else:
            result[key] = value
    return result


def load_config_with_sources(
    path: Path, stack: tuple[Path, ...] = (),
) -> tuple[dict, list[dict[str, str]]]:
    """Return composed data and its ordered include occurrences with content hashes."""
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"cyclic config include: {path}")
    content = path.read_bytes()
    payload = yaml.safe_load(content)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    includes = payload.pop("includes", [])
    if not isinstance(includes, list) or any(not isinstance(p, str) for p in includes):
        raise ValueError(f"includes must be a list of paths: {path}")
    result: dict = {}
    sources = []
    for parent in includes:
        inherited, parents = load_config_with_sources(path.parent / parent, (*stack, path))
        result = merge(result, inherited)
        sources.extend(parents)
    sources.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})
    return merge(result, payload), sources


def load_config(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    return load_config_with_sources(path, stack)[0]


def resolve_arm(name: str, catalog: dict) -> dict:
    """Resolve both direct arms and inherited ablations from the same catalog."""
    if name in catalog["arms"]:
        return catalog["arms"][name]
    if name not in catalog.get("ablations", {}):
        raise ValueError(f"unknown arm: {name!r}")
    ablation = catalog["ablations"][name]
    return merge(catalog["arms"][ablation["base_arm"]], ablation)
