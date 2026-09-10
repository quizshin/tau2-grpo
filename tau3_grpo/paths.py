"""Resolve the flat code root and independently configurable runtime storage."""

from __future__ import annotations

import os
from pathlib import Path

_THIS = Path(__file__).resolve()

# paths.py -> tau3_grpo -> code
CODE_ROOT = _THIS.parents[1]

PROJECT_ROOT = CODE_ROOT
ENV_INFO_ROOT = CODE_ROOT / "env_info"
TAU2_BENCH_ROOT = CODE_ROOT / "tau2-bench"
VERL_ROOT = CODE_ROOT / "verl"

def runtime_root(variable: str, default: str) -> Path:
    value = Path(os.environ.get(variable, default)).expanduser()
    return value.resolve() if value.is_absolute() else (CODE_ROOT / value).resolve()


DATA_ROOT = runtime_root("TAU3_DATA_ROOT", "data")
MODEL_ROOT = runtime_root("TAU3_MODEL_ROOT", "models")
CACHE_ROOT = runtime_root("TAU3_CACHE_ROOT", ".cache")
RAW_DATA_ROOT = DATA_ROOT / "raw"
AREAL_RAW_ROOT = RAW_DATA_ROOT / "areal_tau2"
AREAL_JSONL = AREAL_RAW_ROOT / "tau2_rl_train.jsonl"
AREAL_SFT_JSONL = AREAL_RAW_ROOT / "tau2_sft_train.jsonl"
AREAL_DB_ROOT = AREAL_RAW_ROOT

MANIFEST_ROOT = DATA_ROOT / "manifests"
PARQUET_ROOT = DATA_ROOT / "parquet"
SFT_DATA_ROOT = DATA_ROOT / "sft"
RESULTS_ROOT = runtime_root("TAU3_RUN_ROOT", "results")
CONFIG_ROOT = PROJECT_ROOT / "configs"
DOCS_ROOT = PROJECT_ROOT / "docs"

EXPECTED_ROOTS = ("tau3_grpo", "configs", "scripts", "env_info", "tau2-bench", "verl")


def verify_layout(root: str | Path | None = None) -> Path:
    """Confirm the project and dependency directories and return the code root."""

    base = Path(root).resolve() if root is not None else CODE_ROOT
    missing = [name for name in EXPECTED_ROOTS if not (base / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"code root {base} is missing required directories: {missing}")
    return base


def tau2_src_root(root: str | Path | None = None) -> Path:
    """Return the importable `src/` directory of the vendored tau2-bench."""

    base = Path(root).resolve() if root is not None else CODE_ROOT
    candidate = base / "tau2-bench" / "src"
    if not candidate.is_dir():
        raise FileNotFoundError(f"tau2-bench source tree not found at {candidate}")
    return candidate


def resolve_under(root: str | Path, relative: str) -> Path:
    """Join `relative` under `root`, rejecting traversal outside `root`.

    Used for every dataset-supplied path (AReaL `db_path`) so a crafted record
    cannot read outside the dataset tree.
    """

    base = Path(root).resolve()
    if os.path.isabs(relative):
        raise ValueError(f"absolute paths are not allowed: {relative}")
    candidate = (base / relative).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"path escapes root {base}: {relative}")
    return candidate


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths while keeping runtime storage independent of code."""
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    roots = {"data": DATA_ROOT, "models": MODEL_ROOT, "results": RESULTS_ROOT,
             ".cache": CACHE_ROOT}
    if value.parts and value.parts[0] in roots:
        return roots[value.parts[0]].joinpath(*value.parts[1:])
    return CODE_ROOT / value
