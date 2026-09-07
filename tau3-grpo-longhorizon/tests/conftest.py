"""Shared fixtures.

The real AReaL JSONL lives at `data/raw/areal_tau2/tau2_rl_train.jsonl` and is
Git-ignored, so tests that need it skip when it is absent rather than failing.
`/private/tmp` is checked as a secondary location for sample data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tau3_grpo.paths import AREAL_JSONL
from tau3_grpo.tau2_bridge import tau2_available

SAMPLE_LOCATIONS = (
    AREAL_JSONL,
    Path("/private/tmp/tau2_rl_train.jsonl"),
    Path("/tmp/tau2_rl_train.jsonl"),
)


def find_areal_jsonl() -> Path | None:
    for candidate in SAMPLE_LOCATIONS:
        if candidate.is_file():
            return candidate
    return None


@pytest.fixture(scope="session")
def areal_jsonl() -> Path:
    path = find_areal_jsonl()
    if path is None:
        pytest.skip(
            "real AReaL JSONL not available; expected at "
            f"{AREAL_JSONL} (Git-ignored) or /private/tmp"
        )
    return path


@pytest.fixture(scope="session")
def requires_tau2() -> None:
    if not tau2_available():
        pytest.skip("tau2-bench is not importable in this environment")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "tau3: requires the sibling tau2-bench install")
    config.addinivalue_line("markers", "verl: requires the sibling veRL install")


def verl_available() -> bool:
    try:
        import verl  # noqa: F401
    except Exception:
        return False
    return True
