"""Flat project layout and version pins."""

from __future__ import annotations

import pytest

from tau3_grpo import (
    AREAL_REVISION,
    TAU2_BENCH_COMMIT,
    TAU2_BENCH_VERSION,
    VERL_COMMIT,
    VERL_VERSION,
)
from tau3_grpo.paths import (
    CODE_ROOT,
    ENV_INFO_ROOT,
    EXPECTED_ROOTS,
    PROJECT_ROOT,
    TAU2_BENCH_ROOT,
    VERL_ROOT,
    tau2_src_root,
    verify_layout,
)


def test_code_root_resolves_to_project_root():
    assert CODE_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert PROJECT_ROOT == CODE_ROOT


def test_all_required_roots_exist():
    verify_layout()
    for name in EXPECTED_ROOTS:
        assert (CODE_ROOT / name).is_dir(), f"missing root: {name}"


def test_expected_required_source_directories():
    assert len(EXPECTED_ROOTS) == 6
    assert set(EXPECTED_ROOTS) == {
        "tau3_grpo",
        "configs",
        "scripts",
        "env_info",
        "tau2-bench",
        "verl",
    }


def test_root_shortcuts_point_where_expected():
    assert ENV_INFO_ROOT == CODE_ROOT / "env_info"
    assert TAU2_BENCH_ROOT == CODE_ROOT / "tau2-bench"
    assert VERL_ROOT == CODE_ROOT / "verl"


def test_verify_layout_reports_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing required directories"):
        verify_layout(tmp_path)


def test_tau2_src_root_exists():
    assert tau2_src_root().is_dir()
    assert (tau2_src_root() / "tau2").is_dir()


def test_tau2_src_root_reports_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="tau2-bench source tree"):
        tau2_src_root(tmp_path)


def test_version_pins_are_the_agreed_ones():
    assert AREAL_REVISION == "86971dc03da6e7c1a7933295e05b84aab8215386"
    assert TAU2_BENCH_VERSION == "1.0.1"
    assert TAU2_BENCH_COMMIT == "fc0055d"
    assert VERL_VERSION == "0.7.1"
    assert VERL_COMMIT == "bec9ef7"


def test_no_day_directories_exist():
    # Day progress lives in file headers, docs/implementation_status.md and commit
    # messages only.
    offenders = [
        path
        for path in CODE_ROOT.rglob("day*")
        if path.is_dir() and path.name[3:].isdigit()
    ]
    assert offenders == [], f"temporary day directories found: {offenders}"
