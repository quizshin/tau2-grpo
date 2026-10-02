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


def test_frozen_curriculum_inputs_are_identical_under_data_root():
    import hashlib

    from tau3_grpo.paths import RL_CURRICULUM40_MANIFEST_ROOT, RL_CURRICULUM50_MANIFEST_ROOT

    expected = {
        RL_CURRICULUM40_MANIFEST_ROOT: (
            "79e317a824e8b55f7ce338c9f30a5811d83f2dce17dd896397eaffb003f5efbb",
            "e67447171c385168a01618082bfc08652afdf571b8cf98b22f197a9512dd41db",
        ),
        RL_CURRICULUM50_MANIFEST_ROOT: (
            "641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae",
            "97457389679a7c78d28c08034dd38217f01c20c607dc2cb67923a9ca5763d8b7",
        ),
    }
    for current, hashes in expected.items():
        for name, digest in zip(
            ("areal_airline_train_seed42.jsonl", "areal_airline_split_seed42.json"), hashes
        ):
            assert hashlib.sha256((current / name).read_bytes()).hexdigest() == digest
