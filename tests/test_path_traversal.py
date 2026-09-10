"""Path traversal: a crafted `db_path` must never resolve outside the dataset root."""

from __future__ import annotations

import json

import pytest

from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.paths import resolve_under


def _record(db_path: str) -> ArealTaskRecord:
    return ArealTaskRecord(
        id="airline_1",
        db_path=db_path,
        user_scenario={"instructions": {"domain": "airline"}},
        evaluation_criteria=json.dumps({"actions": []}),
    )


@pytest.mark.parametrize(
    "hostile",
    [
        "../../../../etc/passwd",
        "tau2_rl_database/../../../../etc/passwd",
        "./../../secrets.json",
        "a/b/../../../../../../tmp/evil.json",
    ],
)
def test_record_rejects_traversal(tmp_path, hostile):
    with pytest.raises(ValueError, match="escapes dataset root"):
        _record(hostile).resolve_db_path(tmp_path)


def test_record_accepts_nested_relative_path(tmp_path):
    target = tmp_path / "tau2_rl_database" / "db.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    resolved = _record("tau2_rl_database/db.json").resolve_db_path(tmp_path)
    assert resolved == target.resolve()


@pytest.mark.parametrize(
    "hostile",
    ["../escape.json", "sub/../../escape.json", "../../etc/hosts"],
)
def test_resolve_under_rejects_traversal(tmp_path, hostile):
    with pytest.raises(ValueError, match="escapes root"):
        resolve_under(tmp_path, hostile)


def test_resolve_under_rejects_absolute_path(tmp_path):
    with pytest.raises(ValueError, match="absolute paths"):
        resolve_under(tmp_path, "/etc/passwd")


def test_resolve_under_allows_inside(tmp_path):
    assert resolve_under(tmp_path, "a/b/c.json") == (tmp_path / "a/b/c.json").resolve()


def test_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    root = tmp_path / "root"
    root.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:  # pragma: no cover - platform without symlink permission
        pytest.skip("symlinks not permitted in this environment")
    # resolve() follows the symlink, so the escape is caught.
    with pytest.raises(ValueError, match="escapes root"):
        resolve_under(root, "link/evil.json")
