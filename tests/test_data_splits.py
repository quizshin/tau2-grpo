"""Real-data counts, the frozen 200/60/888 split, and manifest hashing."""

from __future__ import annotations

import json

import pytest

from tau3_grpo.data.dataset import (
    EXPECTED_AIRLINE_RECORDS,
    EXPECTED_RESERVE,
    EXPECTED_SELECTION,
    EXPECTED_TOTAL_RECORDS,
    EXPECTED_TRAIN,
    assert_split_sizes,
    build_manifests,
    load_and_validate,
    write_split_manifests,
)
from tau3_grpo.data.manifest import build_airline_splits, read_manifest
from tau3_grpo.data.schema import ArealTaskRecord


def test_real_jsonl_has_pinned_counts(areal_jsonl):
    records, stats = load_and_validate(areal_jsonl)
    assert stats.total == EXPECTED_TOTAL_RECORDS
    assert stats.airline == EXPECTED_AIRLINE_RECORDS
    assert stats.unique_ids == stats.total
    assert len(records) == EXPECTED_TOTAL_RECORDS


def test_split_is_exactly_200_60_888(areal_jsonl):
    manifest = build_manifests(areal_jsonl, seed=42, require_db=False)
    assert len(manifest.train) == EXPECTED_TRAIN
    assert len(manifest.selection) == EXPECTED_SELECTION
    assert len(manifest.reserve) == EXPECTED_RESERVE
    assert_split_sizes(manifest)


def test_splits_are_disjoint(areal_jsonl):
    manifest = build_manifests(areal_jsonl, seed=42, require_db=False)
    train = {entry.task_id for entry in manifest.train}
    selection = {entry.task_id for entry in manifest.selection}
    reserve = {entry.task_id for entry in manifest.reserve}
    assert not train & selection
    assert not train & reserve
    assert not selection & reserve
    assert len(train | selection | reserve) == EXPECTED_AIRLINE_RECORDS


def test_split_is_deterministic_across_calls(areal_jsonl):
    first = build_manifests(areal_jsonl, seed=42, require_db=False)
    second = build_manifests(areal_jsonl, seed=42, require_db=False)
    assert first.split_hash == second.split_hash
    assert [e.task_id for e in first.train] == [e.task_id for e in second.train]


def test_different_seed_changes_split(areal_jsonl):
    seed42 = build_manifests(areal_jsonl, seed=42, require_db=False)
    seed43 = build_manifests(areal_jsonl, seed=43, require_db=False)
    assert seed42.split_hash != seed43.split_hash
    assert [e.task_id for e in seed42.train] != [e.task_id for e in seed43.train]


def test_manifest_roundtrip_preserves_hash(areal_jsonl, tmp_path):
    manifest = build_manifests(areal_jsonl, seed=42, require_db=False)
    written = write_split_manifests(manifest, tmp_path)
    reloaded = read_manifest(written["train"])
    assert len(reloaded) == EXPECTED_TRAIN
    assert [e.task_id for e in reloaded] == [e.task_id for e in manifest.train]
    assert [e.task_hash for e in reloaded] == [e.task_hash for e in manifest.train]

    sidecar = json.loads(written["split_hash"].read_text(encoding="utf-8"))
    assert sidecar["split_hash"] == manifest.split_hash
    assert sidecar["train"] == EXPECTED_TRAIN
    assert sidecar["reserve"] == EXPECTED_RESERVE


def _record(task_id: str, db: str = "tau2_rl_database/db.json") -> ArealTaskRecord:
    return ArealTaskRecord(
        id=task_id,
        db_path=db,
        user_scenario={"instructions": {"domain": "airline", "task_instructions": "x"}},
        evaluation_criteria=json.dumps({"actions": [], "communicate_info": []}),
    )


def test_wrong_airline_count_is_rejected():
    records = [_record(f"airline_{i}") for i in range(10)]
    with pytest.raises(ValueError, match="expected 1148 Airline records"):
        build_airline_splits(records, seed=42, expected_airline_count=1148)


def test_duplicate_ids_are_rejected():
    records = [_record("airline_1"), _record("airline_1")]
    with pytest.raises(ValueError, match="not unique"):
        build_airline_splits(records, seed=42, expected_airline_count=2)


def test_evaluation_criteria_json_string_is_parsed():
    record = _record("airline_1")
    assert isinstance(record.evaluation_criteria, dict)
    assert "actions" in record.evaluation_criteria


def test_task_hash_is_stable_and_id_sensitive():
    left = _record("airline_1")
    right = _record("airline_1")
    other = _record("airline_2")
    assert left.fingerprint == right.fingerprint
    assert left.fingerprint != other.fingerprint
