"""τ³ official data must never reach a training or internal-selection path."""

from __future__ import annotations

import pytest

from tau3_grpo.data.manifest import ManifestEntry, official_tau3_manifest
from tau3_grpo.data.official import (
    OFFICIAL_AIRLINE_TASK_COUNT,
    SourceIsolationError,
    assert_trainable_entries,
    assert_trainable_source,
)
from tau3_grpo.data.schema import DataSource


def test_areal_source_is_trainable():
    assert_trainable_source(DataSource.AREAL_TAU2_AIRLINE, context="test")


def test_tau3_official_source_is_refused():
    with pytest.raises(SourceIsolationError, match="reserved for the frozen-winner"):
        assert_trainable_source(DataSource.TAU3_OFFICIAL_AIRLINE, context="test")


def test_tau3_official_string_is_refused():
    with pytest.raises(SourceIsolationError):
        assert_trainable_source("tau3_official_airline", context="test")


def test_final_split_entry_is_refused():
    entries = official_tau3_manifest([f"airline_task_{i}" for i in range(50)])
    with pytest.raises(SourceIsolationError):
        assert_trainable_entries(entries, context="test")


def test_train_entries_pass():
    entries = [
        ManifestEntry(
            task_id="airline_1",
            split="train",
            source=DataSource.AREAL_TAU2_AIRLINE,
            source_revision="rev",
            task_hash="hash",
        )
    ]
    assert_trainable_entries(entries, context="test")


def test_official_manifest_requires_exactly_50():
    with pytest.raises(ValueError, match="50 unique tasks"):
        official_tau3_manifest([f"t{i}" for i in range(49)])
    with pytest.raises(ValueError, match="50 unique tasks"):
        official_tau3_manifest([f"t{i}" for i in range(51)])


def test_official_manifest_rejects_duplicates():
    ids = [f"t{i}" for i in range(49)] + ["t0"]
    with pytest.raises(ValueError, match="50 unique tasks"):
        official_tau3_manifest(ids)


def test_official_manifest_shape():
    entries = official_tau3_manifest([f"t{i}" for i in range(OFFICIAL_AIRLINE_TASK_COUNT)])
    assert len(entries) == OFFICIAL_AIRLINE_TASK_COUNT
    assert all(entry.split == "tau3_final" for entry in entries)
    assert all(entry.source is DataSource.TAU3_OFFICIAL_AIRLINE for entry in entries)


@pytest.mark.tau3
def test_official_airline_split_has_50_tasks(requires_tau2):
    from tau3_grpo.data.official import load_official_airline_tasks

    tasks = load_official_airline_tasks()
    assert len(tasks) == OFFICIAL_AIRLINE_TASK_COUNT
    assert len({task.id for task in tasks}) == OFFICIAL_AIRLINE_TASK_COUNT


@pytest.mark.tau3
def test_official_manifest_from_live_tau2(requires_tau2):
    from tau3_grpo.data.official import build_official_manifest

    entries = build_official_manifest()
    assert len(entries) == OFFICIAL_AIRLINE_TASK_COUNT
