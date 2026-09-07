"""The experiment manifest must drive, not merely describe, training order."""

from __future__ import annotations

import json

import pytest

from tau3_grpo.cli.prepare_experiment import _load_split_hash, scheduled_entries
from tau3_grpo.data.manifest import ManifestEntry
from tau3_grpo.data.schema import DataSource
from tau3_grpo.experiment.manifest import create_manifest, flatten_schedule


def _entry(task_id: str) -> ManifestEntry:
    return ManifestEntry(
        task_id=task_id,
        split="train",
        source=DataSource.AREAL_TAU2_AIRLINE,
        source_revision="revision",
        task_hash=f"hash-{task_id}",
        task={"id": task_id},
    )


def test_scheduled_entries_follow_every_update_and_preserve_repeats():
    entries = [_entry("a"), _entry("b")]
    manifest = create_manifest(
        experiment="e0",
        arm="e0",
        seed=42,
        task_ids=[entry.task_id for entry in entries],
        train_split_hash="split",
        group_size=1,
        groups_per_update=3,
        total_updates=2,
    )
    expected = flatten_schedule(manifest.schedule)
    assert [entry.task_id for entry in scheduled_entries(entries, manifest)] == expected
    assert len(expected) == 6


def test_scheduled_entries_reject_duplicate_pool_ids():
    entries = [_entry("a"), _entry("a")]
    manifest = create_manifest(
        experiment="e0",
        arm="e0",
        seed=42,
        task_ids=["a"],
        train_split_hash="split",
        group_size=1,
        groups_per_update=1,
        total_updates=1,
    )
    with pytest.raises(ValueError, match="duplicate task ids"):
        scheduled_entries(entries, manifest)


def test_split_hash_sidecar_is_seed_bound(tmp_path):
    sidecar = tmp_path / "areal_airline_split_seed42.json"
    sidecar.write_text(
        json.dumps({"seed": 42, "split_hash": "split-42"}), encoding="utf-8"
    )
    assert _load_split_hash(tmp_path, 42) == "split-42"
    sidecar.write_text(
        json.dumps({"seed": 43, "split_hash": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="seed does not match"):
        _load_split_hash(tmp_path, 42)
