"""Leakage audit between train and internal selection."""

from __future__ import annotations

from tau3_grpo.data.dataset import build_manifests
from tau3_grpo.data.leakage import audit_exact, normalized_intent
from tau3_grpo.data.manifest import ManifestEntry
from tau3_grpo.data.schema import DataSource


def _entry(task_id: str, *, task_hash: str = "h", db_hash: str | None = None, intent: str = "x"):
    return ManifestEntry(
        task_id=task_id,
        split="train",
        source=DataSource.AREAL_TAU2_AIRLINE,
        source_revision="rev",
        task_hash=task_hash,
        db_hash=db_hash,
        task={
            "user_scenario": {
                "instructions": {
                    "domain": "airline",
                    "reason_for_call": intent,
                    "task_instructions": intent,
                }
            }
        },
    )


def test_identical_task_id_is_flagged():
    findings = audit_exact([_entry("a")], [_entry("a")])
    assert any(finding.reason == "task_id" for finding in findings)


def test_identical_task_hash_is_flagged():
    findings = audit_exact([_entry("a", task_hash="same")], [_entry("b", task_hash="same")])
    assert any(finding.reason == "task_hash" for finding in findings)


def test_identical_db_hash_is_allowed_for_shared_base_template():
    findings = audit_exact(
        [_entry("a", task_hash="left", db_hash="same", intent="change flight")],
        [_entry("b", task_hash="right", db_hash="same", intent="cancel booking")],
    )
    assert findings == []


def test_identical_normalized_intent_is_flagged():
    findings = audit_exact(
        [_entry("a", task_hash="h1", intent="change my flight")],
        [_entry("b", task_hash="h2", intent="change my flight")],
    )
    assert any(finding.reason == "normalized_intent" for finding in findings)


def test_disjoint_entries_are_clean():
    findings = audit_exact(
        [_entry("a", task_hash="h1", db_hash="d1", intent="change flight")],
        [_entry("b", task_hash="h2", db_hash="d2", intent="cancel booking")],
    )
    assert findings == []


def test_normalized_intent_masks_ids_and_dates():
    left = normalized_intent(_entry("a", intent="change HKEG34 on 2024-05-27"))
    right = normalized_intent(_entry("b", intent="change ABCD99 on 2024-06-01"))
    assert left == right
    assert "<ENTITY>" in left


def test_normalized_intent_keeps_real_differences():
    left = normalized_intent(_entry("a", intent="change my flight"))
    right = normalized_intent(_entry("b", intent="cancel my reservation"))
    assert left != right


def test_real_train_selection_split_has_no_exact_leakage(areal_jsonl):
    manifest = build_manifests(areal_jsonl, seed=42, require_db=False)
    findings = audit_exact(manifest.train, manifest.selection)
    id_or_hash = [
        finding for finding in findings if finding.reason in {"task_id", "task_hash"}
    ]
    assert id_or_hash == [], f"exact leakage between train and selection: {id_or_hash[:5]}"


def test_real_train_reserve_split_has_no_exact_leakage(areal_jsonl):
    manifest = build_manifests(areal_jsonl, seed=42, require_db=False)
    findings = audit_exact(manifest.train, manifest.reserve)
    id_or_hash = [
        finding for finding in findings if finding.reason in {"task_id", "task_hash"}
    ]
    assert id_or_hash == []
