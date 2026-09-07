"""veRL parquet rows: real interaction_kwargs, policy in the system prompt."""

from __future__ import annotations

import json

import pytest

from tau3_grpo.data.manifest import ManifestEntry, official_tau3_manifest
from tau3_grpo.data.official import SourceIsolationError
from tau3_grpo.data.parquet_builder import (
    INTERACTION_NAME,
    build_row,
    build_rows,
    build_system_prompt,
)
from tau3_grpo.data.schema import DataSource

POLICY = "1. Always identify the user first.\n2. Never invent reservations."


def _entry(task_id: str = "airline_1") -> ManifestEntry:
    return ManifestEntry(
        task_id=task_id,
        split="train",
        source=DataSource.AREAL_TAU2_AIRLINE,
        source_revision="rev",
        task_hash="hash",
        db_path="tau2_rl_database/db.json",
        db_hash="dbhash",
        task={
            "id": task_id,
            "user_scenario": {
                "instructions": {
                    "domain": "airline",
                    "reason_for_call": "I want to change my flight.",
                    "task_instructions": "Change the flight.",
                }
            },
        },
    )


def test_system_prompt_embeds_policy_verbatim():
    prompt = build_system_prompt(POLICY)
    assert "Always identify the user first." in prompt
    assert "Never invent reservations." in prompt


def test_empty_policy_is_refused():
    with pytest.raises(ValueError, match="policy text is empty"):
        build_system_prompt("   ")


def test_row_has_real_interaction_kwargs():
    row = build_row(_entry(), policy=POLICY, split="train", seed=42)
    kwargs = row["extra_info"]["interaction_kwargs"]
    assert kwargs["name"] == INTERACTION_NAME
    assert kwargs["task_id"] == "airline_1"
    assert kwargs["source"] == DataSource.AREAL_TAU2_AIRLINE.value
    assert kwargs["anchor_mode"] == "structured"
    assert kwargs["seed"] == 42
    assert json.loads(kwargs["record_json"]) == _entry().task


def test_row_prompt_starts_with_system_then_user():
    row = build_row(_entry(), policy=POLICY, split="train")
    roles = [message["role"] for message in row["prompt"]]
    assert roles == ["system", "user"]
    assert "Always identify the user first." in row["prompt"][0]["content"]


def test_row_uses_reason_for_call_as_opening():
    row = build_row(_entry(), policy=POLICY, split="train")
    assert row["prompt"][1]["content"] == "I want to change my flight."
    assert (
        row["extra_info"]["interaction_kwargs"]["initial_user_message"]
        == row["prompt"][1]["content"]
    )


def test_row_falls_back_to_generic_opening():
    entry = _entry()
    entry.task["user_scenario"]["instructions"]["reason_for_call"] = ""
    row = build_row(entry, policy=POLICY, split="train")
    assert row["prompt"][1]["content"].startswith("Hello")


def test_row_carries_provenance():
    row = build_row(_entry(), policy=POLICY, split="train")
    info = row["extra_info"]
    assert info["task_hash"] == "hash"
    assert info["db_hash"] == "dbhash"
    assert info["source_revision"] == "rev"
    assert info["split"] == "train"
    assert info["need_tools_kwargs"] is False
    assert "tools_kwargs" not in info


def test_anchor_mode_is_threaded():
    row = build_row(_entry(), policy=POLICY, split="train", anchor_mode="db_hash_only")
    assert row["extra_info"]["interaction_kwargs"]["anchor_mode"] == "db_hash_only"


def test_build_rows_refuses_tau3_official():
    entries = official_tau3_manifest([f"t{i}" for i in range(50)])
    with pytest.raises(SourceIsolationError):
        build_rows(entries, policy=POLICY, split="tau3_final")


def test_build_rows_accepts_train_entries():
    rows = build_rows([_entry("a"), _entry("b")], policy=POLICY, split="train")
    assert len(rows) == 2
    assert {row["extra_info"]["task_id"] for row in rows} == {"a", "b"}


def test_seed_omitted_when_none():
    row = build_row(_entry(), policy=POLICY, split="train", seed=None)
    assert "seed" not in row["extra_info"]["interaction_kwargs"]
