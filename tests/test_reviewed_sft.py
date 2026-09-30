"""Regressions for the false-ready 500/150 package discovered on 2026-09-30."""

from copy import deepcopy

import pytest

from tau3_grpo.data.reviewed_sft import (
    REVIEW_AREAS,
    audit_reviewed_package,
    validate_frozen_package,
    validate_rendered_evidence,
)
from tau3_grpo.utils.hashing import sha256_json


def package():
    rows, reviews, tokens = [], [], []
    for split in ("train", "validation"):
        row = {
            "messages": [{"role": "system", "content": "policy"},
                         {"role": "user", "content": split},
                         {"role": "assistant", "content": "Verified result"}],
            "metadata": {"source_dialog_id": split, "source_user_id": split,
                         "split": split, "quality_accepted": True,
                         "curriculum_bucket": "foundation"},
            "supervision": {"version": "approved_assistant_v1", "message_indices": [2]},
        }
        identity = sha256_json([row["messages"], [2]])
        common = {"sample_id": split, "split": split, "messages_mask_sha256": identity}
        reviews.append({**common, "source_user_id": split, "decision": "accepted_codex",
                        "reviewer": "Codex", "issues": [], "evidence": ["source-bound review"],
                        "checks": dict.fromkeys(REVIEW_AREAS, "satisfied"),
                        "native": {"passed": True}})
        tokens.append({**common, "n_total_tokens": 10, "n_label_tokens": 3,
                       "max_length": 100, "native_tokens_unchanged": True,
                       "ignore_nonassistant": True})
        rows.append(row)
    return rows[:1], rows[1:], reviews, tokens


def check(parts):
    return audit_reviewed_package(*parts, sizes=(1, 1))


def test_missing_validation_user_cannot_be_counted_as_zero_overlap():
    parts = package()
    del parts[1][0]["metadata"]["source_user_id"]
    with pytest.raises(ValueError, match="Missing source user"):
        check(parts)


def test_nonempty_supervision_is_not_a_valid_contract():
    parts = package()
    del parts[1][0]["supervision"]["version"]
    with pytest.raises(ValueError, match="approved supervision"):
        check(parts)


def test_declared_acceptance_cannot_bypass_stale_identity():
    parts = package()
    parts[1][0]["messages"][2]["content"] = "Different unreviewed answer"
    with pytest.raises(ValueError, match="Stale evidence"):
        check(parts)


def test_actual_user_overlap_blocks_ready_even_with_all_accept_labels():
    parts = package()
    parts[1][0]["metadata"]["source_user_id"] = "train"
    parts[2][1]["source_user_id"] = "train"
    with pytest.raises(ValueError, match="source users overlap"):
        check(parts)


@pytest.mark.parametrize("kind", ["token_count", "max_length", "semantic_check", "native", "extra_review"])
def test_incomplete_evidence_blocks_freeze(kind):
    parts = package()
    if kind == "token_count":
        parts[3][1]["n_total_tokens"] = None
    elif kind == "max_length":
        parts[3][1]["max_length"] = None
    elif kind == "semantic_check":
        parts[2][1]["checks"]["completion"] = "unknown"
    elif kind == "native":
        parts[2][1]["native"]["passed"] = False
    else:
        parts[2].append({**deepcopy(parts[2][1]), "sample_id": "unused"})
    with pytest.raises(ValueError):
        check(parts)


def test_complete_bound_package_can_be_frozen():
    result = check(package())
    assert result["ready_for_training"] is True
    assert result["validation_unique_users"] == 1
    assert result["user_overlap_count"] == 0


def test_not_ready_manifest_fails_before_any_model_load(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"ready_for_training": false}')
    with pytest.raises(ValueError, match="not ready"):
        validate_frozen_package(manifest, root=tmp_path,
                                train_path=tmp_path / "train", validation_path=tmp_path / "dev")


def test_changed_package_file_fails_before_trusting_review_labels(tmp_path):
    import json

    from tau3_grpo.utils.hashing import sha256_file

    data = tmp_path / "train.jsonl"
    data.write_text("original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"ready_for_training": True,
                                    "files": {data.name: sha256_file(data)}}))
    data.write_text("tampered")
    with pytest.raises(ValueError, match="Frozen SFT file changed"):
        validate_frozen_package(manifest, root=tmp_path,
                                train_path=data, validation_path=tmp_path / "dev")


@pytest.mark.parametrize("name,size,steps,previous", [
    ("A109", 109, 14, None), ("B393", 393, 50, "A109"), ("C500", 500, 63, "B393"),
])
def test_cumulative_profiles_preserve_update_budget_and_initialization(name, size, steps, previous):
    from pathlib import Path

    import yaml

    from tau3_grpo.training.sft.train import training_schedule

    config = yaml.safe_load(Path(f"configs/train/sft/curriculum_codex_{name}_1epoch.yaml").read_text())
    assert config["data"]["expected_train_size"] == size
    assert config["data"]["expected_validation_size"] == 150
    assert config["data"]["reviewed_package_manifest"]
    assert config["data"]["require_approved_targets"] is True
    assert training_schedule(config["train"], size, 1, 8) == (1, -1, steps)
    assert config["model"].get("requires_previous_stage") == previous
    if previous:
        assert config["model"]["name_or_path"] is None
        assert "SFT_MODEL_NAME_OR_PATH" not in config["launch"]["environment"]


def test_changed_training_tokens_fail_even_if_counts_match():
    from types import SimpleNamespace

    record = {"metadata": {"source_dialog_id": "sample"}}
    example = {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]}
    tokens = [{"sample_id": "sample", "input_ids_sha256": sha256_json([1, 2, 3]),
               "labels_sha256": sha256_json([-100, -100, 3])}]
    dataset = SimpleNamespace(records=[record], examples=[example])
    validate_rendered_evidence(dataset, tokens)
    example["labels"] = [-100, 2, -100]
    with pytest.raises(ValueError, match="loss mask"):
        validate_rendered_evidence(dataset, tokens)
