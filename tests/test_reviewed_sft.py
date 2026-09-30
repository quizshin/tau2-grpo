"""Regressions for the false-ready 500/150 package discovered on 2026-09-30."""

from copy import deepcopy

import pytest

from tau3_grpo.data.reviewed_sft import REVIEW_AREAS, audit_reviewed_package
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
