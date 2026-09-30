"""Fail-closed checks for a source-bound, reviewed SFT train/dev package.

This validates recorded evidence, never supplies a semantic review decision.
The caller must render the actual messages with the training dataset first.
"""

import json
from collections import Counter
from pathlib import Path

from tau3_grpo.models.qwen35_template import approved_assistant_indices
from tau3_grpo.utils.hashing import sha256_file, sha256_json

REVIEW_AREAS = {"scope", "evidence", "policy", "arithmetic", "completion"}


def audit_reviewed_package(train, validation, reviews, tokens, *, sizes=(500, 150)):
    """Reject missing provenance, stale approval/masks, and unresolved review areas."""
    if (len(train), len(validation)) != tuple(sizes):
        raise ValueError("Unexpected train/validation sizes")
    review_by_id = _unique_index(reviews, "review")
    token_by_id = _unique_index(tokens, "token")
    ids, users, messages = set(), {}, {}
    for split, rows in (("train", train), ("validation", validation)):
        users[split], messages[split] = set(), set()
        for row in rows:
            meta = row.get("metadata") or {}
            sid, uid = meta.get("source_dialog_id"), meta.get("source_user_id")
            if not isinstance(sid, str) or not sid or sid in ids:
                raise ValueError("Missing or duplicate sample identity")
            ids.add(sid)
            if not isinstance(uid, str) or not uid.strip():
                raise ValueError(f"Missing source user: {sid}")
            users[split].add(uid)
            if meta.get("split") != split or meta.get("quality_accepted") is not True:
                raise ValueError(f"Unapproved or incorrect split: {sid}")
            msgs, supervision = row.get("messages"), row.get("supervision")
            if not msgs or msgs[0].get("role") != "system":
                raise ValueError(f"Missing system-prefixed conversation: {sid}")
            if (not isinstance(supervision, dict)
                    or supervision.get("version") != "approved_assistant_v1"):
                raise ValueError(f"Missing approved supervision contract: {sid}")
            indices = supervision.get("message_indices")
            if not isinstance(indices, list):
                raise ValueError(f"Missing explicit target positions: {sid}")
            approved_assistant_indices(msgs, indices)
            identity = sha256_json([msgs, indices])
            conversation = sha256_json(msgs)
            if conversation in messages[split]:
                raise ValueError(f"Duplicate conversation: {sid}")
            messages[split].add(conversation)
            review, token = review_by_id.get(sid), token_by_id.get(sid)
            if not review or not token:
                raise ValueError(f"Missing review/token evidence: {sid}")
            for evidence in (review, token):
                if (evidence.get("split") != split
                        or evidence.get("messages_mask_sha256") != identity):
                    raise ValueError(f"Stale evidence identity: {sid}")
            if review.get("source_user_id") != uid:
                raise ValueError(f"Review user provenance differs: {sid}")
            if (review.get("decision") != "accepted_codex"
                    or review.get("reviewer") != "Codex"
                    or review.get("issues") or not review.get("evidence")):
                raise ValueError(f"Unresolved semantic review: {sid}")
            checks = review.get("checks") or {}
            if (set(checks) != REVIEW_AREAS
                    or any(v not in ("satisfied", "not_applicable") for v in checks.values())):
                raise ValueError(f"Incomplete semantic review: {sid}")
            if review.get("native", {}).get("passed") is not True:
                raise ValueError(f"Unverified native execution: {sid}")
            total, labelled = token.get("n_total_tokens"), token.get("n_label_tokens")
            maximum = token.get("max_length")
            if (type(total) is not int or type(labelled) is not int
                    or type(maximum) is not int
                    or not 0 < labelled <= total <= maximum
                    or token.get("native_tokens_unchanged") is not True
                    or token.get("ignore_nonassistant") is not True):
                raise ValueError(f"Missing/invalid rendered token evidence: {sid}")
    if ids != review_by_id.keys() or ids != token_by_id.keys():
        raise ValueError("Review/token inventory differs from package")
    if users["train"] & users["validation"]:
        raise ValueError("Train/validation source users overlap")
    if messages["train"] & messages["validation"]:
        raise ValueError("Train/validation conversations overlap")
    return {
        "ready_for_training": True,
        "train_count": len(train), "validation_count": len(validation),
        "train_unique_users": len(users["train"]),
        "validation_unique_users": len(users["validation"]),
        "user_overlap_count": 0, "conversation_overlap_count": 0,
        "train_curriculum": dict(Counter(r["metadata"]["curriculum_bucket"] for r in train)),
        "validation_curriculum": dict(Counter(
            r["metadata"]["curriculum_bucket"] for r in validation)),
    }


def _unique_index(rows, kind):
    index = {}
    for row in rows:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in index:
            raise ValueError(f"Missing/duplicate {kind} identity")
        index[sid] = row
    return index


def validate_frozen_package(manifest_path, *, root, train_path, validation_path):
    """Verify a frozen package and its selected cumulative stage before GPU setup."""
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("ready_for_training") is not True:
        raise ValueError("Reviewed SFT package is not ready for training")
    files = manifest["files"]
    for path, digest in files.items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"Frozen SFT file changed: {path}")
    selected = str(Path(train_path).resolve().relative_to(root.resolve()))
    validation = str(Path(validation_path).resolve().relative_to(root.resolve()))
    if selected not in manifest["stage_files"] or validation != manifest["validation_file"]:
        raise ValueError("Configured data does not belong to the frozen SFT package")

    def read_rows(path):
        return [json.loads(line) for line in (root / path).read_text().splitlines() if line]

    train = read_rows(manifest["train_file"])
    dev = read_rows(validation)
    reviews = json.loads((root / manifest["review_file"]).read_text())
    tokens = json.loads((root / manifest["token_file"]).read_text())
    for review in reviews:
        for evidence in review["evidence"]:
            if (not isinstance(evidence, dict)
                    or sha256_file(root / evidence["file"]) != evidence["sha256"]):
                raise ValueError("Frozen SFT review evidence changed")
    result = audit_reviewed_package(train, dev, reviews, tokens)
    if result != manifest["audit"]:
        raise ValueError("Frozen SFT audit differs from package contents")
    full_by_id = {row["metadata"]["source_dialog_id"]: row for row in train}
    stage = read_rows(selected)
    if (len(stage) != manifest["stage_files"][selected]
            or len({r["metadata"]["source_dialog_id"] for r in stage}) != len(stage)
            or any(full_by_id.get(r["metadata"]["source_dialog_id"]) != r for r in stage)):
        raise ValueError("Cumulative stage differs from the reviewed training package")
    return manifest


def validate_rendered_evidence(dataset, tokens):
    """Bind real training renders to the CPU-verified token IDs and loss masks."""
    by_id = _unique_index(tokens, "token")
    for record, example in zip(dataset.records, dataset.examples, strict=True):
        token = by_id[record["metadata"]["source_dialog_id"]]
        if (sha256_json(example["input_ids"]) != token["input_ids_sha256"]
                or sha256_json(example["labels"]) != token["labels_sha256"]):
            raise ValueError("Training tokenizer or loss mask differs from frozen evidence")
