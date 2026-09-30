"""Fail-closed checks for a source-bound, reviewed SFT train/dev package.

This validates recorded evidence, never supplies a semantic review decision.
The caller must render the actual messages with the training dataset first.
"""

from collections import Counter

from tau3_grpo.models.qwen35_template import approved_assistant_indices
from tau3_grpo.utils.hashing import sha256_json

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
