"""Fail-closed checks for a source-bound, reviewed SFT train/dev package.

This validates recorded evidence, never supplies a semantic review decision.
The caller must render the actual messages with the training dataset first.
"""

import json
from collections import Counter
from pathlib import Path

from tau3_grpo.evaluation.rubric_contract import DIMENSIONS
from tau3_grpo.models.qwen35_template import approved_assistant_indices
from tau3_grpo.utils.hashing import sha256_file, sha256_json

REVIEW_AREAS = set(DIMENSIONS)
PORTABLE_SCHEMA = "codex_reviewed_sft_package_v2"


def package_path(directory, name):
    """Resolve a relative artifact without allowing traversal or escaping symlinks."""
    directory = Path(directory).resolve()
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Frozen package paths must stay inside the package")
    resolved = (directory / path).resolve()
    if not resolved.is_relative_to(directory) or resolved == directory:
        raise ValueError("Frozen package paths must stay inside the package")
    return resolved


def load_frozen_tokens(manifest_path, manifest, *, root):
    """Read the verified token inventory under the manifest's declared path scope."""
    directory = Path(manifest_path).resolve().parent if manifest.get("schema") == PORTABLE_SCHEMA else Path(root)
    path = package_path(directory, manifest["token_file"])
    if sha256_file(path) != manifest["files"][manifest["token_file"]]:
        raise ValueError("Frozen SFT token evidence changed")
    return json.loads(path.read_text())


def _portable_source_binding(manifest, directory, reviews):
    """Check the exact accepted ledger inherited at migration, without live archives.

    Historical paths in the source manifest are provenance identifiers only.
    The original source files were verified by the exporter before freezing v2;
    detailed replay inputs are no longer a prerequisite for gradient training.
    """
    source = manifest["source_package"]
    snapshot_path = package_path(directory, source["manifest_file"])
    if sha256_file(snapshot_path) != source["manifest_sha256"]:
        raise ValueError("Frozen source manifest identity changed")
    previous = json.loads(snapshot_path.read_text())
    if previous.get("ready_for_training") is not True or previous["audit"] != manifest["audit"]:
        raise ValueError("Portable package differs from its accepted source audit")
    bindings = {
        manifest["train_file"]: previous["train_file"],
        manifest["validation_file"]: previous["validation_file"],
        manifest["review_file"]: previous["review_file"],
        manifest["token_file"]: previous["token_file"],
    }
    # Check every binding independently: auxiliary or stage mappings must never
    # replace the required identities of the final data and accepted ledgers.
    pairs = list(bindings.items()) + list(source["artifact_bindings"].items())
    if set(source["stage_bindings"]) != set(manifest["stage_files"]):
        raise ValueError("Portable stage inventory differs from source")
    for name, original in source["stage_bindings"].items():
        if previous["stage_files"].get(original) != manifest["stage_files"][name]:
            raise ValueError("Portable stage budget differs from source")
        pairs.append((name, original))
    for name, original in pairs:
        if (name not in manifest["files"] or original not in previous["files"]
                or manifest["files"][name] != previous["files"][original]):
            raise ValueError("Portable SFT artifact differs from its frozen source")
    for review in reviews:
        for evidence in review["evidence"]:
            if (not isinstance(evidence, dict) or not evidence.get("sha256")
                    or previous["files"].get(evidence.get("file")) != evidence["sha256"]):
                raise ValueError("Unbound inherited SFT review evidence")


def _validate_portable_package(manifest_path, manifest, train_path, validation_path):
    directory = Path(manifest_path).resolve().parent
    if manifest.get("path_scope") != "package":
        raise ValueError("Portable SFT manifest requires package-relative paths")
    required = {manifest[key] for key in (
        "train_file", "validation_file", "review_file", "token_file", "tool_config_file",
    )} | set(manifest["stage_files"]) | {manifest["source_package"]["manifest_file"]}
    if not required <= manifest["files"].keys():
        raise ValueError("Missing protected portable package artifact")
    for name, digest in manifest["files"].items():
        if sha256_file(package_path(directory, name)) != digest:
            raise ValueError(f"Frozen SFT file changed: {name}")
    stages = {package_path(directory, name): name for name in manifest["stage_files"]}
    selected = stages.get(Path(train_path).resolve())
    if selected is None or Path(validation_path).resolve() != package_path(directory, manifest["validation_file"]):
        raise ValueError("Configured data does not belong to the frozen SFT package")

    def read_rows(name):
        return [json.loads(line) for line in package_path(directory, name).read_text().splitlines() if line]

    train, dev = read_rows(manifest["train_file"]), read_rows(manifest["validation_file"])
    reviews = json.loads(package_path(directory, manifest["review_file"]).read_text())
    tokens = load_frozen_tokens(manifest_path, manifest, root=directory)
    _portable_source_binding(manifest, directory, reviews)
    if audit_reviewed_package(train, dev, reviews, tokens) != manifest["audit"]:
        raise ValueError("Frozen SFT audit differs from package contents")
    full = {row["metadata"]["source_dialog_id"]: row for row in train}
    stage = read_rows(selected)
    if (len(stage) != manifest["stage_files"][selected]
            or len({r["metadata"]["source_dialog_id"] for r in stage}) != len(stage)
            or any(full.get(r["metadata"]["source_dialog_id"]) != r for r in stage)):
        raise ValueError("Cumulative stage differs from the reviewed training package")
    return manifest


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
    if manifest.get("schema") == PORTABLE_SCHEMA:
        return _validate_portable_package(manifest_path, manifest, train_path, validation_path)
    if manifest.get("schema") not in (None, "codex_reviewed_sft_package_v1"):
        raise ValueError("Unknown reviewed SFT package schema")
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
