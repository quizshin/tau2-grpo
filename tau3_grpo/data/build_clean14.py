"""Rebuild frozen 96/100-dialogue recipes with fail-closed, whole-dataset checks.

Historical JSONL bytes are preserved. Validation evidence is a new sidecar, not
retroactive claims about policy compliance or fresh environment replay.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from tau3_grpo.data.sft import load_complete_airline_dialogues, reason_similarity
from tau3_grpo.data.sft_expansion import audit_tool_calls, coverage, tool_names
from tau3_grpo.prompts import prepare_agent_messages, prompt_provenance
from tau3_grpo.training.sft.dataset import build_supervised_example
from tau3_grpo.utils.hashing import sha256_file, sha256_text

VERSION = "clean14_builder_v2"
BAD_IDS = {"airline_dialog_837", "airline_dialog_357", "airline_dialog_669", "airline_dialog_172"}
INPUT_NAMES = ("source", "anchors", "validation", "supplemental", "blocked_reasons", "tool_config", "train_manifest")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "vocab.json", "merges.txt")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _source_quality(path):
    good, patterns = {}, {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            meta = json.loads(line).get("metadata") or {}
            sid = str(meta.get("source_dialog_id") or "")
            if not sid.startswith("airline_dialog_"):
                continue
            good[sid] = good.get(sid, True) and meta.get("correct") == 1 and meta.get("reward") == 1.0
            pattern = str(meta.get("seed_pattern_task_id") or "")
            if sid in patterns and patterns[sid] != pattern:
                raise ValueError(f"Inconsistent source pattern: {sid}")
            patterns[sid] = pattern
    return good, patterns


def _category(pattern):
    if pattern.endswith("_purpose_01_pos"):
        return "single_pos"
    if pattern.endswith("_purpose_01_neg"):
        return "single_neg"
    if pattern in {"scenario_pos_pos", "scenario_pos_neg", "scenario_neg_pos"}:
        return "double"
    if pattern.startswith("scenario_"):
        return "triple"
    raise ValueError(f"Unknown source task pattern: {pattern}")


def _sid(record):
    value = (record.get("metadata") or {}).get("source_dialog_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Missing source dialogue identity")
    return value


def validate_identities(train, validation):
    train_ids, val_ids = [_sid(r) for r in train], [_sid(r) for r in validation]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicate dialogue identities")
    if set(train_ids) & set(val_ids):
        raise ValueError("Train/validation dialogue overlap")
    if set(train_ids) & BAD_IDS:
        raise ValueError("Known bad dialogue in training")


def validate_supplemental(record, entries):
    """Bind existing execution receipts to messages and a frozen train task.

    This verifies recorded evidence; it does not re-execute tools or claim an
    independent task-success/policy judge. Input recipe hashes prevent drift.
    """
    meta = record.get("metadata") or {}
    if (meta.get("source") != "areal_rl_train_authored_executed"
            or meta.get("verification") != "tool_execution_and_reference_final_db_equality"
            or meta.get("independent_task_success_evaluation") is not False):
        raise ValueError("Supplemental requires explicit recorded execution provenance")
    entry = entries.get(meta.get("source_task_id"))
    if not entry or entry.get("split") != "train":
        raise ValueError("Supplemental task is not in the frozen training manifest")
    for name, key in (("source_task_hash", "task_hash"), ("source_db_hash", "db_hash"), ("source_revision", "source_revision")):
        if not entry.get(key) or meta.get(name) != entry[key]:
            raise ValueError(f"Supplemental identity mismatch: {name}")
    if _sid(record) != "authored_" + entry["task_id"]:
        raise ValueError("Supplemental dialogue/task identity mismatch")
    messages = record.get("messages", [])
    if meta.get("dialogue_hash") != sha256_text(json.dumps(messages, sort_keys=True)):
        raise ValueError("Supplemental dialogue hash mismatch")
    if not isinstance(meta.get("final_db_hash"), str) or len(meta["final_db_hash"]) != 64:
        raise ValueError("Missing supplemental final DB receipt")
    calls, responses = [], []
    for message in messages:
        if message.get("role") == "assistant":
            calls.extend(c.get("function", c) for c in message.get("tool_calls", []))
        elif message.get("role") == "tool":
            responses.append(message)
    executions = meta.get("executions") or []
    if not calls or not len(calls) == len(responses) == len(executions):
        raise ValueError("Supplemental execution/response count mismatch")
    for call, response, receipt in zip(calls, responses, executions, strict=True):
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if (call.get("name") != receipt.get("name") or response.get("name") != receipt.get("name")
                or arguments != receipt.get("arguments")
                or receipt.get("response_sha256") != sha256_text(response.get("content", ""))):
            raise ValueError("Supplemental execution receipt does not match actual messages")
    return "recorded_execution_receipts_bound_to_training_task_and_messages"


def validate_record(record, *, role, good, source_records, entries, blocked,
                    schemas, tokenizer, max_length, similarity_threshold):
    sid, meta = _sid(record), record.get("metadata") or {}
    messages = record.get("messages")
    if (not isinstance(messages, list) or not messages or messages[0].get("role") != "system"
            or messages[-1].get("role") != "assistant"
            or not any(m.get("role") == "user" for m in messages)):
        raise ValueError(f"{sid}: not a complete system/user/assistant dialogue")
    reason = meta.get("reason_for_call")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{sid}: missing intent")
    if meta.get("source") == "areal_tau2_airline_sft":
        if not good.get(sid, False):
            raise ValueError(f"{sid}: source quality missing or correct/reward not all 1")
        original = source_records.get(sid)
        if original is None or prepare_agent_messages(messages) != original["messages"]:
            raise ValueError(f"{sid}: source messages were changed")
        if reason != original["metadata"]["reason_for_call"]:
            raise ValueError(f"{sid}: source intent was changed")
        evidence = "all_source_turns_correct_1_reward_1_and_source_message_identity"
    elif role != "validation":
        evidence = validate_supplemental(record, entries)
    else:
        raise ValueError(f"{sid}: unsupported validation source")
    issues = audit_tool_calls(record, schemas)
    unknown = set(tool_names(record)) - {s["function"]["name"] for s in schemas}
    if issues or unknown:
        raise ValueError(f"{sid}: tool schema/response validation: {issues}; unknown={sorted(unknown)}")
    overlap = max((reason_similarity(reason, text) for text in blocked), default=0.0)
    if role != "validation" and overlap >= similarity_threshold:
        raise ValueError(f"{sid}: heldout lexical overlap {overlap:.4f}")
    rendered = build_supervised_example(prepare_agent_messages(messages), tokenizer,
                                        tools=schemas, max_length=max_length)
    labels, ids = rendered["labels"], rendered["input_ids"]
    n_labels = sum(label != -100 for label in labels)
    if (not len(labels) == len(ids) == rendered["n_total_tokens"] or not n_labels
            or n_labels != rendered["n_label_tokens"] or len(ids) > max_length
            or any(label != -100 and label != token for label, token in zip(labels, ids, strict=True))):
        raise ValueError(f"{sid}: invalid assistant-only render/mask")
    return {"id": sid, "role": role, "source_quality": evidence,
            "heldout_max_lexical_similarity": overlap if role != "validation" else None,
            "rendered_tokens": len(ids), "label_tokens": n_labels,
            "semantic_policy_validation": "not_performed"}


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if recipe.get("schema") != "clean14_frozen_recipe_v1":
        raise ValueError("Unknown clean14 recipe schema")
    if recipe.get("selection") != "frozen_source_ids":
        raise ValueError("Recipe must explicitly freeze source IDs")
    if set(recipe.get("inputs_sha256", {})) != set(INPUT_NAMES):
        raise ValueError("Recipe must bind every input")
    if set(recipe.get("tokenizer_sha256", {})) != set(TOKENIZER_FILES):
        raise ValueError("Recipe must bind tokenizer and template")
    ids = recipe.get("selected_areal_ids", [])
    if len(ids) != len(set(ids)) or not ids or any(not isinstance(x, str) for x in ids):
        raise ValueError("Invalid frozen selection IDs")
    counts = recipe.get("additional_category_counts", {})
    if set(counts) != {"single_pos", "single_neg", "double", "triple"} or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("Invalid category quotas")
    if len(ids) != recipe["anchor_count"] + sum(counts.values()):
        raise ValueError("Recipe category counts disagree with frozen selection")
    if recipe["train_count"] != len(ids) + recipe["supplemental_count"]:
        raise ValueError("Recipe training count mismatch")
    if not 0 < recipe["similarity_threshold"] <= 1 or recipe["max_length"] <= 0:
        raise ValueError("Invalid validation limits")
    return recipe


def verify_inputs(recipe, paths, tokenizer_path):
    actual = {name: sha256_file(paths[name]) for name in INPUT_NAMES}
    if actual != recipe["inputs_sha256"]:
        changed = sorted(k for k in actual if actual[k] != recipe["inputs_sha256"][k])
        raise ValueError(f"Frozen input identity changed: {changed}")
    tokenizer_hashes = {name: sha256_file(Path(tokenizer_path) / name) for name in TOKENIZER_FILES}
    if tokenizer_hashes != recipe["tokenizer_sha256"]:
        raise ValueError("Tokenizer/template identity changed")
    if prompt_provenance() != recipe["prompt_provenance"]:
        raise ValueError("Tool/system prompt protocol changed; create a new recipe")
    return actual, tokenizer_hashes


def build(recipe_path, paths, tokenizer_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite data or historical audit: {output}")
    recipe = load_recipe(recipe_path)
    input_hashes, tokenizer_hashes = verify_inputs(recipe, paths, tokenizer_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    schemas = [x["tool_schema"] for x in yaml.safe_load(Path(paths["tool_config"]).read_text())["tools"]]
    required = {s["function"]["name"] for s in schemas}
    if len(required) != 14:
        raise ValueError("Expected 14 distinct runtime tools")
    good, patterns = _source_quality(paths["source"])
    dialogues, stats = load_complete_airline_dialogues(paths["source"], strict_counts=True)
    source_records = {d.source_dialog_id: d.to_record() for d in dialogues}
    anchors = [r for r in _read_jsonl(paths["anchors"]) if _sid(r) not in BAD_IDS]
    validation, supplemental = _read_jsonl(paths["validation"]), _read_jsonl(paths["supplemental"])
    if (len(anchors), len(validation), len(supplemental)) != (recipe["anchor_count"], recipe["validation_count"], recipe["supplemental_count"]):
        raise ValueError("Recipe input counts changed")
    entries_list = _read_jsonl(paths["train_manifest"])
    entries = {r["task_id"]: r for r in entries_list}
    if len(entries) != len(entries_list):
        raise ValueError("Duplicate training task identity")
    blocked = json.loads(Path(paths["blocked_reasons"]).read_text())
    if not isinstance(blocked, list) or any(not isinstance(x, str) for x in blocked):
        raise ValueError("Blocked intents must be a list of strings")
    blocked += [r["metadata"]["reason_for_call"] for r in validation]
    by_id = {_sid(r): r for r in anchors}
    if recipe["selected_areal_ids"][:len(anchors)] != [_sid(r) for r in anchors]:
        raise ValueError("Frozen anchor order/identity changed")
    selected = []
    for sid in recipe["selected_areal_ids"]:
        if sid not in source_records or not good.get(sid, False):
            raise ValueError(f"Unknown or failed source dialogue: {sid}")
        record = by_id.get(sid, source_records[sid])
        category = _category(patterns.get(sid, ""))
        if sid in by_id:
            record["metadata"]["selection_category"] = category
        else:
            # Preserve historical JSON field insertion order as well as content.
            record["metadata"].update(seed_pattern_task_id=patterns[sid],
                source_quality="all_source_turns_correct_1_reward_1", selection_category=category)
        selected.append(record)
    train = selected + supplemental
    validate_identities(train, validation)
    audits = []
    for role, rows in (("train", train), ("validation", validation)):
        for record in rows:
            checked = validate_record(record, role=role, good=good, source_records=source_records,
                entries=entries, blocked=blocked, schemas=schemas, tokenizer=tokenizer,
                max_length=recipe["max_length"], similarity_threshold=recipe["similarity_threshold"])
            audits.append(checked)
            if role == "train" and _sid(record) not in by_id and record["metadata"]["source"] == "areal_tau2_airline_sft":
                record["metadata"].update(rendered_tokens=checked["rendered_tokens"], label_tokens=checked["label_tokens"])
    categories = dict(Counter(r["metadata"]["selection_category"] for r in selected[len(anchors):]))
    if categories != recipe["additional_category_counts"] or len(train) != recipe["train_count"]:
        raise ValueError("Final dataset does not satisfy recipe quotas")
    if set(coverage(train)) != required:
        raise ValueError("Final dataset does not cover exactly the runtime tool set")
    # Recheck every input after the long render pass before publishing.
    verify_inputs(recipe, paths, tokenizer_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, rows in (("clean14_v1_train.jsonl", train), ("clean14_v1_validation.jsonl", validation)):
            (staging / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        hashes = {n: sha256_file(staging / n) for n in recipe["outputs_sha256"]}
        if hashes != recipe["outputs_sha256"]:
            raise ValueError("Frozen output bytes changed; do not overwrite historical data")
        audit = {"schema": VERSION, "recipe_id": recipe["id"], "recipe_sha256": sha256_file(recipe_path),
                 "builder_sha256": sha256_file(__file__), "inputs_sha256": input_hashes,
                 "tokenizer_sha256": tokenizer_hashes, "prompt_provenance": prompt_provenance(),
                 "source_counts": vars(stats), "outputs_sha256": hashes,
                 "total_train_count": len(train), "validation_count": len(validation),
                 "tool_dialogue_coverage": coverage(train), "rows": audits,
                 "status": "all_rows_source_schema_split_and_mask_verified",
                 "limitations": ["Lexical decontamination only; no semantic leakage guarantee.",
                                 "No new independent policy judge or environment replay.",
                                 "Source quality labels apply to supplied source rows, not every historical context turn.",
                                 "Source success labels and tool coverage do not guarantee SFT improvement."]}
        (staging / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        for name, expected in hashes.items():
            if sha256_file(staging / name) != expected:
                raise ValueError("Written dataset verification failed")
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"recipe_id": recipe["id"], "status": audit["status"], "train": len(train), "validation": len(validation), "outputs_sha256": hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    for name in INPUT_NAMES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.recipe, {n: getattr(args, n) for n in INPUT_NAMES}, args.tokenizer, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
