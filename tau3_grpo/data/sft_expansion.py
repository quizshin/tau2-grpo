"""Build a nested SFT comparison without moving held-out dialogues into training.

Tool coverage means observed calls, not verified task success. Source reward,
schema/response checks and lexical leakage checks are recorded separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.data.sft import load_complete_airline_dialogues, reason_similarity
from tau3_grpo.utils.hashing import sha256_file

SOURCE_SHA256 = "24bc4d479799ef5d5efec7a5388ea2de6bf5dc5ab7c2c83e9b5ee26fa5813804"


def tool_names(record):
    return {call.get("function", call)["name"]
            for message in record["messages"] if message["role"] == "assistant"
            for call in message.get("tool_calls", [])}


def coverage(records):
    return dict(Counter(name for record in records for name in tool_names(record)))


def audit_tool_calls(record, schemas):
    from jsonschema import Draft202012Validator

    validators = {s["function"]["name"]: Draft202012Validator(s["function"]["parameters"])
                  for s in schemas}
    pending = []
    errors = []
    for index, message in enumerate(record["messages"]):
        if message["role"] == "tool":
            name = message.get("name")
            if not pending or name not in pending:
                errors.append(f"{index}:unmatched_tool_response:{name}")
            else:
                pending.remove(name)
            continue
        if pending:
            errors.append(f"{index}:missing_tool_responses:{pending}")
            pending = []
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            name = function.get("name")
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    errors.append(f"{index}:invalid_json:{name}")
                    continue
            if name not in validators:
                errors.append(f"{index}:unknown_tool:{name}")
            else:
                errors.extend(f"{index}:arguments:{name}:{e.message}"
                              for e in validators[name].iter_errors(args))
            pending.append(name)
    if pending:
        errors.append(f"unfinished_tool_calls:{pending}")
    return errors


def select_expansion(anchors, candidates, *, size, required_tools, seed=42,
                     minimum_tool_dialogues=5, similarity_threshold=0.88):
    """Preserve anchors, fill rare-tool deficits, then add distinct intents."""
    if size < len(anchors) or minimum_tool_dialogues <= 0:
        raise ValueError("invalid expansion size or coverage target")
    def sid(r):
        return r["metadata"]["source_dialog_id"]
    if len({sid(r) for r in anchors}) != len(anchors):
        raise ValueError("duplicate anchor dialogue IDs")
    selected = list(anchors)
    ids = {sid(r) for r in selected}
    remaining = {sid(r): r for r in candidates if sid(r) not in ids}
    counts = Counter(coverage(selected))
    names = set(required_tools)
    eligible_coverage = Counter(coverage(list(remaining.values())))
    targets = {name: min(minimum_tool_dialogues, counts[name] + eligible_coverage[name])
               for name in names}
    missing = names - set(counts) - set(eligible_coverage)
    if missing:
        raise ValueError(f"No eligible demonstration for tools: {sorted(missing)}")
    def rank(r):
        return hashlib.sha256(f"{seed}:{sid(r)}".encode()).hexdigest()
    while len(selected) < size and remaining:
        def priority(record):
            benefit = sum(1 / max(eligible_coverage[name], 1)
                          for name in tool_names(record) & names if counts[name] < targets[name])
            return (-benefit, rank(record))
        candidate = min(remaining.values(), key=priority)
        del remaining[sid(candidate)]
        reason = candidate["metadata"]["reason_for_call"]
        if any(reason_similarity(reason, r["metadata"]["reason_for_call"]) >= similarity_threshold
               for r in selected):
            continue
        selected.append(candidate)
        counts.update(tool_names(candidate))
    missing = names - set(counts)
    if len(selected) != size or missing:
        raise ValueError(f"Cannot meet expansion contract: {len(selected)}/{size}, missing={sorted(missing)}")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "anchors", "validation", "blocked-reasons", "tool-config", "output-dir", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--supplemental", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("Raw SFT source does not match the pinned public artifact")

    import yaml
    from transformers import AutoTokenizer

    from tau3_grpo.prompts import prepare_agent_messages
    from tau3_grpo.training.sft.dataset import build_supervised_example

    def read(p):
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    anchors, validation = read(args.anchors), read(args.validation)
    if len(anchors) != 45 or len(validation) != 5:
        raise ValueError("This comparison requires the frozen 45/5 baseline")
    anchor_ids = {r["metadata"]["source_dialog_id"] for r in anchors}
    validation_ids = {r["metadata"]["source_dialog_id"] for r in validation}
    if anchor_ids & validation_ids:
        raise ValueError("Baseline train/validation overlap")
    tools = [e["tool_schema"] for e in yaml.safe_load(args.tool_config.read_text())["tools"]]
    required = {s["function"]["name"] for s in tools}
    if len(required) != 14:
        raise ValueError(f"Expected 14 tools, found {len(required)}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    blocked = json.loads(args.blocked_reasons.read_text())
    blocked += [r["metadata"]["reason_for_call"] for r in validation]
    pool, stats = load_complete_airline_dialogues(args.source)
    source_quality = {}
    with args.source.open() as source:
        for line in source:
            metadata = json.loads(line)["metadata"]
            key = metadata.get("source_dialog_id", "")
            if key.startswith("airline_dialog_"):
                good = metadata.get("correct") == 1 and metadata.get("reward") == 1.0
                source_quality[key] = source_quality.get(key, True) and good
    pool_records = [d.to_record() for d in pool]
    if args.supplemental:
        supplemental = read(args.supplemental)
        for record in supplemental:
            metadata = record["metadata"]
            if (metadata.get("source") != "areal_rl_train_authored_executed"
                    or metadata.get("verification") != "tool_execution_and_reference_final_db_equality"
                    or not metadata.get("executions")):
                raise ValueError("Supplemental records require executed training-task provenance")
            source_quality[metadata["source_dialog_id"]] = True
        pool_records.extend(supplemental)
    if len({r["metadata"]["source_dialog_id"] for r in pool_records}) != len(pool_records):
        raise ValueError("Duplicate source dialogue IDs across pools")
    audit = {"source_sha256": SOURCE_SHA256, "source_counts": vars(stats),
             "anchor_sha256": sha256_file(args.anchors), "validation_sha256": sha256_file(args.validation),
             "tool_config_sha256": sha256_file(args.tool_config),
             "blocked_reasons_sha256": sha256_file(args.blocked_reasons),
             "seed": args.seed, "max_length": args.max_length,
             "pool_coverage": coverage(pool_records), "A_coverage": coverage(anchors),
             "checks": "source correct/reward; tool schema/response pairing; lexical similarity <0.88; native token/mask render",
             "limitations": "No independent task replay or semantic decontamination claim", "candidates": []}
    if args.supplemental:
        audit.update(supplemental_sha256=sha256_file(args.supplemental),
                     supplemental_count=len(supplemental), expanded_pool_size=len(pool_records))
    candidates = []
    for record in pool_records:
        metadata = record["metadata"]
        key = metadata["source_dialog_id"]
        if key in anchor_ids | validation_ids:
            continue
        problems = audit_tool_calls(record, tools)
        if not source_quality[key]:
            problems.append("source_not_success_labeled")
        closest = max((reason_similarity(metadata["reason_for_call"], text) for text in blocked), default=0.0)
        if closest >= 0.88:
            problems.append("heldout_lexical_overlap")
        if not metadata["reason_for_call"].strip():
            problems.append("missing_intent")
        token_stats = {}
        if not problems:
            try:
                example = build_supervised_example(prepare_agent_messages(record["messages"]),
                    tokenizer, tools=tools, max_length=args.max_length)
                token_stats = {"rendered_tokens": example["n_total_tokens"], "label_tokens": example["n_label_tokens"]}
                metadata.update(token_stats)
            except ValueError as exc:
                problems.append(str(exc))
        audit["candidates"].append({"id": key, "tools": sorted(tool_names(record)),
                                    "heldout_similarity": closest, "issues": problems, **token_stats})
        if not problems:
            candidates.append(record)
    audit["eligible_coverage"] = coverage(candidates)
    audit["eligible_count"] = len(candidates)
    audit["anchor_schema_issues"] = {r["metadata"]["source_dialog_id"]: problems
                                      for r in anchors if (problems := audit_tool_calls(r, tools))}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    selected = select_expansion(anchors, candidates, size=args.size, required_tools=required, seed=args.seed)
    for name, rows in (("A_train45.jsonl", anchors), (f"B_train{args.size}.jsonl", selected), ("validation5.jsonl", validation)):
        (args.output_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    # Native render of the immutable baseline must also pass: never silently trim it.
    for record in anchors + validation:
        build_supervised_example(prepare_agent_messages(record["messages"]), tokenizer,
                                 tools=tools, max_length=args.max_length)
    audit.update(B_coverage=coverage(selected), selected_ids=[r["metadata"]["source_dialog_id"] for r in selected],
                 files={p.name: sha256_file(p) for p in args.output_dir.glob("*.jsonl")}, status="cpu_verified")
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in {"candidates", "selected_ids"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
