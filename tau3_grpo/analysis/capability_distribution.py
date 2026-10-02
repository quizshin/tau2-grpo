"""Outcome-blind distribution census and source reuse queue.

Official final tasks are processed in memory and only aggregate counts/hashes
are exported. No final task IDs, text, parameters, answers or examples are
included in the training-side queue. This is distribution exposure, not a
claim that the final distribution remains unseen.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.data.capability_features import (
    CONSTRAINT_RULES as CONSTRAINT_RULES,
)
from tau3_grpo.data.capability_features import (
    INTENT_RULES as INTENT_RULES,
)
from tau3_grpo.data.capability_features import (
    OPERATIONS as OPERATIONS,
)
from tau3_grpo.data.capability_features import (
    annotation_coverage as annotation_coverage,
)
from tau3_grpo.data.capability_features import (
    confirmation_screen as confirmation_screen,
)
from tau3_grpo.data.capability_features import (
    features as features,
)
from tau3_grpo.data.capability_features import (
    sft_feature as sft_feature,
)
from tau3_grpo.data.capability_features import (
    summarize as summarize,
)
from tau3_grpo.data.capability_features import (
    task_feature as task_feature,
)
from tau3_grpo.data.sft import load_complete_airline_dialogues
from tau3_grpo.data.sft_evidence import audit_record
from tau3_grpo.data.sft_expansion import SOURCE_SHA256, audit_tool_calls
from tau3_grpo.data.sft_policy_checks import audit_baggage_allowances
from tau3_grpo.data.staged_sft import ordered_tool_receipts
from tau3_grpo.utils.hashing import sha256_file, sha256_json

VERSION = "capability_distribution_v2"


def read_rows(path):
    return [
        json.loads(lint_item)
        for lint_item in Path(path).read_text().splitlines()
        if lint_item.strip()
    ]


def run(args):
    import yaml

    from tau3_grpo.data.official import load_official_airline_tasks

    if args.output.exists():
        raise FileExistsError("Use a new immutable output directory")
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("Unpinned SFT source")
    args.output.mkdir(parents=True)
    inventory = json.loads(
        Path("results/analysis/areal_sft_strategy_20260925/source_inventory.json").read_text()
    )
    census = {r["source_dialog_id"]: r for r in inventory["records"]}
    schemas = [
        x["tool_schema"]
        for x in yaml.safe_load(Path("configs/envs/tool_config.yaml").read_text())["tools"]
    ]
    # Reuse the earlier frozen protection decisions; do not newly mine reserve/final examples.
    old_pool = read_rows("data/sft/staged_v2_candidates_20260925/source_eligible_pool.jsonl")
    old_dev = read_rows("data/sft/staged_v2_candidates_20260925/offline_dev.jsonl")
    eligible_ids = {r["metadata"]["source_dialog_id"] for r in old_pool}
    protected_ids = {r["metadata"]["source_dialog_id"] for r in old_dev}
    train = read_rows("data/sft/staged_v2_A100_20260925/train.jsonl")
    dev = read_rows("data/sft/staged_v2_A100_20260925/validation.jsonl")
    trained_ids = {r["metadata"]["source_dialog_id"] for r in train}
    protected_ids.update(r["metadata"]["source_dialog_id"] for r in dev)
    protected_family = set().union(*(set(sft_feature(r)["reservation_ids"]) for r in old_dev + dev))
    dialogues, stats = load_complete_airline_dialogues(args.source)
    summaries = {}
    rows_by_set = {}
    reuse = []
    all_features = []
    positive = []
    ready = []
    for d in dialogues:
        row = d.to_record()
        c = census[d.source_dialog_id]
        row["supervision"] = {"message_indices": c["provided_target_message_indices"]}
        f = sft_feature(row)
        all_features.append(f)
        if c["source_eligible"]:
            positive.append(f)
        structural = audit_tool_calls(row, schemas) + ordered_tool_receipts(row)
        audit = audit_record(row)
        baggage = audit_baggage_allowances(row["messages"])
        confirmation = confirmation_screen(row["messages"])
        flags = []
        if not c["source_eligible"]:
            flags.append("source_labels_not_unanimous_1_1")
        if d.source_dialog_id not in eligible_ids:
            flags.append("outside_previous_protected_eligible_pool")
        if d.source_dialog_id in protected_ids:
            flags.append("protected_offline_dev")
        if set(f["reservation_ids"]) & protected_family:
            flags.append("shared_reservation_with_protected_dev_review_required")
        if structural:
            flags.append("runtime_schema_or_receipt_error")
        contradictions = [x for x in audit["findings"] if x["status"] == "contradiction"]
        if contradictions:
            flags.append("deterministic_prefix_contradiction")
        if any(x["status"] == "violated" for x in baggage):
            flags.append("baggage_allowance_contradiction")
        if confirmation:
            flags.append("missing_minimum_confirmation_exchange")
        flags = sorted(set(flags))
        record = {
            "source_dialog_id": d.source_dialog_id,
            "already_in_train100": d.source_dialog_id in trained_ids,
            "source_labels": c["source_labels"],
            "features": f,
            "blocking_flags": flags,
            "schema_receipt_errors": structural,
            "prefix_contradictions": contradictions,
            "prefix_review_flags": audit["findings"],
            "exact_task_db_mapping": False,
            "baggage_checks": baggage,
            "confirmation_screen": confirmation,
            "status": "reuse_candidate_needs_semantic_review" if not flags else "held",
            "source_target_indices": c["provided_target_message_indices"],
            "missing_source_targets": c["assistant_messages_without_source_target"],
            "dialogue_sha256": d.dialogue_hash,
        }
        reuse.append(record)
        if not flags:
            ready.append(f)
    rows_by_set.update(
        areal_all999=all_features,
        areal_label_positive=positive,
        areal_reuse_candidates=ready,
        train100=[sft_feature(r) for r in train],
        offline_dev52=[sft_feature(r) for r in dev],
    )
    for split in ("selection", "train"):
        rows = read_rows(f"data/manifests/areal_airline_{split}_seed42.jsonl")
        rows_by_set[split + "_tasks"] = [task_feature(r["task"]) for r in rows]
    final_tasks = load_official_airline_tasks()
    rows_by_set["official_final50"] = [task_feature(t) for t in final_tasks]
    for name, rows in rows_by_set.items():
        evidence = (
            "reference_actions_not_necessary_minimum"
            if name in ("official_final50", "selection_tasks", "train_tasks")
            else "observed_source_trajectory_actions_not_intrinsic_difficulty"
        )
        summaries[name] = summarize(rows, evidence)
    summaries["official_final50"]["annotation_coverage"] = annotation_coverage(final_tasks)
    for split in ("selection", "train"):
        summaries[split + "_tasks"]["annotation_coverage"] = annotation_coverage(
            [r["task"] for r in read_rows(f"data/manifests/areal_airline_{split}_seed42.jsonl")]
        )
    # Export only aggregate final statistics; no final examples or per-task labels.
    report = {
        "version": VERSION,
        "sets": summaries,
        "final_exposure": "aggregate_distribution_only_user_authorized_20260926",
        "final_task_content_exported": False,
        "final_results_used": False,
        "source_sha256": SOURCE_SHA256,
        "final_task_collection_sha256": sha256_json(
            [t.model_dump(mode="json") for t in final_tasks]
        ),
        "rule_source_sha256": sha256_file(Path(__file__)),
        "legacy_judge_difficulty": {
            name: dict(
                Counter(r["metadata"].get("difficulty", {}).get("level", "unknown") for r in rows)
            )
            for name, rows in [("train100", train), ("offline_dev52", dev)]
        },
        "reuse_queue": {
            "count": len(reuse),
            "candidates": len(ready),
            "new_candidates": sum(
                not r["blocking_flags"] and not r["already_in_train100"] for r in reuse
            ),
            "flags": dict(Counter(x for r in reuse for x in r["blocking_flags"])),
        },
        "limitations": [
            "Regex tags are screening features, not gold business/difficulty annotations.",
            "Reference and observed action lengths are different evidence; neither is intrinsic difficulty.",
            "Shared reservation flags are conservative review holds, not proof of semantic leakage.",
            "Source-positive and static-pass records are candidates, not automatically certified training data.",
            "No source dialogue has a recovered exact task/DB mapping in this census.",
            "Final aggregate distribution has now been consulted; final answer examples remain excluded.",
        ],
    }
    (args.output / "distribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source_reuse_queue.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reuse)
    )
    print(
        json.dumps(
            {
                "counts": {k: v["count"] for k, v in summaries.items()},
                "reuse": report["reuse_queue"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/raw/areal_tau2/tau2_sft_train.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
