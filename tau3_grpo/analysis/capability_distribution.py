"""Outcome-blind distribution census and source reuse queue.

Official final tasks are processed in memory and only aggregate counts/hashes
are exported. No final task IDs, text, parameters, answers or examples are
included in the training-side queue. This is distribution exposure, not a
claim that the final distribution remains unseen.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

from tau3_grpo.analysis.sft_coldstart_audit import audit_record, function
from tau3_grpo.data.sft import load_complete_airline_dialogues
from tau3_grpo.data.sft_expansion import SOURCE_SHA256, audit_tool_calls
from tau3_grpo.data.sft_policy_checks import audit_baggage_allowances
from tau3_grpo.data.staged_sft import ordered_tool_receipts
from tau3_grpo.utils.hashing import sha256_file, sha256_json

VERSION = "capability_distribution_v2"
OPERATIONS = {
    "book_reservation": "booking",
    "cancel_reservation": "cancellation",
    "update_reservation_flights": "flight_or_cabin_change",
    "update_reservation_cabin": "flight_or_cabin_change",
    "update_reservation_baggages": "baggage",
    "update_reservation_passengers": "passenger_change",
    "send_certificate": "compensation",
    "transfer_to_human_agents": "escalation",
}
INTENT_RULES = {
    "booking": r"\b(book|booking|reserve|rebook)\b",
    "cancellation": r"\b(cancel\w*|duplicate\w*|overlapping)\b",
    "flight_or_cabin_change": r"\b(change|move|reschedul\w*|upgrade|downgrade|shift|push)\b",
    "baggage": r"\b(bag\w*|luggage)\b",
    "passenger_change": r"\b(passenger|name|spelling|brother|wife|swap|who is|who appears|who is flying)\b",
    "compensation": r"\b(compensat\w*|goodwill|voucher)\b",
}
CONSTRAINT_RULES = {
    "time_window_or_order": r"\b(before|after|earliest|latest|morning|evening|afternoon|between)\b",
    "price_or_refund": r"\$\d|\b(cheapest|cheaper|budget|refund|save money|extra|difference|cost)\b",
    "preserve_other_state": r"\b(only|same|keeping|keep|unchanged|without|except|just one)\b",
    "conditional_or_policy": r"\b(if|unless|insurance|basic economy|health|medical|illness|24 hour)\b",
    "multiple_targets": r"\b(then|several|multiple|all flights|all reservations|both|two|three|four)\b",
    "payment_constraint": r"\b(payment|credit card|gift card|certificate|original payment)\b",
    "identity_constraint": r"\b(passport|spelling|birth|brother|wife|son|daughter|replace|swap)\b",
}


def read_rows(path):
    return [
        json.loads(lint_item)
        for lint_item in Path(path).read_text().splitlines()
        if lint_item.strip()
    ]


def features(reason, calls, *, messages=None):
    names = [name for name, args in calls]
    reservations = {args["reservation_id"] for name, args in calls if args.get("reservation_id")}
    ops = {OPERATIONS[n] for n in names if n in OPERATIONS}
    constraints = {k for k, pattern in CONSTRAINT_RULES.items() if re.search(pattern, reason, re.I)}
    intent = {k for k, pattern in INTENT_RULES.items() if re.search(pattern, reason, re.I)}
    # A lexical intent proxy shared across every split, not calibrated intrinsic difficulty.
    level = (
        "0-1_constraint_tags"
        if len(constraints) <= 1
        else "2-3_constraint_tags"
        if len(constraints) <= 3
        else "4+_constraint_tags"
    )
    out = dict(
        intent_tags=sorted(intent),
        constraint_tags=sorted(constraints),
        intent_proxy_bin=level,
        tool_presence=sorted(set(names)),
        tool_calls=len(names),
        first_tool=names[0] if names else "none",
        operation_tags=sorted(ops),
        distinct_reservations=len(reservations),
        operation_breadth="0"
        if not ops
        else "1"
        if len(ops) == 1
        else "2"
        if len(ops) == 2
        else "3+",
        reference_or_observed_reservation_count_bin="0"
        if not reservations
        else "1"
        if len(reservations) == 1
        else "2-3"
        if len(reservations) <= 3
        else "4+",
        reservation_ids=sorted(reservations),
    )
    if messages is not None:
        first = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
        user_ids = {args["user_id"] for name, args in calls if args.get("user_id")}
        out.update(
            user_turns=sum(m["role"] == "user" for m in messages),
            assistant_turns=sum(m["role"] == "assistant" for m in messages),
            opening_has_reservation=any(x in first for x in reservations),
            opening_has_user_id=any(x in first for x in user_ids),
            multicall_turns=sum(len(m.get("tool_calls") or []) > 1 for m in messages),
            visible_characters=sum(len(m.get("content") or "") for m in messages),
        )
    return out


def summarize(records, evidence):
    n = len(records)
    out = {
        "count": n,
        "action_evidence": evidence,
        "intrinsic_difficulty": "not_human_calibrated",
        "intent_proxy_basis": "reason_for_call_only_same_regex_all_sets",
        "tool_coverage_is_necessity": False,
    }
    for key in ("intent_tags", "constraint_tags", "tool_presence", "operation_tags"):
        out[key] = dict(sorted(Counter(x for r in records for x in r[key]).items()))
    for key in (
        "intent_proxy_bin",
        "operation_breadth",
        "reference_or_observed_reservation_count_bin",
        "first_tool",
    ):
        out[key] = dict(sorted(Counter(r[key] for r in records).items()))
    for key in ("tool_calls", "user_turns", "assistant_turns", "visible_characters"):
        vals = [r[key] for r in records if key in r]
        if vals:
            vs = sorted(vals)
            out[key] = {
                "n": len(vals),
                "mean": sum(vals) / len(vals),
                "median": median(vals),
                "p90": vs[min(len(vs) - 1, int(0.9 * len(vs)))],
                "max": max(vals),
            }
    for key in ("opening_has_reservation", "opening_has_user_id"):
        if any(key in r for r in records):
            out[key] = sum(r.get(key, False) for r in records)
    return out


def sft_feature(row):
    calls = [function(c) for m in row["messages"] for c in m.get("tool_calls") or []]
    return features(row["metadata"].get("reason_for_call", ""), calls, messages=row["messages"])


def task_feature(task):
    t = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
    ins = t["user_scenario"]["instructions"]
    reason = ins.get("reason_for_call", "") if isinstance(ins, dict) else str(ins)
    return features(
        reason, [(a["name"], a["arguments"]) for a in t["evaluation_criteria"].get("actions") or []]
    )


def annotation_coverage(tasks):
    counts = Counter()
    for task in tasks:
        t = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
        ec = t.get("evaluation_criteria") or {}
        acts = ec.get("actions") or []
        counts["empty_reference_actions"] += not acts
        counts["no_reference_write_or_transfer"] += not any(a["name"] in OPERATIONS for a in acts)
        counts["has_communicate_info"] += bool(ec.get("communicate_info"))
        counts["has_nl_assertions"] += bool(ec.get("nl_assertions"))
        counts["has_env_assertions"] += bool(ec.get("env_assertions"))
        for basis in ec.get("reward_basis") or ["DB", "COMMUNICATE"]:
            counts["reward_basis_" + str(basis)] += 1
    return dict(counts)


def confirmation_screen(messages):
    """Flag only absence of any earlier assistant prose/user response pair.

    This is a weak necessary-condition screen. A passing result does not prove
    the right action was described or approved. Never auto-insert a yes.
    """
    described = False
    user_after_description = False
    flags = []
    for i, m in enumerate(messages):
        if (
            m["role"] == "assistant"
            and (m.get("content") or "").strip()
            and not m.get("tool_calls")
        ):
            described = True
        elif m["role"] == "user" and described:
            user_after_description = True
        elif m["role"] == "assistant":
            writes = [
                function(c)[0]
                for c in m.get("tool_calls") or []
                if function(c)[0] in OPERATIONS and function(c)[0] != "transfer_to_human_agents"
            ]
            if writes and not user_after_description:
                flags.append(
                    {
                        "event_id": f"m{i:03d}",
                        "tools": writes,
                        "kind": "no_prior_assistant_description_user_reply_pair",
                    }
                )
    return flags


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
