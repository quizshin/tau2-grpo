"""CPU compilation of exact accepted outcomes; unresolved tasks block live use.

Uses only the exposed selection manifest, never final/reserve or model rewards.
The two alternative-set rules come from the original task wording. All other
tasks keep their original reference actions pending semantic adjudication.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.outcome_recipes import (
    duplicate_cancellation_allowed as duplicate_cancellation_allowed,
)
from tau3_grpo.data.outcome_recipes import (
    repair_known_reference_actions as repair_known_reference_actions,
)
from tau3_grpo.data.outcome_recipes import (
    variants as variants,
)
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.evaluation.outcome_contract import (
    USER_SCOPE,
    VERSION,
    action_policy_flags,
    execute_actions,
    fresh_environment,
    outcome_hash,
    runtime_identity,
)
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def run(args):
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    entries = read_manifest(args.manifest)
    args.output.mkdir(parents=True)
    bundle = {
        "version": VERSION,
        "user_scope_sha256": sha256_json(USER_SCOPE),
        "tasks": {},
        "purpose": "candidate_outcome_sets_not_semantically_approved",
        "runtime_identity": runtime_identity(),
        "selection_manifest_sha256": sha256_file(args.manifest),
        "source_sha256": sha256_file(Path(__file__)),
        "old_rewards_changed": False,
        "final_or_reserve_read": False,
    }
    if args.repair_known_references:
        bundle["reference_repair_version"] = "evidence_bound_reference_repairs_v1"
    previous = json.loads(args.reuse_bundle.read_text()) if args.reuse_bundle else None
    if previous:
        if previous.get("runtime_identity") != bundle["runtime_identity"]:
            raise ValueError("Cannot reuse outcomes from a changed runtime")
        bundle["reuse_bundle_sha256"] = sha256_file(args.reuse_bundle)
    for e in entries:
        if e.split != "selection":
            raise ValueError("Only exposed selection may be compiled")
        r = ArealTaskRecord.model_validate(e.task)
        adapted = adapt_record(r)
        if r.fingerprint != e.task_hash or sha256_file(adapted.db_path) != e.db_hash:
            raise ValueError("Manifest drift")
        repaired_ids = (
            {"airline_698", "airline_921", "airline_1143", "airline_925", "airline_976"}
            if args.repair_known_references
            else set()
        )
        if (
            previous
            and e.task_id not in set(args.recompile_task) | repaired_ids
            and not previous["tasks"][e.task_id]["reference_execution_errors"]
        ):
            old = previous["tasks"][e.task_id]
            if old["record_sha256"] != sha256_json(e.task) or old["db_hash"] != e.db_hash:
                raise ValueError("Stale reused task")
            bundle["tasks"][e.task_id] = old
            continue
        db = fresh_environment(adapted.db_path).tools.db.model_dump(mode="json")
        item = {
            "task_hash": e.task_hash,
            "record_sha256": sha256_json(e.task),
            "db_hash": e.db_hash,
            "status": "semantic_review_required",
            "accepted_outcomes": [],
            "issues": [],
            "reference_execution_errors": [],
        }
        source = json.dumps(r.user_scenario, ensure_ascii=False)
        # Outcome-blind contradiction detection, not a model-specific exception.
        if re.search(r"8\s*am.{0,3}9\s*pm", source, re.I):
            for number in set(re.findall(r"\bHAT\d{3}\b", source)):
                flight = db["flights"].get(number)
                if (
                    flight
                    and not "08:00:00" <= flight["scheduled_departure_time_est"] <= "21:00:00"
                ):
                    item["issues"].append(
                        {
                            "kind": "named_flight_time_window_conflict",
                            "flight": number,
                            "departure": flight["scheduled_departure_time_est"],
                        }
                    )
        try:
            if r.id in repaired_ids:
                repaired, changes = repair_known_reference_actions(r, db)
                options = [("evidence_reviewed_repair_v1", repaired)]
                item["reference_repairs"] = changes
                item["remaining_semantic_gate"] = (
                    "full_goal_communication_and_simulator_acceptance_pending"
                )
            else:
                options = variants(r, db)
        except Exception as exc:
            options = [("original_reference", r.evaluation_criteria.get("actions") or [])]
            item["issues"].append({"kind": "alternative_set_unresolved", "error": str(exc)})
        for label, actions in options:
            try:

                def inspect_action(env, action):
                    for flag in action_policy_flags(env, action):
                        item["issues"].append({"kind": flag, "variant": label, "action": action})

                env, receipts = execute_actions(
                    adapted.db_path,
                    actions,
                    adapted.task.initial_state,
                    before_action=inspect_action,
                )
                item["accepted_outcomes"].append(
                    {
                        "variant": label,
                        "outcome_sha256": outcome_hash(env),
                        "actions_sha256": sha256_json(actions),
                        "actions": actions,
                        "receipts": receipts,
                    }
                )
            except Exception as exc:
                item["reference_execution_errors"].append({"variant": label, "error": str(exc)})
        if item["issues"] or item["reference_execution_errors"]:
            item["status"] = "blocked_quality_issue"
        bundle["tasks"][e.task_id] = item
    counts = Counter(x["status"] for x in bundle["tasks"].values())
    (args.output / "bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    )
    summary = {
        "tasks": len(entries),
        "statuses": dict(counts),
        "issues": dict(Counter(i["kind"] for x in bundle["tasks"].values() for i in x["issues"])),
        "reference_execution_error_tasks": sum(
            bool(x["reference_execution_errors"]) for x in bundle["tasks"].values()
        ),
        "variants": {
            k: len(v["accepted_outcomes"])
            for k, v in bundle["tasks"].items()
            if len(v["accepted_outcomes"]) > 1
        },
        "live_ready": False,
        "explanation": "Executable references do not prove semantic validity; unresolved tasks cannot be silently dropped.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/areal_airline_selection_seed42.jsonl")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-bundle", type=Path)
    p.add_argument("--recompile-task", action="append", default=[])
    p.add_argument("--repair-known-references", action="store_true")
    run(p.parse_args())
