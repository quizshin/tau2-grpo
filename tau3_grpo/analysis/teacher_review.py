"""Prepare evidence-bound review packets, never auto-accept teacher candidates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tau3_grpo.data.staged_sft import ordered_tool_receipts
from tau3_grpo.envs.adapter import build_environment, load_flight_db
from tau3_grpo.envs.tau2_bridge import message_models
from tau3_grpo.tracking.judge_budget import dump
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def _replay_runtime(record, db_root):
    task = record["task"]
    path = (Path(db_root) / task["db_path"]).resolve()
    if sha256_file(path) != task["db_hash"] or sha256_json(task) != record["task_snapshot_sha256"]:
        raise ValueError("Task/database identity changed")
    db = load_flight_db(path)
    env = build_environment(db)
    if (
        env.tools.db.model_dump(mode="json") != record["initial_db"]
        or env.get_db_hash() != record["initial_db_hash"]
    ):
        raise ValueError("Initial state mismatch")
    messages = record["messages"]
    if sha256_json(messages) != record["messages_sha256"]:
        raise ValueError("Saved conversation identity mismatch")
    if ordered_tool_receipts({"messages": messages}):
        raise ValueError("Tool receipt order is invalid")
    calls = [c for m in messages for c in m.get("tool_calls", [])]
    receipts = [m for m in messages if m["role"] == "tool"]
    executions = record["executions"]
    if len(calls) != len(receipts) or len(calls) != len(executions):
        raise ValueError("Tool trace/receipt count mismatch")
    hashes = []
    for call, visible, execution in zip(calls, receipts, executions):
        function = call["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        native = message_models()["ToolCall"](
            id=call["id"], name=function["name"], arguments=arguments, requestor="assistant"
        )
        if native.model_dump(mode="json") != execution["call"]:
            raise ValueError("Visible action differs from executed action")
        if env.get_db_hash() != execution["before_db_hash"]:
            raise ValueError("Pre-action DB mismatch")
        result = env.get_response(native)
        if (
            result.content != visible["content"]
            or result.error != visible.get("error", False)
            or visible["tool_call_id"] != native.id
            or visible["name"] != native.name
            or result.content != execution["native_receipt"]["content"]
            or result.error != execution["native_receipt"]["error"]
        ):
            raise ValueError("Native tool receipt mismatch")
        if env.get_db_hash() != execution["after_db_hash"]:
            raise ValueError("Post-action DB mismatch")
        hashes.append(env.get_db_hash())
    if env.get_db_hash() != record["final_db_hash"]:
        raise ValueError("Final environment hash mismatch")
    actual_final = env.tools.db.model_dump(mode="json")
    return (
        dict(
            passed=True,
            native_tools_replayed=len(executions),
            initial_db_hash=record["initial_db_hash"],
            final_db_hash=record["final_db_hash"],
            intermediate_db_hashes=hashes,
            semantic_success_not_established=True,
        ),
        actual_final,
    )


def replay_candidate(record, db_root):
    result, actual_final = _replay_runtime(record, db_root)
    if actual_final != record["final_db"]:
        raise ValueError(
            "Final complete state mismatch: legacy snapshot is untrusted; independently reconstruct into a separate artifact"
        )
    result["complete_state_verified_against"] = "native_environment_owned_db"
    return result


def reconstruct_legacy_snapshot(record, db_root, *, source_file_sha256):
    """Produce a separate trace-bound derivation; never overwrite original data.

    All native responses and intermediate/final runtime hashes must match first.
    Reconstruction establishes state provenance, never semantic acceptance.
    """
    if not source_file_sha256 or len(source_file_sha256) != 64:
        raise ValueError("Original candidate file SHA256 required")
    replay, actual_final = _replay_runtime(record, db_root)
    derived = copy.deepcopy(record)
    derived["final_db"] = actual_final
    derived.setdefault("metadata", {}).update(
        quality_accepted=False,
        state_snapshot_version="native_environment_owned_db_v2_reconstructed",
    )
    derived["snapshot_reconstruction"] = {
        "original_file_sha256": source_file_sha256,
        "original_record_sha256": sha256_json(record),
        "original_final_db_sha256": sha256_json(record["final_db"]),
        "reconstructed_final_db_sha256": sha256_json(actual_final),
        "original_final_snapshot_matched": record["final_db"] == actual_final,
        "method": "independent native replay; exact receipts and all runtime hash checkpoints",
        "quality_accepted": False,
        "requires_fresh_rubric_review": True,
        "replay": replay,
    }
    return derived


def prepare(run, rubrics, db_root, output):
    if output.exists():
        raise FileExistsError("Never overwrite review evidence")
    frozen = [json.loads(line) for line in rubrics.read_text().splitlines() if line.strip()]
    rubric_index = {r["task_id"]: r for r in frozen}
    if len(rubric_index) != len(frozen):
        raise ValueError("Duplicate task rubric")
    output.mkdir(parents=True)
    summary = dict(candidate_snapshots=[], review_only=True, accepted_count=0)
    for path in sorted((run / "candidates").glob("*.json")):
        record = json.loads(path.read_text())
        if record["status"] == "running":
            continue
        task = record["task"]
        rubric = rubric_index[task["task_id"]]
        if rubric["task_hash"] != task["task_hash"]:
            raise ValueError("Rubric does not belong to exact candidate task")
        result = dict(
            candidate_id=record["candidate_id"],
            candidate_file=str(path),
            candidate_sha256=sha256_file(path),
            rubric=rubric,
            rubric_file_sha256=sha256_file(rubrics),
            status="not_reviewed",
            source_status=record["status"],
            quality_accepted=False,
            messages=record["messages"],
            executions=record["executions"],
        )
        try:
            result["native_replay"] = replay_candidate(record, db_root)
        except (ValueError, KeyError, TypeError) as exc:
            result["native_replay"] = dict(
                passed=False, error_type=type(exc).__name__, reason=str(exc)
            )
        dump(output / (record["candidate_id"] + ".json"), result)
        summary["candidate_snapshots"].append(
            {
                k: result[k]
                for k in (
                    "candidate_id",
                    "candidate_sha256",
                    "source_status",
                    "status",
                    "native_replay",
                )
            }
        )
    dump(output / "summary.json", summary)
    print(json.dumps(dict(prepared=len(summary["candidate_snapshots"]), accepted=0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "rubrics", "db-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    prepare(args.run, args.rubrics, args.db_root, args.output)


if __name__ == "__main__":
    main()
