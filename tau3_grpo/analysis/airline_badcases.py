"""Raw Airline analysis: observations are not correctness or causal labels.

V2 replaces v1 heuristic labels. Missing telemetry is unknown, not zero. Output
bundles are immutable and records, aggregate statistics and report are verified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from tau3_grpo.evaluation.eligibility import (
    FALLBACK_TERMINATIONS,
    POLICY_VERSION,
    execution_eligibility,
)
from tau3_grpo.utils.hashing import sha256_file

VERSION = "airline_badcase_analysis_v2"
WRITE_NAMES = {
    "book_reservation", "cancel_reservation", "update_reservation_flights",
    "update_reservation_passengers", "update_reservation_baggages", "send_certificate",
}
READ_NAMES = {
    "get_reservation_details", "get_user_details", "get_flight_status",
    "search_direct_flight", "search_onestop_flight", "list_all_airports",
}
INFRA_TERMINATIONS = {"infrastructure_error", "unexpected_error", "simulator_error"}
INFRA_FAILURES = {"infrastructure", "infrastructure_error", "simulator_error"}
LIMITATIONS = [
    "Repeated calls may be necessary state re-reads; no redundancy judgment is made.",
    "Database changes and write attempts do not prove correct or partial completion.",
    "Schema, argument binding, policy compliance and first-failure causality need separate review.",
    "Task buckets describe observed trials, not calibrated difficulty or guaranteed stability.",
    "Completed fallback-zero trials remain failures; unresolved attempts invalidate the observed success rate.",
    "Rates cover supplied records only; planned-budget completeness and cross-run comparability require separate evidence.",
]


def decode_object(value, *, field):
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _first(*values):
    return next((v for v in values if v is not None), None)


def normalize_call(call):
    if not isinstance(call, dict):
        raise ValueError("Tool call must be an object")
    function = call.get("function")
    function = function if isinstance(function, dict) else call
    name = function.get("name") or call.get("tool_name")
    if not isinstance(name, str) or not name:
        raise ValueError("Tool call has no function name")
    args = function.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"invalid_json_arguments": args}
    return {"name": name, "arguments": args, "error": call.get("error")}


def extract_calls(row, simulation, facts):
    if "turns" in facts:
        turns = facts["turns"]
        if not isinstance(turns, list):
            raise ValueError("Facts turns must be a list")
        if all(isinstance(t, dict) and isinstance(t.get("tool_calls"), list) for t in turns):
            return [normalize_call(c) for t in turns for c in t["tool_calls"]], "trajectory_facts"
    container = simulation if simulation else row
    if "messages" in container:
        messages = container["messages"]
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        calls = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Message must be an object")
            if message.get("role") != "assistant":
                continue
            values = message.get("tool_calls") or []
            if not isinstance(values, list):
                raise ValueError("Message tool_calls must be a list")
            calls.extend(normalize_call(c) for c in values)
        return calls, "simulation.messages" if simulation else "messages"
    trajectory = decode_object(row.get("trajectory_json"), field="trajectory_json")
    if isinstance(trajectory.get("tool_calls"), list):
        return [normalize_call(c) for c in trajectory["tool_calls"]], "trajectory_json"
    return None, "unavailable"


def normalize_row(row, source, *, kind="rollout", sft=None):
    """Consume original input, not a lossy v1 records.jsonl row."""
    if row.get("record_kind") or row.get("analysis_version"):
        raise ValueError("Expected raw input; normalized records cannot reconstruct missing evidence")
    if sft is not None:
        kind = "demonstration" if sft and "messages" in row and "simulation" not in row else "rollout"
    if kind not in {"demonstration", "rollout"}:
        raise ValueError("Unknown record kind")
    simulation = decode_object(row.get("simulation"), field="simulation")
    info = decode_object(simulation.get("info"), field="simulation.info")
    facts = decode_object(_first(row.get("trajectory_facts_json"), info.get("trajectory_facts")),
                          field="trajectory_facts")
    terminal = decode_object(facts.get("terminal"), field="terminal")
    identity = decode_object(facts.get("identity"), field="identity")
    calls, call_source = extract_calls(row, simulation, facts)
    names = None if calls is None else [c["name"] for c in calls]
    messages = (simulation or row).get("messages", [])
    text = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "assistant")
    if not messages:
        text = str(row.get("output") or "")
    termination = _first(row.get("termination_reason"), simulation.get("termination_reason"),
                         terminal.get("termination_reason"))
    failure = _first(row.get("failure_category"), terminal.get("failure_category"))
    initial_hash = _first(row.get("initial_db_hash"), info.get("initial_db_hash"), terminal.get("initial_db_hash"))
    final_hash = _first(row.get("db_hash"), info.get("final_db_hash"), info.get("db_hash"), terminal.get("db_hash"))
    reward_info = decode_object(simulation.get("reward_info"), field="simulation.reward_info")
    raw_score = _first(row.get("score"), row.get("reward"), reward_info.get("reward"), terminal.get("reward"))
    score = None
    if raw_score is not None:
        if isinstance(raw_score, bool):
            raise ValueError("Reward must be numeric, not bool")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("Reward must be finite")
    eligibility = decode_object(_first(row.get("execution_eligibility"), terminal.get("execution_eligibility")),
                                field="execution_eligibility")
    infra = (termination in INFRA_TERMINATIONS or failure in INFRA_FAILURES
             or bool(row.get("error")) or eligibility.get("category") == "infrastructure_error")
    policy = execution_eligibility(termination, reward=score if kind == "rollout" else None, exception=infra)
    officially_scored = (kind == "rollout" and score is not None and policy["officially_scored"]
                         and row.get("scored") is not False and terminal.get("scored") is not False
                         and eligibility.get("officially_scored") is not False)
    # Verifier.scored=False is expected for capped/model-error fallback zeros.
    # Those are complete trials, unlike infrastructure/unknown/missing outcomes.
    complete = (kind == "rollout" and score is not None and policy["evaluation_trial_complete"]
                and eligibility.get("evaluation_trial_complete") is not False
                and (officially_scored or termination in FALLBACK_TERMINATIONS))
    success = abs(score - 1.0) <= 1e-6 if complete else None
    labels = set()
    if kind == "rollout":
        if not complete:
            labels.add("infrastructure_error" if infra else "unresolved")
        if termination == "agent_error" or failure == "agent_error":
            labels.add("agent_generation_failure")
        if (failure == "tool_errors" or termination in {"tool_error", "too_many_errors"}
                or any(c["error"] is True for c in calls or [])
                or any(m.get("role") == "tool" and m.get("error") is True for m in messages)):
            labels.add("tool_execution_error")
        if termination in {"max_steps", "timeout"}:
            labels.add("turn_or_time_limit")
        if termination in {"context_window_exceeded", "length"} or any(
                t.get("finish_reason") == "length" for t in facts.get("turns", [])):
            labels.add("length_or_context_limit")
        if re.search(r"(?:✅\s*){20,}", text) or any(
                t.get("generation_guard") for t in facts.get("turns", [])):
            labels.add("repetition_detected")
        if complete and not success:
            if initial_hash and final_hash:
                labels.add("db_changed_on_failure" if initial_hash != final_hash else "db_unchanged_on_failure")
            if calls is not None:
                if not calls:
                    labels.add("no_tool_calls_observed_on_failure")
                if any(n in WRITE_NAMES for n in names):
                    labels.add("write_attempt_on_failure")
        if calls is not None:
            signatures = [(c["name"], json.dumps(c["arguments"], sort_keys=True, ensure_ascii=False)) for c in calls]
            if len(signatures) != len(set(signatures)):
                labels.add("repeated_call_signature")
            writes = [i for i, n in enumerate(names) if n in WRITE_NAMES]
            if writes and not any(n in READ_NAMES for n in names[writes[-1] + 1:]):
                labels.add("write_attempt_without_later_read")
    return {
        "analysis_version": VERSION, "source": source, "record_kind": kind,
        "task_id": _first(row.get("task_id"), identity.get("task_id"), simulation.get("task_id")),
        "trajectory_id": _first(identity.get("trajectory_id"), simulation.get("id")),
        "sample_group_uid": identity.get("sample_group_uid"),
        "trial": _first(row.get("trial"), identity.get("trial"), simulation.get("trial")),
        "seed": _first(row.get("seed"), identity.get("seed"), simulation.get("seed")),
        "global_step": _first(identity.get("global_step"), row.get("step")),
        "source_dialog_id": (row.get("metadata") or {}).get("source_dialog_id"),
        "scored": officially_scored, "evaluation_trial_complete": complete,
        "eligibility_category": policy["category"],
        "success": success, "score": score if kind == "rollout" else None,
        "termination_reason": termination, "failure_category": failure,
        "termination_detail": terminal.get("termination_detail"),
        "badcase_labels": sorted(labels), "tool_names": names, "tool_evidence_source": call_source,
        "tool_call_count": None if calls is None else len(calls),
        "read_call_count": None if names is None else sum(n in READ_NAMES for n in names),
        "write_call_count": None if names is None else sum(n in WRITE_NAMES for n in names),
        "initial_db_hash": initial_hash, "final_db_hash": final_hash, "text_length": len(text),
    }


def _counts(values):
    return dict(sorted(Counter("unknown" if v is None else str(v) for v in values).items()))


def task_groups(records):
    by_task = defaultdict(list)
    for row in records:
        if row["record_kind"] == "rollout" and row["task_id"] is not None:
            by_task[str(row["task_id"])].append(row)
    result = {}
    for task, rows in sorted(by_task.items()):
        successes = sum(r["success"] is True for r in rows)
        unknown = sum(not r["evaluation_trial_complete"] for r in rows)
        bucket = ("environment_or_unscorable" if unknown else "stable_success" if successes == len(rows)
                  else "stable_failure" if successes == 0 else "mixed_success")
        result[task] = {"bucket": bucket, "trials": len(rows), "unresolved": unknown,
                        "successes": successes, "success_rate": None if unknown else successes / len(rows),
                        "badcase_counts": _counts(label for r in rows for label in r["badcase_labels"])}
    return result


def summarize(records):
    sources = {}
    for source in sorted({r["source"] for r in records}):
        rows = [r for r in records if r["source"] == source]
        rollouts = [r for r in rows if r["record_kind"] == "rollout"]
        demos = [r for r in rows if r["record_kind"] == "demonstration"]
        successes = sum(r["success"] is True for r in rollouts)
        unresolved = sum(not r["evaluation_trial_complete"] for r in rollouts)
        sources[source] = {
            "records": len(rows), "rollouts": len(rollouts), "demonstrations": len(demos),
            "successes": successes, "unresolved": unresolved,
            "officially_scored": sum(r["scored"] for r in rollouts),
            "completed_fallback_zeros": sum(r["evaluation_trial_complete"] and not r["scored"] for r in rollouts),
            "success_rate": successes / len(rollouts) if rollouts and not unresolved else None,
            "tool_calls_observed": sum(r["tool_call_count"] or 0 for r in rows),
            "tool_telemetry_missing_records": sum(r["tool_call_count"] is None for r in rows),
            "tool_counts": _counts(n for r in rows for n in r["tool_names"] or []),
            "termination_counts": _counts(r["termination_reason"] for r in rollouts),
            "failure_counts": _counts(r["failure_category"] for r in rollouts),
            "badcase_counts": _counts(label for r in rollouts for label in r["badcase_labels"]),
            "task_groups": task_groups(rollouts),
        }
    return {"version": VERSION, "limitations": LIMITATIONS, "sources": sources}


def render_report(summary):
    lines = ["# Airline badcase analysis v2", "", "事实观测，不是已验证的能力缺陷或训练标签。", ""]
    for name, s in summary["sources"].items():
        rate = "N/A" if s["success_rate"] is None else f"{s['success_rate']:.4%}"
        lines += [f"## {name}", "", f"Records: {s['records']}; rollouts: {s['rollouts']}; demonstrations: {s['demonstrations']}",
                  f"Successes: {s['successes']}; unresolved: {s['unresolved']}; observed success rate: {rate}",
                  f"Officially scored: {s['officially_scored']}; completed fallback zeros: {s['completed_fallback_zeros']}",
                  f"Observed tool calls: {s['tool_calls_observed']}; missing telemetry records: {s['tool_telemetry_missing_records']}", ""]
        lines += [f"- `{label}`: {count}" for label, count in s["badcase_counts"].items()]
        lines += ["", "Observed task buckets:"]
        lines += [f"- `{k}`: {v}" for k, v in _counts(t["bucket"] for t in s["task_groups"].values()).items()]
        lines.append("")
    return "\n".join(lines + ["## Limitations", ""] + [f"- {s}" for s in LIMITATIONS]) + "\n"


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("records.jsonl", "summary.json", "report.md"):
        if sha256_file(directory / name) != manifest["outputs_sha256"][name]:
            raise ValueError(f"Bundle hash mismatch: {name}")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    summary = json.loads((directory / "summary.json").read_text())
    if summary != summarize(records) or (directory / "report.md").read_text() != render_report(summary):
        raise ValueError("Bundle records, summary and report disagree")
    return {"status": "verified", "records": len(records), "sources": len(summary["sources"])}


def build_bundle(sources, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {output}")
    if not sources or len({s["name"] for s in sources}) != len(sources):
        raise ValueError("Require nonempty, uniquely named sources (one run/step per source)")
    records, provenance = [], []
    for spec in sources:
        path = Path(spec["path"])
        paths = [path]
        if spec.get("errors_path"):
            paths.append(Path(spec["errors_path"]))
        elif path.name == "trajectories.jsonl" and path.with_name("errors.jsonl").exists():
            paths.append(path.with_name("errors.jsonl"))
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError("Result and error paths must differ")
        rows, ids, trial_ids, input_files = [], set(), set(), []
        for file_index, input_path in enumerate(paths):
            before = sha256_file(input_path)
            expected_hash = spec.get("sha256" if file_index == 0 else "errors_sha256", before)
            if expected_hash != before:
                raise ValueError(f"Source hash mismatch: {input_path}")
            count = 0
            for line_number, line in enumerate(input_path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = normalize_row(json.loads(line), spec["name"], kind=spec["kind"])
                identity = row["trajectory_id"] or row["source_dialog_id"]
                trial_identity = ((row["task_id"], row["sample_group_uid"], row["global_step"], row["trial"])
                                  if row["trial"] is not None else None)
                if (identity is not None and identity in ids) or (trial_identity is not None and trial_identity in trial_ids):
                    raise ValueError(f"Duplicate trajectory/trial identity in {spec['name']}")
                if identity is not None:
                    ids.add(identity)
                if trial_identity is not None:
                    trial_ids.add(trial_identity)
                row.update(source_file=input_path.name, source_line=line_number)
                rows.append(row)
                count += 1
            if sha256_file(input_path) != before:
                raise ValueError(f"Input changed during analysis: {input_path}")
            input_files.append({"path": str(input_path.resolve()), "sha256": before, "records": count})
        if not rows or len(rows) != spec.get("expected_records", len(rows)):
            raise ValueError(f"Unexpected record count for {spec['name']}: {len(rows)}")
        records.extend(rows)
        provenance.append({**spec, "path": str(path.resolve()), "sha256": input_files[0]["sha256"],
                           "records": len(rows), "input_files": input_files})
    summary = summarize(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (staging / "report.md").write_text(render_report(summary))
        manifest = {"version": VERSION, "sources": provenance, "code_sha256": sha256_file(__file__),
                    "eligibility_policy": POLICY_VERSION,
                    "eligibility_code_sha256": sha256_file(Path(__file__).resolve().parents[1] / "evaluation/eligibility.py"),
                    "outputs_sha256": {name: sha256_file(staging / name) for name in ("records.jsonl", "summary.json", "report.md")}}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        verification = verify_bundle(staging)
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--verify-dir", type=Path)
    for flag in ("sft-train", "sft-eval", "grpo-train", "grpo-eval"):
        parser.add_argument(f"--{flag}", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.verify_dir:
        print(json.dumps(verify_bundle(args.verify_dir)))
        return 0
    sources = []
    if args.input_manifest:
        sources = json.loads(args.input_manifest.read_text())["sources"]
        sources = [{**s, **{key: str((args.input_manifest.parent / s[key]).resolve())
                            for key in ("path", "errors_path") if key in s}} for s in sources]
    for flag in ("sft_train", "sft_eval", "grpo_train", "grpo_eval"):
        for item in getattr(args, flag):
            name, path = item.split("=", 1)
            sources.append({"name": name, "path": path,
                            "kind": "demonstration" if flag == "sft_train" else "rollout"})
    if not args.out:
        parser.error("--out is required unless verifying an existing bundle")
    print(json.dumps(build_bundle(sources, args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
