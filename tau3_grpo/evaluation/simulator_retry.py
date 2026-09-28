"""Append-only, evidence-approved simulator drift retries under one queue deadline.

Never selects attempts by reward. Full initial attempts and derived subset contracts
are kept separately. The adjusted comparison needs an explicit all-attempt audit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from tau3_grpo.utils.hashing import sha256_json

REASONS = {
    "invented_fixed_identity",
    "changed_fixed_goal",
    "omitted_required_goal_with_opportunity",
    "unauthorized_user_escalation",
    "contradicted_explicit_payment_constraint",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_retry_plan(plan):
    if plan.get("probe_failure_policy") != "record_and_review_each_trajectory":
        raise ValueError("v3 requires per-trajectory simulator review")
    if not plan.get("retry_contract") or not plan.get("retry_contract_sha256"):
        raise ValueError("Missing frozen retry contract")
    if plan["simulator_probe"].get("semantic_review"):
        raise ValueError("Diagnostic probe must not be mislabeled as an all-pass semantic gate")
    for arm in plan["arms"]:
        argv = arm["evaluation"]["argv"]
        for flag in ("--user-temperature", "--policy-temperature"):
            if flag not in argv or float(argv[argv.index(flag) + 1]) != 0.7:
                raise ValueError("v3 requires fixed user/policy temperature0.7")


def load_contract(controller):
    p = Path(controller.plan["retry_contract"])
    if digest(p) != controller.plan["retry_contract_sha256"]:
        raise ValueError("Frozen retry contract changed")
    contract = json.loads(p.read_text())
    if (
        contract.get("version") != "evidence_simulator_retry_v1"
        or contract.get("selection_rule") != "first_scope_valid_attempt_regardless_of_reward"
    ):
        raise ValueError("Unsupported retry selection rule")
    if (
        not 1 <= contract["max_additional_trajectories"] <= 60
        or not 2 <= contract["max_attempts_per_cell"] <= 4
    ):
        raise ValueError("Unbounded retry scope")
    return contract


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for x in value.values() for s in strings(x)]
    if isinstance(value, list):
        return [s for x in value for s in strings(x)]
    return []


def validate_request(request, *, root, plan_hash, entries, completed, contract):
    """Mechanical provenance gate; reviewer's causal diagnosis remains auditable."""
    if (
        request.get("plan_sha256") != plan_hash
        or request.get("verdict") != "confirmed_simulator_drift"
    ):
        raise ValueError("Stale plan or no confirmed drift verdict")
    arm, tid, attempt = request["arm"], request["task_id"], request["rejected_attempt"]
    if arm not in ("base", "sft1", "repair_base", "repair_sft1") or tid not in entries:
        raise ValueError("Unknown arm/task")
    if type(attempt) is not int or not 0 <= attempt < contract["max_attempts_per_cell"] - 1:
        raise ValueError("Attempt cap reached")
    if (
        request.get("reason_code") not in REASONS
        or not request.get("explanation")
        or not request.get("reviewer")
    ):
        raise ValueError("Missing simulator-specific evidence review")
    source = (root / request["source_file"]).resolve()
    if not source.is_relative_to(root.resolve()) or source.name not in (
        "trajectories.jsonl",
        "errors.jsonl",
    ):
        raise ValueError("Source outside this run")
    expected = (
        root / arm / "eval" if attempt == 0 else root / "retry_runs" / arm / f"attempt{attempt}"
    )
    valid_parent = (
        source.parent == expected.resolve()
        if attempt == 0
        else source.parent.is_relative_to(expected.resolve()) and source.parent.name == "eval"
    )
    if not valid_parent:
        raise ValueError("Source not the previous attempt")
    if attempt and (arm, tid, attempt) not in completed:
        raise ValueError("Previous retry not completed")
    matches = [
        x
        for x in map(json.loads, source.read_text().splitlines())
        if x["task_id"] == tid and x["trial"] == 0 and x["seed"] == 42 + attempt
    ]
    if len(matches) != 1 or sha256_json(matches[0]) != request.get("trajectory_sha256"):
        raise ValueError("Missing or changed prior trajectory")
    row = matches[0]
    sim = row.get("simulation") or (row.get("execution_evidence") or {}).get("simulation")
    if not sim or not sim.get("reward_info"):
        raise ValueError("Incomplete simulation; infrastructure recovery is separate")
    quote = request.get("scenario_quote", "")
    if not quote or not any(quote in s for s in strings(entries[tid]["task"]["user_scenario"])):
        raise ValueError("Missing exact frozen scenario evidence")
    evidence = request.get("user_evidence", [])
    if not evidence:
        raise ValueError("Missing exact user-message evidence")
    for e in evidence:
        m = sim["messages"][e["message_index"]]
        if m["role"] != "user" or not e.get("quote") or e["quote"] not in (m.get("content") or ""):
            raise ValueError("Evidence is not the actual user utterance")
    return arm, tid, attempt + 1


def build_retry_arm(controller, arm_name, attempt, tids, entries, requests):
    from tau3_grpo.evaluation.bounded import save

    parent = next(x for x in controller.plan["arms"] if x["name"] == arm_name)
    parent_folder = controller.root / "retry_runs" / arm_name / f"attempt{attempt}"
    batch = len(list(parent_folder.glob("batch*"))) + 1
    folder = parent_folder / f"batch{batch:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    manifest = folder / "manifest"
    manifest.mkdir()
    (manifest / "areal_airline_selection_seed42.jsonl").write_text(
        "".join(json.dumps(entries[t], ensure_ascii=False) + "\n" for t in tids)
    )
    bundle = json.loads(Path(controller.plan["bundle"]).read_text())
    bundle["tasks"] = {t: bundle["tasks"][t] for t in tids}
    bundle["retry_provenance"] = dict(
        parent_bundle_sha256=controller.plan["bundle_sha256"],
        selection="exact task subset; task predicates and accepted outcomes unchanged",
        attempt=attempt,
    )
    save(folder / "bundle.json", bundle)
    save(folder / "authorization_evidence.json", requests)
    arm = copy.deepcopy(parent)
    arm.update(name=f"{arm_name}_retry_{attempt}_batch{batch:03d}", output=str(folder / "eval"))
    argv = arm["evaluation"]["argv"]
    for flag, value in {
        "--manifest-dir": str(manifest),
        "--quality-bundle": str(folder / "bundle.json"),
        "--output-dir": str(folder / "eval"),
        "--results-dir": str(folder),
        "--seed": str(42 + attempt),
        "--policy-attestation": str(folder / "policy_service_attestation.json"),
    }.items():
        argv[argv.index(flag) + 1] = value
    arm["policy"]["environment"]["TAU3_POLICY_ATTESTATION_PATH"] = str(
        folder / "policy_service_attestation.json"
    )
    save(folder / "launch_arm.json", arm)
    return arm


def process_retry_queue(controller, user):
    from tau3_grpo.evaluation.bounded import save

    contract = load_contract(controller)
    entries = {
        x["task_id"]: x
        for x in map(json.loads, Path(controller.plan["manifest"]).read_text().splitlines())
    }
    queue = controller.root / "drift_retry_requests"
    queue.mkdir(exist_ok=True)
    completed = set()
    used = set()
    additional = 0
    controller.status("awaiting_drift_review", initial_sampling_complete=True)
    while True:
        controller.guard([user])
        load_contract(controller)
        if (controller.root / "FINISH_RETRY_REVIEW.json").exists():
            finish = json.loads((controller.root / "FINISH_RETRY_REVIEW.json").read_text())
            if (
                finish.get("plan_sha256") != controller.state["plan_sha256"]
                or finish.get("all_planned_scope_reviewed") is not True
            ):
                raise ValueError("Stale or incomplete retry finish receipt")
            controller.state["retry_review_finished_receipt"] = digest(
                controller.root / "FINISH_RETRY_REVIEW.json"
            )
            break
        pending = []
        for path in sorted(queue.glob("*.json")):
            if path.name in used:
                continue
            request = json.loads(path.read_text())
            try:
                key = validate_request(
                    request,
                    root=controller.root,
                    plan_hash=controller.state["plan_sha256"],
                    entries=entries,
                    completed=completed,
                    contract=contract,
                )
                if key in completed:
                    raise ValueError("Duplicate retry for task and attempt")
                pending.append((key, path, request))
            except Exception as exc:
                used.add(path.name)
                save(
                    queue / (path.stem + ".rejected"),
                    dict(error=str(exc), request_sha256=digest(path)),
                )
                controller.state.setdefault("retry_request_errors", []).append(
                    dict(path=str(path), error=str(exc))
                )
                controller.status("retry_request_rejected")
        if not pending:
            if controller.deadline - time.monotonic() < 120:
                controller.state["retry_budget_remaining_insufficient"] = True
                break
            time.sleep(5)
            continue
        group_key = pending[0][0][0], pending[0][0][2]
        selected = [x for x in pending if (x[0][0], x[0][2]) == group_key]
        tids = [x[0][1] for x in selected]
        if len(set(tids)) != len(tids):
            raise ValueError("Duplicate pending retry requests")
        if additional + len(tids) > contract["max_additional_trajectories"]:
            controller.state["retry_count_cap_reached"] = True
            break
        arm = build_retry_arm(controller, *group_key, tids, entries, [x[2] for x in selected])
        additional += len(tids)
        controller.state["additional_trajectories_started"] = additional
        controller.evaluate(arm, user)
        for key, path, _ in selected:
            completed.add(key)
            used.add(path.name)
        save(
            controller.root / "retry_ledger.json",
            dict(
                completed=[list(k) for k in sorted(completed)],
                started=additional,
                selection_rule=contract["selection_rule"],
                initial_records_replaced=False,
            ),
        )
        controller.status("awaiting_drift_review", initial_sampling_complete=True)
    controller.state["additional_trajectories_started"] = additional
    controller.state["adjusted_metrics_require_complete_scope_audit"] = True
