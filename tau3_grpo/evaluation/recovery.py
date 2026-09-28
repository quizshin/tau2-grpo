"""Explicit saved-rollout rescoring and sealed continuation inputs.

Never regenerates a completed trajectory. Preserves the original run and all
unresolved identities. Invocation is a separate authorized scoring attempt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau3_grpo.utils.hashing import sha256_file, sha256_json


def identity(row):
    return tuple(row[k] for k in ("task_id", "trial", "seed"))


def collect_complete(artifact):
    planned = {identity(x) for x in artifact.metadata["planned"]}
    simulations = []
    seen = set()
    for row in artifact.trajectories + artifact.errors:
        key = identity(row)
        if key not in planned or key in seen:
            raise ValueError("Unplanned or duplicate source attempt")
        seen.add(key)
        sim = row.get("simulation") or (row.get("execution_evidence") or {}).get("simulation")
        if sim is not None:
            simulations.append((row, sim))
    return simulations


def validate_recovery(path, metadata, quality):
    from tau3_grpo.evaluation.artifacts import read_evaluation
    from tau3_grpo.evaluation.dual_metrics import receipt

    data = json.loads(Path(path).read_text())
    if data.get("version") != "saved_rollout_recovery_v1" or data.get("errors"):
        raise ValueError("Unresolved recovery input")
    if data["bundle_sha256"] != sha256_json(quality):
        raise ValueError("Recovery uses another scoring bundle")
    source = read_evaluation(Path(data["source_evaluation"]))
    if source.files != data["source_files"]:
        raise ValueError("Recovery source files changed")
    original = dict((identity(row), sim) for row, sim in collect_complete(source))
    recovered = data["results"]
    if len(recovered) != len(original) or {identity(x) for x in recovered} != set(original):
        raise ValueError(
            "Recovery must include all complete source trajectories, no cherry-picking"
        )
    if source.metadata["planned"] != metadata["planned"]:
        raise ValueError("Recovery schedule changed")
    for field in (
        "target",
        "trials",
        "seed",
        "max_steps",
        "max_errors",
        "max_concurrency",
        "harness_protocol",
    ):
        if source.metadata["spec"][field] != metadata["spec"][field]:
            raise ValueError("Recovery protocol changed: " + field)
    for name in ("policy", "user"):
        if source.metadata["endpoints"][name] != metadata["endpoints"][name]:
            raise ValueError("Recovery endpoint settings changed")
    for field in (
        "checkpoint_hash",
        "task_manifest_sha256",
        "task_db_sha256",
        "agent_system_prompt_sha256",
        "user_scope_sha256",
        "evaluator_source_sha256",
    ):
        if source.metadata["provenance"].get(field) != metadata["provenance"].get(field):
            raise ValueError("Recovery provenance changed: " + field)
    # Only evaluation/scoring/persistence code may change. No policy/tools/user edits.
    old = source.metadata["provenance"]["harness_source_sha256"]
    new = metadata["provenance"]["harness_source_sha256"]
    changed = [k for k in old if old[k] != new.get(k)]
    if any(not k.startswith("tau3_grpo/evaluation/") for k in changed):
        raise ValueError("Recovery generation implementation changed")
    for row in recovered:
        receipt(row)
        if row["simulation"]["messages"] != original[identity(row)]["messages"]:
            raise ValueError("Recovery altered messages")
        if row["simulation"]["reward_info"] != original[identity(row)]["reward_info"]:
            raise ValueError("Recovery altered native evaluator result")
    return recovered, dict(
        receipt_file=str(Path(path).resolve()),
        receipt_sha256=sha256_file(path),
        original_evaluation=data["source_evaluation"],
        original_files=source.files,
        original_provenance=source.metadata["provenance"],
        complete_rollouts_reused=len(recovered),
        scoring_only_source_changes=changed,
        disclosure="Original generation provenance retained; new metadata describes continuation runtime. "
        "This is a revised development evaluation, not a new blind test.",
    )


def rescore(source, bundle_path, manifest, output):
    from tau2.data_model.simulation import SimulationRun

    from tau3_grpo.data.manifest import read_manifest
    from tau3_grpo.data.schema import ArealTaskRecord
    from tau3_grpo.envs.adapter import adapt_record
    from tau3_grpo.evaluation.artifacts import read_evaluation
    from tau3_grpo.evaluation.communication_contract import judge_communication
    from tau3_grpo.evaluation.eligibility import SCORABLE_TERMINATIONS, execution_eligibility
    from tau3_grpo.evaluation.outcome_contract import load_bundle, score_simulation
    from tau3_grpo.prompts import build_system_prompt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifact = read_evaluation(Path(source))
    entries = read_manifest(Path(manifest))
    bundle = load_bundle(bundle_path, entries, require_ready=False)
    by_id = {e.task_id: e for e in entries}
    data = dict(
        version="saved_rollout_recovery_v1",
        source_evaluation=str(Path(source).resolve()),
        source_files=artifact.files,
        bundle_sha256=sha256_json(bundle),
        results=[],
        errors=[],
    )
    for row, raw in collect_complete(artifact):
        job = {k: row[k] for k in ("task_id", "trial", "seed")}
        try:
            sim = SimulationRun.model_validate(raw)
            task = adapt_record(ArealTaskRecord.model_validate(by_id[row["task_id"]].task))
            contract = bundle["tasks"][row["task_id"]]
            behavior = None
            if sim.termination_reason.value in SCORABLE_TERMINATIONS:
                behavior = judge_communication(
                    raw["messages"],
                    scenario=task.task.user_scenario.model_dump(mode="json"),
                    policy=build_system_prompt(),
                    contract=contract,
                    output=output / "behavior",
                    budget_directory=bundle["budget_directory"],
                )
            sim.info = dict(sim.info or {})
            sim.info["outcome_contract"] = score_simulation(
                sim, db_path=task.db_path, task=task.task, contract=contract, behavior=behavior
            )
            reward = sim.info["outcome_contract"]["reward"]
            data["results"].append(
                dict(
                    job,
                    reward=reward,
                    termination_reason=sim.termination_reason.value,
                    execution_eligibility=execution_eligibility(
                        sim.termination_reason.value, reward=reward
                    ),
                    simulation=sim.model_dump(mode="json"),
                    generation_reused_from=str(source),
                )
            )
            print(
                json.dumps(
                    dict(
                        job,
                        status="rescored",
                        behavior_passed=behavior["passed"] if behavior else None,
                    )
                ),
                flush=True,
            )
        except Exception as exc:
            data["errors"].append(dict(job, error_type=type(exc).__name__, error=str(exc)))
            print(json.dumps(dict(job, status="unresolved", error=str(exc))), flush=True)
        (output / "recovery.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "bundle", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    d = rescore(a.source, a.bundle, a.manifest, a.output)
    return int(bool(d["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
