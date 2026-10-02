"""CPU-only strict database replay for diagnostic evidence; never changes scores."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.analysis.selection_audit import prepare
from tau3_grpo.envs.adapter import build_environment, load_flight_db
from tau3_grpo.envs.tau2_bridge import message_models, task_model
from tau3_grpo.tracking.judge_budget import dump
from tau3_grpo.utils.hashing import sha256_file


def differences(expected, actual, path=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                out.append(
                    {
                        "path": path + "/" + key,
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                        "kind": "missing_key",
                    }
                )
            else:
                out.extend(differences(expected[key], actual[key], path + "/" + key))
        return out
    if expected == actual:
        return []
    kind = "value"
    if isinstance(expected, list) and isinstance(actual, list):

        def canonical(v):
            return json.dumps(v, sort_keys=True)

        kind = (
            "list_order_only"
            if Counter(map(canonical, expected)) == Counter(map(canonical, actual))
            else "list_content"
        )
    return [{"path": path, "expected": expected, "actual": actual, "kind": kind}]


def run(root, output):
    tasks, cases, _ = prepare(root)
    by_task = {t["task_id"]: t for t in tasks}
    models = message_models()
    role_models = {
        k.lower().replace("message", ""): v for k, v in models.items() if k.endswith("Message")
    }
    output.mkdir(parents=True, exist_ok=True)
    for item in cases:
        rid, row = item["record_id"], item["row"]
        result = {"record_id": rid, "status": "unresolved", "strict": True}
        record = by_task[row["task_id"]]
        try:
            db_path = root / "data/raw/areal_tau2" / record["db_path"]
            result["db_sha256"] = sha256_file(db_path)
            if result["db_sha256"] != record["db_hash"]:
                raise ValueError("Source DB hash differs from manifest")
            task = task_model().model_validate(record["task"])
            db = load_flight_db(db_path)
            pred, gold = build_environment(db), build_environment(db)
            initial = task.initial_state
            init = dict(
                initialization_data=initial.initialization_data if initial else None,
                initialization_actions=initial.initialization_actions if initial else None,
                strict=True,
            )
            messages = [
                role_models[m["role"]].model_validate(m) for m in row["simulation"]["messages"]
            ]
            pred.set_state(message_history=messages, **init)
            gold.set_state(
                message_history=(initial.message_history or []) if initial else [], **init
            )
            gold_errors = []
            for action in task.evaluation_criteria.actions or []:
                try:
                    response = gold.make_tool_call(
                        tool_name=action.name, requestor=action.requestor, **action.arguments
                    )
                    if isinstance(response, str) and response.lower().startswith("error"):
                        gold_errors.append({"action": action.model_dump(), "response": response})
                except Exception as exc:
                    gold_errors.append({"action": action.model_dump(), "error": str(exc)})
            match = (
                pred.get_db_hash() == gold.get_db_hash()
                and pred.get_user_db_hash() == gold.get_user_db_hash()
            )
            saved = (row["simulation"].get("reward_info") or {}).get("db_check")
            diff = differences(gold.tools.db.model_dump(), pred.tools.db.model_dump())
            result.update(
                status="strict_replay_complete",
                db_match=match,
                saved_db_match=saved.get("db_match") if saved else None,
                agrees_with_saved_db_check=(match == saved["db_match"]) if saved else None,
                gold_execution_errors=gold_errors,
                differences=diff,
                difference_kinds=dict(Counter(x["kind"] for x in diff)),
                predicted_db_hash=pred.get_db_hash(),
                gold_db_hash=gold.get_db_hash(),
            )
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc)[:1500])
        dump(output / f"{rid}.json", result)
    rows = [json.loads(p.read_text()) for p in output.glob("*.json") if p.name != "summary.json"]
    dump(
        output / "summary.json",
        {
            "cases": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "saved_db_disagreements": sum(
                r.get("agrees_with_saved_db_check") is False for r in rows
            ),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
