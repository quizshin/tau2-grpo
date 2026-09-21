"""Execute explicitly authored training-task demonstrations in isolated databases.

This produces authored SFT examples, not teacher rollouts or benchmark scores.
Tool observations are real outputs and the final DB must match a separate
execution of the training task's expected actions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau3_grpo.data.sft import reason_similarity
from tau3_grpo.envs.adapter import build_environment, load_flight_db
from tau3_grpo.prompts import build_system_prompt, prompt_provenance
from tau3_grpo.utils.hashing import sha256_file, sha256_text


def build_demonstration(recipe, entry, *, db_root, heldout_text):
    if entry["split"] != "train" or entry["task"].get("initial_state"):
        raise ValueError("Only fresh, explicitly designated training tasks are supported")
    path = Path(db_root) / entry["db_path"]
    if sha256_file(path) != entry["db_hash"]:
        raise ValueError("Training DB identity changed")
    identities = {value for action in entry["task"]["evaluation_criteria"]["actions"]
                  for key, value in action["arguments"].items() if key in {"user_id", "reservation_id"}}
    if any(value in heldout_text for value in identities):
        raise ValueError("Training demonstration shares user/reservation identity with held-out tasks")
    db, gold_db = load_flight_db(path), load_flight_db(path)
    environment, gold = build_environment(db), build_environment(gold_db)
    messages = [{"role": "system", "content": build_system_prompt()}]
    executions = []
    for step in recipe["steps"]:
        if "role" in step:
            if step["role"] not in {"user", "assistant"} or set(step) != {"role", "content"}:
                raise ValueError("Only user/assistant text may be authored; tools must be executed")
            messages.append(dict(step))
            continue
        name, arguments = step["tool"], step["arguments"]
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": arguments}]})
        result = environment.use_tool(name, **arguments)
        value = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        messages.append({"role": "tool", "name": name, "content": content})
        executions.append({"name": name, "arguments": arguments, "response_sha256": sha256_text(content)})
    for action in entry["task"]["evaluation_criteria"]["actions"]:
        gold.use_tool(action["name"], **action["arguments"])
    actual, expected = db.model_dump(mode="json"), gold_db.model_dump(mode="json")
    if actual != expected:
        raise ValueError("Executed demonstration disagrees with expected final training-task DB")
    return {"messages": messages, "metadata": {
        **prompt_provenance(), "source": "areal_rl_train_authored_executed",
        "source_revision": entry["source_revision"],
        "source_dialog_id": "authored_" + entry["task_id"], "source_task_id": entry["task_id"],
        "source_task_hash": entry["task_hash"], "source_db_hash": entry["db_hash"],
        "reason_for_call": entry["task"]["user_scenario"]["instructions"]["reason_for_call"],
        "construction": "authored dialogue, actual tools, isolated DB; no teacher model",
        "verification": "tool_execution_and_reference_final_db_equality",
        "independent_task_success_evaluation": False,
        "dialogue_hash": sha256_text(json.dumps(messages, sort_keys=True)),
        "final_db_hash": sha256_text(json.dumps(actual, sort_keys=True)), "executions": executions,
    }}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipes", "train-manifest", "heldout-tasks", "blocked-reasons", "db-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    entries = {r["task_id"]: r for r in map(json.loads, args.train_manifest.read_text().splitlines())}
    blocked = json.loads(args.blocked_reasons.read_text())
    result = []
    for recipe in json.loads(args.recipes.read_text()):
        entry = entries[recipe["task_id"]]
        reason = entry["task"]["user_scenario"]["instructions"]["reason_for_call"]
        if max((reason_similarity(reason, r) for r in blocked), default=0) >= 0.88:
            raise ValueError(f"Held-out intent overlap: {recipe['task_id']}")
        result.append(build_demonstration(recipe, entry, db_root=args.db_root,
                                           heldout_text=args.heldout_tasks.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result))
    print(json.dumps({"dialogues": len(result), "sha256": sha256_file(args.output),
                      "recipes_sha256": sha256_file(args.recipes), "verification": "real tools and final DB equality"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
