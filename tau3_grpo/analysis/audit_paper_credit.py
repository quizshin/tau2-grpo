"""Train-buffer diagnostics for paper soft/gold tiers and duplicate credit.

Read-only counterfactual analysis: no weight fitting, model calls or training.
The previous IRC split is retained for descriptive reporting, not a fresh test.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
from tau3_grpo.analysis.calibrate_paper_rewards import correlation, load_round
from tau3_grpo.evaluation.paper_reward import match_arguments, normalize
from tau3_grpo.evaluation.process_reward import deep_equal, execution_arguments, score_turns
from tau3_grpo.paths import CODE_ROOT

COMPONENTS = ("duplicate_immediate", "other_immediate", "future_process", "terminal")


def official_tool_types(path):
    result = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "is_tool" and decorator.args):
                value = decorator.args[0]
                if isinstance(value, ast.Attribute):
                    result[node.name] = value.attr
    return result


def argument_differences(actual, expected):
    """Top-level fields after the scorer's exact recursive normalization.

    Nested mismatches retain their full values in the case artifacts. A differing
    field is a reference mismatch, not proof of an invalid or unauthorized call.
    """
    left, right = dict(normalize(actual)[1]), dict(normalize(expected)[1])
    return {
        "changed": sorted(k for k in left.keys() & right.keys() if left[k] != right[k]),
        "missing": sorted(right.keys() - left.keys()),
        "extra": sorted(left.keys() - right.keys()),
    }


def execution_comparison(event, golds, previous):
    """Compare pinned tool-model inputs; no DB execution or policy judgment."""
    effective = execution_arguments(event["name"], event["arguments"])
    matches = [i for i, g in enumerate(golds) if g["name"] == event["name"]
               and deep_equal(effective, execution_arguments(g["name"], g["arguments"]))]
    prior = [i for i, e in enumerate(previous) if not e["error"] and e["name"] == event["name"]
             and deep_equal(effective, execution_arguments(e["name"], e["arguments"]))]
    return {"effective_arguments": effective, "execution_equivalent_gold_indices": matches,
            "prior_equivalent_success_call_indices": prior}


def duplicate_history(process, types):
    """Repeated arguments need not imply an unchanged response after a write."""
    previous, result = {}, defaultdict(list)
    write_count = 0
    for turn in process["turn_records"]:
        for index, event in enumerate(turn["tool_calls"]):
            signature = (event["name"], normalize(event["arguments"]))
            observation = event.get("observation")
            if isinstance(observation, str):
                try:
                    observation = json.loads(observation)
                except json.JSONDecodeError:
                    pass
            obs_hash = (hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()
                        if observation is not None else None)
            if event["reward_type"] == "duplicate":
                prior = previous.get(signature)
                result[turn["turn_index"]].append({
                    "tool": event["name"], "tool_type": types.get(event["name"], "UNKNOWN"),
                    "call_index": index, "prior": prior,
                    "response_changed": (obs_hash != prior["observation_sha256"]
                                         if prior and obs_hash and prior["observation_sha256"] else None),
                    "intervening_successful_writes": write_count - prior["write_count"] if prior else None})
            if not event["error"]:
                write_count += types.get(event["name"]) == "WRITE"
                previous[signature] = {"turn_index": turn["turn_index"], "call_index": index,
                                       "observation_sha256": obs_hash, "write_count": write_count}
    return result


def decompose(processes, outcomes, uids, settings):
    """Additive attribution using the *same* actual-return std for all terms.

    Separately normalizing each component would not reconstruct the algorithm.
    Missing positions/singletons/zero-variance groups follow compute_mt_gtpo.
    """
    n = len(processes)
    mask = np.zeros((n, max(len(p["turn_rewards"]) for p in processes)))
    spans = []
    for i, p in enumerate(processes):
        size = len(p["turn_rewards"])
        mask[i, :size] = 1
        spans.append([[k, k + 1] for k in range(size)])
    _, _, details = compute_mt_gtpo(
        outcomes, uids, [p["turn_rewards"] for p in processes], spans, mask, **settings)
    groups = defaultdict(list)
    records = []
    for i, p in enumerate(processes):
        for k, turn in enumerate(p["turn_records"]):
            reward = p["turn_rewards"][k]
            value = details["turn_returns"][i][k]
            dup = sum(e["reward"] for e in turn["tool_calls"] if e["reward_type"] == "duplicate")
            if p["settings"]["paper_options"]["aggregation"] == "mean":
                dup /= max(1, len(turn["tool_calls"]))
            terminal = settings["gamma"] ** (len(p["turn_rewards"]) - k) * outcomes[i]
            raw = {"duplicate_immediate": dup, "other_immediate": reward - dup,
                   "future_process": value - reward - terminal, "terminal": terminal}
            row = {"row_index": i, "turn_index": k, "return": value, "reward": reward,
                   "raw": raw, "advantage": details["turn_advantages"][i][k],
                   "episode_component": settings["lambda_outcome"] * details["episode_advantages"][i]}
            groups[(str(uids[i]), k)].append(len(records))
            records.append(row)
    max_error = 0.0
    for indices in groups.values():
        values = np.array([records[j]["return"] for j in indices])
        std = float(values.std())
        means = {c: float(np.mean([records[j]["raw"][c] for j in indices])) for c in COMPONENTS}
        active = len(indices) >= settings["min_group_size"] and std > 0
        for j in indices:
            row = records[j]
            parts = {c: (row["raw"][c] - means[c]) / (std + settings["eps"]) if active else 0.0
                     for c in COMPONENTS}
            error = abs(sum(parts.values()) + row["episode_component"] - row["advantage"])
            max_error = max(max_error, error)
            if not np.isclose(error, 0, atol=1e-9, rtol=0):
                raise ValueError("credit decomposition does not reconstruct Hybrid advantage")
            row.update(components=parts, group_size=len(indices), return_mean=float(values.mean()),
                       return_std=std, component_means=means)
    return records, max_error


def association(rows, feature):
    x = np.array([r[feature] for r in rows], dtype=float)
    y = np.array([r["outcome"] for r in rows], dtype=float)
    xc, yc = x.copy(), y.copy()
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["task_id"]].append(i)
    for indices in groups.values():
        xc[indices] -= x[indices].mean()
        yc[indices] -= y[indices].mean()
    return {"rho": correlation(x, y), "within_task_centered_rho": correlation(xc, yc),
            "present": int((x > 0).sum()), "absent": int((x == 0).sum()),
            "mean_success": float(x[y == 1].mean()) if (y == 1).any() else None,
            "mean_failure": float(x[y == 0].mean()) if (y == 0).any() else None}


def aggregate(rows, soft, duplicates):
    result = {"trajectories": len(rows), "successes": sum(r["outcome"] for r in rows),
              "features": {}, "soft": {}, "duplicates": {}}
    features = ("gold_any", "gold_read", "gold_write", "gold_generic", "gold_count",
                "gold_fraction", "gold_all", "gold_read_only", "soft_any")
    for f in features:
        result["features"][f] = association(rows, f)
    result["gold_tools"] = dict(sum((Counter(r["gold_tools"]) for r in rows), Counter()))
    by_tool = {}
    for name in sorted({e["tool"] for e in soft}):
        events = [e for e in soft if e["tool"] == name]
        fields = Counter(k for e in events for k in e["best_matches"][0]["differences"]["changed"])
        by_tool[name] = {"calls": len(events), "failed_trajectory_calls": sum(not e["outcome"] for e in events),
                         "first_best_match_changed_fields": dict(fields),
                         "execution_equivalent_gold_calls": sum(bool(e["execution_equivalent_gold_indices"]) for e in events),
                         "equivalent_gold_success_trajectory_calls": sum(bool(e["execution_equivalent_gold_indices"]) and bool(e["outcome"]) for e in events),
                         "prior_equivalent_success_calls": sum(bool(e["prior_equivalent_success_call_indices"]) for e in events),
                         "target_id_mismatch_all_best": sum(e["target_id_mismatch_all_best"] for e in events)}
    result["soft"] = {"calls": len(soft), "tools": by_tool,
                      "target_id_mismatch_all_best": sum(e["target_id_mismatch_all_best"] for e in soft)}
    subclasses = {}
    for kind in ("execution_equivalent_gold", "reference_different_write", "reference_different_read"):
        events = [e for e in soft if ("execution_equivalent_gold" if e["execution_equivalent_gold_indices"]
                                     else "reference_different_write" if e["tool_type"] == "WRITE"
                                     else "reference_different_read") == kind]
        identities = {(e["source"], e["line"]) for e in events}
        obs = [{**r, "present": int((r["source"], r["line"]) in identities)} for r in rows]
        subclasses[kind] = {"calls": len(events), "calls_on_successful_trajectories": sum(e["outcome"] for e in events),
                            "association": association(obs, "present")}
    result["soft"]["subclasses"] = subclasses
    for version in sorted({d["recipe"] for d in duplicates}):
        all_d = [d for d in duplicates if d["recipe"] == version]
        positive = [d for d in all_d if d["advantage"] > 1e-12]
        components = [*COMPONENTS, "episode"]
        def parts(d):
            return {**d["components"], "episode": d["episode_component"]}
        dominant = Counter(max(parts(d), key=parts(d).get) for d in positive)
        result["duplicates"][version] = {
            "turns": len(all_d), "positive_turns": len(positive),
            "mean_advantage": float(np.mean([d["advantage"] for d in all_d])) if all_d else None,
            "positive_failed_turns": sum(not d["outcome"] for d in positive),
            "positive_with_other_positive_calls": sum(d["other_positive_calls"] > 0 for d in positive),
            "positive_with_nonpositive_turn_reward": sum(d["reward"] <= 0 for d in positive),
            "dominant_positive_component": dict(dominant),
            "positive_component_means": {c: float(np.mean([parts(d)[c] for d in positive])) if positive else None
                                         for c in components},
            "without_component_nonpositive": {c: sum(d["advantage"] - parts(d)[c] <= 1e-12 for d in positive)
                                               for c in components},
            "tools": dict(Counter(e["name"] for d in all_d for e in d["calls"] if e["tier"] == "duplicate")),
            "read_calls_with_changed_response": sum(h["tool_type"] == "READ" and h["response_changed"] is True
                                                   for d in all_d for h in d["history"]),
            "positive_read_calls_with_changed_response": sum(h["tool_type"] == "READ" and h["response_changed"] is True
                                                            for d in positive for h in d["history"]),
            "read_calls_after_successful_write": sum(h["tool_type"] == "READ" and bool(h["intervening_successful_writes"])
                                                    for d in all_d for h in d["history"]),
        }
    return result


def audit(updates, sources, report, types):
    trajectories, soft, duplicates = [], [], []
    max_error = 0.0
    last = report["rounds"][-1]
    for batch, update in enumerate(updates):
        for version, recipe in (("initializer", last["initial_recipe"]), ("candidate", last["candidate_recipe"])):
            processes = [score_turns(p["turn_records"], p["golden_actions"], p["reward_basis"], recipe,
                                    official_outcome=y)
                         for p, y in zip(update["processes"], update["outcomes"], strict=True)]
            records, error = decompose(processes, update["outcomes"], update["uids"], update["algorithm"])
            histories = [duplicate_history(p, types) for p in processes]
            max_error = max(max_error, error)
            for i, p in enumerate(processes):
                identity = {"source": sources[batch]["path"], "line": i + 1,
                            "task_id": update["task_ids"][i], "uid": update["uids"][i],
                            "outcome": update["outcomes"][i]}
                events = [e for t in p["turn_records"] for e in t["tool_calls"]]
                gold = [e for e in events if e["reward_type"] == "gold_exact"]
                if version == "initializer":
                    official = Counter(types.get(e["name"], "UNKNOWN") for e in gold)
                    n_gold = len(p["golden_actions"])
                    trajectories.append({**identity, "gold_any": int(bool(gold)), "gold_count": len(gold),
                                         "gold_read": int(official["READ"] > 0), "gold_write": int(official["WRITE"] > 0),
                                         "gold_generic": int(official["GENERIC"] > 0),
                                         "gold_fraction": len(gold) / n_gold if n_gold else 0,
                                         "gold_all": int(n_gold > 0 and len(gold) == n_gold),
                                         "gold_read_only": int(bool(gold) and official["READ"] == len(gold)),
                                         "gold_tools": dict(Counter(e["name"] for e in gold)),
                                         "reference_actions": n_gold,
                                         "soft_any": int(any(e["reward_type"] == "soft_match" for e in events))})
                previous = []
                for turn in p["turn_records"]:
                    if version != "initializer":
                        continue
                    for call_index, event in enumerate(turn["tool_calls"]):
                        preceding = list(previous)
                        previous.append(event)
                        if event["reward_type"] != "soft_match":
                            continue
                        matches = []
                        for gi, g in enumerate(p["golden_actions"]):
                            _, overlap = match_arguments(event, g)
                            if overlap > 0 and overlap == event["paper_match_fraction"]:
                                matches.append({"gold_index": gi, "arguments": g["arguments"],
                                                "differences": argument_differences(event["arguments"], g["arguments"])})
                        if not matches:
                            raise ValueError("soft match has no matching reference")
                        # Explicit narrow flag, not a general authorization/correctness label.
                        id_mismatch = all(any(k in m["differences"]["changed"] + m["differences"]["missing"]
                                              for k in ("reservation_id", "user_id")) for m in matches)
                        soft.append({**identity, "turn_index": turn["turn_index"], "call_index": call_index,
                                     "tool": event["name"], "tool_type": types.get(event["name"], "UNKNOWN"),
                                     "arguments": event["arguments"], "overlap": event["paper_match_fraction"],
                                     "reward": event["reward"], "best_matches": matches,
                                     **execution_comparison(event, p["golden_actions"], preceding),
                                     "target_id_mismatch_all_best": id_mismatch})
            for row in records:
                i, k = row["row_index"], row["turn_index"]
                turn = processes[i]["turn_records"][k]
                if not any(e["reward_type"] == "duplicate" for e in turn["tool_calls"]):
                    continue
                calls = [{"name": e["name"], "arguments": e["arguments"], "tier": e["reward_type"],
                          "reward": e["reward"]} for e in turn["tool_calls"]]
                duplicates.append({**row, "source": sources[batch]["path"], "line": i + 1,
                                   "task_id": update["task_ids"][i], "uid": update["uids"][i],
                                   "outcome": update["outcomes"][i], "recipe": version, "calls": calls,
                                   "history": histories[i][k],
                                   "other_positive_calls": sum(e["reward"] > 0 and e["tier"] != "duplicate" for e in calls)})
    splits = {"all_development": {r["task_id"] for r in trajectories},
              "previous_calibration": set(report["calibration_tasks"]),
              "previous_holdout_now_inspected": set(report["holdout_tasks"])}
    summaries = {name: aggregate([r for r in trajectories if r["task_id"] in tasks],
                                [r for r in soft if r["task_id"] in tasks],
                                [r for r in duplicates if r["task_id"] in tasks])
                 for name, tasks in splits.items()}
    per_task = {t: aggregate([r for r in trajectories if r["task_id"] == t], [], [])
                for t in sorted(splits["all_development"])}
    return {"summary": summaries, "per_task": per_task, "decomposition_max_error": max_error}, trajectories, soft, duplicates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--irc-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Do not overwrite previous diagnostic evidence")
    report = json.loads(args.irc_report.read_text())
    manifest = Path(report["train_manifest"]["path"])
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != report["train_manifest"]["sha256"]:
        raise ValueError("training manifest changed")
    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(e.get("split") != "train" for e in entries):
        raise ValueError("training-only manifest required")
    tasks = {e["task_id"] for e in entries}
    fit, holdout = set(report["calibration_tasks"]), set(report["holdout_tasks"])
    if fit & holdout or fit | holdout != tasks:
        raise ValueError("invalid task split")
    sources = report["rounds"][-1]["sources"]
    updates, verified = load_round([Path(s["path"]) for s in sources], tasks, set())
    if verified != sources or any(u["algorithm"] != report["algorithm"] for u in updates):
        raise ValueError("source buffers or algorithm changed")
    tools_path = CODE_ROOT / "tau2-bench/src/tau2/domains/airline/tools.py"
    types = official_tool_types(tools_path)
    result, rows, soft, duplicates = audit(updates, sources, report, types)
    result.update(schema="paper_credit_audit_v1", sources=sources, tool_types=types,
                  recipes={k: report["rounds"][-1][k] for k in ("initial_recipe", "candidate_recipe")},
                  note="Historical train-only development diagnostics. No fresh holdout, fitting, sampling or causal effect estimate.")
    from tau3_grpo.analysis.calibrate_paper_rewards import implementation_hashes
    result["sha256"] = {**implementation_hashes(),
                        "audit_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "irc_report": hashlib.sha256(args.irc_report.read_bytes()).hexdigest(),
                        "official_tools": hashlib.sha256(tools_path.read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, records in (("trajectories", rows), ("soft_calls", soft), ("duplicate_turns", duplicates)):
        with (args.output_dir / f"{name}.jsonl").open("x") as stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({"trajectories": len(rows), "soft_calls": len(soft),
                      "duplicate_turn_recipe_records": len(duplicates),
                      "decomposition_max_error": result["decomposition_max_error"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
