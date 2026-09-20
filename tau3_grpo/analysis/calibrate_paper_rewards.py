"""Offline IRC rounds on complete training buffers, with task-held-out checks.

Each --round supplies newly collected complete update files. This program never
collects rollouts, trains, or treats repeated passes over a fixed buffer as fresh
IRC evidence. Only a passing last round outside development-only mode exports
a frozen, hashed recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo, group_normalize
from tau3_grpo.analysis.calibrate_process_rewards import analyze_update
from tau3_grpo.evaluation.process_reward import default_weights, reward_settings, score_turns

# Re-exports preserve historical analysis imports; runtime imports the public reader.
from tau3_grpo.evaluation.rewards.recipe import (
    DEFAULT_IRC as DEFAULT_IRC,
)
from tau3_grpo.evaluation.rewards.recipe import (
    alignment_issues as alignment_issues,
)
from tau3_grpo.evaluation.rewards.recipe import (
    digest as digest,
)
from tau3_grpo.evaluation.rewards.recipe import (
    implementation_hashes as implementation_hashes,
)
from tau3_grpo.evaluation.rewards.recipe import (
    load_frozen_recipe as load_frozen_recipe,
)
from tau3_grpo.evaluation.rewards.recipe import (
    validate_irc as validate_irc,
)


def task_split(task_ids, *, fraction, seed):
    ordered = sorted(task_ids, key=lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest())
    if len(ordered) < 2:
        raise ValueError("IRC requires multiple training tasks")
    count = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
    return set(ordered[count:]), set(ordered[:count])


def load_round(paths, allowed_tasks, seen_hashes):
    """Validate original rewards/advantages before any counterfactual scoring."""
    updates, sources = [], []
    for path in paths:
        path = Path(path).resolve()
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen_hashes:
            raise ValueError("duplicate buffer: a new IRC round requires new rollout evidence")
        seen_hashes.add(sha)
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not rows or any(row.get("task_id") not in allowed_tasks for row in rows):
            raise ValueError("buffer contains task outside the declared training manifest")
        analyze_update(rows)
        saved = [json.loads(row["mt_gtpo_replay_json"]) for row in rows]
        group_tasks = defaultdict(set)
        for row, data in zip(rows, saved, strict=True):
            group_tasks[data["uid"]].add(row["task_id"])
        if any(len(tasks) != 1 for tasks in group_tasks.values()):
            raise ValueError("rollout group crosses task identities")
        if any(data["settings"] != saved[0]["settings"] for data in saved):
            raise ValueError("mixed Hybrid settings")
        updates.append({
            "task_ids": [row["task_id"] for row in rows],
            "uids": [data["uid"] for data in saved],
            "outcomes": [data["outcome"] for data in saved],
            "processes": [data["process"] for data in saved],
            "algorithm": saved[0]["settings"],
        })
        sources.append({"path": str(path), "sha256": sha, "rows": len(rows)})
    return updates, sources


def rescore(updates, recipe):
    """Same Hybrid turn values using compact masks, preserving token weights.

    Group normalization is independent of token widths. One synthetic token per
    real turn saves memory; original spans supply token-weighted diagnostics.
    """
    observations = []
    for update in updates:
        processes = [score_turns(p["turn_records"], p["golden_actions"], p["reward_basis"],
                                recipe, official_outcome=outcome)
                     for p, outcome in zip(update["processes"], update["outcomes"], strict=True)]
        length = max(len(p["turn_rewards"]) for p in processes)
        masks = np.zeros((len(processes), length))
        spans = []
        for i, p in enumerate(processes):
            n = len(p["turn_rewards"])
            masks[i, :n] = 1
            spans.append([[k, k + 1] for k in range(n)])
        _, _, detail = compute_mt_gtpo(
            update["outcomes"], update["uids"], [p["turn_rewards"] for p in processes],
            spans, masks, **update["algorithm"],
        )
        # Algorithm 1's immediate-reward proxy is reported separately from Eq.(2).
        keys, immediate = [], []
        for uid, p in zip(update["uids"], processes, strict=True):
            for k, reward in enumerate(p["turn_rewards"]):
                keys.append((uid, k))
                immediate.append(reward)
        settings = update["algorithm"]
        proxy = group_normalize(immediate, keys, eps=settings["eps"],
                                min_group_size=settings["min_group_size"])
        cursor = 0
        for i, p in enumerate(processes):
            tiers, values, proxies = set(), defaultdict(list), defaultdict(list)
            mixed_positive = errors_positive = 0
            positive_tokens = all_tokens = 0
            for turn, advantage, span in zip(p["turn_records"], detail["turn_advantages"][i],
                                             p["turn_spans"], strict=True):
                present = set(turn["reward_types"])
                tiers.update(present)
                immediate_adv = proxy[cursor] + settings["lambda_outcome"] * detail["episode_advantages"][i]
                cursor += 1
                for tier in present:
                    values[tier].append(advantage)
                    proxies[tier].append(float(immediate_adv))
                events = turn["tool_calls"]
                good = any(e["reward"] > 0 for e in events)
                risky = any(e["reward_type"] in {"error", "state_change", "duplicate"} for e in events)
                mixed_positive += int(good and risky and advantage > 1e-12)
                errors_positive += sum(e["error"] and advantage > 1e-12 for e in events)
                n = span[1] - span[0]
                all_tokens += n
                positive_tokens += n * (advantage > 1e-12)
            observations.append({
                "task_id": update["task_ids"][i], "outcome": update["outcomes"][i],
                "tiers": tiers, "advantages": dict(values), "proxy_advantages": dict(proxies),
                "mean_reward": float(np.mean(p["turn_rewards"])),
                "mixed_positive_turns": mixed_positive, "error_positive_calls": errors_positive,
                "positive_tokens": positive_tokens, "tokens": all_tokens,
            })
    return observations


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(observations, config):
    outcomes = np.array([o["outcome"] for o in observations])
    result = {}
    for tier in default_weights(config.get("reward_version")):
        presence = np.array([int(tier in o["tiers"]) for o in observations])
        n = int(presence.sum())
        support = min(n, len(presence) - n, int(outcomes.sum()),
                      len(outcomes) - int(outcomes.sum()))
        values = [a for o in observations for a in o["advantages"].get(tier, [])]
        proxies = [a for o in observations for a in o["proxy_advantages"].get(tier, [])]
        result[tier] = {
            "trajectory_presence": n, "support": support,
            "supported": support >= config["min_support"],
            "rho": correlation(presence, outcomes),
            "mean_hybrid_advantage": float(np.mean(values)) if values else None,
            "mean_proxy_advantage": float(np.mean(proxies)) if proxies else None,
            "positive_turns": sum(a > 1e-12 for a in values),
            "negative_turns": sum(a < -1e-12 for a in values),
        }
    return {
        "trajectories": len(observations), "tasks": len({o["task_id"] for o in observations}),
        "successes": int(outcomes.sum()), "tiers": result,
        "reward_outcome_correlation": correlation([o["mean_reward"] for o in observations], outcomes),
        "mixed_positive_turns": sum(o["mixed_positive_turns"] for o in observations),
        "error_positive_calls": sum(o["error_positive_calls"] for o in observations),
        "failed_positive_tokens": sum(o["positive_tokens"] for o in observations if not o["outcome"]),
        "failed_tokens": sum(o["tokens"] for o in observations if not o["outcome"]),
    }


def calibrate_round(updates, recipe, calibration_tasks, holdout_tasks, config):
    validate_irc(config)
    if (config.get("reward_version", recipe["version"]) != recipe["version"]
            or set(recipe["weights"]) != set(config["intended_signs"]) | set(config["fixed_weights"])):
        raise ValueError("IRC tier schema/version differs from reward recipe")
    before = rescore(updates, recipe)
    fit = summarize([o for o in before if o["task_id"] in calibration_tasks], config)
    candidate = deepcopy(recipe)
    # Algorithm 1 uses categorical r_c. Explicit transition from Appendix A's
    # overlap initializer, instead of silently scaling r_c again by overlap.
    candidate["paper_options"]["soft_scoring"] = "constant"
    proposal_issues = []
    for tier in config["intended_signs"]:
        row = fit["tiers"][tier]
        if row["supported"] and row["rho"] is not None:
            rho = row["rho"]
            candidate["weights"][tier] = config["alpha"] * rho if abs(rho) > config["delta"] else 0.0
        else:
            proposal_issues.append(f"{tier}: insufficient_support_for_weight")
    candidate["weights"].update(config["fixed_weights"])
    candidate = reward_settings(candidate)
    after = rescore(updates, candidate)
    summaries = {
        "calibration": summarize([o for o in after if o["task_id"] in calibration_tasks], config),
        "holdout": summarize([o for o in after if o["task_id"] in holdout_tasks], config),
    }
    issues = {name: alignment_issues(summary, config) for name, summary in summaries.items()}
    issues["proposal"] = proposal_issues
    # Never derive intended signs from rho: that would bless inverted semantics.
    for tier, sign in config["intended_signs"].items():
        value = candidate["weights"][tier]
        if (sign and sign * value <= 0) or (sign == 0 and value != 0):
            issues["proposal"].append(f"{tier}: proposed_weight_conflicts_with_intended_sign")
    return {"initial_recipe": recipe, "initial_calibration": fit, "candidate_recipe": candidate,
            "checks": summaries, "issues": issues, "passed": not any(issues.values())}


def freeze_recipe(report):
    if report.get("development_only", False):
        raise ValueError("development-only evidence cannot freeze a training recipe")
    last = report["rounds"][-1]
    if not last["passed"] or any(last["issues"].values()):
        raise ValueError("IRC did not pass; cannot freeze a training recipe")
    artifact = {
        "schema": "mt_gtpo_irc_recipe_v2", "status": "passed",
        "reward": last["candidate_recipe"], "algorithm": report["algorithm"],
        "train_manifest_sha256": report["train_manifest"]["sha256"],
        "calibration_tasks": report["calibration_tasks"], "holdout_tasks": report["holdout_tasks"],
        "irc": report["irc"], "checks": last["checks"], "issues": last["issues"],
        "source_buffers": last["sources"], "report_sha256": digest(report),
        "implementation_sha256": implementation_hashes(),
        "scope": "offline_buffer_calibration_not_new_policy_performance",
    }
    artifact["sha256"] = digest(artifact)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-version", choices=("paper_v1", "paper_env_v2", "paper_env_split_v3"), default="paper_v1")
    parser.add_argument("--development-only", action="store_true",
                        help="Previously inspected buffers/splits: diagnostics only, never freeze")
    args = parser.parse_args(argv)
    raw_config = args.config.read_text()
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError:
        config = yaml.safe_load(raw_config)
    validate_irc(config)
    manifest_raw = args.train_manifest.read_bytes()
    entries = [json.loads(line) for line in manifest_raw.splitlines() if line.strip()]
    if not entries or any(e.get("split") != "train" for e in entries):
        raise ValueError("IRC accepts a training-only task manifest, never selection/final")
    task_ids = [e["task_id"] for e in entries]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task in training manifest")
    fit, holdout = task_split(task_ids, fraction=config["holdout_fraction"], seed=config["split_seed"])
    if args.output_dir.exists():
        raise FileExistsError("Use a new IRC output directory; do not overwrite audit evidence")
    recipe = reward_settings({"mode": "paper", "version": args.reward_version})
    report = {
        "schema": "mt_gtpo_irc_rounds_v1", "irc": config,
        "development_only": args.development_only,
        "implementation_sha256": implementation_hashes(),
        "config_sha256": hashlib.sha256(raw_config.encode()).hexdigest(),
        "train_manifest": {"path": str(args.train_manifest.resolve()),
                           "sha256": hashlib.sha256(manifest_raw).hexdigest()},
        "calibration_tasks": sorted(fit), "holdout_tasks": sorted(holdout), "rounds": [],
        "note": "Offline counterfactual on training buffers; no training or policy improvement claimed.",
    }
    seen = set()
    for paths in args.round:
        updates, sources = load_round(paths, set(task_ids), seen)
        algorithm = updates[0]["algorithm"]
        if any(u["algorithm"] != algorithm for u in updates) or report.get("algorithm", algorithm) != algorithm:
            raise ValueError("cannot mix Hybrid settings across calibration rounds")
        report["algorithm"] = algorithm
        result = calibrate_round(updates, recipe, fit, holdout, config)
        result["sources"] = sources
        report["rounds"].append(result)
        recipe = result["candidate_recipe"]
    passed = report["rounds"][-1]["passed"]
    report["status"] = ("development_checks_passed" if args.development_only else "passed") if passed else "needs_new_buffer_or_reward_revision"
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] == "passed":
        artifact = freeze_recipe(report)
        (args.output_dir / "frozen-recipe.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "report.json")}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
