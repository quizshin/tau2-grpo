"""Public, versioned IRC recipe validation; fitting stays in analysis.

Legacy v1 acceptance is restricted to the exact pre-migration calibration source
and unchanged numerical/benchmark identities. Old artifacts are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
from tau3_grpo.evaluation.process_reward import default_weights, reward_settings

LEGACY_CALIBRATION_SHA256 = "def9e690cccce3e3297269a98d9177403e26ea7697b9dc3427a5f90ce0281f8d"

DEFAULT_IRC = {
    "alpha": 1.0, "delta": 0.05, "eta": 0.1, "min_support": 20,
    "holdout_fraction": 0.2, "split_seed": 42, "alignment_tolerance": 1e-8,
    "intended_signs": {"gold_exact": 1, "soft_match": 1, "state_change": -1,
                       "error": -1, "duplicate": -1},
    "fixed_weights": {"read_only": 0.0, "message": 0.0, "unknown": 0.0},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ("evaluation/process_reward.py", "evaluation/paper_reward.py", "evaluation/environment_reward.py",
             "algorithms/mt_gtpo.py", "analysis/calibrate_paper_rewards.py", "evaluation/rewards/recipe.py")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    for name in ("tau2-bench/src/tau2/domains/airline/data_model.py", "tau2-bench/src/tau2/domains/airline/tools.py"):
        hashes[name] = hashlib.sha256((root.parent / name).read_bytes()).hexdigest()
    return hashes


def validate_irc(config):
    if set(config) - {"fixed_anchor_policy", "reward_version"} != set(DEFAULT_IRC):
        raise ValueError("IRC configuration must explicitly specify all documented fields")
    for key in ("alpha", "delta", "eta", "holdout_fraction", "alignment_tolerance"):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]):
            raise ValueError(f"non-finite IRC {key}")
    if not (config["alpha"] > 0 and 0 <= config["delta"] <= 1 and 0 <= config["eta"] < 1
            and 0 < config["holdout_fraction"] < 1 and config["alignment_tolerance"] >= 0):
        raise ValueError("invalid IRC thresholds")
    if type(config["min_support"]) is not int or config["min_support"] < 2:
        raise ValueError("min_support must be at least 2")
    if type(config["split_seed"]) is not int:
        raise ValueError("split_seed must be an integer")
    signs, fixed = config["intended_signs"], config["fixed_weights"]
    version = config.get("reward_version", "paper_v1")
    reward_settings({"mode": "paper", "version": version})
    if set(signs) & set(fixed) or set(signs) | set(fixed) != set(default_weights(version)):
        raise ValueError("every tier must be either calibrated or explicitly fixed")
    if any(type(s) is not int or s not in (-1, 0, 1) for s in signs.values()):
        raise ValueError("intended signs must be -1, 0, or 1")
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in fixed.values()):
        raise ValueError("fixed reward weights must be finite")
    anchor_policy = config.get("fixed_anchor_policy")
    if anchor_policy not in (None, "gold_reference_v1"):
        raise ValueError("unsupported fixed anchor policy")
    nonzero = {tier: value for tier, value in fixed.items() if value != 0}
    if anchor_policy == "gold_reference_v1":
        if nonzero != {"gold_exact": 1.0}:
            raise ValueError("gold_reference_v1 requires only a gold_exact=1 fixed anchor")
    elif nonzero:
        raise ValueError("nonzero fixed weights require an explicit fixed anchor policy")


def alignment_issues(summary, config):
    issues = []
    tol = config["alignment_tolerance"]
    for tier, intended in config["intended_signs"].items():
        row = summary["tiers"][tier]
        if not row["supported"] or row["rho"] is None:
            issues.append(f"{tier}: insufficient_support")
            continue
        for key in ("mean_proxy_advantage", "mean_hybrid_advantage"):
            value = row[key]
            aligned = value is not None and (
                abs(value) <= tol if intended == 0 else intended * value > tol
            )
            if not aligned:
                issues.append(f"{tier}: {key}_misaligned")
    if config.get("fixed_anchor_policy") == "gold_reference_v1":
        # A prescribed anchor is not fitted from presence/outcome rho. Still
        # require observed support and both advantage directions on each split.
        row = summary["tiers"]["gold_exact"]
        if row["trajectory_presence"] < config["min_support"]:
            issues.append("gold_exact: insufficient_anchor_occurrences")
        for key in ("mean_proxy_advantage", "mean_hybrid_advantage"):
            if row[key] is None or row[key] <= tol:
                issues.append(f"gold_exact: {key}_misaligned")
    rho = summary["reward_outcome_correlation"]
    if rho is None or rho <= config["eta"]:
        issues.append("reward_outcome_correlation_below_eta_or_undefined")
    if summary["tiers"]["unknown"]["trajectory_presence"]:
        issues.append("unknown_tool_tier_present")
    return issues


def load_frozen_recipe(path, *, manifest_sha256=None):
    artifact = json.loads(Path(path).read_text())
    hashed = {k: v for k, v in artifact.items() if k != "sha256"}
    if artifact.get("schema") not in {"mt_gtpo_irc_recipe_v1", "mt_gtpo_irc_recipe_v2"} or artifact.get("status") != "passed":
        raise ValueError("requires a passed frozen IRC recipe")
    if artifact.get("sha256") != digest(hashed):
        raise ValueError("frozen recipe hash mismatch")
    expected = implementation_hashes()
    if artifact["schema"] == "mt_gtpo_irc_recipe_v1":
        expected.pop("evaluation/rewards/recipe.py")
        expected["analysis/calibrate_paper_rewards.py"] = LEGACY_CALIBRATION_SHA256
    if artifact.get("implementation_sha256") != expected:
        raise ValueError("reward/Hybrid/calibration implementation changed; re-audit frozen recipe")
    validate_irc(artifact["irc"])
    if (any(artifact["issues"].values()) or any(
        alignment_issues(artifact["checks"][part], artifact["irc"])
        for part in ("calibration", "holdout")
    )):
        raise ValueError("frozen recipe contains failed IRC checks")
    reward = reward_settings(artifact["reward"])
    if (artifact["irc"].get("reward_version", reward["version"]) != reward["version"]
            or set(reward["weights"]) != set(artifact["irc"]["intended_signs"]) | set(artifact["irc"]["fixed_weights"])):
        raise ValueError("frozen IRC tier schema/version differs from reward recipe")
    if reward["mode"] != "paper" or reward != artifact["reward"]:
        raise ValueError("invalid frozen paper reward configuration")
    if reward["paper_options"]["soft_scoring"] != "constant":
        raise ValueError("frozen IRC weights must be categorical")
    from tau3_grpo.integrations.verl.mt_gtpo import settings_from_config

    algorithm = artifact["algorithm"]
    if settings_from_config({"mt_gtpo": algorithm}) != algorithm:
        raise ValueError("incomplete frozen Hybrid settings")
    # Reuse the numerical implementation's full parameter validation.
    compute_mt_gtpo([0, 1], ["g", "g"], [[0], [0]], [[[0, 1]], [[0, 1]]],
                    np.ones((2, 1)), **algorithm)
    for tier, sign in artifact["irc"]["intended_signs"].items():
        value = reward["weights"][tier]
        if (sign and sign * value <= 0) or (sign == 0 and value != 0):
            raise ValueError("frozen weight contradicts intended sign")
    if any(reward["weights"][tier] != value for tier, value in artifact["irc"]["fixed_weights"].items()):
        raise ValueError("frozen fixed weights changed")
    fit, holdout = set(artifact["calibration_tasks"]), set(artifact["holdout_tasks"])
    if not fit or not holdout or fit & holdout:
        raise ValueError("invalid frozen task split")
    if manifest_sha256 is not None and artifact["train_manifest_sha256"] != manifest_sha256:
        raise ValueError("IRC training manifest differs from training run")
    return artifact
