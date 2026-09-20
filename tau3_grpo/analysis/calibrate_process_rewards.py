"""Audit saved mt_gtpo rollout JSONL; replay rules and advantages without models.

Each input file is one complete update, as emitted by veRL. Calibration uses
trajectory-level tier presence (not call-count weighting). Undefined correlations
remain null; suggestions never rewrite an active reward configuration.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tau3_grpo.algorithms.dynamic_filtering import apply_advantage_filter
from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
from tau3_grpo.evaluation.process_reward import DEFAULT_WEIGHTS, score_turns


def analyze_update(records):
    if not records:
        raise ValueError("empty update")
    replay = [json.loads(r["mt_gtpo_replay_json"]) for r in records]
    if any(r.get("schema") != "mt_gtpo_replay_v1" or r["process"] is None for r in replay):
        raise ValueError("expected complete real rollout update with replay metadata")
    settings, df = replay[0]["settings"], replay[0]["dynamic_filter"]
    if any(r["settings"] != settings or r["dynamic_filter"] != df for r in replay):
        raise ValueError("mixed algorithm/filter settings")
    outcomes = [r["outcome"] for r in replay]
    uids = [r["uid"] for r in replay]
    groups = Counter(uids)
    if any(
        groups[r["uid"]] != r["sampled_group_size"] or r["sampled_group_size"] < 2 for r in replay
    ):
        raise ValueError("incomplete or singleton rollout group")
    masks = np.asarray([r["response_mask"] for r in replay])
    processes = [r["process"] for r in replay]
    for p, outcome in zip(processes, outcomes, strict=True):
        if p["settings"]["mode"] in {"paper", "reference_write"} and p.get("official_outcome") != outcome:
            raise ValueError("saved process official outcome mismatch")
        recalculated = score_turns(
            p["turn_records"], p["golden_actions"], p["reward_basis"], p["settings"],
            official_outcome=p.get("official_outcome"),
        )
        if any(
            recalculated[key] != p[key] for key in ("turn_records", "turn_rewards", "turn_spans")
        ):
            raise ValueError("saved process rewards cannot be reproduced")
    adv, _, detail = compute_mt_gtpo(
        outcomes,
        uids,
        [p["turn_rewards"] for p in processes],
        [p["turn_spans"] for p in processes],
        masks,
        **settings,
    )
    filtered = masks
    if df.get("enable", False):
        filtered, _, _ = apply_advantage_filter(
            masks,
            adv,
            uids,
            group_size=int(df["group_size"]),
            tolerance=float(df.get("tolerance", 1e-12)),
        )
    for i, r in enumerate(replay):
        if not np.isclose(r["episode_advantage"], detail["episode_advantages"][i], atol=1e-10):
            raise ValueError("episode advantage replay mismatch")
        for key in ("turn_returns", "turn_advantages"):
            if not np.allclose(r[key], detail[key][i], atol=1e-10, rtol=1e-10):
                raise ValueError(f"{key} replay mismatch")
        if not np.array_equal(filtered[i], r["filtered_response_mask"]):
            raise ValueError("filter replay mismatch")
    observations = []
    for r, p in zip(replay, processes):
        tiers, by_tier = set(), defaultdict(list)
        for turn, advantage in zip(p["turn_records"], r["turn_advantages"], strict=True):
            for tier in set(turn["reward_types"]):
                tiers.add(tier)
                by_tier[tier].append(advantage)
        observations.append({"outcome": r["outcome"], "tiers": tiers, "advantages": dict(by_tier)})
    return observations


def summarize(observations, *, delta=0.05, min_support=20):
    outcomes = np.asarray([o["outcome"] for o in observations], dtype=float)
    if not np.isin(outcomes, [0, 1]).all():
        raise ValueError("IRC point-biserial analysis requires binary official outcomes")
    result = {}
    tiers = set(DEFAULT_WEIGHTS).union(*(o["tiers"] for o in observations))
    for tier in sorted(tiers):
        presence = np.asarray([tier in o["tiers"] for o in observations], dtype=float)
        count = int(presence.sum())
        defined = len(outcomes) > 1 and outcomes.std() > 0 and presence.std() > 0
        rho = float(np.corrcoef(presence, outcomes)[0, 1]) if defined else None
        values = [a for o in observations for a in o["advantages"].get(tier, [])]
        enough = (
            min(
                count,
                len(outcomes) - count,
                int(outcomes.sum()),
                int(len(outcomes) - outcomes.sum()),
            )
            >= min_support
        )
        result[tier] = {
            "trajectory_count": count,
            "correlation": rho,
            "presence_in_success": float(presence[outcomes == 1].mean())
            if (outcomes == 1).any()
            else None,
            "presence_in_failure": float(presence[outcomes == 0].mean())
            if (outcomes == 0).any()
            else None,
            "positive_advantage_turns": sum(a > 1e-12 for a in values),
            "negative_advantage_turns": sum(a < -1e-12 for a in values),
            "zero_advantage_turns": sum(abs(a) <= 1e-12 for a in values),
            "suggested_weight": (rho if abs(rho) > delta else 0.0)
            if rho is not None and enough
            else None,
        }
    return {
        "schema": "irc_audit_v1",
        "trajectories": len(observations),
        "tiers": result,
        "min_support": min_support,
        "delta": delta,
        "note": "Association only; proposed alpha=1 weights require separate task-held-out audit. No auto-apply.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="one complete saved training-update JSONL; repeat for more updates",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.min_support < 1 or not 0 <= args.delta <= 1:
        parser.error("invalid support/delta")
    observations = []
    recipe = None
    for path in args.input:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            saved = json.loads(row["mt_gtpo_replay_json"])
            current = (saved["settings"], saved["dynamic_filter"], saved["process"]["settings"])
            if recipe is not None and current != recipe:
                raise ValueError("mixed reward/algorithm versions; audit each recipe separately")
            recipe = current
        observations.extend(analyze_update(rows))
    report = summarize(observations, delta=args.delta, min_support=args.min_support)
    from tau3_grpo.utils.hashing import sha256_file

    report["sources"] = [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in args.input]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
