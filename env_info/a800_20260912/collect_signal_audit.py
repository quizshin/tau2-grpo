"""Read-only, CPU-only extraction of existing training evidence; emit JSON to stdout."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, code):
    out = {"captured_at": datetime.now(timezone.utc).isoformat(),
           "root": str(root), "arms": {}, "source_sha256": {}}
    for name in ["tau3_grpo/algorithms/tau_gigpo.py", "tau3_grpo/algorithms/dynamic_filtering.py",
                 "tau3_grpo/algorithms/verl_estimator.py", "tau3_grpo/integrations/anchor_hook.py",
                 "tau3_grpo/tracking/trainer_telemetry.py", "verl/verl/trainer/ppo/ray_trainer.py",
                 "verl/verl/trainer/ppo/metric_utils.py", "tau3_grpo/algorithms/anchors/encoder.py",
                 "tau3_grpo/envs/interaction.py", "verl/verl/experimental/agent_loop/tool_agent_loop.py"]:
        out["source_sha256"][name] = digest(code / name)
    for arm in ["e0", "e1", "e2", "e3"]:
        run = root / (arm + "_seed42")
        data = {"files_sha256": {}, "metrics": [], "updates": [], "raw_signal_files": []}
        for name in ["metrics.jsonl", "telemetry.jsonl", "experiment_manifest.json", "launch.json"]:
            p = run / name
            data["files_sha256"][name] = digest(p)
            if name == "metrics.jsonl":
                data["metrics"] = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            elif name == "experiment_manifest.json":
                data["experiment_manifest"] = json.loads(p.read_text())
            elif name == "launch.json":
                launch = json.loads(p.read_text()).get("configuration", {}).get("launch", {})
                data["debug_batch_dir_setting"] = launch.get("environment", {}).get("TAU3_GRPO_DEBUG_BATCH_DIR")
                data["algorithm_overrides"] = [x for x in launch.get("overrides", []) if x.startswith("algorithm.")]
        for p in sorted((run / "rollouts").glob("*.jsonl"), key=lambda x: int(x.stem)):
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            data["files_sha256"][str(p.relative_to(run))] = digest(p)
            compact = []
            for row in rows:
                trajectory = json.loads(row["trajectory_json"])
                compact.append({"task_id": row["task_id"], "score": row["score"],
                                "gts": row["gts"], "scored": row["scored"],
                                "termination_reason": row["termination_reason"],
                                "failure_category": row["failure_category"],
                                "trajectory": trajectory,
                                "has_anchor_ids": "anchor_ids" in row,
                                "has_anchor_spans": "anchor_spans" in row,
                                "has_response_mask": "response_mask" in row})
            data["updates"].append({"step": int(p.stem), "rows": compact,
                                    "row_keys": sorted(set().union(*(row.keys() for row in rows)))})
        # Inventory only: never load large checkpoints or tensors.
        for p in run.rglob("*"):
            if p.is_file() and ("anchor" in p.name or p.suffix == ".pkl" or "batch" in p.name):
                data["raw_signal_files"].append({"path": str(p.relative_to(run)), "bytes": p.stat().st_size})
        out["arms"][arm] = data
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.code), ensure_ascii=False))
