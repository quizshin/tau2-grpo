"""Recover readable examples from saved veRL rollouts in a finished SwanLab run."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tau3_grpo.tracking.swanlab import (
    create_source_archive,
    load_tracking_env,
    log_examples,
)


def read_rollouts(directory: Path) -> list[tuple[int, Path, list[dict]]]:
    batches = []
    for path in sorted(directory.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        if not path.stem.isdigit():
            continue
        step = int(path.stem)
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("tau3_is_padding"):
                continue
            if value.get("step") != step:
                raise ValueError(f"Step mismatch in {path}")
            if not isinstance(value.get("input"), str) or not isinstance(value.get("output"), str):
                raise ValueError(f"Missing decoded prompt/response in {path}")
            if not value["input"] or not value["output"]:
                raise ValueError(f"Empty decoded prompt/response in {path}")
            rows.append({
                "task_id": value.get("task_id", ""), "reward": value["score"],
                "termination_reason": value.get("termination_reason", ""),
                "update_index": step, "source_file": path.name,
                "trajectory": {"prompt": value["input"], "response": value["output"]},
                "trajectory_metadata": value.get("trajectory_json"),
            })
        if rows:
            batches.append((step, path, rows))
    if not batches:
        raise ValueError(f"No real rollout batches in {directory}")
    return batches


def scalar_records(snapshot: dict) -> dict:
    # Equal-valued minima/maxima may resolve to different steps across queries.
    # Compare the recorded points, not those nondeterministic summary locations.
    return {entry["key"]: entry.get("metrics", []) for entry in snapshot.get("list", [])}


def config_values(config: dict) -> dict:
    """The public API returns SDK config entries wrapped in value/sort/desc."""
    return {key: entry["value"] for key, entry in config.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="workspace/project/run-slug")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    workspace, project, _ = args.run.split("/")
    batches = read_rollouts(args.rollout_dir)
    plan = {"run": args.run, "steps": [step for step, _, _ in batches],
            "trajectories": sum(len(rows) for _, _, rows in batches),
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for _, path, _ in batches}}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    receipt = args.output / "upload-complete.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("source") != plan:
            raise ValueError("Existing receipt describes different rollouts; choose another output directory")
        print(json.dumps(previous, indent=2))
        return 0
    load_tracking_env()
    import swanlab

    remote = swanlab.Api().run(args.run)
    if remote.state != "FINISHED":
        raise RuntimeError(f"Only backfill a finished run; current state is {remote.state}")
    original_config = remote.profile["config"]
    config = config_values(original_config)
    if config.get("trainer", {}).get("phase") != "rl":
        raise ValueError("Expected an RL run with its original training configuration")
    scalar_keys = ["trainer/global_step", "actor/grad_norm", "critic/rewards/mean"]
    steps = remote.metrics(keys=scalar_keys, all=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "recovered-dialogues-source.json"
    manifest.write_text(json.dumps({**plan, "training_metrics_before": steps,
        "original_config": original_config,
        "original_finished_at": remote.finished_at,
        "note": "Media-only recovery at original update indices; existing scalar metrics are not replayed."
    }, ensure_ascii=False, indent=2) + "\n")
    run = swanlab.init(workspace=workspace, project=project, id=remote.run_id,
                       resume="must", name=remote.name, mode="online",
                       config=config,
                       log_dir=str(args.output / "swanlog"))
    try:
        swanlab.define_metric("cases/recovered_*", section_name="cases")
        # New media keys accept original steps; SDK forbids rewriting existing keys/steps.
        for step, path, rows in batches:
            log_examples(run, rows, key="cases/recovered_dialogues", step=step,
                         output=args.output / "examples", count=4)
            run.save(str(path), base_path=str(args.rollout_dir), policy="now")
        archive = create_source_archive(args.output / "logging-code")
        run.save(str(archive), base_path=str(args.output), policy="now")
        run.save(str(manifest), base_path=str(args.output), policy="now")
    finally:
        # This is an upload session; do not mislabel the completed training if upload fails.
        run.finish()
    verified_steps = set()
    for attempt in range(3):
        media = remote.medias(keys=["cases/recovered_dialogues"], all=True)
        verified_steps = {point["index"] for entry in media.get("list", [])
                          for point in entry.get("metrics", []) if point.get("items")}
        if set(plan["steps"]) <= verified_steps:
            break
        if attempt < 2:
            time.sleep(2)
    if not set(plan["steps"]) <= verified_steps:
        raise RuntimeError("Cloud media has not arrived for every step; retry before declaring upload complete")
    if scalar_records(remote.metrics(keys=scalar_keys, all=True)) != scalar_records(steps):
        raise RuntimeError("Training metric readback changed during media-only recovery; inspect before retrying")
    if config_values(swanlab.Api().run(args.run).profile["config"]) != config:
        raise RuntimeError("Training configuration changed during media recovery")
    result = {"source": plan, "url": remote.url, "media_key": "cases/recovered_dialogues",
              "cloud_verified_steps": sorted(verified_steps), "training_metrics_unchanged": True,
              "training_config_unchanged": True}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
