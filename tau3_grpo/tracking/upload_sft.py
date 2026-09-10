"""Upload completed SFT runs without retraining or relabelling dataset examples as generations."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from tau3_grpo.paths import PROJECT_ROOT
from tau3_grpo.tracking.swanlab import (
    experiment_name,
    log_examples,
    log_sft_dataset,
    run_url,
    scalar_metrics,
    sft_metrics,
    start_run,
)


def historical_metrics(summary: dict, state: dict) -> dict[int, dict]:
    steps = {0: sft_metrics(summary.get("baseline_validation") or {})}
    for row in state["log_history"]:
        step = int(row.get("step", 0))
        steps.setdefault(step, {}).update(sft_metrics(row))
    for step, values in steps.items():
        values["train/global_step"] = step
    return dict(sorted(steps.items()))


def upload_method(root: Path, method: str, *, require_evaluation: bool, dry_run: bool) -> dict:
    checkpoint = root / method / f"sft_{method}_seed42"
    summary = json.loads((checkpoint / "train_summary.json").read_text())
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    evaluation_dir = root / method / "evaluation/selection-isolated"
    evaluation = None
    if (evaluation_dir / "summary.json").exists():
        evaluation = json.loads((evaluation_dir / "summary.json").read_text())
        if evaluation.get("score_version") != "isolated_airline_db_v1":
            raise ValueError("Refusing evaluation scores without isolated database replay")
        if evaluation.get("replay_errors") or evaluation["scored_trajectories"] != evaluation["planned_trajectories"]:
            raise ValueError("Evaluation is incomplete or contains replay errors")
    elif require_evaluation:
        raise FileNotFoundError(evaluation_dir / "summary.json")
    steps = historical_metrics(summary, state)
    plan = {"method": method, "logged_steps": list(steps), "evaluation": evaluation,
            "best_validation_loss": summary["best_validation_loss"]}
    if dry_run:
        return plan
    output = root / method / "swanlab-upload"
    receipt = output / "upload-complete.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    run = start_run(
        name=experiment_name(stage="sft", model=summary["model_name_or_path"], method=method,
                             lr=summary["learning_rate"], seed=summary["seed"]), output=output,
        config={"stage": "sft", "historical_upload": True, "method": method,
                "seed": summary["seed"],
                "actor": {"model": summary["model_name_or_path"], "training_mode": method,
                          "optim": {"lr": summary["learning_rate"]}},
                "trainer": {"phase": "sft", "backend": "veRL / local HF Trainer",
                            "seed": summary["seed"], "epochs": 5},
                "training_summary": summary,
                "source_snapshot_note": "Current code including fixes added after this SFT run",
                "evaluation_protocol": evaluation},
    )
    # Dataset references at step 0 are labelled separately from model-generated trajectories.
    for split in ("train", "validation"):
        path = PROJECT_ROOT / f"data/sft/airline_sft_{split}_seed42.jsonl"
        if path.exists():
            log_sft_dataset(run, path, output / "examples", split)
    for step, metrics in steps.items():
        if metrics:
            run.log(metrics, step=step)
    final_step = int(summary["actual_optimizer_steps"])
    run.log(scalar_metrics(summary, "summary"), step=final_step)
    for filename in ("train_summary.json", "trainer_state.json"):
        run.save(str(checkpoint / filename), base_path=str(checkpoint), policy="now")
    if evaluation is not None:
        run.log(scalar_metrics(evaluation, "selection"), step=final_step)
        rows = [json.loads(line) for line in (evaluation_dir / "trajectories.jsonl").read_text().splitlines()]
        log_examples(run, rows, key="selection/generated_dialogues", step=final_step,
                     output=output / "examples", count=12)
        archive = output / "selection-full-trajectories.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for filename in ("summary.json", "trajectories.jsonl", "runtime-errors.jsonl", "errors.jsonl"):
                path = evaluation_dir / filename
                if path.exists():
                    handle.write(path, filename)
        run.save(str(archive), base_path=str(output), policy="now")
    comparison = root / "comparison-report.json"
    if comparison.exists():
        run.save(str(comparison), base_path=str(root), policy="now")
    result = {**plan, "url": run_url(run), "local_dir": str(run.dir)}
    run.finish()
    result["uploaded_online"] = bool(result["url"])
    if not result["uploaded_online"]:
        receipt = output / "offline-complete.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["full", "lora"], default=["full", "lora"])
    parser.add_argument("--require-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    # Preflight all methods before creating any cloud experiments.
    for method in args.methods:
        upload_method(args.results_root, method, require_evaluation=args.require_evaluation, dry_run=True)
    results = [upload_method(args.results_root, method, require_evaluation=args.require_evaluation,
                             dry_run=args.dry_run) for method in args.methods]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
