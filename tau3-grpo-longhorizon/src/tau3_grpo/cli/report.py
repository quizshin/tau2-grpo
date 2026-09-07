"""CLI: build the deterministic run report.

Milestone D14. Reads one telemetry JSONL per arm, computes pass@k and the paired
bootstrap between two arms, and writes `report.json`.

    python -m tau3_grpo.cli.report --arm e0 results/e0/telemetry.jsonl \
        --arm e3 results/e3/telemetry.jsonl --baseline e0 --treatment e3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tau3_grpo.experiment.metrics import paired_bootstrap_by_task
from tau3_grpo.experiment.telemetry import build_report, write_report
from tau3_grpo.paths import RESULTS_ROOT, verify_four_root_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the run report")
    parser.add_argument(
        "--arm",
        action="append",
        nargs=2,
        metavar=("NAME", "TELEMETRY_JSONL"),
        required=True,
        help="repeatable: arm name and its telemetry file",
    )
    parser.add_argument("--experiment", default="tau3-grpo")
    parser.add_argument("--baseline", default=None, help="arm name for the bootstrap baseline")
    parser.add_argument("--treatment", default=None, help="arm name for the bootstrap treatment")
    parser.add_argument(
        "--per-task",
        type=Path,
        default=None,
        help="optional JSON: {arm: {task_id: [rewards...]}} for pass@k",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"telemetry not found at {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_four_root_layout()

    arms: dict[str, list[dict]] = {}
    for name, path in args.arm:
        try:
            arms[name] = _read_jsonl(Path(path))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    per_task = None
    if args.per_task is not None:
        if not args.per_task.is_file():
            print(f"error: per-task file not found at {args.per_task}", file=sys.stderr)
            return 2
        per_task = json.loads(args.per_task.read_text(encoding="utf-8"))

    extra: dict = {}
    if args.baseline and args.treatment:
        if per_task is None:
            print(
                "error: --baseline/--treatment need --per-task scores to pair by task",
                file=sys.stderr,
            )
            return 2
        missing = [name for name in (args.baseline, args.treatment) if name not in per_task]
        if missing:
            print(f"error: per-task file lacks arms {missing}", file=sys.stderr)
            return 2
        baseline_scores = {task: float(sum(v) / len(v)) for task, v in per_task[args.baseline].items()}
        treatment_scores = {
            task: float(sum(v) / len(v)) for task, v in per_task[args.treatment].items()
        }
        result = paired_bootstrap_by_task(
            baseline_scores,
            treatment_scores,
            resamples=args.resamples,
            seed=args.bootstrap_seed,
        )
        extra["paired_bootstrap"] = {
            "baseline": args.baseline,
            "treatment": args.treatment,
            **result.to_dict(),
        }

    report = build_report(
        experiment=args.experiment,
        arms=arms,
        per_task_rewards=per_task,
        extra=extra or None,
    )
    path = write_report(report, args.output_dir)
    print(f"wrote {path}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
