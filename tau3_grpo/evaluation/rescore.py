"""Recompute pass@k/pass^k from saved independent evaluation, without a GPU.

    python -m tau3_grpo.evaluation.rescore --run-dir results/evaluation/sft \
        --ks 1 2 4 --include-pass-hat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tau3_grpo.evaluation.artifacts import read_evaluation
from tau3_grpo.evaluation.scoring import summarize_trials


def rescore_run(
    run_dir: Path, *, ks: list[int] | None = None, include_pass_hat: bool | None = None,
) -> dict:
    artifact = read_evaluation(run_dir)
    metadata = artifact.metadata
    spec = metadata["spec"]
    return {
        "target": spec["target"],
        **summarize_trials(
            planned=metadata["planned"],
            results=artifact.trajectories,
            errors=artifact.errors,
            trials=spec["trials"],
            ks=ks if ks is not None else spec["ks"],
            include_pass_hat=spec["include_pass_hat"] if include_pass_hat is None else include_pass_hat,
        ),
        "policy_model": metadata["endpoints"]["policy"]["model"],
        "user_model": metadata["endpoints"]["user"]["model"],
        "benchmark_revision": metadata["benchmark_revision"],
        "provenance": metadata["provenance"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rescore saved independent evaluation; no model calls")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=None)
    parser.add_argument("--include-pass-hat", action="store_true", default=None)
    parser.add_argument("--output", type=Path, default=None, help="optional new JSON file; never overwrite")
    args = parser.parse_args(argv)
    try:
        summary = rescore_run(args.run_dir, ks=args.ks, include_pass_hat=args.include_pass_hat)
        text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(text, end="")
    return 0 if summary["metrics_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
