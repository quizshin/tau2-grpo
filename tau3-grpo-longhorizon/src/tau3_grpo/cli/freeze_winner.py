"""CLI: freeze the winner before any τ³ official run.

Milestone D13. The lock records the checkpoint, the internal-selection evidence
and a hash over both, so the τ³ final guard can refuse an unfrozen or edited lock.

    python -m tau3_grpo.cli.freeze_winner --experiment e3 \
        --checkpoint results/e3_seed42/global_step_40/merged_hf \
        --metric selection_solve_rate --score 0.42 --task-count 60 --seeds 42 43
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tau3_grpo.experiment.winner_lock import WinnerLockError, freeze_winner
from tau3_grpo.paths import RESULTS_ROOT, verify_four_root_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the winning checkpoint")
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="merged Hugging Face checkpoint directory; its full content is hashed",
    )
    parser.add_argument("--metric", required=True, help="internal-selection metric name")
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument(
        "--task-count", type=int, required=True, help="number of internal-selection tasks"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing lock (only before any τ³ run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_four_root_layout()
    try:
        lock, path = freeze_winner(
            experiment=args.experiment,
            checkpoint_path=args.checkpoint,
            selection_metric=args.metric,
            selection_score=args.score,
            selection_task_count=args.task_count,
            seeds=args.seeds,
            notes=args.notes,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except WinnerLockError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"froze winner at {path}")
    print(json.dumps(lock.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
