"""CLI: evaluation entrypoints for internal selection and the τ³ official final.

Milestone D13. `--target selection` scores the 60 internal-selection tasks and may
run freely. `--target tau3-final` loads the official 50-task Airline base split and
**requires a valid winner lock naming the exact checkpoint**, so the final set can
never be used to choose a model.

    python -m tau3_grpo.evaluation.run --target selection --seed 42 \
        --checkpoint results/e3_seed42/global_step_40/merged_hf
    python -m tau3_grpo.evaluation.run --target tau3-final \
        --checkpoint results/e3_seed42/global_step_40/merged_hf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.official import (
    OFFICIAL_AIRLINE_TASK_COUNT,
    build_official_manifest,
    official_airline_task_ids,
)
from tau3_grpo.envs.tau2_bridge import Tau2Unavailable
from tau3_grpo.evaluation.runtime import Endpoint, EvalSpec, run_evaluation
from tau3_grpo.evaluation.service_attestation import (
    ServiceAttestationError,
    assert_service_matches_checkpoint,
)
from tau3_grpo.experiments.winner_lock import WinnerLockError, assert_final_run_allowed
from tau3_grpo.paths import MANIFEST_ROOT, RESULTS_ROOT, verify_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--target", choices=["selection", "tau3-final"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="frozen AReaL split seed; independent of evaluation sampling seed",
    )
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--policy-base-url",
        default=os.environ.get("TAU3_POLICY_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--policy-model",
        default=os.environ.get("TAU3_POLICY_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--user-base-url",
        default=os.environ.get("TAU3_USER_BASE_URL", "http://127.0.0.1:8100/v1"),
    )
    parser.add_argument(
        "--user-model",
        default=os.environ.get("TAU3_USER_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--policy-attestation",
        type=Path,
        default=(
            Path(os.environ["TAU3_POLICY_ATTESTATION_PATH"])
            if os.environ.get("TAU3_POLICY_ATTESTATION_PATH")
            else None
        ),
        help="checkpoint/content/PID attestation written by serve_policy_eval.sh",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve tasks and guards, then stop before any rollout",
    )
    return parser


def _selection_tasks(manifest_dir: Path, seed: int) -> list[str]:
    path = manifest_dir / f"areal_airline_selection_seed{seed}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"selection manifest not found at {path}")
    return [entry.task_id for entry in read_manifest(path)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_layout()

    if args.target == "selection":
        try:
            manifest_path = (
                args.manifest_dir / f"areal_airline_selection_seed{args.data_seed}.jsonl"
            )
            selection_entries = read_manifest(manifest_path)
            task_ids = [entry.task_id for entry in selection_entries]
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        payload = {
            "target": "selection",
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "task_count": len(task_ids),
        }
    else:
        try:
            task_ids = official_airline_task_ids()
            manifest = build_official_manifest()
        except Tau2Unavailable as exc:
            print(f"error: tau2-bench unavailable: {exc}", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            lock = assert_final_run_allowed(
                checkpoint_path=args.checkpoint,
                task_count=len(task_ids),
                directory=args.results_dir,
            )
        except WinnerLockError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        payload = {
            "target": "tau3_final",
            "checkpoint": args.checkpoint,
            "task_count": len(manifest),
            "expected_task_count": OFFICIAL_AIRLINE_TASK_COUNT,
            "winner_lock_hash": lock.lock_hash,
            "frozen_at": lock.frozen_at,
        }

    if args.dry_run:
        payload["dry_run"] = True
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    attestation_path = args.policy_attestation or (
        args.results_dir / "policy_service_attestation.json"
    )
    try:
        attestation = assert_service_matches_checkpoint(
            attestation_path=attestation_path,
            checkpoint_path=args.checkpoint,
            served_model_name=args.policy_model,
            base_url=args.policy_base_url,
        )
    except ServiceAttestationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    payload["policy_service_attestation_hash"] = attestation.attestation_hash

    output_dir = args.output_dir or (
        args.results_dir
        / "evaluation"
        / f"{args.target}_{Path(args.checkpoint).name}_seed{args.seed}"
    )
    policy = Endpoint(
        model=args.policy_model,
        base_url=args.policy_base_url,
        api_key=os.environ.get("TAU3_POLICY_API_KEY", "EMPTY"),
        temperature=0.4,
    )
    user = Endpoint(
        model=args.user_model,
        base_url=args.user_base_url,
        api_key=os.environ.get("TAU3_USER_API_KEY", "EMPTY"),
        temperature=1.0,
    )
    try:
        summary = run_evaluation(
            spec=EvalSpec(
                target=args.target,
                trials=args.trials,
                seed=args.seed,
                max_steps=args.max_steps,
                max_concurrency=args.max_concurrency,
            ),
            policy=policy,
            user=user,
            output_dir=output_dir,
            selection_entries=selection_entries if args.target == "selection" else (),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary["checkpoint"] = args.checkpoint
    summary["policy_service_attestation_hash"] = attestation.attestation_hash
    summary["output_dir"] = str(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed_trajectories"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
