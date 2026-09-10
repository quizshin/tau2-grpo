"""CLI: verify the veRL local patches and the tau_gigpo registration.

Milestone D5. Run this before launching training: a veRL upgrade that dropped a
patch would otherwise produce quietly wrong advantages.

    python -m tau3_grpo.integrations.verify_patches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tau3_grpo.integrations.patch_contract import (
    REQUIREMENTS,
    vanilla_grpo_untouched,
    verify_all,
)
from tau3_grpo.paths import VERL_ROOT, verify_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the local veRL patches")
    parser.add_argument("--verl-root", type=Path, default=VERL_ROOT)
    parser.add_argument(
        "--check-registration",
        action="store_true",
        help="also import veRL and confirm tau_gigpo registers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_layout()

    problems = verify_all(args.verl_root)
    for requirement in REQUIREMENTS:
        status = "ok"
        if any(str(requirement.relative_path) in problem for problem in problems):
            status = "FAIL"
        print(f"[{status}] {requirement.relative_path}: {requirement.description}")

    if not vanilla_grpo_untouched(args.verl_root):
        problems.append(
            "ray_trainer.py: the vanilla GRPO branch is missing; the patch must not "
            "change the default path"
        )

    if args.check_registration:
        try:
            from tau3_grpo.algorithms.verl_estimator import is_registered, register

            register()
            print(f"[{'ok' if is_registered() else 'FAIL'}] tau_gigpo estimator registered")
            if not is_registered():
                problems.append("tau_gigpo did not register in ADV_ESTIMATOR_REGISTRY")
        except ImportError as exc:
            print(f"[skip] veRL not importable: {exc}")

    if problems:
        print("\npatch contract violations:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nall patch contracts satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
