"""CLI: turn frozen split manifests into veRL parquet datasets.

Milestone D8. The system prompt carries the official tau2 Airline policy, and the
builder refuses any τ³ official entry.

    python -m tau3_grpo.cli.build_parquet --seed 42 --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.official import SourceIsolationError
from tau3_grpo.data.parquet_builder import build_and_write
from tau3_grpo.env.adapter import airline_policy
from tau3_grpo.paths import MANIFEST_ROOT, PARQUET_ROOT, verify_four_root_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build veRL parquet from a split manifest")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["train", "selection", "reserve"], default="train")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PARQUET_ROOT)
    parser.add_argument(
        "--anchor-mode",
        choices=["structured", "db_hash_only", "similarity"],
        default="structured",
    )
    parser.add_argument("--policy-file", type=Path, default=None, help="override policy text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_four_root_layout()

    manifest_path = args.manifest_dir / f"areal_airline_{args.split}_seed{args.seed}.jsonl"
    if not manifest_path.is_file():
        print(f"error: manifest not found at {manifest_path}", file=sys.stderr)
        return 2
    entries = read_manifest(manifest_path)

    if args.policy_file is not None:
        policy = args.policy_file.read_text(encoding="utf-8")
    else:
        policy = airline_policy()

    try:
        path = build_and_write(
            entries,
            policy=policy,
            split=args.split,
            output_dir=args.output_dir,
            filename=f"airline_{args.split}_seed{args.seed}.parquet",
            anchor_mode=args.anchor_mode,
            seed=args.seed,
        )
    except SourceIsolationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(entries)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
