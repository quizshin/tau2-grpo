"""CLI: download AReaL, validate DB paths, write the frozen split manifests.

Milestone D1–D2. Every path defaults to the four-root layout, so the command works
from any working directory.

    python -m tau3_grpo.data.prepare --seed 42
    python -m tau3_grpo.data.prepare --seed 42 --download --require-db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tau3_grpo.data.dataset import (
    build_manifests,
    download_areal_dataset,
    load_and_validate,
    validate_db_paths,
    write_split_manifests,
)
from tau3_grpo.data.leakage import audit_exact
from tau3_grpo.paths import AREAL_DB_ROOT, AREAL_JSONL, MANIFEST_ROOT, verify_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the AReaL Airline training pool")
    parser.add_argument("--seed", type=int, required=True, help="split seed (42 or 43)")
    parser.add_argument("--jsonl", type=Path, default=AREAL_JSONL)
    parser.add_argument("--dataset-root", type=Path, default=AREAL_DB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--download", action="store_true", help="fetch the pinned revision first")
    parser.add_argument(
        "--require-db", action="store_true", help="hash every record-specific FlightDB"
    )
    parser.add_argument(
        "--allow-count-drift",
        action="store_true",
        help="skip the 1982/1148 assertions (diagnostics only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_layout()

    if args.download:
        download_areal_dataset(args.dataset_root)

    if not args.jsonl.is_file():
        print(f"error: AReaL JSONL not found at {args.jsonl}", file=sys.stderr)
        return 2

    records, stats = load_and_validate(args.jsonl, strict_counts=not args.allow_count_drift)
    print(
        f"loaded {stats.total} records ({stats.airline} airline, "
        f"{stats.non_airline} other), file hash {stats.source_file_hash[:12]}"
    )

    if args.require_db:
        resolved = validate_db_paths(records, args.dataset_root, require_exists=True)
        print(f"validated {len(resolved)} record-specific FlightDB paths")

    manifest = build_manifests(
        args.jsonl,
        seed=args.seed,
        dataset_root=args.dataset_root,
        require_db=args.require_db,
        strict_counts=not args.allow_count_drift,
    )
    written = write_split_manifests(manifest, args.output_dir)

    findings = audit_exact(manifest.train, manifest.selection)
    if findings:
        print(f"error: {len(findings)} leakage findings between train and selection", file=sys.stderr)
        for finding in findings[:10]:
            print(f"  {finding.left_task_id} ~ {finding.right_task_id}: {finding.reason}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "seed": manifest.seed,
                "split_hash": manifest.split_hash,
                "train": len(manifest.train),
                "selection": len(manifest.selection),
                "reserve": len(manifest.reserve),
                "files": {key: str(value) for key, value in written.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
