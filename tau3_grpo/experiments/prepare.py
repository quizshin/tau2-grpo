"""Build and verify the exact task schedule consumed by one veRL run.

The 200-task manifest freezes the eligible pool. This command expands that pool
into exactly ``total_updates * groups_per_update`` parquet rows, in manifest
schedule order, and writes the matching experiment manifest next to checkpoints.
With ``data.shuffle=false`` the dataloader now consumes the documented schedule
rather than leaving it as unused metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau3_grpo.data.manifest import ManifestEntry
from tau3_grpo.data.manifest import read_manifest as read_data_manifest
from tau3_grpo.data.parquet_builder import build_rows, write_parquet
from tau3_grpo.envs.adapter import airline_policy
from tau3_grpo.experiments.manifest import (
    MANIFEST_FILENAME,
    ExperimentManifest,
    create_manifest,
    flatten_schedule,
    verify_extension,
    verify_resume,
)
from tau3_grpo.experiments.manifest import read_manifest as read_experiment_manifest
from tau3_grpo.experiments.manifest import write_manifest as write_experiment_manifest
from tau3_grpo.paths import MANIFEST_ROOT, RESULTS_ROOT, verify_layout

SCHEDULE_PARQUET_FILENAME = "train_schedule.parquet"


def _load_split_hash(manifest_dir: Path, data_seed: int) -> str:
    sidecar = manifest_dir / f"areal_airline_split_seed{data_seed}.json"
    if not sidecar.is_file():
        raise FileNotFoundError(f"split hash sidecar not found at {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if int(payload.get("seed", -1)) != data_seed:
        raise ValueError(f"split sidecar seed does not match {data_seed}: {sidecar}")
    split_hash = payload.get("split_hash")
    if not isinstance(split_hash, str) or not split_hash:
        raise ValueError(f"split sidecar has no split_hash: {sidecar}")
    return split_hash


def scheduled_entries(
    entries: list[ManifestEntry], manifest: ExperimentManifest
) -> list[ManifestEntry]:
    """Expand unique train entries in the manifest's exact update order."""

    by_id = {entry.task_id: entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("training manifest contains duplicate task ids")
    ordered: list[ManifestEntry] = []
    for task_id in flatten_schedule(manifest.schedule):
        try:
            ordered.append(by_id[task_id])
        except KeyError as exc:
            raise ValueError(f"scheduled task is absent from train manifest: {task_id}") from exc
    return ordered


def _verify_existing(
    *,
    output_dir: Path,
    parquet_path: Path,
    expected: ExperimentManifest,
    expected_task_ids: list[str],
) -> bool:
    actual = read_experiment_manifest(output_dir)
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"experiment manifest exists but schedule parquet is missing: {parquet_path}"
        )

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - setup installs the data extra
        raise RuntimeError("pandas is required to verify the schedule parquet") from exc
    frame = pd.read_parquet(parquet_path, columns=["extra_info"])
    actual_task_ids = [str(info["task_id"]) for info in frame["extra_info"]]

    if actual.schedule_hash == expected.schedule_hash:
        verify_resume(expected, actual)
        if actual_task_ids != expected_task_ids:
            raise ValueError("existing train_schedule.parquet does not match experiment manifest")
        return False

    verify_extension(actual, expected)
    prefix_size = len(flatten_schedule(actual.schedule))
    if actual_task_ids != expected_task_ids[:prefix_size]:
        raise ValueError("existing train_schedule.parquet does not match experiment manifest")
    return True


def prepare_experiment_inputs(
    *,
    arm: str,
    seed: int,
    data_seed: int,
    group_size: int,
    groups_per_update: int,
    total_updates: int,
    anchor_mode: str,
    manifest_dir: Path = MANIFEST_ROOT,
    output_dir: Path | None = None,
) -> tuple[Path, Path, bool]:
    """Create or verify a schedule parquet and experiment manifest.

    Returns ``(parquet_path, manifest_path, created)``. Existing artifacts are
    never silently overwritten; they must match the newly computed schedule.
    """

    train_manifest_path = manifest_dir / f"areal_airline_train_seed{data_seed}.jsonl"
    if not train_manifest_path.is_file():
        raise FileNotFoundError(f"training manifest not found at {train_manifest_path}")
    entries = read_data_manifest(train_manifest_path)
    split_hash = _load_split_hash(manifest_dir, data_seed)
    expected = create_manifest(
        experiment=arm,
        arm=arm,
        seed=seed,
        task_ids=[entry.task_id for entry in entries],
        train_split_hash=split_hash,
        group_size=group_size,
        groups_per_update=groups_per_update,
        total_updates=total_updates,
        anchor_mode=anchor_mode,
    )
    ordered_entries = scheduled_entries(entries, expected)
    expected_task_ids = [entry.task_id for entry in ordered_entries]

    destination = output_dir or RESULTS_ROOT / f"{arm}_seed{seed}"
    parquet_path = destination / SCHEDULE_PARQUET_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    if manifest_path.exists() or parquet_path.exists():
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"schedule parquet exists but experiment manifest is missing: {manifest_path}"
            )
        extend = _verify_existing(
            output_dir=destination,
            parquet_path=parquet_path,
            expected=expected,
            expected_task_ids=expected_task_ids,
        )
        if not extend:
            return parquet_path, manifest_path, False

        # Phase B extends the frozen screening schedule (normally 40→60)
        # without changing one task in the already-consumed prefix.
        rows = build_rows(
            ordered_entries,
            policy=airline_policy(),
            split="train",
            anchor_mode=anchor_mode,
            seed=seed,
        )
        write_parquet(rows, parquet_path)
        write_experiment_manifest(expected, destination)
        return parquet_path, manifest_path, True

    destination.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        ordered_entries,
        policy=airline_policy(),
        split="train",
        anchor_mode=anchor_mode,
        seed=seed,
    )
    write_parquet(rows, parquet_path)
    write_experiment_manifest(expected, destination)
    return parquet_path, manifest_path, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one deterministic veRL run")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--groups-per-update", type=int, default=16)
    parser.add_argument("--total-updates", type=int, required=True)
    parser.add_argument(
        "--anchor-mode",
        choices=["structured", "db_hash_only", "similarity"],
        default="structured",
    )
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_layout()
    parquet, manifest, created = prepare_experiment_inputs(
        arm=args.arm,
        seed=args.seed,
        data_seed=args.data_seed,
        group_size=args.group_size,
        groups_per_update=args.groups_per_update,
        total_updates=args.total_updates,
        anchor_mode=args.anchor_mode,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "created": created,
                "experiment_manifest": str(manifest),
                "train_parquet": str(parquet),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
