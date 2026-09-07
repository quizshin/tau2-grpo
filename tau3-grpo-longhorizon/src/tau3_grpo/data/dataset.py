"""AReaL dataset download, DB validation and strict-count manifest build.

Milestone D1–D2: the immutable data boundary is
`inclusionAI/AReaL-tau2-data` revision 86971dc03da6e7c1a7933295e05b84aab8215386,
1,982 records of which 1,148 are Airline, split into exactly 200 train,
60 internal-selection and 888 reserve. AReaL is τ²-style synthetic data; the
converted records are never labelled τ³ official.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tau3_grpo.data.manifest import (
    AREAL_REVISION,
    ManifestEntry,
    SplitManifest,
    build_airline_splits,
    load_areal_records,
    write_jsonl,
)
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.paths import AREAL_DB_ROOT, AREAL_JSONL, MANIFEST_ROOT
from tau3_grpo.utils.hashing import sha256_file

AREAL_REPO_ID = "inclusionAI/AReaL-tau2-data"
AREAL_TRAIN_FILE = "tau2_rl_train.jsonl"
AREAL_SFT_FILE = "tau2_sft_train.jsonl"

EXPECTED_TOTAL_RECORDS = 1982
EXPECTED_AIRLINE_RECORDS = 1148
EXPECTED_TRAIN = 200
EXPECTED_SELECTION = 60
EXPECTED_RESERVE = 888


@dataclass(frozen=True)
class DatasetStats:
    """Counts used to fail fast when the dataset drifts from the pinned revision."""

    total: int
    airline: int
    non_airline: int
    unique_ids: int
    source_file_hash: str

    def assert_expected(self) -> None:
        if self.total != EXPECTED_TOTAL_RECORDS:
            raise ValueError(
                f"expected {EXPECTED_TOTAL_RECORDS} AReaL records, found {self.total}"
            )
        if self.airline != EXPECTED_AIRLINE_RECORDS:
            raise ValueError(
                f"expected {EXPECTED_AIRLINE_RECORDS} Airline records, found {self.airline}"
            )
        if self.unique_ids != self.total:
            raise ValueError("AReaL task IDs are not unique")


def download_areal_dataset(
    target_dir: str | Path = AREAL_DB_ROOT,
    *,
    revision: str = AREAL_REVISION,
    repo_id: str = AREAL_REPO_ID,
) -> Path:
    """Download the pinned AReaL revision.

    The revision is passed explicitly so a moved branch head cannot silently
    change the training pool.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface-hub is required to download the AReaL dataset; "
            "install the 'data' extra"
        ) from exc

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target),
    )
    return target


def describe_records(records: list[ArealTaskRecord], *, source_file_hash: str) -> DatasetStats:
    """Compute the counts that gate every downstream step."""

    airline = [record for record in records if record.domain == "airline"]
    return DatasetStats(
        total=len(records),
        airline=len(airline),
        non_airline=len(records) - len(airline),
        unique_ids=len({record.id for record in records}),
        source_file_hash=source_file_hash,
    )


def validate_db_paths(
    records: list[ArealTaskRecord],
    dataset_root: str | Path = AREAL_DB_ROOT,
    *,
    require_exists: bool = True,
) -> dict[str, Path]:
    """Resolve every record-specific DB path under `dataset_root`.

    `ArealTaskRecord.resolve_db_path` rejects traversal outside the dataset root,
    so a crafted `db_path` cannot reach arbitrary files.
    """

    resolved: dict[str, Path] = {}
    for record in records:
        path = record.resolve_db_path(dataset_root)
        if require_exists and not path.is_file():
            raise FileNotFoundError(f"missing FlightDB for task {record.id}: {path}")
        resolved[record.id] = path
    return resolved


def load_and_validate(
    jsonl_path: str | Path = AREAL_JSONL,
    *,
    strict_counts: bool = True,
) -> tuple[list[ArealTaskRecord], DatasetStats]:
    """Load the real JSONL and assert the pinned counts."""

    path = Path(jsonl_path)
    records = load_areal_records(path)
    stats = describe_records(records, source_file_hash=sha256_file(path))
    if strict_counts:
        stats.assert_expected()
    return records, stats


def build_manifests(
    jsonl_path: str | Path = AREAL_JSONL,
    *,
    seed: int,
    dataset_root: str | Path | None = AREAL_DB_ROOT,
    require_db: bool = False,
    strict_counts: bool = True,
) -> SplitManifest:
    """Build the deterministic 200/60/888 manifest from the real JSONL."""

    records, stats = load_and_validate(jsonl_path, strict_counts=strict_counts)
    # Diagnostics mode still has to name an expected count, so use the observed
    # Airline count rather than the total record count.
    expected_airline = EXPECTED_AIRLINE_RECORDS if strict_counts else stats.airline
    manifest = build_airline_splits(
        records,
        seed=seed,
        train_size=EXPECTED_TRAIN,
        selection_size=EXPECTED_SELECTION,
        expected_airline_count=expected_airline,
        dataset_root=dataset_root if require_db else None,
        source_file_hash=stats.source_file_hash,
    )
    if strict_counts:
        assert_split_sizes(manifest)
    return manifest


def assert_split_sizes(manifest: SplitManifest) -> None:
    """Enforce the frozen 200/60/888 partition."""

    actual = (len(manifest.train), len(manifest.selection), len(manifest.reserve))
    expected = (EXPECTED_TRAIN, EXPECTED_SELECTION, EXPECTED_RESERVE)
    if actual != expected:
        raise ValueError(f"split sizes {actual} do not match frozen {expected}")
    ids = [entry.task_id for entry in manifest.train + manifest.selection + manifest.reserve]
    if len(ids) != len(set(ids)):
        raise ValueError("splits overlap: duplicate task ids across train/selection/reserve")


def write_split_manifests(
    manifest: SplitManifest,
    output_dir: str | Path = MANIFEST_ROOT,
) -> dict[str, Path]:
    """Write one JSONL per split plus a split-hash sidecar."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, entries in (
        ("train", manifest.train),
        ("selection", manifest.selection),
        ("reserve", manifest.reserve),
    ):
        path = out / f"areal_airline_{name}_seed{manifest.seed}.jsonl"
        write_jsonl(entries, path)
        written[name] = path
    sidecar = out / f"areal_airline_split_seed{manifest.seed}.json"
    sidecar.write_text(
        "\n".join(
            [
                "{",
                f'  "seed": {manifest.seed},',
                f'  "source_revision": "{manifest.source_revision}",',
                f'  "source_file_hash": "{manifest.source_file_hash}",',
                f'  "split_hash": "{manifest.split_hash}",',
                f'  "train": {len(manifest.train)},',
                f'  "selection": {len(manifest.selection)},',
                f'  "reserve": {len(manifest.reserve)}',
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    written["split_hash"] = sidecar
    return written


def entries_by_id(entries: list[ManifestEntry]) -> dict[str, ManifestEntry]:
    return {entry.task_id: entry for entry in entries}
