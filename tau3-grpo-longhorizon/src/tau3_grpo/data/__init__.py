"""Data contracts and deterministic manifest construction."""

from .manifest import ManifestEntry, SplitManifest, build_airline_splits, read_manifest
from .schema import ArealTaskRecord, DataSource

__all__ = [
    "ArealTaskRecord",
    "DataSource",
    "ManifestEntry",
    "SplitManifest",
    "build_airline_splits",
    "read_manifest",
]

