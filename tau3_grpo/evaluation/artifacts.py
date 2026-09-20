"""Read independent-evaluation evidence without trusting cached summary scores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationArtifacts:
    directory: Path
    metadata: dict
    trajectories: list[dict]
    errors: list[dict]
    files: dict[str, str | None]


def read_evaluation(directory: Path) -> EvaluationArtifacts:
    directory = Path(directory).resolve()
    files = {}

    def read(name, *, required=False):
        path = directory / name
        if not path.exists() and not required:
            files[name] = None
            return b""
        content = path.read_bytes()
        files[name] = hashlib.sha256(content).hexdigest()
        return content

    metadata = json.loads(read("run.json", required=True))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported independent evaluation run schema")
    trajectories = [json.loads(line) for line in read("trajectories.jsonl").splitlines()
                    if line.strip()]
    errors = [json.loads(line) for line in read("errors.jsonl").splitlines() if line.strip()]
    return EvaluationArtifacts(directory, metadata, trajectories, errors, files)
