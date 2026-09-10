"""Task leakage checks used before any training job is launched.

Milestone D2: exact task IDs/fingerprints and normalized intents are hard
failures. AReaL deliberately reuses a small set of immutable base FlightDB
templates across many independent tasks, so a shared *initial* DB hash is
provenance metadata rather than evidence that two task specifications leaked.
Each rollout still loads that template into a private mutable environment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from tau3_grpo.data.manifest import ManifestEntry
from tau3_grpo.utils.hashing import sha256_text


@dataclass(frozen=True)
class LeakageFinding:
    left_task_id: str
    right_task_id: str
    reason: str


def normalized_intent(entry: ManifestEntry) -> str:
    task = entry.task or {}
    instructions = task.get("user_scenario", {}).get("instructions", {})
    if isinstance(instructions, dict):
        text = " ".join(
            str(instructions.get(key, ""))
            for key in ("domain", "reason_for_call", "task_instructions")
        )
    else:
        text = str(instructions)
    # Mask dates and identifier-like tokens (at least one letter and one
    # digit), while preserving ordinary words such as "change" and "cancel".
    entity_pattern = (
        r"\b\d{4}-\d{2}-\d{2}\b|"
        r"\b(?=[a-z0-9_]{5,}\b)(?=[a-z0-9_]*[a-z])(?=[a-z0-9_]*\d)[a-z0-9_]+\b"
    )
    return re.sub(entity_pattern, "<ENTITY>", text.lower())


def audit_exact(left: Iterable[ManifestEntry], right: Iterable[ManifestEntry]) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    right_by_id = {entry.task_id: entry for entry in right}
    right_task_hashes = {entry.task_hash: entry for entry in right}
    right_intents = {sha256_text(normalized_intent(entry)): entry for entry in right}
    for entry in left:
        if entry.task_id in right_by_id:
            findings.append(LeakageFinding(entry.task_id, entry.task_id, "task_id"))
        if entry.task_hash in right_task_hashes:
            findings.append(
                LeakageFinding(entry.task_id, right_task_hashes[entry.task_hash].task_id, "task_hash")
            )
        intent_hash = sha256_text(normalized_intent(entry))
        if intent_hash in right_intents:
            findings.append(
                LeakageFinding(entry.task_id, right_intents[intent_hash].task_id, "normalized_intent")
            )
    return findings
