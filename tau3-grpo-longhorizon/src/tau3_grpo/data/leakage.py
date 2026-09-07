"""Leakage checks used before any training job is launched.

Milestone D2: exact IDs/hashes are hard failures; semantic fields are exported
for a separate audit rather than being silently declared clean.
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
    return re.sub(r"\b[A-Z0-9_]{5,}\b|\d{4}-\d{2}-\d{2}", "<ENTITY>", text.lower())


def audit_exact(left: Iterable[ManifestEntry], right: Iterable[ManifestEntry]) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    right_by_id = {entry.task_id: entry for entry in right}
    right_task_hashes = {entry.task_hash: entry for entry in right}
    right_db_hashes = {entry.db_hash: entry for entry in right if entry.db_hash}
    right_intents = {sha256_text(normalized_intent(entry)): entry for entry in right}
    for entry in left:
        if entry.task_id in right_by_id:
            findings.append(LeakageFinding(entry.task_id, entry.task_id, "task_id"))
        if entry.task_hash in right_task_hashes:
            findings.append(
                LeakageFinding(entry.task_id, right_task_hashes[entry.task_hash].task_id, "task_hash")
            )
        if entry.db_hash and entry.db_hash in right_db_hashes:
            findings.append(
                LeakageFinding(entry.task_id, right_db_hashes[entry.db_hash].task_id, "db_hash")
            )
        intent_hash = sha256_text(normalized_intent(entry))
        if intent_hash in right_intents:
            findings.append(
                LeakageFinding(entry.task_id, right_intents[intent_hash].task_id, "normalized_intent")
            )
    return findings

