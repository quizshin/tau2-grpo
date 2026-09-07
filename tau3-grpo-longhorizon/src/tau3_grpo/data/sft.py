"""Build the frozen 45+5 complete-dialogue Airline SFT split.

AReaL's SFT JSONL contains one row per assistant turn.  Selecting 50 rows would
leak turns from the same source dialogue across train/validation and would not
match the v2-2 budget.  This module first reconstructs one complete trajectory
per ``source_dialog_id`` from its final turn, then performs dialogue-level
selection and leakage checks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tau3_grpo.data.manifest import AREAL_REVISION
from tau3_grpo.utils.hashing import sha256_text

EXPECTED_SFT_ROWS = 33_531
EXPECTED_AIRLINE_SFT_ROWS = 12_842
EXPECTED_AIRLINE_DIALOGUES = 999
SFT_TRAIN_DIALOGUES = 45
SFT_VALIDATION_DIALOGUES = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_INTENT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_MAX_DIALOGUES_PER_INTENT = 1
DEFAULT_MAX_RENDERED_TOKENS = 16_384

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class SFTDialogue:
    """One reconstructed, correct AReaL Airline conversation."""

    source_dialog_id: str
    scenario_id: str
    turn_index: int
    reason_for_call: str
    messages: tuple[dict[str, Any], ...]

    @property
    def dialogue_hash(self) -> str:
        payload = json.dumps(self.messages, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_record(self, *, rendered_tokens: int | None = None) -> dict[str, Any]:
        record = {
            "messages": list(self.messages),
            "metadata": {
                "source": "areal_tau2_airline_sft",
                "source_revision": AREAL_REVISION,
                "source_dialog_id": self.source_dialog_id,
                "scenario_id": self.scenario_id,
                "final_turn_index": self.turn_index,
                "reason_for_call": self.reason_for_call,
                "dialogue_hash": self.dialogue_hash,
            },
        }
        if rendered_tokens is not None:
            record["metadata"]["rendered_tokens"] = int(rendered_tokens)
        return record


@dataclass(frozen=True)
class SFTPoolStats:
    total_rows: int
    airline_rows: int
    airline_dialogues: int

    def assert_expected(self) -> None:
        expected = (
            EXPECTED_SFT_ROWS,
            EXPECTED_AIRLINE_SFT_ROWS,
            EXPECTED_AIRLINE_DIALOGUES,
        )
        actual = (self.total_rows, self.airline_rows, self.airline_dialogues)
        if actual != expected:
            raise ValueError(f"AReaL SFT counts {actual} do not match pinned {expected}")


@dataclass(frozen=True)
class LeakageAudit:
    source_dialog_id: str
    max_similarity: float
    closest_blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dialog_id": self.source_dialog_id,
            "max_similarity": self.max_similarity,
            "closest_blocked_reason": self.closest_blocked_reason,
        }


def normalize_reason(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def reason_similarity(left: str, right: str) -> float:
    """Conservative lexical near-duplicate score for the audit gate."""

    a = normalize_reason(left)
    b = normalize_reason(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    union = a_tokens | b_tokens
    jaccard = len(a_tokens & b_tokens) / len(union) if union else 0.0
    return max(jaccard, SequenceMatcher(None, a, b).ratio())


def _assistant_answer(answer: dict[str, Any]) -> dict[str, Any]:
    if answer.get("role") != "assistant":
        raise ValueError(f"SFT answer role must be assistant, got {answer.get('role')!r}")
    message: dict[str, Any] = {
        "role": "assistant",
        "content": answer.get("content") or "",
    }
    if answer.get("tool_calls"):
        message["tool_calls"] = answer["tool_calls"]
    # Qwen2.5-Instruct's chat template consumes content/tool_calls, not the
    # dataset's private thinking field.  Preserve it as provenance without
    # placing hidden reasoning in the supervised output.
    if answer.get("thinking"):
        message["reasoning"] = answer["thinking"]
    return message


def load_complete_airline_dialogues(
    path: str | Path,
    *,
    strict_counts: bool = True,
) -> tuple[list[SFTDialogue], SFTPoolStats]:
    """Load final-turn rows and reconstruct one complete dialogue per source id."""

    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    total_rows = 0
    airline_rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            dialog_id = str(metadata.get("source_dialog_id") or "")
            if not dialog_id.startswith("airline_dialog_"):
                continue
            airline_rows += 1
            turn_index = int(metadata.get("turn_index", -1))
            if turn_index < 0:
                raise ValueError(f"line {line_number} has invalid turn_index")
            previous = latest.get(dialog_id)
            if previous is None or turn_index > previous[0]:
                latest[dialog_id] = (turn_index, row)

    dialogues: list[SFTDialogue] = []
    for dialog_id, (turn_index, row) in sorted(latest.items()):
        metadata = row["metadata"]
        messages = list(row.get("messages") or [])
        messages.append(_assistant_answer(row.get("answer") or {}))
        if not messages or messages[0].get("role") != "system":
            raise ValueError(f"{dialog_id} is not a complete system-prefixed dialogue")
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError(f"{dialog_id} has no user turn")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"{dialog_id} does not end in an assistant answer")
        dialogues.append(
            SFTDialogue(
                source_dialog_id=dialog_id,
                scenario_id=str(metadata.get("scenario_id") or ""),
                turn_index=turn_index,
                reason_for_call=str(metadata.get("reason_for_call") or ""),
                messages=tuple(messages),
            )
        )

    stats = SFTPoolStats(total_rows, airline_rows, len(dialogues))
    if strict_counts:
        stats.assert_expected()
    return dialogues, stats


def _closest_reason(reason: str, blocked_reasons: Sequence[str]) -> tuple[float, str]:
    if not blocked_reasons:
        return 0.0, ""
    scored = [(reason_similarity(reason, blocked), blocked) for blocked in blocked_reasons]
    return max(scored, key=lambda item: item[0])


def select_dialogues(
    dialogues: Sequence[SFTDialogue],
    *,
    seed: int,
    blocked_reasons: Iterable[str] = (),
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    rendered_token_counts: Mapping[str, int] | None = None,
    max_rendered_tokens: int | None = None,
    intent_similarity_threshold: float = DEFAULT_INTENT_SIMILARITY_THRESHOLD,
    max_dialogues_per_intent: int = DEFAULT_MAX_DIALOGUES_PER_INTENT,
    train_size: int = SFT_TRAIN_DIALOGUES,
    validation_size: int = SFT_VALIDATION_DIALOGUES,
) -> tuple[list[SFTDialogue], list[SFTDialogue], list[LeakageAudit]]:
    """Choose complete, length-safe and intent-diverse dialogues deterministically.

    ``reason_for_call`` is the only usable intent key in the pinned AReaL SFT
    artifact: all 999 Airline rows have an empty ``scenario_id``. Candidates
    are ranked by a seed-bound stable hash and greedily assigned to semantic
    intent buckets. The default cap of one prevents repeated easy intents from
    dominating a 45-dialogue warm start.
    """

    if train_size <= 0 or validation_size <= 0:
        raise ValueError("SFT train and validation sizes must be positive")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1]")
    if not 0.0 <= intent_similarity_threshold <= 1.0:
        raise ValueError("intent_similarity_threshold must be in [0, 1]")
    if max_dialogues_per_intent <= 0:
        raise ValueError("max_dialogues_per_intent must be positive")
    if max_rendered_tokens is not None:
        if max_rendered_tokens <= 0:
            raise ValueError("max_rendered_tokens must be positive")
        if rendered_token_counts is None:
            raise ValueError("rendered_token_counts are required for a length gate")

    blocked = sorted({text for text in blocked_reasons if normalize_reason(text)})
    eligible: list[tuple[SFTDialogue, LeakageAudit]] = []
    for dialogue in sorted(dialogues, key=lambda value: value.source_dialog_id):
        score, closest = _closest_reason(dialogue.reason_for_call, blocked)
        if score >= similarity_threshold:
            continue
        if max_rendered_tokens is not None:
            if dialogue.source_dialog_id not in rendered_token_counts:
                raise ValueError(
                    f"missing rendered token count for {dialogue.source_dialog_id}"
                )
            if int(rendered_token_counts[dialogue.source_dialog_id]) > max_rendered_tokens:
                continue
        eligible.append(
            (
                dialogue,
                LeakageAudit(dialogue.source_dialog_id, score, closest),
            )
        )

    required = train_size + validation_size
    if len(eligible) < required:
        raise ValueError(
            f"only {len(eligible)} leakage-clean SFT dialogues remain; need {required}"
        )
    ranked = sorted(
        eligible,
        key=lambda item: sha256_text(
            f"{seed}:{item[0].source_dialog_id}:{item[0].dialogue_hash}"
        ),
    )
    intent_buckets: list[dict[str, Any]] = []
    chosen: list[tuple[SFTDialogue, LeakageAudit]] = []
    for item in ranked:
        dialogue = item[0]
        best_bucket: dict[str, Any] | None = None
        best_score = -1.0
        for bucket in intent_buckets:
            score = reason_similarity(dialogue.reason_for_call, bucket["representative"])
            if score >= intent_similarity_threshold and score > best_score:
                best_bucket = bucket
                best_score = score
        if best_bucket is not None:
            if int(best_bucket["count"]) >= max_dialogues_per_intent:
                continue
            best_bucket["count"] = int(best_bucket["count"]) + 1
        else:
            intent_buckets.append(
                {"representative": dialogue.reason_for_call, "count": 1}
            )
        chosen.append(item)
        if len(chosen) == required:
            break
    if len(chosen) < required:
        raise ValueError(
            f"only {len(chosen)} intent-diverse SFT dialogues remain; need {required}"
        )
    train = [item[0] for item in chosen[:train_size]]
    validation = [item[0] for item in chosen[train_size:]]
    audits = [item[1] for item in chosen]
    return train, validation, audits


def reason_texts_from_manifest(path: str | Path) -> list[str]:
    """Extract user intents from a project JSONL manifest."""

    reasons: list[str] = []
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"selection manifest not found: {manifest}")
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = (json.loads(line).get("task") or {})
            instructions = (task.get("user_scenario") or {}).get("instructions") or {}
            reason = instructions.get("reason_for_call")
            if reason:
                reasons.append(str(reason))
    return reasons


def write_dialogue_split(
    train: Sequence[SFTDialogue],
    validation: Sequence[SFTDialogue],
    audits: Sequence[LeakageAudit],
    output_dir: str | Path,
    *,
    seed: int,
    source_file_hash: str,
    blocked_reason_count: int,
    similarity_threshold: float,
    rendered_token_counts: Mapping[str, int] | None = None,
    max_rendered_tokens: int | None = None,
    intent_similarity_threshold: float = DEFAULT_INTENT_SIMILARITY_THRESHOLD,
    max_dialogues_per_intent: int = DEFAULT_MAX_DIALOGUES_PER_INTENT,
) -> dict[str, Path]:
    """Write two JSONLs and an auditable deterministic split sidecar."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, records in (("train", train), ("validation", validation)):
        target = out / f"airline_sft_{name}_seed{seed}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for dialogue in records:
                rendered_tokens = None
                if rendered_token_counts is not None:
                    rendered_tokens = rendered_token_counts[dialogue.source_dialog_id]
                handle.write(
                    json.dumps(
                        dialogue.to_record(rendered_tokens=rendered_tokens), sort_keys=True
                    )
                    + "\n"
                )
        written[name] = target

    split_payload = {
        "source": "inclusionAI/AReaL-tau2-data",
        "source_revision": AREAL_REVISION,
        "source_file_hash": source_file_hash,
        "seed": seed,
        "train_dialogues": len(train),
        "validation_dialogues": len(validation),
        "blocked_reason_count": blocked_reason_count,
        "similarity_threshold": similarity_threshold,
        "selection_method": "seeded_hash_rank_with_greedy_semantic_intent_cap",
        "max_rendered_tokens": max_rendered_tokens,
        "intent_similarity_threshold": intent_similarity_threshold,
        "max_dialogues_per_intent": max_dialogues_per_intent,
        "train_ids": [dialogue.source_dialog_id for dialogue in train],
        "validation_ids": [dialogue.source_dialog_id for dialogue in validation],
        "selected_rendered_tokens": {
            dialogue.source_dialog_id: int(rendered_token_counts[dialogue.source_dialog_id])
            for dialogue in (*train, *validation)
        }
        if rendered_token_counts is not None
        else {},
        "leakage_audit": [audit.to_dict() for audit in audits],
    }
    canonical = json.dumps(split_payload, sort_keys=True, separators=(",", ":"))
    split_payload["split_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    sidecar = out / f"airline_sft_split_seed{seed}.json"
    sidecar.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["manifest"] = sidecar
    return written
