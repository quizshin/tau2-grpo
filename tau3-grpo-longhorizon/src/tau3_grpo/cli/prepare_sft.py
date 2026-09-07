"""Prepare the frozen 45 train + 5 validation complete-dialogue SFT split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from tau3_grpo.data.official import load_official_airline_tasks
from tau3_grpo.data.sft import (
    DEFAULT_INTENT_SIMILARITY_THRESHOLD,
    DEFAULT_MAX_DIALOGUES_PER_INTENT,
    DEFAULT_MAX_RENDERED_TOKENS,
    DEFAULT_SIMILARITY_THRESHOLD,
    load_complete_airline_dialogues,
    reason_texts_from_manifest,
    select_dialogues,
    write_dialogue_split,
)
from tau3_grpo.paths import AREAL_SFT_JSONL, CONFIG_ROOT, MANIFEST_ROOT, SFT_DATA_ROOT
from tau3_grpo.sft.dataset import _render_ids
from tau3_grpo.utils.hashing import sha256_file


def _official_reasons() -> list[str]:
    reasons: list[str] = []
    for task in load_official_airline_tasks():
        payload = task.model_dump() if hasattr(task, "model_dump") else dict(task)
        instructions = (payload.get("user_scenario") or {}).get("instructions") or {}
        reason = instructions.get("reason_for_call")
        if reason:
            reasons.append(str(reason))
    return reasons


def _tool_schemas(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [entry["tool_schema"] for entry in payload["tools"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-audited AReaL Airline SFT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl", type=Path, default=AREAL_SFT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=SFT_DATA_ROOT)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=MANIFEST_ROOT / "areal_airline_selection_seed42.jsonl",
    )
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="frozen tokenizer used for exact chat-template length accounting",
    )
    parser.add_argument(
        "--tool-config", type=Path, default=CONFIG_ROOT / "verl/tool_config.yaml"
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_RENDERED_TOKENS)
    parser.add_argument(
        "--intent-similarity-threshold",
        type=float,
        default=DEFAULT_INTENT_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--max-dialogues-per-intent",
        type=int,
        default=DEFAULT_MAX_DIALOGUES_PER_INTENT,
    )
    parser.add_argument("--skip-official-audit", action="store_true")
    parser.add_argument("--allow-count-drift", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.jsonl.is_file():
        print(f"error: AReaL SFT JSONL not found at {args.jsonl}", file=sys.stderr)
        return 2

    dialogues, stats = load_complete_airline_dialogues(
        args.jsonl, strict_counts=not args.allow_count_drift
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=True, trust_remote_code=False
    )
    tools = _tool_schemas(args.tool_config)
    rendered_token_counts = {
        dialogue.source_dialog_id: len(
            _render_ids(
                tokenizer,
                list(dialogue.messages),
                tools=tools,
                add_generation_prompt=False,
            )
        )
        for dialogue in dialogues
    }
    blocked = reason_texts_from_manifest(args.selection_manifest)
    if not args.skip_official_audit:
        blocked.extend(_official_reasons())
    train, validation, audits = select_dialogues(
        dialogues,
        seed=args.seed,
        blocked_reasons=blocked,
        similarity_threshold=args.similarity_threshold,
        rendered_token_counts=rendered_token_counts,
        max_rendered_tokens=args.max_length,
        intent_similarity_threshold=args.intent_similarity_threshold,
        max_dialogues_per_intent=args.max_dialogues_per_intent,
    )
    source_hash = sha256_file(args.jsonl)
    written = write_dialogue_split(
        train,
        validation,
        audits,
        args.output_dir,
        seed=args.seed,
        source_file_hash=source_hash,
        blocked_reason_count=len(blocked),
        similarity_threshold=args.similarity_threshold,
        rendered_token_counts=rendered_token_counts,
        max_rendered_tokens=args.max_length,
        intent_similarity_threshold=args.intent_similarity_threshold,
        max_dialogues_per_intent=args.max_dialogues_per_intent,
    )
    print(
        json.dumps(
            {
                "source_rows": stats.total_rows,
                "airline_turn_rows": stats.airline_rows,
                "airline_complete_dialogues": stats.airline_dialogues,
                "length_clean_dialogues": sum(
                    value <= args.max_length for value in rendered_token_counts.values()
                ),
                "max_rendered_tokens": args.max_length,
                "max_dialogues_per_intent": args.max_dialogues_per_intent,
                "train": len(train),
                "validation": len(validation),
                "files": {key: str(value) for key, value in written.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
