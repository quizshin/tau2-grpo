"""Qwen chat-template SFT dataset with strict assistant-only loss masking.

The AReaL output contains complete multi-turn conversations with tool calls.
Role-string token slicing is unsafe because chat templates insert special tokens.
For each assistant turn we render (a) the preceding dialogue with a generation
prompt and (b) the dialogue including that assistant turn.  The exact suffix is
the supervised span; system, user and tool tokens remain ``IGNORE_INDEX``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from tau3_grpo.models.compat import is_qwen35_tokenizer
from tau3_grpo.models.qwen35_template import build_qwen35_example, thinking_options
from tau3_grpo.models.tokenization import render_chat_ids as _render_ids
from tau3_grpo.prompts import prepare_agent_messages, prompt_provenance
from tau3_grpo.utils.hashing import sha256_text

IGNORE_INDEX = -100


def build_supervised_example(
    messages: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    max_length: int = 16_384,
    enable_thinking: bool = False,
    supervise_reasoning: bool = False,
    preserve_historical_reasoning: bool = False,
) -> dict[str, Any]:
    """Render one complete dialogue and label assistant output spans only."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("dialogue has no assistant turn")

    options = thinking_options(dict(enable_thinking=enable_thinking,
        supervise_reasoning=supervise_reasoning,
        preserve_historical_reasoning=preserve_historical_reasoning))
    if is_qwen35_tokenizer(tokenizer):
        return build_qwen35_example(messages, tokenizer, tools=tools, max_length=max_length, **options)
    if any(options.values()):
        raise ValueError("Thinking SFT is supported only for Qwen3.5 tokenizers")

    full_ids = _render_ids(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=False,
    )
    if len(full_ids) > max_length:
        raise ValueError(
            f"rendered dialogue has {len(full_ids)} tokens, exceeding max_length={max_length}"
        )
    labels = [IGNORE_INDEX] * len(full_ids)

    for assistant_index in assistant_indices:
        prefix_ids = _render_ids(
            tokenizer,
            messages[:assistant_index],
            tools=tools,
            add_generation_prompt=True,
        )
        completed_ids = _render_ids(
            tokenizer,
            messages[: assistant_index + 1],
            tools=tools,
            add_generation_prompt=False,
        )
        if completed_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"assistant turn {assistant_index} does not extend its generation prompt; "
                "tokenizer chat template is incompatible with strict masking"
            )
        if full_ids[: len(completed_ids)] != completed_ids:
            raise ValueError(
                f"assistant turn {assistant_index} is not an exact prefix of the full dialogue"
            )
        start = len(prefix_ids)
        end = len(completed_ids)
        if end <= start:
            raise ValueError(f"assistant turn {assistant_index} rendered no supervised tokens")
        labels[start:end] = full_ids[start:end]

    supervised_tokens = sum(label != IGNORE_INDEX for label in labels)
    if supervised_tokens == 0:
        raise ValueError("dialogue rendered no assistant supervision")
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
        "n_total_tokens": len(full_ids),
        "n_label_tokens": supervised_tokens,
    }


class TrajectorySFTDataset:
    """Eagerly render the small 45/5 complete-dialogue split."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        max_length: int = 16_384,
        expected_size: Optional[int] = None,
        enable_thinking: bool = False,
        supervise_reasoning: bool = False,
        preserve_historical_reasoning: bool = False,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        path = Path(jsonl_path)
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if expected_size is not None and len(records) != expected_size:
            raise ValueError(f"{path} contains {len(records)} dialogues, expected {expected_size}")
        for record in records:
            effective = {
                **record,
                "messages": prepare_agent_messages(record["messages"]),
                "metadata": {**(record.get("metadata") or {}), **prompt_provenance()},
            }
            example = build_supervised_example(
                effective["messages"], tokenizer, tools=tools, max_length=max_length,
                enable_thinking=enable_thinking, supervise_reasoning=supervise_reasoning,
                preserve_historical_reasoning=preserve_historical_reasoning
            )
            effective["metadata"].update({
                "source_dialogue_hash": (record.get("metadata") or {}).get("dialogue_hash"),
                "dialogue_hash": sha256_text(json.dumps(
                    effective["messages"], sort_keys=True, separators=(",", ":")
                )),
                "rendered_tokens": example["n_total_tokens"],
            })
            example["metadata"] = effective["metadata"]
            self.records.append(effective)
            self.examples.append(example)

    def write_effective_jsonl(self, path: Path) -> None:
        """Persist the actual prompt-bearing records used by this training run."""

        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        example = self.examples[index]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
        }

    def token_stats(self) -> dict[str, float | int]:
        if not self.examples:
            return {"dialogues": 0, "total_tokens": 0, "label_tokens": 0}
        total = sum(int(example["n_total_tokens"]) for example in self.examples)
        labels = sum(int(example["n_label_tokens"]) for example in self.examples)
        return {
            "dialogues": len(self.examples),
            "total_tokens": total,
            "label_tokens": labels,
            "mean_total_tokens": total / len(self.examples),
            "mean_label_tokens": labels / len(self.examples),
        }


def collate_fn_padding(batch: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad a batch while keeping every padding label ignored."""

    import torch

    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_length = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    for index, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[index, :length] = item["input_ids"]
        labels[index, :length] = item["labels"]
        attention_mask[index, :length] = item["attention_mask"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
