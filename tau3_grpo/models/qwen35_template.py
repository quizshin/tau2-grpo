"""Assistant supervision in one native Qwen3.5 full-dialogue render.

Historical thinking scaffolding changes as the last user turn moves. Prefix
comparisons therefore cannot locate assistant spans. Jinja generation markers
mark those spans without altering the model's native text or token sequence.
"""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=8)
def supervised_template(template: str) -> str:
    marker = '{%- elif message.role == "assistant" %}'
    before, found, rest = template.partition(marker)
    body, end, after = rest.partition('{%- elif message.role == "tool" %}')
    if not found or not end or body.count(" + content }}") != 2:
        raise ValueError(
            "Unsupported Qwen3.5 assistant template; review supervision before training"
        )
    body = body.replace(" + content }}", " }}{%- generation %}{{- content }}{%- endgeneration %}")
    tool_marker = "{%- if message.tool_calls and"
    eos_marker = "{{- '<|im_end|>\\n' }}"
    if body.count(tool_marker) != 1 or body.count(eos_marker) != 1:
        raise ValueError(
            "Unsupported Qwen3.5 tool/EOS template; review supervision before training"
        )
    body = body.replace(tool_marker, "{%- generation %}" + tool_marker)
    body = body.replace(eos_marker, eos_marker + "{%- endgeneration %}")
    return before + found + body + end + after


def token_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected one rendered dialogue")
        encoded = encoded[0]
    return [int(value) for value in encoded]


def build_qwen35_example(messages, tokenizer, *, tools, max_length):
    params = dict(tools=tools, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    native = token_ids(tokenizer.apply_chat_template(list(messages), **params))
    if len(native) > max_length:
        raise ValueError(
            f"rendered dialogue has {len(native)} tokens, exceeding max_length={max_length}"
        )
    encoded = tokenizer.apply_chat_template(
        list(messages),
        chat_template=supervised_template(tokenizer.chat_template),
        return_dict=True,
        return_assistant_tokens_mask=True,
        **params,
    )
    tokens = token_ids(encoded)
    mask = encoded["assistant_masks"]
    if mask and isinstance(mask[0], list):
        mask = mask[0]
    if tokens != native or len(mask) != len(tokens) or not any(mask[1:]):
        raise ValueError("Qwen3.5 supervision changed native tokens or omitted assistant labels")
    return {
        "input_ids": tokens,
        "labels": [
            value if supervised else -100 for value, supervised in zip(tokens, mask, strict=True)
        ],
        "attention_mask": [1] * len(tokens),
        "n_total_tokens": len(tokens),
        "n_label_tokens": sum(bool(value) for value in mask),
    }
