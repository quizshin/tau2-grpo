"""Assistant supervision in one native Qwen3.5 full-dialogue render.

Historical thinking scaffolding changes as the last user turn moves. Prefix
comparisons therefore cannot locate assistant spans. Jinja generation markers
mark those spans without altering the model's native text or token sequence.
"""

from copy import deepcopy
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=8)
def supervised_template(template: str, supervise_reasoning: bool = False) -> str:
    marker = '{%- elif message.role == "assistant" %}'
    before, found, rest = template.partition(marker)
    body, end, after = rest.partition('{%- elif message.role == "tool" %}')
    if not found or not end or body.count(" + content }}") != 2:
        raise ValueError(
            "Unsupported Qwen3.5 assistant template; review supervision before training"
        )
    if supervise_reasoning:
        reasoning = "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}"
        plain = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
        if body.count(reasoning) != 1 or body.count(plain) != 1:
            raise ValueError("Unsupported Qwen3.5 reasoning template")
        header = "{{- '<|im_start|>' + message.role + '\\n' }}"
        body = body.replace(reasoning, header + "{%- generation %}{{- '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}{%- endgeneration %}")
        body = body.replace(plain, header + "{%- generation %}{{- content }}{%- endgeneration %}")
    else:
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


THINKING_KEYS = ("enable_thinking", "supervise_reasoning", "preserve_historical_reasoning")


def thinking_options(config):
    options = {key: config.get(key, False) for key in THINKING_KEYS}
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("SFT thinking options must be YAML booleans")
    if not options["enable_thinking"] and any(options[k] for k in THINKING_KEYS[1:]):
        raise ValueError("Reasoning supervision/history requires enable_thinking=true")
    return options


def build_qwen35_example(messages, tokenizer, *, tools, max_length,
                         enable_thinking=False, supervise_reasoning=False,
                         preserve_historical_reasoning=False):
    thinking_options(dict(enable_thinking=enable_thinking,
                         supervise_reasoning=supervise_reasoning,
                         preserve_historical_reasoning=preserve_historical_reasoning))
    messages = deepcopy(list(messages))
    template = tokenizer.chat_template
    if enable_thinking:
        for message in messages:
            if message.get("role") == "assistant":
                message["reasoning_content"] = message.get("reasoning_content", message.get("reasoning", ""))
    if preserve_historical_reasoning:
        condition = "loop.index0 > ns.last_query_index"
        if template.count(condition) != 1:
            raise ValueError("Unsupported Qwen3.5 history template")
        template = template.replace(condition, "true")
    params = dict(tools=tools, tokenize=True, add_generation_prompt=False,
                  enable_thinking=enable_thinking)
    native = token_ids(tokenizer.apply_chat_template(messages, chat_template=template, **params))
    if len(native) > max_length:
        raise ValueError(
            f"rendered dialogue has {len(native)} tokens, exceeding max_length={max_length}"
        )
    encoded = tokenizer.apply_chat_template(
        list(messages),
        chat_template=supervised_template(template, supervise_reasoning),
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
