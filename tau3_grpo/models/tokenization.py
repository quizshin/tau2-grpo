"""Public chat rendering shared by data preparation and SFT."""

from __future__ import annotations

from typing import Any, Sequence

from tau3_grpo.models.compat import chat_template_kwargs
from tau3_grpo.models.qwen35_template import token_ids


def render_chat_ids(
    tokenizer: Any, messages: Sequence[dict[str, Any]], *,
    tools: list[dict[str, Any]] | None, add_generation_prompt: bool,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        list(messages), tools=tools, tokenize=True,
        add_generation_prompt=add_generation_prompt, **chat_template_kwargs(tokenizer),
    )
    return token_ids(rendered)
