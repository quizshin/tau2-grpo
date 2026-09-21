"""Token accounting for append-only multi-turn trajectories, without model calls."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from tau3_grpo.configuration import load_config
from tau3_grpo.paths import CONFIG_ROOT

VERSION = "tau3_token_budget_v1"


@dataclass(frozen=True)
class TokenBudget:
    version: str
    prompt: int
    response: int
    context: int
    per_turn: int
    user_per_turn: int
    user_context: int

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError(f"Unknown token budget: {self.version}")
        for name, value in asdict(self).items():
            if name != "version" and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.prompt > self.context or self.user_per_turn >= self.user_context:
            raise ValueError("Token budget exceeds its context")

    def allowance(self, prompt_tokens: int, appended_tokens: int) -> int:
        if not 0 <= appended_tokens <= prompt_tokens:
            raise ValueError("Invalid append-only token counts")
        return max(0, min(self.per_turn, self.response - appended_tokens,
                          self.context - prompt_tokens))

    def fits(self, prompt_tokens: int, appended_tokens: int, observation_tokens: int) -> bool:
        if min(prompt_tokens, appended_tokens, observation_tokens) < 0:
            raise ValueError("Negative token count")
        # Equality is allowed: the observation is retained, but no next generation.
        return (appended_tokens + observation_tokens <= self.response
                and prompt_tokens + observation_tokens <= self.context)


def default_budget() -> TokenBudget:
    return TokenBudget(**load_config(CONFIG_ROOT / "protocols/token_budget_v1.yaml"))


def token_digest(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def complete_tool_envelopes(text: str) -> bool:
    """Never let the permissive native XML parser execute a truncated call batch."""
    import re

    # Even a cut inside the tag name must invalidate earlier complete calls.
    tail = text[text.rfind("<"):] if "<" in text else ""
    prefixes = ("<tool_call", "</tool_call", "<function=", "</function", "<parameter=", "</parameter")
    if (tail and ">" not in tail and (tail != "<" or "<tool_call>" in text)
            and any(p.startswith(tail) or tail.startswith(p) for p in prefixes)):
        return False
    stack = []
    for tag in re.findall(r"</?(?:tool_call|function|parameter)(?:=[^>]*)?>", text):
        name = tag.strip("<>").lstrip("/").split("=", 1)[0]
        if tag.startswith("</"):
            if not stack or stack.pop() != name:
                return False
        else:
            if ((name == "tool_call" and stack)
                    or (name == "function" and stack != ["tool_call"])
                    or (name == "parameter" and stack != ["tool_call", "function"])):
                return False
            stack.append(name)
    return not stack
