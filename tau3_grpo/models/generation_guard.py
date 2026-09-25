"""Versioned, bounded repetition diagnostics on actual generated tokens.

Detection never edits tokens or assigns rewards. Tool-bearing turns are excluded
from cancellation until their repetition rules have separate validation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationGuard:
    version: str = "tau3_generation_guard_v1"
    mode: str = "off"
    min_tokens: int = 128
    repeats: int = 32
    max_token_period: int = 8
    max_text_period: int = 64
    min_text_chars: int = 128

    def __post_init__(self):
        if self.version != "tau3_generation_guard_v1" or self.mode not in {"off", "observe", "abort"}:
            raise ValueError("Invalid generation guard version/mode")
        for name in ("min_tokens", "repeats", "max_token_period", "max_text_period", "min_text_chars"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid repetition setting: {name}")
        if (self.repeats < 4 or self.max_token_period > 32 or self.max_text_period > 128
                or self.repeats > 128 or self.min_tokens > 16384):
            raise ValueError("Repetition settings outside bounded detector limits")

    @classmethod
    def from_config(cls, config=None):
        return cls(**dict(config or {}))


def repetition_penalty(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("repetition_penalty must be finite and positive")
    return float(value)


def sampling_penalty(config, *, validation=False):
    value = config.get("repetition_penalty", 1.0)
    if validation:
        override = (config.get("val_kwargs") or {}).get("repetition_penalty")
        if override is not None:
            value = override
    return repetition_penalty(value)


def detect_repetition(ids, text, config: GenerationGuard):
    """Return suffix evidence; normalized text is diagnostic only."""
    if config.mode == "off" or len(ids) < config.min_tokens:
        return None
    # Even a partial tool envelope excludes the entire turn from this v1 guard.
    if "<tool" in text or "<function" in text or "<parameter" in text:
        return None
    base = {"version": config.version, "mode": config.mode, "detected_at_tokens": len(ids),
            "scope": "ordinary_assistant_turn", "settings": asdict(config)}
    for period in range(1, config.max_token_period + 1):
        size = period * config.repeats
        if size <= len(ids) and list(ids[-size:]) == list(ids[-period:]) * config.repeats:
            return {**base, "kind": "token_cycle", "period": period, "repeat_count": config.repeats}
    # Bound the tail, and normalize whitespace/variation selectors only here.
    tail = "".join(c for c in text[-32768:] if not c.isspace() and c != "\ufe0f")
    for period in range(1, config.max_text_period + 1):
        count = max(config.repeats, (config.min_text_chars + period - 1) // period)
        size = period * count
        if size <= len(tail) and tail[-size:] == tail[-period:] * count:
            return {**base, "kind": "text_cycle", "period": period, "repeat_count": count}
    return None
