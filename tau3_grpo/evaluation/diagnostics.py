"""Observed execution costs with explicit missing-data coverage.

These are diagnostics of scored trajectories, not new benchmark rewards. Token
counts are supplied usage only; repeated prompts count repeated input processing.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _distribution(values, expected):
    return {
        "observed": len(values), "expected": expected,
        "coverage": len(values) / expected if expected else None,
        "sum_observed": float(sum(values)) if values else None,
        "mean_observed": float(np.mean(values)) if values else None,
        "p50_observed": float(np.quantile(values, .5)) if values else None,
        "p95_observed": float(np.quantile(values, .95)) if values else None,
    }


def trajectory_diagnostics(rows: list[dict]) -> dict:
    turns, calls, durations = [], [], []
    tokens = {role: {kind: [] for kind in ("prompt_tokens", "completion_tokens")}
              for role in ("assistant", "user")}
    messages_seen = Counter()
    termination = Counter()
    matched = known_errors = error_count = requested = 0
    unknown_tools = 0
    for row in rows:
        reason = row.get("termination_reason")
        termination[reason if isinstance(reason, str) and reason else "unknown"] += 1
        simulation = row.get("simulation") or {}
        if _number(simulation.get("duration")):
            durations.append(simulation["duration"])
        messages = simulation.get("messages")
        if not isinstance(messages, list):
            continue
        assistants = [m for m in messages if m.get("role") == "assistant"]
        turns.append(len(assistants))
        assistant_calls = [call for m in assistants for call in (m.get("tool_calls") or [])]
        calls.append(len(assistant_calls))
        requested += len(assistant_calls)
        # Match ordered responses to assistant calls, keeping user tools separate.
        pending = Counter()
        for message in messages:
            role = message.get("role")
            if role in tokens:
                messages_seen[role] += 1
                usage = message.get("usage") or {}
                for kind in tokens[role]:
                    if _number(usage.get(kind)):
                        tokens[role][kind].append(usage[kind])
            if role == "assistant":
                pending.update(c["id"] for c in (message.get("tool_calls") or [])
                               if c.get("id"))
            elif role == "tool":
                identity = message.get("tool_call_id") or message.get("id")
                if message.get("requestor") == "user":
                    continue
                if not identity or pending[identity] <= 0:
                    unknown_tools += 1
                    continue
                pending[identity] -= 1
                matched += 1
                error = message.get("error")
                if isinstance(error, bool):
                    known_errors += 1
                    error_count += int(error)
    return {
        "scope": "scored trajectories only; incomplete runs are not comparable",
        "assistant_messages_per_trajectory": _distribution(turns, len(rows)),
        "assistant_tool_calls_per_trajectory": _distribution(calls, len(rows)),
        "simulation_duration_seconds": _distribution(durations, len(rows)),
        "duration_note": "sum is accumulated trajectory duration, not run wall time or GPU hours",
        "termination_reasons": dict(sorted(termination.items())),
        "assistant_tools": {
            "requested": requested if turns else None,
            "matched_responses": matched if turns else None,
            "response_coverage": matched / requested if requested else None,
            "responses_with_error_flag": known_errors if turns else None,
            "errors": error_count if known_errors else None,
            "error_rate_observed": error_count / known_errors if known_errors else None,
            "error_flag_coverage": known_errors / requested if requested else None,
            "unattributed_responses": unknown_tools if turns else None,
        },
        "usage_by_message": {
            role: {kind: _distribution(values, messages_seen[role])
                   for kind, values in by_kind.items()}
            for role, by_kind in tokens.items()
        },
        "usage_note": "observed usage, not retokenized; coverage includes scripted messages; missing is not zero",
    }
