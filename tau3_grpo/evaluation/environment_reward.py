"""Versioned Airline execution-input and repeated-read matching for paper_env_v2.

No DB execution or user-authorization judgment. A changed query response removes
the duplicate penalty; it never renews a consumed gold reward.
"""

from __future__ import annotations

import json
import math

ENVIRONMENT_OPTIONS = {
    "normalization": "execution_v1",
    "aggregation": "sum",
    "duplicate_scope": "successful_call_read_response_v1",
    "soft_scoring": "overlap",
    "precedence": "error_gold_duplicate_soft_read_state",
}


def typed_json(value):
    """Strict JSON identity: preserve IDs, empty values, types and list order."""
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("JSON keys must be strings")
        return ("object", tuple(sorted((k, typed_json(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ("list", tuple(typed_json(v) for v in value))
    if value is None:
        return ("null",)
    if type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("non-finite JSON value")
        return (type(value).__name__, value)
    raise ValueError("unsupported JSON value")


def observation_identity(event):
    if event.get("observation_truncated") or event.get("observation") is None:
        return None
    value = event["observation"]
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        return typed_json(value)
    except ValueError:
        return None


class EnvironmentMatcher:
    def __init__(self, gold):
        from tau3_grpo.evaluation.process_reward import execution_arguments

        self.gold = [(g["name"], execution_arguments(g["name"], g["arguments"])) for g in gold]
        self.seen = {}

    def classify(self, event, used):
        from tau3_grpo.evaluation.process_reward import READ_TOOLS, WRITE_TOOLS, execution_arguments

        # Real execution errors dominate and never consume gold or duplicate history.
        if event["error"]:
            return "error", None, 0.0, {"environment_match": "execution_error"}
        try:
            effective = execution_arguments(event["name"], event["arguments"])
            actual = typed_json(effective)
        except (ValueError, TypeError):
            return "unknown", None, 0.0, {"environment_match": "invalid_successful_arguments"}
        matches = []
        for name, expected in self.gold:
            if event["name"] != name:
                matches.append((False, 0.0))
                continue
            exact = actual == typed_json(expected)
            overlap = sum(k in effective and typed_json(effective[k]) == typed_json(v)
                          for k, v in expected.items())
            matches.append((exact, overlap / len(expected) if expected else 0.0))
        fresh = next((i for i, (exact, _) in enumerate(matches) if exact and i not in used), None)
        partial = max((fraction for _, fraction in matches), default=0.0)
        signature = (event["name"], actual)
        repeated = signature in self.seen
        is_read = event["name"] in READ_TOOLS or event["name"] == "get_flight_status"
        observation = observation_identity(event) if is_read else None
        evidence = "first_success"
        if repeated:
            if not is_read:
                evidence = "same_execution_arguments"
            elif observation is None or self.seen[signature] is None:
                evidence = "read_response_unavailable"
            else:
                evidence = "read_response_unchanged" if observation == self.seen[signature] else "read_response_changed"
        if fresh is not None:
            tier, match = "gold_exact", fresh
            used.add(fresh)
        elif repeated:
            match = None
            tier = ("unknown" if evidence == "read_response_unavailable" else
                    "read_only" if evidence == "read_response_changed" else "duplicate")
        elif partial > 0:
            tier, match = "soft_match", None
        elif is_read:
            tier, match = "read_only", None
        elif event["name"] in WRITE_TOOLS:
            tier, match = "state_change", None
        else:
            tier, match = "unknown", None
        self.seen[signature] = observation
        return tier, match, partial, {"effective_arguments": effective, "environment_match": "execution_v1",
                                     "duplicate_evidence": evidence}
