"""Explicit interpretation of arXiv:2604.02869v1, not author-code parity.

Unspecified choices are versioned: sum/mean aggregation; successful-call duplicate
tracking; gold before soft before read/state; whole argument objects (compare_args
is not used). Numeric coercion can collapse identifiers; this mode deliberately
does not claim execution equivalence or policy/authorization validation.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from decimal import Decimal

PAPER_OPTIONS = {
    "normalization": "recursive_v1",
    "aggregation": "sum",
    "duplicate_scope": "successful_call",
    "soft_scoring": "overlap",
    "precedence": "error_gold_duplicate_soft_read_state",
}
_NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def validate_options(options, version="paper_v1"):
    from tau3_grpo.evaluation.environment_reward import ENVIRONMENT_OPTIONS
    from tau3_grpo.evaluation.process_reward import ENVIRONMENT_VERSIONS

    defaults = ENVIRONMENT_OPTIONS if version in ENVIRONMENT_VERSIONS else PAPER_OPTIONS
    if set(options) != set(defaults):
        raise ValueError("incomplete paper reward options")
    for key, value in options.items():
        choices = {
            "aggregation": {"sum", "mean"},
            "soft_scoring": {"overlap", "constant"},
        }.get(key, {defaults[key]})
        if value not in choices:
            raise ValueError(f"unsupported paper option {key}={value}")


def _empty(value):
    return value is None or value == "" or value == [] or value == {}


def normalize(value):
    """Hashable, type-tagged recursive normalization with exact decimal numbers.

    Empty entries are removed; lists of dicts are sorted, other lists retain
    order; booleans never equal numbers. Scalars and extra nonempty keys remain.
    """
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("JSON object keys must be strings")
        return ("object", tuple(sorted((k, normalize(v)) for k, v in value.items() if not _empty(v))))
    if isinstance(value, list):
        items = [normalize(v) for v in value if not _empty(v)]
        if all(isinstance(v, dict) for v in value):
            items.sort(key=repr)
        return ("list", tuple(items))
    if isinstance(value, bool):
        return ("bool", value)
    if value is None:
        return ("null",)
    if isinstance(value, (int, float)) or (
        isinstance(value, str) and _NUMERIC.fullmatch(value.strip())
    ):
        number = Decimal(str(value).strip())
        if not number.is_finite():
            raise ValueError("non-finite JSON number")
        # Canonical repr for sorting dict lists, without Decimal.normalize()'s
        # context precision rounding (identifiers can exceed 28 digits).
        sign, digits, exponent = number.as_tuple()
        digits = list(digits)
        while digits and digits[-1] == 0:
            digits.pop()
            exponent += 1
        number = Decimal((sign, tuple(digits), exponent)) if digits else Decimal(0)
        return ("number", number)  # Decimal equality is exact, including trailing zeroes.
    if isinstance(value, str):
        return ("string", value)
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def match_arguments(call, gold):
    if call["name"] != gold["name"] or not isinstance(call["arguments"], dict):
        return False, 0.0
    actual, expected = normalize(call["arguments"]), normalize(gold["arguments"])
    left, right = dict(actual[1]), dict(expected[1])
    overlap = sum(k in left and left[k] == v for k, v in right.items())
    return actual == expected, overlap / len(right) if right else 0.0


def score_paper_turns(records, golden_actions, reward_basis, settings, *, official_outcome):
    from tau3_grpo.evaluation.process_reward import (
        DB_WRITE_TOOLS,
        ENVIRONMENT_VERSIONS,
        READ_TOOLS,
        SPLIT_VERSION,
        WRITE_TOOLS,
    )

    if official_outcome not in (0.0, 1.0):
        raise ValueError("paper reward requires the binary official_outcome")
    gold = [deepcopy(g) for g in golden_actions if g.get("requestor", "assistant") == "assistant"]
    if any(not isinstance(g["arguments"], dict) for g in gold):
        raise ValueError("gold arguments must be objects")
    used, seen = set(), set()
    environment = None
    if settings["version"] in ENVIRONMENT_VERSIONS:
        from tau3_grpo.evaluation.environment_reward import EnvironmentMatcher

        environment = EnvironmentMatcher(gold)
    rows = []
    weights, options = settings["weights"], settings["paper_options"]
    for k, original in enumerate(records):
        row = deepcopy(original)
        # Deepcopy each occurrence: callers may reuse the same event object.
        row["tool_calls"] = [deepcopy(e) for e in original["tool_calls"]]
        if row.get("schema") != "tau3_turn_v1" or row["turn_index"] != k:
            raise ValueError("invalid turn record schema/order")
        values, tiers = [], []
        for event in row["tool_calls"]:
            if type(event.get("error")) is not bool:
                raise ValueError("tool event requires explicit execution error status")
            # Keep historical paper_v1 matching and payouts byte-for-byte replayable.
            matches = [match_arguments(event, g) for g in gold] if environment is None else []
            exact = [j for j, (matched, _) in enumerate(matches) if matched]
            fresh = next((j for j in exact if j not in used), None)
            partial = max((overlap for _, overlap in matches), default=0.0)
            signature = (event["name"], normalize(event["arguments"])) if environment is None else None
            match = None
            metadata = {}
            if environment is not None:
                tier, match, partial, metadata = environment.classify(event, used)
            elif event["error"]:
                tier = "error"
            elif fresh is not None:
                tier, match = "gold_exact", fresh
                used.add(fresh)
            elif exact or signature in seen:
                tier = "duplicate"
            elif partial > 0:
                tier = "soft_match"
            elif event["name"] in READ_TOOLS or event["name"] == "get_flight_status":
                tier = "read_only"
            elif event["name"] in WRITE_TOOLS:
                tier = "state_change"
            else:
                tier = "unknown"
            if settings["version"] == SPLIT_VERSION and tier == "gold_exact":
                # Matching/one-time consumption stay unchanged. Transfer-to-human
                # and other non-DB actions retain the separate generic exact tier.
                if event["name"] in READ_TOOLS or event["name"] == "get_flight_status":
                    tier = "gold_read"
                elif event["name"] in DB_WRITE_TOOLS:
                    tier = "gold_write"
            if environment is None and not event["error"]:
                seen.add(signature)
            # Default soft weight .5 gives Appendix A's .5 + .5 * overlap.
            # IRC constant mode instead uses the calibrated categorical r_c.
            scale = 1 + partial if tier == "soft_match" and options["soft_scoring"] == "overlap" else 1
            reward = float(weights[tier]) * scale
            if not math.isfinite(reward):
                raise ValueError("non-finite paper call reward")
            # Drop fields from the execution-equivalence scorer when reclassifying a buffer.
            event.pop("effective_arguments", None)
            event.pop("environment_match", None)
            event.pop("duplicate_evidence", None)
            event.update(reward_type=tier, matched_gold_index=match,
                         candidate_reward=reward, reward=reward, paper_match_fraction=partial)
            event.update(metadata)
            values.append(reward)
            tiers.append(tier)
        if not values:
            values = [float(weights["message"])]
        total = sum(values)
        if options["aggregation"] == "mean":
            total /= len(values)
        row.update(reward=total, candidate_reward=total, reward_types=tiers or ["message"])
        rows.append(row)
    return {
        "schema": "tau3_process_reward_v1", "settings": deepcopy(settings),
        "matching": "environment_execution_split_v1" if settings["version"] == SPLIT_VERSION else
                    "environment_execution_v1" if environment is not None else "paper_recursive_v1",
        "official_outcome": float(official_outcome),
        "golden_actions": gold, "reward_basis": list(reward_basis), "turn_records": rows,
        "turn_rewards": [r["reward"] for r in rows], "turn_spans": [r["token_span"] for r in rows],
    }
