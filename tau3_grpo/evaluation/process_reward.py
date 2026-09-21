"""Deterministic, versioned process reward. No judge or learned reward model.

Reference actions may describe just one legal path. Conservative positives are
limited to tasks whose official basis explicitly requires ACTION. Opt-in v2
rewards exact reference DB writes only on officially successful trajectories,
with a total positive budget of one. V3 removes that success gate and compares
the nested objects after the same model conversion performed by the tools.
None of these modes certifies natural-language authorization or policy compliance.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy

READ_TOOLS = frozenset(
    {
        "get_user_details",
        "get_reservation_details",
        "search_direct_flight",
        "search_onestop_flight",
        "list_all_airports",
        "calculate",
    }
)
WRITE_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "transfer_to_human_agents",
        "update_reservation_flights",
        "update_reservation_passengers",
        "update_reservation_baggages",
    }
)
# v2 is a DB-reference shaping experiment, not a policy/authorization verifier.
DB_WRITE_TOOLS = WRITE_TOOLS - {"transfer_to_human_agents"}
DEFAULT_WEIGHTS = {
    "gold_exact": 1.0,
    "soft_match": 0.5,
    "read_only": 0.0,
    "state_change": -0.1,
    "error": -0.1,
    "duplicate": -0.2,
    "message": 0.0,
    "unknown": 0.0,
}
SPLIT_VERSION = "paper_env_split_v3"
DB_SEMANTICS_VERSION = "paper_env_split_v4"
SPLIT_VERSIONS = frozenset({SPLIT_VERSION, DB_SEMANTICS_VERSION})
ENVIRONMENT_VERSIONS = frozenset({"paper_env_v2", *SPLIT_VERSIONS})


def default_weights(version=None):
    """Keep legacy schemas exact; the read/write split is explicitly versioned."""
    weights = dict(DEFAULT_WEIGHTS)
    if version in SPLIT_VERSIONS:
        weights.update(gold_exact=0.0, gold_read=0.0, gold_write=1.0,
                       soft_match=0.0, duplicate=0.0)
    if version == DB_SEMANTICS_VERSION:
        weights["generic"] = 0.0
    return weights


def reward_settings(config=None):
    settings = {"mode": "audit", "version": "v1",
                "weights": default_weights((config or {}).get("version"))}
    if config and config.get("mode") == "paper":
        from tau3_grpo.evaluation.paper_reward import PAPER_OPTIONS

        settings["paper_options"] = dict(PAPER_OPTIONS)
        if config.get("version") in ENVIRONMENT_VERSIONS:
            from tau3_grpo.evaluation.environment_reward import ENVIRONMENT_OPTIONS

            settings["paper_options"] = dict(ENVIRONMENT_OPTIONS)
            if config.get("version") in SPLIT_VERSIONS:
                settings["paper_options"]["soft_scoring"] = "constant"
    if config:
        if set(config) - set(settings):
            raise ValueError("unknown process_reward setting")
        settings.update({k: v for k, v in config.items() if k not in {"weights", "paper_options"}})
        if "paper_options" in config:
            if set(config["paper_options"]) - set(settings["paper_options"]):
                raise ValueError("unknown paper reward option")
            settings["paper_options"].update(config["paper_options"])
        if set(config.get("weights", {})) - set(settings["weights"]):
            raise ValueError("unknown reward tier")
        settings["weights"].update(config.get("weights", {}))
    if (settings["mode"], settings["version"]) not in {
        ("audit", "v1"), ("conservative", "v1"), ("reference_write", "v2"),
        ("reference_write", "v3"),
        ("paper", "paper_v1"),
        ("paper", "paper_env_v2"),
        ("paper", SPLIT_VERSION),
        ("paper", DB_SEMANTICS_VERSION),
    }:
        raise ValueError("unsupported process reward mode/version")
    if settings["version"] == DB_SEMANTICS_VERSION and settings["weights"]["generic"] != 0:
        raise ValueError("generic reward must remain neutral")
    if not all(math.isfinite(float(x)) for x in settings["weights"].values()):
        raise ValueError("non-finite reward weight")
    if settings["mode"] == "paper":
        from tau3_grpo.evaluation.paper_reward import validate_options

        validate_options(settings["paper_options"], settings["version"])
    if settings["version"] in {"v2", "v3"} and (
        not 0 < float(settings["weights"]["gold_exact"]) <= 1
        or float(settings["weights"]["error"]) > 0
    ):
        raise ValueError("reference_write requires a positive reward budget <= 1 and nonpositive error weight")
    return settings


def deep_equal(left, right):
    """Strict types and list order, ignoring only dictionary key order."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(deep_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(deep_equal(a, b) for a, b in zip(left, right))
    return left == right


def match_arguments(call, gold):
    if call["name"] != gold["name"] or not isinstance(call["arguments"], dict):
        return False, 0.0
    actual, expected = call["arguments"], gold["arguments"]
    keys = gold.get("compare_args")
    if keys is None:
        exact = deep_equal(actual, expected)
        keys = list(expected)
    else:
        exact = all(
            k in actual and k in expected and deep_equal(actual[k], expected[k]) for k in keys
        )
    partial = (
        (
            sum(k in actual and k in expected and deep_equal(actual[k], expected[k]) for k in keys)
            / len(keys)
        )
        if keys
        else 0.0
    )
    return exact, partial


def execution_arguments(name, arguments):
    """Mirror ONLY the nested model conversions in the pinned Airline tools.

    Top-level keys, IDs and list order remain strict. The official models decide
    nested validation, extra-field handling and coercion; no heuristic soft match.
    This does not execute a tool, access its DB, or validate user authorization.
    """
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    fields = {
        "book_reservation": {"flights": "FlightInfo", "passengers": "Passenger",
                             "payment_methods": "Payment"},
        "update_reservation_flights": {"flights": "FlightInfo"},
        "update_reservation_passengers": {"passengers": "Passenger"},
    }.get(name, {})
    result = deepcopy(arguments)
    if fields:
        from tau3_grpo.envs.tau2_bridge import import_tau2

        models = import_tau2("tau2.domains.airline.data_model")
        for field, model_name in fields.items():
            values = result.get(field)
            if not isinstance(values, list) or not all(isinstance(x, dict) for x in values):
                raise ValueError(f"{name}.{field} requires a list of objects")
            model = getattr(models, model_name)
            result[field] = [model(**value).model_dump(mode="json") for value in values]
    return result


def score_turns(records, golden_actions, reward_basis, config=None, *, official_outcome=None):
    """Sum call rewards within each turn; each gold occurrence can pay only once.

    Audit computes candidate rewards but pays zero. Conservative pays only
    successful first-time required exact actions and explicit execution errors.
    Duplicate/soft/non-gold-write penalties remain audit-only in v1.
    """
    settings = reward_settings(config)
    if settings["mode"] == "paper":
        from tau3_grpo.evaluation.paper_reward import score_paper_turns

        return score_paper_turns(
            records, golden_actions, reward_basis, settings, official_outcome=official_outcome
        )
    reference_write = settings["mode"] == "reference_write"
    execution_match = settings["version"] == "v3"
    if reference_write and official_outcome not in (0.0, 1.0):
        raise ValueError("reference_write requires the binary official_outcome")
    gold = [deepcopy(g) for g in golden_actions if g.get("requestor", "assistant") == "assistant"]
    effective_gold = [execution_arguments(g["name"], g["arguments"]) for g in gold] if execution_match else []
    bases = {str(getattr(b, "value", b)).upper() for b in reward_basis}
    required = "ACTION" in bases
    reference_writes = sum(g["name"] in DB_WRITE_TOOLS for g in gold)
    used = set()
    rows = deepcopy(records)
    for k, row in enumerate(rows):
        # Each occurrence is independent even if callers reused a dict object.
        row = rows[k] = deepcopy(row)
        row["tool_calls"] = [deepcopy(event) for event in row["tool_calls"]]
        if row["turn_index"] != k or row.get("schema") != "tau3_turn_v1":
            raise ValueError("invalid turn record schema/order")
        row["reward"] = row["candidate_reward"] = 0.0
        tiers = []
        for event in row["tool_calls"]:
            if type(event.get("error")) is not bool:
                raise ValueError("tool event requires explicit execution error status")
            effective = None
            if execution_match:
                try:
                    effective = execution_arguments(event["name"], event["arguments"])
                except (ValueError, TypeError):
                    pass  # Invalid arguments cannot match or receive a positive reward.
            exact = [j for j, g in enumerate(gold) if (
                event["name"] == g["name"] and effective is not None
                and deep_equal(effective, effective_gold[j])
                if execution_match else (
                event["name"] == g["name"] and deep_equal(event["arguments"], g["arguments"])
                if reference_write else match_arguments(event, g)[0]
            ))]
            fresh = next((j for j in exact if j not in used), None)
            partial = max((match_arguments(event, g)[1] for g in gold), default=0.0)
            match = None
            if event["error"]:
                tier = "error"
            elif fresh is not None:
                tier, match = "gold_exact", fresh
                used.add(fresh)
            elif exact:
                tier = "duplicate"
            elif partial > 0:
                tier = "soft_match"
            elif event["name"] in READ_TOOLS or (reference_write and event["name"] == "get_flight_status"):
                tier = "read_only"
            elif event["name"] in WRITE_TOOLS:
                tier = "state_change"
            else:
                tier = "unknown"
            candidate = float(settings["weights"][tier])
            reward = (
                candidate
                if (
                    settings["mode"] == "conservative"
                    and (tier == "error" or (tier == "gold_exact" and required))
                )
                else 0.0
            )
            if reference_write:
                # A reference read is never paid. Alternative legal writes are
                # neutral, and compare_args cannot hide a wrong write argument.
                reward = 0.0
                if tier == "error":
                    reward = float(settings["weights"]["error"])
                elif (tier == "gold_exact" and event["name"] in DB_WRITE_TOOLS
                      and "DB" in bases and (execution_match or official_outcome == 1.0)):
                    reward = float(settings["weights"]["gold_exact"]) / reference_writes
            event.update(
                reward_type=tier,
                matched_gold_index=match,
                candidate_reward=candidate,
                reward=reward,
            )
            if execution_match:
                event["effective_arguments"] = effective
            row["candidate_reward"] += candidate
            row["reward"] += reward
            tiers.append(tier)
        row["reward_types"] = tiers or ["message"]
    payload = {
        "schema": "tau3_process_reward_v1",
        "settings": settings,
        "golden_actions": gold,
        "reward_basis": list(reward_basis),
        "turn_records": rows,
        "turn_rewards": [row["reward"] for row in rows],
        "turn_spans": [row["token_span"] for row in rows],
    }
    if reference_write:
        payload["official_outcome"] = float(official_outcome)
        payload["reference_write_count"] = reference_writes
        payload["positive_reward_budget"] = float(settings["weights"]["gold_exact"])
    if execution_match:
        payload["matching"] = "airline_execution_models_v1"
        payload["effective_golden_arguments"] = effective_gold
    return payload


def payload_json(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
