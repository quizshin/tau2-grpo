"""Outcome-blind capability features shared by data builders and census tools."""

import re
from collections import Counter
from statistics import median

from tau3_grpo.data.sft_evidence import function

OPERATIONS = {
    "book_reservation": "booking",
    "cancel_reservation": "cancellation",
    "update_reservation_flights": "flight_or_cabin_change",
    "update_reservation_cabin": "flight_or_cabin_change",
    "update_reservation_baggages": "baggage",
    "update_reservation_passengers": "passenger_change",
    "send_certificate": "compensation",
    "transfer_to_human_agents": "escalation",
}



INTENT_RULES = {
    "booking": r"\b(book|booking|reserve|rebook)\b",
    "cancellation": r"\b(cancel\w*|duplicate\w*|overlapping)\b",
    "flight_or_cabin_change": r"\b(change|move|reschedul\w*|upgrade|downgrade|shift|push)\b",
    "baggage": r"\b(bag\w*|luggage)\b",
    "passenger_change": r"\b(passenger|name|spelling|brother|wife|swap|who is|who appears|who is flying)\b",
    "compensation": r"\b(compensat\w*|goodwill|voucher)\b",
}



CONSTRAINT_RULES = {
    "time_window_or_order": r"\b(before|after|earliest|latest|morning|evening|afternoon|between)\b",
    "price_or_refund": r"\$\d|\b(cheapest|cheaper|budget|refund|save money|extra|difference|cost)\b",
    "preserve_other_state": r"\b(only|same|keeping|keep|unchanged|without|except|just one)\b",
    "conditional_or_policy": r"\b(if|unless|insurance|basic economy|health|medical|illness|24 hour)\b",
    "multiple_targets": r"\b(then|several|multiple|all flights|all reservations|both|two|three|four)\b",
    "payment_constraint": r"\b(payment|credit card|gift card|certificate|original payment)\b",
    "identity_constraint": r"\b(passport|spelling|birth|brother|wife|son|daughter|replace|swap)\b",
}



def features(reason, calls, *, messages=None):
    names = [name for name, args in calls]
    reservations = {args["reservation_id"] for name, args in calls if args.get("reservation_id")}
    ops = {OPERATIONS[n] for n in names if n in OPERATIONS}
    constraints = {k for k, pattern in CONSTRAINT_RULES.items() if re.search(pattern, reason, re.I)}
    intent = {k for k, pattern in INTENT_RULES.items() if re.search(pattern, reason, re.I)}
    # A lexical intent proxy shared across every split, not calibrated intrinsic difficulty.
    level = (
        "0-1_constraint_tags"
        if len(constraints) <= 1
        else "2-3_constraint_tags"
        if len(constraints) <= 3
        else "4+_constraint_tags"
    )
    out = dict(
        intent_tags=sorted(intent),
        constraint_tags=sorted(constraints),
        intent_proxy_bin=level,
        tool_presence=sorted(set(names)),
        tool_calls=len(names),
        first_tool=names[0] if names else "none",
        operation_tags=sorted(ops),
        distinct_reservations=len(reservations),
        operation_breadth="0"
        if not ops
        else "1"
        if len(ops) == 1
        else "2"
        if len(ops) == 2
        else "3+",
        reference_or_observed_reservation_count_bin="0"
        if not reservations
        else "1"
        if len(reservations) == 1
        else "2-3"
        if len(reservations) <= 3
        else "4+",
        reservation_ids=sorted(reservations),
    )
    if messages is not None:
        first = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
        user_ids = {args["user_id"] for name, args in calls if args.get("user_id")}
        out.update(
            user_turns=sum(m["role"] == "user" for m in messages),
            assistant_turns=sum(m["role"] == "assistant" for m in messages),
            opening_has_reservation=any(x in first for x in reservations),
            opening_has_user_id=any(x in first for x in user_ids),
            multicall_turns=sum(len(m.get("tool_calls") or []) > 1 for m in messages),
            visible_characters=sum(len(m.get("content") or "") for m in messages),
        )
    return out



def summarize(records, evidence):
    n = len(records)
    out = {
        "count": n,
        "action_evidence": evidence,
        "intrinsic_difficulty": "not_human_calibrated",
        "intent_proxy_basis": "reason_for_call_only_same_regex_all_sets",
        "tool_coverage_is_necessity": False,
    }
    for key in ("intent_tags", "constraint_tags", "tool_presence", "operation_tags"):
        out[key] = dict(sorted(Counter(x for r in records for x in r[key]).items()))
    for key in (
        "intent_proxy_bin",
        "operation_breadth",
        "reference_or_observed_reservation_count_bin",
        "first_tool",
    ):
        out[key] = dict(sorted(Counter(r[key] for r in records).items()))
    for key in ("tool_calls", "user_turns", "assistant_turns", "visible_characters"):
        vals = [r[key] for r in records if key in r]
        if vals:
            vs = sorted(vals)
            out[key] = {
                "n": len(vals),
                "mean": sum(vals) / len(vals),
                "median": median(vals),
                "p90": vs[min(len(vs) - 1, int(0.9 * len(vs)))],
                "max": max(vals),
            }
    for key in ("opening_has_reservation", "opening_has_user_id"):
        if any(key in r for r in records):
            out[key] = sum(r.get(key, False) for r in records)
    return out



def sft_feature(row):
    calls = [function(c) for m in row["messages"] for c in m.get("tool_calls") or []]
    return features(row["metadata"].get("reason_for_call", ""), calls, messages=row["messages"])



def task_feature(task):
    t = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
    ins = t["user_scenario"]["instructions"]
    reason = ins.get("reason_for_call", "") if isinstance(ins, dict) else str(ins)
    return features(
        reason, [(a["name"], a["arguments"]) for a in t["evaluation_criteria"].get("actions") or []]
    )



def annotation_coverage(tasks):
    counts = Counter()
    for task in tasks:
        t = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
        ec = t.get("evaluation_criteria") or {}
        acts = ec.get("actions") or []
        counts["empty_reference_actions"] += not acts
        counts["no_reference_write_or_transfer"] += not any(a["name"] in OPERATIONS for a in acts)
        counts["has_communicate_info"] += bool(ec.get("communicate_info"))
        counts["has_nl_assertions"] += bool(ec.get("nl_assertions"))
        counts["has_env_assertions"] += bool(ec.get("env_assertions"))
        for basis in ec.get("reward_basis") or ["DB", "COMMUNICATE"]:
            counts["reward_basis_" + str(basis)] += 1
    return dict(counts)



def confirmation_screen(messages):
    """Flag only absence of any earlier assistant prose/user response pair.

    This is a weak necessary-condition screen. A passing result does not prove
    the right action was described or approved. Never auto-insert a yes.
    """
    described = False
    user_after_description = False
    flags = []
    for i, m in enumerate(messages):
        if (
            m["role"] == "assistant"
            and (m.get("content") or "").strip()
            and not m.get("tool_calls")
        ):
            described = True
        elif m["role"] == "user" and described:
            user_after_description = True
        elif m["role"] == "assistant":
            writes = [
                function(c)[0]
                for c in m.get("tool_calls") or []
                if function(c)[0] in OPERATIONS and function(c)[0] != "transfer_to_human_agents"
            ]
            if writes and not user_after_description:
                flags.append(
                    {
                        "event_id": f"m{i:03d}",
                        "tools": writes,
                        "kind": "no_prior_assistant_description_user_reply_pair",
                    }
                )
    return flags
