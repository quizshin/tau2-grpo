"""Shared prefix-only SFT evidence checks; findings never supply acceptance."""

import hashlib
import json
from collections import Counter

WRITES = {
    "book_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "cancel_reservation",
    "send_certificate",
}



def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()



def function(call):
    fn = call.get("function", call)
    args = fn.get("arguments", {})
    return fn.get("name"), json.loads(args) if isinstance(args, str) else args



def flight_keys(flights):
    return [(f.get("flight_number"), f.get("date")) for f in flights]



def audit_record(row):
    messages, meta = row["messages"], row["metadata"]
    approved = row.get("supervision", {}).get("message_indices", [])
    findings, checks, calls, pending = [], [], [], []
    users, reservations, seen_ids = {}, {}, set()
    prefix = ""
    tool_counts = Counter()
    first_tool, multi = None, 0

    def flag(kind, event, status, **evidence):
        findings.append(dict(kind=kind, event_id=f"m{event:03d}", status=status, **evidence))

    if len(set(approved)) != len(approved) or any(
        not isinstance(i, int)
        or i < 0
        or i >= len(messages)
        or messages[i].get("role") != "assistant"
        for i in approved
    ):
        flag("invalid_supervision_indices", 0, "contradiction")

    for i, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content") or ""
        if role != "tool" and pending:
            flag("missing_tool_receipt", i, "contradiction", pending=[x["name"] for x in pending])
            pending = []
        if role == "tool":
            if not pending:
                flag("orphan_tool_receipt", i, "contradiction")
                continue
            request = pending.pop(0)
            if (message.get("name") and message["name"] != request["name"]) or (
                request["id"] and message.get("tool_call_id") != request["id"]
            ):
                flag("tool_receipt_mismatch", i, "contradiction", expected=request["name"])
                continue
            request["response_sha256"] = hashlib.sha256(content.encode()).hexdigest()
            request["response_event"] = f"m{i:03d}"
            try:
                payload = json.loads(content)
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict):
                if payload.get("error") or payload.get("is_error"):
                    flag("tool_error_response", i, "review", tool=request["name"])
                elif request["name"] == "get_user_details" and payload.get("user_id"):
                    users[payload["user_id"]] = (payload, i)
                if (
                    not payload.get("error")
                    and payload.get("reservation_id")
                    and "cabin" in payload
                ):
                    reservations[payload["reservation_id"]] = (payload, i)
            prefix += "\n" + content
        elif role == "user":
            prefix += "\n" + content
        elif role == "assistant":
            batch = message.get("tool_calls") or []
            multi += len(batch) > 1
            for position, call in enumerate(batch):
                try:
                    name, args = function(call)
                    if not isinstance(args, dict):
                        raise ValueError("non-object arguments")
                except (ValueError, TypeError):
                    flag("invalid_arguments", i, "contradiction", batch_position=position)
                    continue
                first_tool = first_tool or name
                tool_counts[name] += 1
                call_id = call.get("id")
                if call_id and call_id in seen_ids:
                    flag("duplicate_tool_call_id", i, "contradiction", call_id=call_id)
                if call_id:
                    seen_ids.add(call_id)
                record = dict(
                    event_id=f"m{i:03d}",
                    batch_position=position,
                    name=name,
                    arguments=args,
                    id=call_id,
                    supervised=i in approved,
                )
                calls.append(record)
                pending.append(record)
                if name not in WRITES:
                    continue
                # Text occurrence is only a screening check, never proof of entity binding.
                for key in ("user_id", "reservation_id", "payment_id"):
                    value = args.get(key)
                    if value and str(value) not in prefix:
                        flag(
                            "write_identifier_absent_from_prior_user_or_tool",
                            i,
                            "unknown",
                            tool=name,
                            key=key,
                            value=value,
                            batch_position=position,
                        )
                state, state_i = reservations.get(args.get("reservation_id"), ({}, None))
                if name == "update_reservation_flights" and state:
                    changed = flight_keys(state.get("flights", [])) != flight_keys(
                        args.get("flights", [])
                    )
                    if state.get("cabin") == "basic_economy" and changed:
                        flag(
                            "basic_economy_flight_change",
                            i,
                            "contradiction" if args.get("cabin") == "basic_economy" else "review",
                            prior_event=f"m{state_i:03d}",
                            prior_cabin=state["cabin"],
                            requested_cabin=args.get("cabin"),
                            tool=name,
                            prior_flights=flight_keys(state.get("flights", [])),
                            requested_flights=flight_keys(args.get("flights", [])),
                        )
                if name == "update_reservation_passengers" and state and "passengers" in args:
                    if len(state.get("passengers", [])) != len(args["passengers"]):
                        flag(
                            "passenger_count_change",
                            i,
                            "contradiction",
                            prior_event=f"m{state_i:03d}",
                        )
                if name == "update_reservation_baggages" and state:
                    before, after = state.get("total_baggages"), args.get("total_baggages")
                    if type(before) is int and type(after) is int and after < before:
                        flag("baggage_removal", i, "contradiction", before=before, after=after)
                if name not in {"book_reservation", "update_reservation_flights"}:
                    continue
                uid = args.get("user_id") if name == "book_reservation" else state.get("user_id")
                user, user_i = users.get(uid, ({}, None))
                payment_ids = (
                    [p.get("payment_id") for p in args.get("payment_methods", [])]
                    if name == "book_reservation"
                    else [args.get("payment_id")]
                )
                for payment in payment_ids:
                    methods = user.get("payment_methods", {})
                    status = (
                        "unknown"
                        if not user or not payment
                        else "satisfied"
                        if payment in methods
                        else "contradiction"
                    )
                    checks.append(
                        dict(
                            kind="payment_profile_membership",
                            event_id=f"m{i:03d}",
                            status=status,
                            payment_id=payment,
                            profile_event=None if user_i is None else f"m{user_i:03d}",
                        )
                    )
                    if status != "satisfied":
                        flag("payment_profile_membership", i, status, payment_id=payment)
    if pending:
        flag("unfinished_tool_calls", len(messages) - 1, "contradiction")
    executions = meta.get("executions", [])
    execution_match = None
    if executions:
        actual = [{k: c.get(k) for k in ("name", "arguments", "response_sha256")} for c in calls]
        execution_match = actual == executions
        if not execution_match:
            flag("stored_execution_receipt_mismatch", 0, "contradiction")
    mapping = bool(
        meta.get("source_task_id") and meta.get("source_task_hash") and meta.get("source_db_hash")
    )
    return {
        "source_id": meta["source_dialog_id"],
        "messages_sha256": digest(messages),
        "historical_difficulty": meta.get("difficulty"),
        "difficulty_status": "legacy_trajectory_judge_not_calibrated_intrinsic_difficulty",
        "source_task_id": meta.get("source_task_id"),
        "exact_task_db_mapping_present": mapping,
        "evidence_tier": (
            "stored_execution_receipts_match_not_fresh_replay"
            if execution_match
            else "source_and_old_judge_only"
        ),
        "execution_receipts_match": execution_match,
        "trajectory_complexity": {
            "assistant_messages": sum(m["role"] == "assistant" for m in messages),
            "tool_calls": sum(tool_counts.values()),
            "multicall_batches": multi,
            "label_tokens_recorded": meta.get("label_tokens"),
        },
        "first_tool": first_tool,
        "tool_counts": dict(tool_counts),
        "supervised_call_count": sum(c["supervised"] for c in calls),
        "findings": findings,
        "checks": checks,
        "semantic_reviews_required": [
            "explicit_confirmation_and_scope",
            "policy_eligibility",
            "price_refund_and_claim_grounding",
            "task_completion",
        ],
    }
