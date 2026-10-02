"""Evidence-reviewed outcome recipes; exact rules, not new semantic approval."""

import copy
from datetime import datetime


def duplicate_cancellation_allowed(reservation, db):
    """Change-of-plans cancellation at the pinned benchmark policy time."""
    now = datetime.fromisoformat("2024-05-15T15:00:00")
    for segment in reservation["flights"]:
        flight = db["flights"][segment["flight_number"]]
        state = flight["dates"].get(segment["date"], {})
        if segment["date"] < "2024-05-15" or state.get("status") in ("flying", "landed"):
            return False
    age = (now - datetime.fromisoformat(reservation["created_at"])).total_seconds()
    cancelled = any(
        db["flights"][s["flight_number"]]["dates"][s["date"]]["status"] == "cancelled"
        for s in reservation["flights"]
    )
    return reservation["cabin"] == "business" or 0 <= age <= 86400 or cancelled



def variants(record, db):
    original = copy.deepcopy(record.evaluation_criteria.get("actions") or [])
    result = [("original_reference", original)]
    if record.id == "airline_162":
        # Original scope: keep exactly one ORD->ATL on May25; all other trips unchanged.
        ins = record.user_scenario["instructions"]
        if "keep exactly one ORD" not in ins["task_instructions"]:
            raise ValueError("Duplicate acceptance source wording changed")
        ids = ["90A522", "203327", "6DC4D8"]
        for rid in ids:
            r = db["reservations"][rid]
            if (
                (r["origin"], r["destination"]) != ("ORD", "ATL")
                or len(r["flights"]) != 1
                or r["flights"][0]["date"] != "2024-05-25"
            ):
                raise ValueError("Duplicate acceptance DB scope changed")
            if not duplicate_cancellation_allowed(r, db):
                raise ValueError("Duplicate cancellation policy needs separate review")
        for keep in ids:
            result.append(
                (
                    "keep_" + keep,
                    [
                        {"name": "cancel_reservation", "arguments": {"reservation_id": rid}}
                        for rid in ids
                        if rid != keep
                    ],
                )
            )
    if record.id == "airline_923":
        ins = record.user_scenario["instructions"]["task_instructions"]
        if "available later date in May" not in ins:
            raise ValueError("Return-date acceptance wording changed")
        rid = "044B85"
        r = db["reservations"][rid]
        user = db["users"][r["user_id"]]
        if (
            r["flight_type"] != "round_trip"
            or r["cabin"] == "basic_economy"
            or len(r["flights"]) != 2
        ):
            raise ValueError("Return-date variant assumptions changed")
        out, old_return = r["flights"]
        cards = [k for k, v in user["payment_methods"].items() if v["source"] == "credit_card"]
        for number, f in sorted(db["flights"].items()):
            if (f["origin"], f["destination"]) != (old_return["origin"], old_return["destination"]):
                continue
            for date, state in sorted(f["dates"].items()):
                if not old_return["date"] < date <= "2024-05-31" or state["status"] != "available":
                    continue
                if state["available_seats"][r["cabin"]] < len(r["passengers"]):
                    continue
                for card in cards:
                    result.append(
                        (
                            f"return_{date}_{number}_{card}",
                            [
                                {
                                    "name": "update_reservation_flights",
                                    "arguments": {
                                        "reservation_id": rid,
                                        "cabin": r["cabin"],
                                        "flights": [
                                            {
                                                "flight_number": out["flight_number"],
                                                "date": out["date"],
                                            },
                                            {"flight_number": number, "date": date},
                                        ],
                                        "payment_id": card,
                                    },
                                }
                            ],
                        )
                    )
        if len(result) == 1:
            raise ValueError("No feasible later-date variants")
    if record.id in ("airline_728", "airline_458"):
        rid = "2F8EEF" if record.id == "airline_728" else "84263E"
        r = db["reservations"][rid]
        old = r["flights"][0]
        flight = db["flights"][old["flight_number"]]
        threshold = flight["scheduled_departure_time_est"]
        if record.id == "airline_458":
            # Arrival and onward departure are both Boston local time.
            retained = db["reservations"]["D76B65"]["flights"][0]
            arrival = db["flights"][retained["flight_number"]]["scheduled_arrival_time_est"]
            threshold = max(threshold, arrival)
        args = next(a["arguments"] for a in original if a["name"] == "update_reservation_flights")
        for fid, f in sorted(db["flights"].items()):
            day = f["dates"].get(old["date"])
            if (f["origin"], f["destination"]) != (old["origin"], old["destination"]):
                continue
            if f["scheduled_departure_time_est"] <= threshold:
                continue
            if (
                not day
                or day["status"] != "available"
                or day["available_seats"][r["cabin"]] < len(r["passengers"])
            ):
                continue
            changed = copy.deepcopy(args)
            changed["flights"] = [dict(flight_number=fid, date=old["date"])] + [
                dict(flight_number=f["flight_number"], date=f["date"]) for f in r["flights"][1:]
            ]
            result.append(
                ("later_" + fid, [dict(name="update_reservation_flights", arguments=changed)])
            )
    # Legal flight/cabin changes can produce different intermediate ledgers.
    # Enumerate both permitted two-step orders instead of masking payment history.
    expanded = []
    for label, recipe in result:
        positions = [i for i, a in enumerate(recipe) if a["name"] == "update_reservation_flights"]
        if len(positions) != 1:
            continue
        i = positions[0]
        a = recipe[i]
        args = a["arguments"]
        r = db["reservations"][args["reservation_id"]]
        old = [dict(flight_number=f["flight_number"], date=f["date"]) for f in r["flights"]]
        if r["cabin"] == args["cabin"] or old == args["flights"] or r["cabin"] == "basic_economy":
            continue
        old_keys = {(f["flight_number"], f["date"]) for f in old}
        can_move_first = all(
            (f["flight_number"], f["date"]) in old_keys
            or (
                db["flights"][f["flight_number"]]["dates"][f["date"]]["status"] == "available"
                and db["flights"][f["flight_number"]]["dates"][f["date"]]["available_seats"][
                    r["cabin"]
                ]
                >= len(r["passengers"])
            )
            for f in args["flights"]
        )
        if can_move_first:
            first = copy.deepcopy(a)
            first["arguments"]["cabin"] = r["cabin"]
            expanded.append(
                (label + "_flight_then_cabin", recipe[:i] + [first, a] + recipe[i + 1 :])
            )
        can_cabin_first = all(
            db["flights"][f["flight_number"]]["dates"][f["date"]]["available_seats"][args["cabin"]]
            >= len(r["passengers"])
            for f in old
        )
        if args["cabin"] != "basic_economy" and can_cabin_first:
            first = copy.deepcopy(a)
            first["arguments"]["flights"] = old
            expanded.append(
                (label + "_cabin_then_flight", recipe[:i] + [first, a] + recipe[i + 1 :])
            )
    result += expanded
    return result



def repair_known_reference_actions(record, db):
    """Narrow evidence-reviewed recipe repairs; never an approval of task semantics.

    Keep the original task and database identities. A new bundle explicitly uses
    these recipes instead of accepting the policy-invalid original state. Payment
    histories remain exact, so a legal two-step path is represented by execution,
    not by erasing its transactions from the scorer.
    """
    actions = copy.deepcopy(record.evaluation_criteria.get("actions") or [])
    changes = []
    if record.id in ("airline_698", "airline_921"):
        positions = [i for i, a in enumerate(actions) if a["name"] == "update_reservation_flights"]
        if len(positions) != 1:
            raise ValueError("Reviewed upgrade/change recipe changed")
        i = positions[0]
        action = actions[i]
        args = action["arguments"]
        r = db["reservations"][args["reservation_id"]]
        old = [dict(flight_number=f["flight_number"], date=f["date"]) for f in r["flights"]]
        if r["cabin"] != "basic_economy" or args["cabin"] != "economy" or old == args["flights"]:
            raise ValueError("Reviewed basic-economy repair no longer applies")
        upgrade = copy.deepcopy(action)
        upgrade["arguments"]["flights"] = old
        upgrade["action_id"] = "quality_repair_separate_cabin_upgrade"
        actions.insert(i, upgrade)
        changes.append("execute_cabin_only_upgrade_before_changing_flights")
    elif record.id in ("airline_1143", "airline_925"):
        r = db["reservations"]["GCZ58I"]
        n = len(r["passengers"])
        cancelled = [
            f
            for f in r["flights"]
            if db["flights"][f["flight_number"]]["dates"][f["date"]]["status"] == "cancelled"
        ]
        if (
            n != 2
            or not cancelled
            or any(
                a["name"] in ("cancel_reservation", "update_reservation_flights") for a in actions
            )
        ):
            raise ValueError("Reviewed compensation scope changed")
        certificates = [a for a in actions if a["name"] == "send_certificate"]
        if len(certificates) != 1 or certificates[0]["arguments"]["amount"] != 300:
            raise ValueError("Reviewed compensation amount changed")
        certificates[0]["arguments"]["amount"] = 100 * n
        changes.append(
            "cancelled_flight_compensation_200_no_delayed_extra_without_change_or_cancel"
        )
        if record.id == "airline_925":
            status_calls = [
                a
                for a in actions
                if a["name"] == "get_flight_status" and a["arguments"]["flight_number"] == "HAT273"
            ]
            if len(status_calls) != 1 or status_calls[0]["arguments"]["date"] != "2024-05-15":
                raise ValueError("Reviewed incorrect status date changed")
            status_calls[0]["arguments"]["date"] = "2024-05-11"
            changes.append("check_status_on_reserved_segment_date")
    elif record.id == "airline_976":
        certs = [(i, a) for i, a in enumerate(actions) if a["name"] == "send_certificate"]
        writes = [i for i, a in enumerate(actions) if a["name"] == "update_reservation_flights"]
        if len(certs) != 1 or len(writes) != 1 or certs[0][0] >= writes[0]:
            raise ValueError("Reviewed delayed compensation ordering changed")
        certificate = actions.pop(certs[0][0])
        actions.append(certificate)
        changes.append("issue_delayed_compensation_only_after_successful_flight_change")
    else:
        raise ValueError("No adjudicated reference repair for " + record.id)
    return actions, changes
