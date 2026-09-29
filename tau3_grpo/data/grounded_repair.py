"""Entity-disjoint capability train/dev candidates using real training DBs.

Families are locked before dialogue generation. Source selection uses only train,
old dev and exposed selection protections. These are procedural development probes,
not an unseen distribution, independent templates, or final benchmark tasks.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.analysis.capability_distribution import task_feature
from tau3_grpo.data.grounded_gap_pilot import Builder, build_case, feasible, raw_database, read_rows
from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.data.sft_expansion import audit_tool_calls
from tau3_grpo.data.sft_policy_checks import ALLOWANCE
from tau3_grpo.data.staged_sft import ordered_tool_receipts
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.paths import AREAL_DB_ROOT
from tau3_grpo.utils.hashing import sha256_file, sha256_json

KINDS = (
    "baggage_free",
    "baggage_paid",
    "cabin_basic_upgrade",
    "cabin_economy_upgrade",
    "cancel_allowed",
    "cancel_denied",
    "compensation_1",
    "passenger_correction",
    "flight_change",
    "flight_refusal",
    "airport_lookup",
    "connection_lookup",
    "book_direct",
    "transfer_flown",
)
BASE_KINDS = set(KINDS[:7])


def build_curriculum_read(entry, rid, kind):
    """Foundation goals with read-only native evidence and a no-change contract."""
    adapted = adapt_record(ArealTaskRecord.model_validate(entry.task))
    if adapted.db_file_hash != entry.db_hash:
        raise ValueError("Source DB changed")
    b = Builder(adapted, entry, rid, kind)
    b.expected = copy.deepcopy(b.before)
    r = b.before["reservations"][rid]
    uid = r["user_id"]
    _ = b.before["users"][uid]
    goals = {
        "allowance_quote": "Tell me my free checked-bag allowance for this booking. Do not change any bags.",
        "bag_removal_refusal": "Can you remove one checked bag? If the policy forbids removal, explain and leave it unchanged; I do not want a transfer.",
        "passenger_count_refusal": "Can you add another passenger to this existing reservation? Do not create a new booking; explain if the passenger count cannot be changed.",
        "certificate_update_refusal": "Can I pay for adding a checked bag using a travel certificate? Do not add any bag or charge anything if certificates are not permitted for this update.",
        "insurance_update_refusal": "Can I add travel insurance to this existing booking now? If insurance can only be purchased during booking, leave it unchanged.",
        "reservation_summary": "Please list the existing flight numbers, dates, cabin and passenger count. I only need the recorded details, not a change.",
        "payment_inventory": "Which stored non-certificate payment methods can I choose for a future reservation update? List their IDs and any gift-card balances. Do not process an update.",
        "passenger_identity": "Please read back all the passenger names and dates of birth on this reservation. Do not replace them with saved profile passengers.",
    }
    if kind not in goals:
        raise ValueError(kind)
    b.say("user", f"My user ID is {uid}. For reservation {rid}, {goals[kind]}")
    observed = b.tool("get_reservation_details", {"reservation_id": rid})
    if kind in ("allowance_quote", "payment_inventory"):
        profile = b.tool("get_user_details", {"user_id": uid})
    if kind == "allowance_quote":
        n = len(observed["passengers"])
        per = ALLOWANCE[profile["membership"]][observed["cabin"]]
        b.say(
            "assistant",
            f"Your {profile['membership']} membership and {observed['cabin']} cabin allow {per} free checked bags per passenger. For {n} passengers that is {per * n} free bags. Currently {observed['total_baggages']} bags are recorded. I made no changes.",
        )
    elif kind == "bag_removal_refusal":
        if not observed["total_baggages"]:
            raise ValueError("No recorded bag to remove")
        b.say(
            "assistant",
            f"The policy allows adding checked bags but does not allow removing them. The reservation remains at {observed['total_baggages']} bags; no update or charge was made.",
        )
    elif kind == "passenger_count_refusal":
        b.say(
            "assistant",
            f"The number of passengers on an existing reservation cannot be changed. Your reservation remains at {len(observed['passengers'])} passengers. I have not created a new booking.",
        )
    elif kind == "certificate_update_refusal":
        b.say(
            "assistant",
            "Travel certificates cannot pay for reservation updates, including adding checked bags. A stored credit card or gift card can be used subject to its available balance. I have not added bags or charged a payment method.",
        )
    elif kind == "insurance_update_refusal":
        if observed["insurance"] != "no":
            raise ValueError("Already insured")
        b.say(
            "assistant",
            "Travel insurance is purchased during booking. I cannot add it to this existing reservation. Its insurance remains no, and I made no changes.",
        )
    elif kind == "reservation_summary":
        fs = "; ".join(
            f"{f['flight_number']} on {f['date']}, {f['origin']} to {f['destination']}"
            for f in observed["flights"]
        )
        b.say(
            "assistant",
            f"Reservation {rid}: {fs}. Cabin: {observed['cabin']}; passengers: {len(observed['passengers'])}. These are the recorded details. No change was made.",
        )
    elif kind == "payment_inventory":
        methods = [
            dict(
                payment_id=k,
                source=v["source"],
                **(
                    {"balance": v["amount"]}
                    if v["source"] == "gift_card"
                    else {"brand": v["brand"], "last_four": v["last_four"]}
                ),
            )
            for k, v in profile["payment_methods"].items()
            if v["source"] in ("gift_card", "credit_card")
        ]
        b.say(
            "assistant",
            f"Your stored non-certificate payment methods are {json.dumps(methods)}. A gift card must have enough balance for a charge. No payment or reservation update was made.",
        )
    else:
        b.say(
            "assistant",
            f"The booked passengers are {json.dumps(observed['passengers'])}. These are the reservation passengers; I made no changes.",
        )
    b.say("user", "That answers my question. Please leave the booking unchanged.")
    b.say("assistant", "The booking is unchanged. No further action was taken.")
    return b.finish(
        goals[kind],
        [
            "read_only_contract_full_DB_equal",
            "all_facts_from_prior_native_receipts",
            "explicit_user_stop_respected",
        ],
    )


def curriculum_pool():
    """Only source-task-linked training users; never inspect final/reserve tasks."""
    protections = json.loads(Path("data/sft/decision_repair_20260926_v1/plan.json").read_text())
    protected = set(protections["protected_users"]) | set(protections["validation_users"])
    protected.update(
        json.loads(
            Path(
                "data/sft/repaired_source_grounded_20260926_v1/protected_entities.json"
            ).read_text()
        )["users"]
    )
    historical = user_ids(read_rows("data/sft/staged_v2_A100_20260925/train.jsonl")) | user_ids(
        read_rows("data/sft/decision_repair_20260926_v1/train.jsonl")
    )
    # Other materialized candidates also exclude their users from "unseen" dev.
    historical.update(
        user_ids(read_rows("data/sft/repaired_source_grounded_20260926_v1/train.jsonl"))
    )
    # Exclude active formal RL reference entities from the development split.
    for e in read_manifest(
        "data/manifests/rl_curriculum50_20260912/areal_airline_train_seed42.jsonl"
    ):
        raw = raw_database(
            str(ArealTaskRecord.model_validate(e.task).resolve_db_path(AREAL_DB_ROOT))
        )
        historical.update(
            raw["reservations"][rid]["user_id"]
            for rid in task_feature(e.task)["reservation_ids"]
            if rid in raw["reservations"]
        )
        historical.update(
            a["arguments"]["user_id"]
            for a in e.task["evaluation_criteria"].get("actions", [])
            if a.get("arguments", {}).get("user_id")
        )
    entries = {
        e.task_id: e for e in read_manifest("data/manifests/areal_airline_train_seed42.jsonl")
    }
    pool = {}
    dbs = {}
    for entry in entries.values():
        if entry.task.get("initial_state"):
            continue
        db = raw_database(
            str(ArealTaskRecord.model_validate(entry.task).resolve_db_path(AREAL_DB_ROOT))
        )
        relevant = {
            db["reservations"][rid]["user_id"]
            for rid in task_feature(entry.task)["reservation_ids"]
            if rid in db["reservations"]
        } - protected
        for rid, r in db["reservations"].items():
            if (
                r["user_id"] not in relevant
                or r.get("status") == "cancelled"
                or not r["passengers"]
            ):
                continue
            key = (r["user_id"], rid)
            if key in pool:
                continue
            pool[key] = dict(
                task_id=entry.task_id, rid=rid, user_id=r["user_id"], db_hash=entry.db_hash
            )
            dbs[entry.task_id] = db
    return list(pool.values()), entries, dbs, protected, historical


def curriculum_signature(db, rid, kind):
    """Meaningful parameters, excluding names, IDs and decorative wording."""
    r = db["reservations"][rid]
    u = db["users"][r["user_id"]]
    if kind in SEARCH_KINDS | BOOK_KINDS:
        o = curriculum_search_option(db, rid, kind)
        return (
            kind,
            o["origin"],
            o["destination"],
            o["date"],
            o["return_date"],
            o["preferred_date"],
            o["cabin"],
            o["window"],
            o["passengers"],
            o["total_bags"],
            o["insurance"],
            o["gift_amount"],
            o["total"],
        )
    if kind == "allowance_quote":
        return (kind, u["membership"], r["cabin"], len(r["passengers"]), r["total_baggages"])
    if kind == "bag_removal_refusal":
        return (kind, r["cabin"], len(r["passengers"]), r["total_baggages"], r["nonfree_baggages"])
    if kind == "passenger_count_refusal":
        return (kind, len(r["passengers"]), len(r["flights"]), r["flight_type"])
    if kind == "certificate_update_refusal":
        return (kind, r["cabin"], u["membership"], len(r["passengers"]), r["total_baggages"])
    if kind == "insurance_update_refusal":
        return (kind, r["cabin"], len(r["passengers"]), len(r["flights"]))
    if kind == "payment_inventory":
        return (
            kind,
            tuple(
                sorted(
                    (v["source"], v.get("amount"), v.get("brand"))
                    for v in u["payment_methods"].values()
                )
            ),
        )
    if kind == "passenger_identity":
        return (
            kind,
            len(r["passengers"]),
            sum(
                p["first_name"] == u["name"]["first_name"]
                and p["last_name"] == u["name"]["last_name"]
                for p in r["passengers"]
            ),
            len(u.get("saved_passengers", [])),
        )
    if kind == "reservation_summary":
        return (
            kind,
            r["flight_type"],
            r["cabin"],
            len(r["passengers"]),
            tuple((f["origin"], f["destination"], f["date"]) for f in r["flights"]),
        )
    return (
        kind,
        u["membership"],
        r["cabin"],
        len(r["passengers"]),
        len(r["flights"]),
        r["insurance"],
        r["total_baggages"],
        r["nonfree_baggages"],
        tuple(f["price"] for f in r["flights"]),
        tuple(sorted(v["source"] for v in u["payment_methods"].values())),
    )


def curriculum_strategy_option(db, rid, kind):
    r = db["reservations"][rid]
    u = db["users"][r["user_id"]]
    n = len(r["passengers"])
    if (
        not n
        or r.get("status") == "cancelled"
        or any(f["date"] <= "2024-05-15" for f in r["flights"])
    ):
        return None
    states = [
        db["flights"].get(f["flight_number"], {}).get("dates", {}).get(f["date"], {})
        for f in r["flights"]
    ]
    if any(s.get("status") != "available" for s in states):
        return None
    cards = sorted(k for k, v in u["payment_methods"].items() if v["source"] == "credit_card")
    gifts = sorted(k for k, v in u["payment_methods"].items() if v["source"] == "gift_card")
    if not cards:
        return None
    oldfree = ALLOWANCE[u["membership"]][r["cabin"]] * n
    if r["nonfree_baggages"] != max(0, r["total_baggages"] - oldfree):
        return None
    if kind == "cancel_refund_rebook":
        if r["cabin"] != "business" or len(r["flights"]) != 1 or len(r["passengers"]) > 5:
            return None
        if states[0]["available_seats"]["economy"] < n:
            return None
        if not gifts:
            return None
        refund = sum(x["amount"] for x in r["payment_history"] if x["payment_id"] == gifts[0])
        if refund <= 0:
            return None
        price = states[0]["prices"]["economy"] * n
        return dict(
            target="economy",
            new_prices=[states[0]["prices"]["economy"]],
            delta=price,
            card=cards[-1],
            gift=gifts[0],
            replacement=None,
        )
    target = "economy" if r["cabin"] == "basic_economy" else "business"
    replacement = None
    if kind == "downgrade_gift_bags":
        if r["cabin"] != "business" or not gifts:
            return None
        target = "economy"
    elif kind == "partial_change_bags":
        if r["cabin"] == "basic_economy":
            return None
        target = r["cabin"]
        old = r["flights"][0]
        options = []
        for fid, f in db["flights"].items():
            s = f["dates"].get(old["date"], {})
            if fid == old["flight_number"] or (f["origin"], f["destination"]) != (
                old["origin"],
                old["destination"],
            ):
                continue
            if (
                s.get("status") == "available"
                and s["available_seats"][target] >= n
                and s["prices"][target] > old["price"]
            ):
                options.append((s["prices"][target], fid))
        if not options:
            return None
        price, fid = min(options)
        replacement = dict(
            flight_number=fid,
            date=old["date"],
            origin=old["origin"],
            destination=old["destination"],
            price=price,
        )
    elif kind == "no_inventory_cabin_bags":
        if r["cabin"] != "basic_economy" or not any(
            s["available_seats"]["business"] < n for s in states
        ):
            return None
        target = "economy"
    elif r["cabin"] == "business":
        return None
    if replacement:
        prices = [replacement["price"]] + [f["price"] for f in r["flights"][1:]]
    else:
        if any(s["available_seats"][target] < n for s in states):
            return None
        prices = [s["prices"][target] for s in states]
    delta = (sum(prices) - sum(f["price"] for f in r["flights"])) * n
    if kind == "downgrade_gift_bags" and delta >= 0:
        return None
    if kind not in ("downgrade_gift_bags", "partial_change_bags") and delta <= 0:
        return None
    gift = None
    if kind == "gift_limit_upgrade_bags":
        options = [k for k in gifts if u["payment_methods"][k]["amount"] < delta]
        if not options:
            return None
        gift = options[0]
    if kind == "downgrade_gift_bags":
        gift = gifts[0]
    newfree = ALLOWANCE[u["membership"]][target] * n
    total = max(r["total_baggages"] + 1, newfree + r["nonfree_baggages"] + 1)
    paid = max(0, total - newfree)
    cost = 50 * (paid - r["nonfree_baggages"])
    if (
        gift
        and kind == "downgrade_gift_bags"
        and u["payment_methods"][gift]["amount"] - delta < cost
    ):
        return None
    return dict(
        target=target,
        new_prices=prices,
        delta=delta,
        card=cards[-1],
        gift=gift,
        replacement=replacement,
        total_bags=total,
        paid_bags=paid,
        bag_cost=cost,
    )


def build_curriculum_strategy(entry, rid, kind):
    """Causal two-write goals; never intentionally call a known-failing tool."""
    adapted = adapt_record(ArealTaskRecord.model_validate(entry.task))
    if adapted.db_file_hash != entry.db_hash:
        raise ValueError("DB identity changed")
    b = Builder(adapted, entry, rid, kind)
    db = b.before
    r = db["reservations"][rid]
    uid = r["user_id"]
    _ = db["users"][uid]
    option = curriculum_strategy_option(db, rid, kind)
    if not option:
        raise ValueError("Strategy eligibility changed")
    b.expected = copy.deepcopy(db)
    er = b.expected["reservations"][rid]
    eu = b.expected["users"][uid]
    n = len(r["passengers"])
    target = option["target"]
    card = option["card"]
    gift = option["gift"]
    if kind == "cancel_refund_rebook":
        flight = r["flights"][0]
        fid = flight["flight_number"]
        date = flight["date"]
        price = option["delta"]
        b.say(
            "user",
            f"My user ID is {uid}. I want to replace business reservation {rid} with a NEW one-way economy booking on {fid}, {date}, for the same {n} passengers, with no checked bags. Please get rid of the old booking so I do not keep both. I would prefer to use {gift} for the new fare. What can you do?",
        )
        observed = b.tool("get_reservation_details", {"reservation_id": rid})
        profile = b.tool("get_user_details", {"user_id": uid})
        b.tool("get_flight_status", {"flight_number": fid, "date": date})
        refund = sum(p["amount"] for p in observed["payment_history"])
        b.say(
            "assistant",
            f"This unflown business booking can be cancelled. I would cancel the entire {rid} and reverse ${refund} to its original payment destinations: {json.dumps(observed['payment_history'])}. May I cancel it?",
        )
        b.say(
            "user",
            f"Yes, cancel {rid} and refund the original methods. Then check my actual updated gift-card balance before booking.",
        )
        for payment in observed["payment_history"]:
            er["payment_history"].append(
                dict(payment_id=payment["payment_id"], amount=-payment["amount"])
            )
        # Native cancellation records reversal entries but does not immediately
        # credit profile gift balances. The next decision must use a fresh read.
        er["status"] = "cancelled"
        b.tool("cancel_reservation", {"reservation_id": rid})
        updated = b.tool("get_user_details", {"user_id": uid})
        options = b.tool(
            "search_direct_flight",
            {"origin": flight["origin"], "destination": flight["destination"], "date": date},
        )
        chosen = next(x for x in options if x["flight_number"] == fid)
        if chosen["prices"]["economy"] * n != price or chosen["available_seats"]["economy"] < n:
            raise ValueError("New booking quote changed")
        b.say(
            "assistant",
            f"The cancellation recorded refund entries. The freshly read {gift} balance is ${updated['payment_methods'][gift]['amount']}; I will not assume an uncredited refund is spendable. Economy costs ${price} for {n} passengers. Would you like insurance at $30 per passenger?",
        )
        b.say(
            "user",
            "No insurance. Use the gift card if it covers the full fare; otherwise ask me to approve a credit card.",
        )
        method = gift
        if updated["payment_methods"][gift]["amount"] < price:
            b.say(
                "assistant",
                f"The gift card cannot cover ${price}. May I use stored credit card {card} for the full fare instead?",
            )
            b.say("user", f"Yes, use {card} instead.")
            method = card
        b.say(
            "assistant",
            f"Please confirm a new one-way economy booking: {fid} on {date}, {flight['origin']} to {flight['destination']}, passengers {json.dumps(observed['passengers'])}, zero checked bags and no insurance, total ${price} using {method}. Shall I book it?",
        )
        b.say("user", "Yes, book exactly those confirmed details at that price.")
        receipt = b.tool(
            "book_reservation",
            dict(
                user_id=uid,
                origin=flight["origin"],
                destination=flight["destination"],
                flight_type="one_way",
                cabin="economy",
                flights=[dict(flight_number=fid, date=date)],
                passengers=observed["passengers"],
                payment_methods=[dict(payment_id=method, amount=price)],
                total_baggages=0,
                nonfree_baggages=0,
                insurance="no",
            ),
        )
        newid = receipt["reservation_id"]
        if newid in db["reservations"]:
            raise ValueError("New booking overwrote existing ID")
        b.expected["reservations"][newid] = dict(
            reservation_id=newid,
            user_id=uid,
            origin=flight["origin"],
            destination=flight["destination"],
            flight_type="one_way",
            cabin="economy",
            flights=[
                dict(
                    flight_number=fid,
                    date=date,
                    origin=flight["origin"],
                    destination=flight["destination"],
                    price=chosen["prices"]["economy"],
                )
            ],
            passengers=copy.deepcopy(observed["passengers"]),
            payment_history=[dict(payment_id=method, amount=price)],
            created_at="2024-05-15T15:00:00",
            total_baggages=0,
            nonfree_baggages=0,
            insurance="no",
            status=None,
        )
        eu["reservations"].append(newid)
        if method == gift:
            eu["payment_methods"][gift]["amount"] -= price
        b.expected["flights"][fid]["dates"][date]["available_seats"]["economy"] -= n
        b.say(
            "assistant",
            f"Old reservation {rid} is cancelled. New reservation {newid} is economy on {fid}, {date}, for the confirmed passengers, with zero bags and no insurance. The new fare ${price} used {method}.",
        )
    else:
        delta = option["delta"]
        replacement = option["replacement"]
        total = option["total_bags"]
        initial = (
            f"I need flight {replacement['flight_number']} on {replacement['date']} instead of the first listed segment; the other segments and cabin should stay as booked."
            if replacement
            else f"I would like {target} on all my existing flights, keeping their numbers and dates."
        )
        if kind == "no_inventory_cabin_bags":
            initial = "I would like business on all my existing flights. Keeping these flights and all passengers together matters more to me than the cabin if business is not available."
        payment_request = (
            f"I would prefer to pay with {gift}."
            if kind == "gift_limit_upgrade_bags"
            else f"Please use {gift} for any fare refund and baggage charge."
            if gift
            else f"You can use my stored card {card}."
        )
        budget = abs(delta) + option["bag_cost"] + 20
        b.say(
            "user",
            f"My user ID is {uid}. For reservation {rid}: {initial} I also need {total} checked bags in total. {payment_request} I can spend at most ${budget} more today, without counting a refund as extra spending money. Keep my passengers and insurance unchanged. Please work out the options for me.",
        )
        observed = b.tool("get_reservation_details", {"reservation_id": rid})
        profile = b.tool("get_user_details", {"user_id": uid})
        for f in observed["flights"]:
            options = b.tool(
                "search_direct_flight", {k: f[k] for k in ("origin", "destination", "date")}
            )
            existing = next(x for x in options if x["flight_number"] == f["flight_number"])
            if kind == "no_inventory_cabin_bags" and existing["available_seats"]["business"] < n:
                b.say(
                    "assistant",
                    f"{f['flight_number']} has only {existing['available_seats']['business']} business seats for {n} passengers, so I will not attempt that unavailable cabin. I can quote economy for all segments.",
                )
            if replacement and f == observed["flights"][0]:
                new = next(x for x in options if x["flight_number"] == replacement["flight_number"])
                if new["available_seats"][target] < n:
                    raise ValueError("Replacement lacks seats")
        b.tool(
            "calculate",
            {
                "expression": f"({sum(option['new_prices'])}-{sum(f['price'] for f in observed['flights'])})*{n}"
            },
        )
        if target != observed["cabin"]:
            b.say(
                "assistant",
                f"The cabin changes your baggage allowance, so I will handle the cabin change before pricing the baggage update against the resulting reservation. The available whole-reservation cabin is {target}. I will check the actual updated state and the remaining budget before adding bags.",
            )
        else:
            b.say(
                "assistant",
                "I will handle the flight change first, then use its actual charge to determine how much of your budget remains for bags. The other flight segments, passengers and insurance will stay unchanged.",
            )
        payment = gift if kind == "downgrade_gift_bags" else card
        if kind == "gift_limit_upgrade_bags":
            b.say(
                "assistant",
                f"The first change costs ${delta}, but {gift} has only ${profile['payment_methods'][gift]['amount']}. I cannot use it for this charge. Would you like stored credit card {card} instead? No update has been attempted.",
            )
            b.say(
                "user",
                f"Yes, use {card} for both steps. Please quote the change for final confirmation.",
            )
        signed = f"charge ${delta}" if delta >= 0 else f"refund ${-delta}"
        flights = [
            dict(flight_number=f["flight_number"], date=f["date"]) for f in observed["flights"]
        ]
        if replacement:
            flights[0] = {k: replacement[k] for k in ("flight_number", "date")}
        b.say(
            "assistant",
            f"First step: {target} cabin, flights {json.dumps(flights)}, {n} passengers. The stored airfare difference is a {signed} using {payment}. Bags, passengers and insurance remain unchanged in this step. May I proceed?",
        )
        b.say(
            "user",
            f"Yes, make that exact first change with the stated {signed} and payment method.",
        )
        er["cabin"] = target
        if replacement:
            er["flights"][0] = copy.deepcopy(replacement)
        else:
            for f, price in zip(er["flights"], option["new_prices"]):
                f["price"] = price
        if delta:
            er["payment_history"].append(dict(payment_id=payment, amount=delta))
        if gift and payment == gift:
            eu["payment_methods"][gift]["amount"] -= delta
        updated = b.tool(
            "update_reservation_flights",
            dict(reservation_id=rid, cabin=target, flights=flights, payment_id=payment),
        )
        free = ALLOWANCE[profile["membership"]][updated["cabin"]] * len(updated["passengers"])
        paid = max(0, total - free)
        cost = 50 * max(0, paid - updated["nonfree_baggages"])
        remaining = budget - max(0, delta)
        if cost > remaining:
            raise ValueError("User combined budget exceeded")
        b.tool("calculate", {"expression": f"{budget}-{max(0, delta)}-{cost}"})
        balance_text = ""
        if payment == gift:
            fresh = b.tool("get_user_details", {"user_id": uid})
            balance = fresh["payment_methods"][gift]["amount"]
            if balance < cost:
                raise ValueError("Updated gift balance insufficient")
            balance_text = (
                f" The actual updated {gift} balance is ${balance}, sufficient for this charge."
            )
        b.say(
            "assistant",
            f"The first update succeeded. With {profile['membership']} membership and the resulting {updated['cabin']} cabin, the allowance is {free} bags for {n} passengers. The new total {total} means {paid} paid bags, compared with {updated['nonfree_baggages']} already paid; the incremental charge is ${cost}. ${remaining} of your positive-charge budget remains.{balance_text} May I add those bags using {payment}?",
        )
        b.say(
            "user",
            f"Yes, set {total} bags in total for the additional ${cost} using {payment}, retaining the first change.",
        )
        er.update(total_baggages=total, nonfree_baggages=paid)
        if cost:
            er["payment_history"].append(dict(payment_id=payment, amount=cost))
        if payment == gift:
            eu["payment_methods"][gift]["amount"] -= cost
        b.tool(
            "update_reservation_baggages",
            dict(
                reservation_id=rid, total_baggages=total, nonfree_baggages=paid, payment_id=payment
            ),
        )
        b.say(
            "assistant",
            f"Both changes succeeded: {rid} now has the confirmed {target} flights and {total} checked bags, including {paid} paid bags. The fare change was a {signed}; the added-bag charge was ${cost}. Total new charges were ${max(0, delta) + cost}, within your ${budget} limit; refunds are accounted for separately. Passengers and insurance are unchanged.",
        )
    row = b.finish(
        f"Execute dependent {kind} goal with state-aware second action.",
        [
            "native_observations_before_quotes",
            "two_explicit_write_confirmations",
            "second_action_uses_actual_updated_state",
            "full_independent_expected_DB_and_protected_state",
            "fresh_independent_native_replay",
            "no_known_invalid_write_for_artificial_failure",
        ],
    )
    row["metadata"]["strategy_branch"] = kind
    row["metadata"]["strategy_demonstration_scope"] = (
        "natural_goal_assistant_decomposition_with_observed_state_branch"
    )
    return row


def curriculum_valid_date(value):
    from datetime import date

    try:
        return (
            isinstance(value, str)
            and len(value) == 10
            and date.fromisoformat(value).isoformat() == value
        )
    except ValueError:
        return False


def audit_curriculum_dates(messages):
    """Native DB keys can have synthetic suffixes; they are not user dates."""
    import re

    problems = []

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "date" and item is not None and not curriculum_valid_date(item):
                    problems.append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for message in messages:
        for call in message.get("tool_calls", []):
            visit(call["function"]["arguments"])
        text = message.get("content", "") or ""
        if message.get("role") == "tool":
            try:
                visit(json.loads(text))
            except (ValueError, TypeError):
                pass
        problems.extend(
            x
            for x in re.findall(r"\d{4}-\d{2}-\d{2}(?:_[A-Za-z0-9]+)?", text)
            if not curriculum_valid_date(x)
        )
    return sorted(set(problems))


SEARCH_KINDS = {"search_morning", "search_afternoon", "search_evening", "search_price"}
BOOK_KINDS = {
    "book_direct_credit",
    "book_direct_split",
    "book_roundtrip_split",
    "book_date_replan",
    "book_connection_split",
}


def curriculum_search_option(db, rid, kind):
    """Freeze real route/date/window and payment constraints, never flight IDs in goals."""
    r = db["reservations"][rid]
    u = db["users"][r["user_id"]]
    n = len(r["passengers"])
    if not 1 <= n <= 5:
        return None
    old = r["flights"][0]
    origin = old["origin"]
    destination = r["destination"] if kind == "book_connection_split" else old["destination"]
    cabin = r["cabin"]
    window = {
        "search_morning": ("00:00:00", "12:00:00"),
        "search_afternoon": ("12:00:00", "18:00:00"),
        "search_evening": ("18:00:00", "24:00:00"),
    }.get(kind, ("00:00:00", "24:00:00"))
    cards = sorted(k for k, v in u["payment_methods"].items() if v["source"] == "credit_card")
    gifts = sorted(
        k for k, v in u["payment_methods"].items() if v["source"] == "gift_card" and v["amount"] > 0
    )
    if kind in BOOK_KINDS and (not cards or ("split" in kind and not gifts)):
        return None

    def options(a, z, date):
        return sorted(
            [
                (s["prices"][cabin], fid)
                for fid, f in db["flights"].items()
                if (f["origin"], f["destination"]) == (a, z)
                and window[0] <= f["scheduled_departure_time_est"] < window[1]
                for d, s in f["dates"].items()
                if d == date and s.get("status") == "available" and s["available_seats"][cabin] >= n
            ]
        )

    dates = sorted(
        {
            d
            for f in db["flights"].values()
            if (f["origin"], f["destination"]) == (origin, destination)
            for d in f["dates"]
            if curriculum_valid_date(d) and d > "2024-05-15" and d >= old["date"]
        }
    )
    for date in dates:
        if kind == "book_connection_split":
            pairs = []
            raw_count = 0
            for fid, f in db["flights"].items():
                state = f["dates"].get(date, {})
                if f["origin"] != origin or state.get("status") != "available":
                    continue
                arrival = f["scheduled_arrival_time_est"]
                if "+1" in arrival:
                    continue
                for gid, g in db["flights"].items():
                    gs = g["dates"].get(date, {})
                    if (
                        g["origin"] != f["destination"]
                        or g["destination"] != destination
                        or gs.get("status") != "available"
                        or g["scheduled_departure_time_est"] < arrival
                    ):
                        continue
                    raw_count += 1
                    layover = sum(
                        int(v) * m
                        for v, m in zip(g["scheduled_departure_time_est"].split(":"), (3600, 60, 1))
                    ) - sum(int(v) * m for v, m in zip(arrival.split(":"), (3600, 60, 1)))
                    if (
                        state["available_seats"][cabin] >= n
                        and gs["available_seats"][cabin] >= n
                        and layover >= 3600
                    ):
                        pairs.append((state["prices"][cabin] + gs["prices"][cabin], fid, gid))
            # Bound only source choice, never truncate the actual receipt.
            if not pairs or raw_count > 12:
                continue
            _, first, second = min(pairs)
            picked = [dict(flight_number=first, date=date), dict(flight_number=second, date=date)]
        else:
            choices = options(origin, destination, date)
            if not choices:
                continue
            picked = [dict(flight_number=choices[0][1], date=date)]
        return_date = None
        if kind == "book_roundtrip_split":
            returns = [(d, options(destination, origin, d)) for d in dates if d > date]
            returns = [(d, x) for d, x in returns if x]
            if not returns:
                continue
            return_date, back = returns[0]
            picked.append(dict(flight_number=back[0][1], date=return_date))
        preferred = None
        if kind == "book_date_replan":
            # A real infeasible preferred date/seat requirement drives a later alternative.
            for prior in dates:
                if prior >= date:
                    break
                if not options(origin, destination, prior):
                    preferred = prior
                    break
            if preferred is None:
                continue
        free = ALLOWANCE[u["membership"]][cabin] * n
        bags = n + 1 if int(sha256_json([rid, kind])[:2], 16) % 2 else 0
        paid = max(0, bags - free)
        insurance = "yes" if int(sha256_json([rid, kind])[-2:], 16) % 2 else "no"
        total = (
            sum(
                db["flights"][x["flight_number"]]["dates"][x["date"]]["prices"][cabin]
                for x in picked
            )
            * n
            + 50 * paid
            + (30 * n if insurance == "yes" else 0)
        )
        gift = gifts[0] if gifts and "split" in kind else None
        gift_amount = min(u["payment_methods"][gift]["amount"], max(1, total // 2)) if gift else 0
        return dict(
            origin=origin,
            destination=destination,
            date=date,
            return_date=return_date,
            preferred_date=preferred,
            cabin=cabin,
            window=window,
            flights=picked,
            passengers=n,
            total_bags=bags,
            paid_bags=paid,
            insurance=insurance,
            total=total,
            budget=total + 25,
            card=cards[-1] if cards else None,
            gift=gift,
            gift_amount=gift_amount,
        )
    return None


def build_curriculum_search(entry, rid, kind):
    adapted = adapt_record(ArealTaskRecord.model_validate(entry.task))
    if adapted.db_file_hash != entry.db_hash:
        raise ValueError("Source DB changed")
    b = Builder(adapted, entry, rid, kind)
    db = b.before
    r = db["reservations"][rid]
    uid = r["user_id"]
    _ = db["users"][uid]
    o = curriculum_search_option(db, rid, kind)
    if o is None:
        raise ValueError("Search eligibility changed")
    b.expected = copy.deepcopy(db)
    n = o["passengers"]
    cabin = o["cabin"]
    booking = kind in BOOK_KINDS
    date = o["preferred_date"] or o["date"]
    trip = "round-trip" if o["return_date"] else "one-way"
    extra = f" Return on {o['return_date']}." if o["return_date"] else ""
    flexibility = (
        f" If nothing fits on that day, I can travel on {o['date']}." if o["preferred_date"] else ""
    )
    routing = (
        "one-stop trip with both departures on the same date and at least one hour between flights"
        if kind == "book_connection_split"
        else "nonstop trip in each direction"
    )
    goal = f"I need a {trip} {routing} from {o['origin']} to {o['destination']} on {date}{extra} for {n} passengers in {cabin}, departing between {o['window'][0]} and {o['window'][1]} EST. Choose the cheapest option with seats for everyone.{flexibility}"
    if booking:
        goal += f" We need {o['total_bags']} checked bags total, and my budget including bags and any insurance is ${o['budget']}. The passengers are {json.dumps(r['passengers'])}."
        goal += (
            f" Use ${o['gift_amount']} from my stored gift card if possible and put the rest on a stored credit card."
            if o["gift"]
            else " Use a stored credit card."
        )
    else:
        goal += " Only search and quote the airfare; do not book or change anything."
    b.say("user", f"My user ID is {uid}. {goal}")
    profile = b.tool("get_user_details", {"user_id": uid}) if booking else None
    chosen = []
    requests = [(o["origin"], o["destination"], date)]
    if o["return_date"]:
        requests.append((o["destination"], o["origin"], o["return_date"]))
    if kind == "book_connection_split":
        pairs = b.tool(
            "search_onestop_flight",
            dict(origin=o["origin"], destination=o["destination"], date=date),
        )
        feasible_pairs = []
        for pair in pairs:
            first, second = pair
            if "+1" in first["scheduled_arrival_time_est"] or first["date"] != second["date"]:
                continue
            layover = sum(
                int(v) * m
                for v, m in zip(second["scheduled_departure_time_est"].split(":"), (3600, 60, 1))
            ) - sum(
                int(v) * m
                for v, m in zip(first["scheduled_arrival_time_est"].split(":"), (3600, 60, 1))
            )
            if (
                first["destination"] == second["origin"]
                and min(x["available_seats"][cabin] for x in pair) >= n
                and layover >= 3600
            ):
                feasible_pairs.append(pair)
        chosen = min(
            feasible_pairs,
            key=lambda pair: (
                sum(x["prices"][cabin] for x in pair),
                pair[0]["flight_number"],
                pair[1]["flight_number"],
            ),
        )
        requests = []
    for origin, destination, search_date in requests:
        options = b.tool(
            "search_direct_flight", dict(origin=origin, destination=destination, date=search_date)
        )
        feasible_options = [
            x
            for x in options
            if x["available_seats"][cabin] >= n
            and o["window"][0] <= x["scheduled_departure_time_est"] < o["window"][1]
        ]
        if not feasible_options:
            if not o["preferred_date"]:
                raise ValueError("No feasible receipt option")
            b.say(
                "assistant",
                f"The returned flights on {search_date} do not meet your seat and departure constraints. I will check your allowed alternative {o['date']} before attempting any booking.",
            )
            search_date = o["date"]
            options = b.tool(
                "search_direct_flight",
                dict(origin=origin, destination=destination, date=search_date),
            )
            feasible_options = [
                x
                for x in options
                if x["available_seats"][cabin] >= n
                and o["window"][0] <= x["scheduled_departure_time_est"] < o["window"][1]
            ]
        selected = min(feasible_options, key=lambda x: (x["prices"][cabin], x["flight_number"]))
        chosen.append(dict(selected, date=search_date))
    flights = [dict(flight_number=x["flight_number"], date=x["date"]) for x in chosen]
    if flights != o["flights"]:
        raise ValueError("Receipt-derived cheapest choice differs")
    fare = sum(x["prices"][cabin] for x in chosen) * n
    b.tool(
        "calculate", {"expression": f"({'+'.join(str(x['prices'][cabin]) for x in chosen)})*{n}"}
    )
    evidence = [
        dict(
            flight_number=x["flight_number"],
            date=x["date"],
            departure=x["scheduled_departure_time_est"],
            arrival=x["scheduled_arrival_time_est"],
            seats=x["available_seats"][cabin],
            per_passenger=x["prices"][cabin],
        )
        for x in chosen
    ]
    if not booking:
        b.say(
            "assistant",
            f"The cheapest returned itinerary meeting your departure window and {n}-seat requirement is {json.dumps(evidence)}. Airfare for all passengers is ${fare}. No booking, payment or change was made.",
        )
        b.say("user", "Thanks, that is all.")
        b.say("assistant", "No changes have been made.")
    else:
        b.say(
            "assistant",
            f"The cheapest feasible flights are {json.dumps(evidence)}, airfare ${fare}. Would you like travel insurance at $30 per passenger, ${30 * n} total?",
        )
        b.say(
            "user",
            ("Yes, include insurance." if o["insurance"] == "yes" else "No insurance, please."),
        )
        free = ALLOWANCE[profile["membership"]][cabin] * n
        paid = max(0, o["total_bags"] - free)
        total = fare + 50 * paid + (30 * n if o["insurance"] == "yes" else 0)
        if total != o["total"] or total > o["budget"]:
            raise ValueError("Quote/budget mismatch")
        payments = []
        if o["gift"]:
            if profile["payment_methods"][o["gift"]]["amount"] < o["gift_amount"]:
                raise ValueError("Insufficient observed gift balance")
            payments.append(dict(payment_id=o["gift"], amount=o["gift_amount"]))
        payments.append(dict(payment_id=o["card"], amount=total - o["gift_amount"]))
        b.tool(
            "calculate",
            {"expression": f"{fare}+{paid}*50+{30 * n if o['insurance'] == 'yes' else 0}"},
        )
        b.say(
            "assistant",
            f"Please confirm this NEW {trip} {cabin} booking: {json.dumps(evidence)}; passengers {json.dumps(r['passengers'])}; {o['total_bags']} checked bags total, with allowance {free} and {paid} paid bags costing ${50 * paid}; insurance {o['insurance']}. Total ${total}, within your ${o['budget']} budget. Payment allocations: {json.dumps(payments)}. Existing reservations stay unchanged. Shall I book?",
        )
        b.say(
            "user",
            "Yes, book the exact itinerary, passengers, bags, insurance and payment amounts just quoted.",
        )
        receipt = b.tool(
            "book_reservation",
            dict(
                user_id=uid,
                origin=o["origin"],
                destination=o["destination"],
                flight_type="round_trip" if o["return_date"] else "one_way",
                cabin=cabin,
                flights=flights,
                passengers=r["passengers"],
                payment_methods=payments,
                total_baggages=o["total_bags"],
                nonfree_baggages=paid,
                insurance=o["insurance"],
            ),
        )
        newid = receipt["reservation_id"]
        if newid in db["reservations"]:
            raise ValueError("Existing reservation overwritten")
        expected_flights = [
            dict(
                flight_number=x["flight_number"],
                date=x["date"],
                origin=x["origin"],
                destination=x["destination"],
                price=x["prices"][cabin],
            )
            for x in chosen
        ]
        b.expected["reservations"][newid] = dict(
            reservation_id=newid,
            user_id=uid,
            origin=o["origin"],
            destination=o["destination"],
            flight_type="round_trip" if o["return_date"] else "one_way",
            cabin=cabin,
            flights=expected_flights,
            passengers=copy.deepcopy(r["passengers"]),
            payment_history=copy.deepcopy(payments),
            created_at="2024-05-15T15:00:00",
            total_baggages=o["total_bags"],
            nonfree_baggages=paid,
            insurance=o["insurance"],
            status=None,
        )
        b.expected["users"][uid]["reservations"].append(newid)
        if o["gift"]:
            b.expected["users"][uid]["payment_methods"][o["gift"]]["amount"] -= o["gift_amount"]
        for x in chosen:
            b.expected["flights"][x["flight_number"]]["dates"][x["date"]]["available_seats"][
                cabin
            ] -= n
        b.say(
            "assistant",
            f"New reservation {newid} is confirmed for {json.dumps(flights)} in {cabin}, with the confirmed passengers, {o['total_bags']} bags and insurance {o['insurance']}. Total paid ${total} using {json.dumps(payments)}. Your existing bookings were not changed.",
        )
    row = b.finish(
        goal,
        [
            "receipt_derived_cheapest_feasible_itinerary",
            "seats_window_budget_checked_before_write",
            "explicit_complete_booking_confirmation" if booking else "no_write_read_only_contract",
            "independent_full_expected_database_and_native_replay",
        ],
    )
    row["metadata"]["strategy_demonstration_scope"] = (
        "natural_goal_assistant_search_selection_payment_decomposition"
        if booking
        else "read_only_search_filter_calculate"
    )
    return row


def run_curriculum(args):
    """Versioned local CPU expansion through existing builders; append only."""
    if (args.output / "summary.json").exists():
        raise FileExistsError("Finalized curriculum is immutable; use a new output directory")
    from loguru import logger

    from tau3_grpo.data.decision_repair import build_decision, eligible

    logger.remove()
    pool, entries, dbs, protected, historical = curriculum_pool()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "entity_split.json"
    if lock.exists():
        split = json.loads(lock.read_text())
    else:
        fresh = {x["user_id"] for x in pool} - historical
        # Hold out ten users before construction, with broad reservation coverage.
        counts = Counter(x["user_id"] for x in pool)
        dev = sorted(fresh, key=lambda uid: (-counts[uid], sha256_json([42, uid])))[:10]
        split = dict(
            dev_users=dev,
            protected_users=sorted(protected),
            historical_training_users=sorted(historical),
            exposure_scope=[
                "old_train100",
                "repair72_train",
                "repaired149_train",
                "active_formal_RL50_reference_users",
            ],
            candidate_exposure_note="Previously generated v1 candidates are not gradient exposure; all final train users are disjoint from these dev users",
            split_scope="entity_heldout_template_shared_not_semantic_family_independent",
            seed=42,
        )
        lock.write_text(json.dumps(split, indent=2) + "\n")
    if set(split["dev_users"]) & historical:
        raise ValueError("Historical dev exposure collision")
    foundation = {
        "search_morning": 10,
        "search_afternoon": 10,
        "search_evening": 10,
        "search_price": 10,
        "allowance_quote": 15,
        "reservation_summary": 15,
        "payment_inventory": 10,
        "passenger_identity": 5,
        "bag_removal_refusal": 5,
        "passenger_count_refusal": 5,
        "certificate_update_refusal": 5,
    }
    constraints = {
        "book_direct_credit": 20,
        "book_direct_split": 20,
        "baggage_free": 8,
        "baggage_paid": 10,
        "cabin_basic_upgrade": 8,
        "cabin_economy_upgrade": 10,
        "cancel_allowed": 8,
        "cancel_after24": 8,
        "cancel_within24": 8,
        "historical_upgrade": 8,
        "downgrade_refund": 8,
        "multileg_cabin": 8,
        "card_tail_binding": 8,
        "passenger_correction": 8,
        "flight_change": 5,
        "flight_refusal": 5,
    }
    strategy = {
        "book_connection_split": 15,
        "book_roundtrip_split": 35,
        "book_date_replan": 5,
        "upgrade_then_bags": 25,
        "gift_limit_upgrade_bags": 25,
        "downgrade_gift_bags": 15,
        "partial_change_bags": 20,
        "no_inventory_cabin_bags": 5,
        "cancel_refund_rebook": 5,
    }
    plans = {"foundation": foundation, "constraints": constraints, "strategy": strategy}
    phase = args.curriculum_phase
    if phase not in plans:
        raise ValueError("Unknown curriculum phase")
    quantities = plans[phase]
    candidates = []
    for item in pool:
        db = dbs[item["task_id"]]
        r = db["reservations"][item["rid"]]
        if any(not curriculum_valid_date(f["date"]) for f in r["flights"]):
            continue
        for kind in quantities:
            if kind in SEARCH_KINDS | BOOK_KINDS:
                okay = curriculum_search_option(db, item["rid"], kind) is not None
            elif phase == "foundation":
                okay = (kind != "bag_removal_refusal" or r["total_baggages"] > 0) and (
                    kind != "insurance_update_refusal" or r["insurance"] == "no"
                )
            elif phase == "strategy":
                okay = curriculum_strategy_option(db, item["rid"], kind) is not None
            elif kind in (
                "cancel_after24",
                "cancel_within24",
                "historical_upgrade",
                "downgrade_refund",
                "multileg_cabin",
                "card_tail_binding",
            ):
                okay = eligible(db, item["rid"], kind)
            else:
                okay = suitable(db, item["rid"], kind)
            if okay:
                candidates.append(
                    dict(
                        item,
                        kind=kind,
                        split="validation" if item["user_id"] in split["dev_users"] else "train",
                        signature=list(curriculum_signature(db, item["rid"], kind)),
                    )
                )
    selected = []
    used_signatures = set()
    entity_uses = Counter()
    for dest in ("train", "validation"):
        dev_total = {"foundation": 16, "constraints": 14, "strategy": 30}[phase]
        for ki, (kind, quota) in enumerate(quantities.items()):
            quota = (
                quota
                if dest == "train"
                else dev_total // len(quantities) + int(ki < dev_total % len(quantities))
            )
            options = sorted(
                [x for x in candidates if x["kind"] == kind and x["split"] == dest],
                key=lambda x: sha256_json([42, kind, x["user_id"], x["rid"]]),
            )
            count = 0
            while options and count < quota:
                options.sort(key=lambda x: entity_uses[(x["user_id"], x["rid"])])
                item = options.pop(0)
                signature = sha256_json([dest, item["signature"]])
                if signature in used_signatures:
                    continue
                used_signatures.add(signature)
                entity_uses[(item["user_id"], item["rid"])] += 1
                selected.append(item)
                count += 1
        # Scarce native branches are not fabricated. Redistribute only to other
        # eligible, parameter-distinct goals and publish the actual branch mix.
        desired = (
            sum(quantities.values())
            if dest == "train"
            else (16 if phase == "foundation" else 14 if phase == "constraints" else 30)
        )
        remaining = [
            x
            for x in candidates
            if x["split"] == dest and sha256_json([dest, x["signature"]]) not in used_signatures
        ]
        while sum(x["split"] == dest for x in selected) < desired and remaining:
            branch_counts = Counter(x["kind"] for x in selected if x["split"] == dest)
            remaining.sort(
                key=lambda x: (
                    branch_counts[x["kind"]],
                    entity_uses[(x["user_id"], x["rid"])],
                    sha256_json(x),
                )
            )
            item = remaining.pop(0)
            signature = sha256_json([dest, item["signature"]])
            if signature in used_signatures:
                continue
            used_signatures.add(signature)
            entity_uses[(item["user_id"], item["rid"])] += 1
            selected.append(item)
    plan_path = output / f"{phase}_plan.json"
    plan = dict(
        phase=phase,
        requested=quantities,
        selected=selected,
        split_sha256=sha256_file(lock),
        code_sha256=sha256_file(__file__),
    )
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        if sha256_json(saved["selected"]) != sha256_json(selected):
            raise ValueError("Frozen plan drift")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    paths = {dest: output / f"{phase}_{dest}.jsonl" for dest in ("train", "validation")}
    done = {
        r["metadata"]["source_dialog_id"]
        for p in paths.values()
        if p.exists()
        for r in read_rows(p)
    }
    done.update(
        r["metadata"]["replaces_source_dialog_id"]
        for p in paths.values()
        if p.exists()
        for r in read_rows(p)
        if r["metadata"].get("replaces_source_dialog_id")
    )
    historical_identities = {
        sha256_json([r["messages"], r["supervision"]["message_indices"]])
        for directory in ("staged_v2_A100_20260925", "decision_repair_20260926_v1")
        for r in read_rows("data/sft/" + directory + "/train.jsonl")
    }
    import yaml
    from transformers import AutoTokenizer

    from tau3_grpo.prompts import prepare_agent_messages
    from tau3_grpo.training.sft.dataset import build_supervised_example

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3.5-4B", local_files_only=True)
    schemas = [
        x["tool_schema"]
        for x in yaml.safe_load(Path("configs/envs/tool_config.yaml").read_text())["tools"]
    ]
    provenance = dict(
        tokenizer_config_sha256=sha256_file("models/Qwen3.5-4B/tokenizer_config.json"),
        tokenizer_sha256=sha256_file("models/Qwen3.5-4B/tokenizer.json"),
        schema_sha256=sha256_file("configs/envs/tool_config.yaml"),
        template_code_sha256=sha256_file("tau3_grpo/models/qwen35_template.py"),
        max_length=24576,
        mode="actual_local_tokenizer_complete_dialogue_no_truncation",
    )
    audit_output = Path("results/analysis") / output.name.replace(
        "generated_curriculum", "sft_generated_curriculum"
    )
    audit_output.mkdir(parents=True, exist_ok=True)
    (audit_output / f"executed_source_{sha256_file(__file__)}.py").write_text(
        Path(__file__).read_text()
    )
    errors = []
    completed = 0
    reserve = [
        x for x in candidates if sha256_json([x["split"], x["signature"]]) not in used_signatures
    ]
    reserve.sort(key=lambda x: sha256_json(x))
    reserve_path = output / f"{phase}_reserve_plan.json"
    if reserve_path.exists():
        if sha256_json(json.loads(reserve_path.read_text())) != sha256_json(reserve):
            raise ValueError("Reserve plan drift")
    else:
        reserve_path.write_text(json.dumps(reserve, indent=2) + "\n")
    reuse = {}
    if args.reuse_curriculum:
        for dest in ("train", "validation"):
            source_path = args.reuse_curriculum / f"{phase}_{dest}.jsonl"
            if not source_path.exists():
                continue
            source_hash = sha256_file(source_path)
            for old in read_rows(source_path):
                reuse[old["metadata"]["source_dialog_id"]] = (old, str(source_path), source_hash)
    queue = list(selected)
    accepted_signatures = {
        sha256_json([dest, r["metadata"]["parameter_signature"]])
        for dest, p in paths.items()
        if p.exists()
        for r in read_rows(p)
    }
    for item in queue:
        row = None
        if sha256_json([item["split"], item["signature"]]) in accepted_signatures:
            continue

        sid = f"curriculum_{phase}_{item['kind']}_{item['task_id']}_{item['rid']}"
        if sid in done:
            continue
        if args.limit and completed >= args.limit:
            break
        try:
            reuse_evidence = None
            if sid in reuse:
                old, source_path, source_hash = reuse[sid]
                if sha256_json(old["metadata"]["parameter_signature"]) != sha256_json(
                    item["signature"]
                ):
                    raise ValueError("Reused parameter mismatch")
                if old["metadata"]["source_db_hash"] != entries[item["task_id"]].db_hash:
                    raise ValueError("Reused DB mismatch")
                row = copy.deepcopy(old)
                reuse_evidence = dict(
                    path=source_path,
                    file_sha256=source_hash,
                    exact_messages_mask_sha256=sha256_json(
                        [old["messages"], old["supervision"]["message_indices"]]
                    ),
                    original_generation_code_sha256=old["metadata"]["generation_code_sha256"],
                    original_plan_sha256=old["metadata"]["plan_sha256"],
                    scope="Exact current-session native replay evidence retained; no new native execution claimed for reused rows",
                )
            elif item["kind"] in SEARCH_KINDS | BOOK_KINDS:
                row = build_curriculum_search(entries[item["task_id"]], item["rid"], item["kind"])
            elif phase == "foundation":
                row = build_curriculum_read(entries[item["task_id"]], item["rid"], item["kind"])
            elif phase == "strategy":
                row = build_curriculum_strategy(entries[item["task_id"]], item["rid"], item["kind"])
            elif item["kind"] in (
                "cancel_after24",
                "cancel_within24",
                "historical_upgrade",
                "downgrade_refund",
                "multileg_cabin",
                "card_tail_binding",
            ):
                row = build_decision(entries[item["task_id"]], item["rid"], item["kind"])
            else:
                row = (build_case if item["kind"] in BASE_KINDS else build_extra)(
                    entries[item["task_id"]], item["rid"], item["kind"]
                )
            if row is None:
                raise ValueError("No live eligible dialogue")
            row["metadata"].update(
                source_dialog_id=sid,
                split=item["split"],
                entity_group=item["user_id"],
                construction_kind=item["kind"],
                semantic_family=item["kind"],
                curriculum_bucket=phase,
                parameter_signature=item["signature"],
                difficulty={
                    "level": {"foundation": "easy", "constraints": "medium", "strategy": "hard"}[
                        phase
                    ],
                    "basis": "authored_goal_and_policy_branch_rubric_not_tool_count",
                },
                semantic_review_scope="programmatic_branch_contract_and_native_replay_not_per_record_independent_human_read",
                generation_code_sha256=plan["code_sha256"],
                plan_sha256=sha256_file(plan_path),
                status="native_replayed_full_dialogue_candidate_not_training_run_authorization",
            )
            if reuse_evidence:
                row["metadata"]["exact_candidate_reuse"] = reuse_evidence
                row["metadata"]["generation_code_sha256"] = reuse_evidence[
                    "original_generation_code_sha256"
                ]
                row["metadata"]["materialization_code_sha256"] = plan["code_sha256"]
            if item.get("replaces_source_dialog_id"):
                row["metadata"]["replaces_source_dialog_id"] = item["replaces_source_dialog_id"]
            if (
                sha256_json([row["messages"], row["supervision"]["message_indices"]])
                in historical_identities
            ):
                raise ValueError("Historical exact messages+mask duplicate")
            if audit_curriculum_dates(row["messages"]):
                raise ValueError(
                    f"Non-ISO or invalid visible dates: {audit_curriculum_dates(row['messages'])}"
                )
            example = build_supervised_example(
                prepare_agent_messages(row["messages"]),
                tokenizer,
                tools=schemas,
                max_length=24576,
                approved_indices=row["supervision"]["message_indices"],
            )
            row["metadata"]["render_token_counts"] = dict(
                provenance,
                total_tokens=example["n_total_tokens"],
                assistant_tokens=example["n_label_tokens"],
                input_ids_sha256=sha256_json(example["input_ids"]),
                labels_sha256=sha256_json(example["labels"]),
            )
            if audit_tool_calls(row, schemas) or ordered_tool_receipts(row):
                raise ValueError("Tool schema/receipt audit failed")
            with paths[item["split"]].open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.add(sid)
            completed += 1
            accepted_signatures.add(sha256_json([item["split"], item["signature"]]))
        except Exception as exc:
            errors.append(dict(item, error=str(exc)[:1000]))
            if row is not None:
                with (output / f"{phase}_rejected_full_dialogues.jsonl").open("a") as f:
                    f.write(json.dumps(dict(error=str(exc), record=row), ensure_ascii=False) + "\n")
            replacement = next(
                (
                    x
                    for x in reserve
                    if x["split"] == item["split"]
                    and x["kind"] == item["kind"]
                    and sha256_json([x["split"], x["signature"]]) not in accepted_signatures
                ),
                None,
            )
            if replacement:
                reserve.remove(replacement)
                queue.append(
                    dict(
                        replacement,
                        replaces_source_dialog_id=item.get("replaces_source_dialog_id", sid),
                    )
                )

        print(
            json.dumps(dict(phase=phase, completed=len(done), new=completed, errors=len(errors))),
            flush=True,
        )
    report = dict(
        phase=phase,
        counts={dest: len(read_rows(p)) if p.exists() else 0 for dest, p in paths.items()},
        errors=errors,
        requested_train=sum(quantities.values()),
        planned_train=sum(x["split"] == "train" for x in selected),
        paid_api_calls=0,
        gpu_hours=0,
        split_scope=split["split_scope"],
    )
    (output / f"{phase}_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def finalize_curriculum(output):
    """Freeze generated dialogue artifacts with honest review and isolation scope."""
    import math

    from tau3_grpo.data.curriculum_split import build_split

    audit_output = Path("results/analysis") / output.name.replace(
        "generated_curriculum", "sft_generated_curriculum"
    )
    audit_output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((output / "entity_split.json").read_text())
    phases = ("foundation", "constraints", "strategy")
    counts = {"foundation": 100, "constraints": 150, "strategy": 150}
    dev_counts = {"foundation": 16, "constraints": 14, "strategy": 30}
    rows = {
        dest: {phase: read_rows(output / f"{phase}_{dest}.jsonl") for phase in phases}
        for dest in ("train", "validation")
    }
    for dest, groups in rows.items():
        for phase, values in groups.items():
            if len(values) != (counts if dest == "train" else dev_counts)[phase]:
                raise ValueError(f"Incomplete {dest}/{phase}: {len(values)}")
            plan = json.loads((output / f"{phase}_plan.json").read_text())
            planned = {
                f"curriculum_{phase}_{x['kind']}_{x['task_id']}_{x['rid']}"
                for x in plan["selected"]
                if x["split"] == dest
            }
            covered = {r["metadata"]["source_dialog_id"] for r in values} | {
                r["metadata"]["replaces_source_dialog_id"]
                for r in values
                if r["metadata"].get("replaces_source_dialog_id")
            }
            if planned - covered:
                raise ValueError("Unresolved planned rows after replacements")

    merged = {dest: sum(groups.values(), []) for dest, groups in rows.items()}

    def identities(values):
        return {sha256_json([r["messages"], r["supervision"]["message_indices"]]) for r in values}

    def users(values):
        return {r["metadata"]["entity_group"] for r in values}

    def source_rids(values):
        return {r["metadata"]["source_family"] for r in values}

    if users(merged["train"]) & users(merged["validation"]):
        raise ValueError("Train/dev user collision")
    if source_rids(merged["train"]) & source_rids(merged["validation"]):
        raise ValueError("Train/dev source reservation collision")
    if users(merged["validation"]) & set(lock["historical_training_users"]):
        raise ValueError("Historical dev exposure")
    formal_rl_users = set()
    formal_rl_manifest = Path(
        "data/manifests/rl_curriculum50_20260912/areal_airline_train_seed42.jsonl"
    )
    for entry in read_manifest(formal_rl_manifest):
        raw = raw_database(
            str(ArealTaskRecord.model_validate(entry.task).resolve_db_path(AREAL_DB_ROOT))
        )
        formal_rl_users.update(
            raw["reservations"][rid]["user_id"]
            for rid in task_feature(entry.task)["reservation_ids"]
            if rid in raw["reservations"]
        )
        formal_rl_users.update(
            a["arguments"]["user_id"]
            for a in entry.task["evaluation_criteria"].get("actions", [])
            if a.get("arguments", {}).get("user_id")
        )
    if users(merged["validation"]) & formal_rl_users:
        raise ValueError("Formal RL dev exposure")

    if (users(merged["train"]) | users(merged["validation"])) & set(lock["protected_users"]):
        raise ValueError("Protected user collision")
    allrows = merged["train"] + merged["validation"]
    allids = identities(allrows)
    if len(allids) != len(allrows):
        raise ValueError("Exact dialogue/mask duplicate")
    historical = read_rows("data/sft/staged_v2_A100_20260925/train.jsonl") + read_rows(
        "data/sft/decision_repair_20260926_v1/train.jsonl"
    )
    historical_visible = historical + read_rows(
        "data/sft/repaired_source_grounded_20260926_v1/train.jsonl"
    )
    historical_text = "\n".join(json.dumps(r["messages"]) for r in historical_visible)
    if any(uid in historical_text for uid in users(merged["validation"])):
        raise ValueError("Historical visible-message dev identity exposure")
    if allids & identities(historical):
        raise ValueError("Historical exact duplicate")

    def write(name, value):
        text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        target = output / name
        if target.exists() and target.read_text() != text:
            raise ValueError(f"Frozen output drift: {target}")
        target.write_text(text)

    def write_rows(name, values):
        target = output / name
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values)
        if target.exists() and target.read_text() != text:
            raise ValueError(f"Frozen output drift: {target}")
        target.write_text(text)

    labels = []
    write_counts = Counter()
    missing_confirmation = []
    newids = Counter()
    reuse_sources = {}
    for r in allrows:
        md = r["metadata"]
        m = r["messages"]
        indices = r["supervision"]["message_indices"]
        if "exact_candidate_reuse" in md:
            evidence = md["exact_candidate_reuse"]
            path = evidence["path"]
            if path not in reuse_sources:
                if sha256_file(path) != evidence["file_sha256"]:
                    raise ValueError("Reused source file changed")
                reuse_sources[path] = {
                    x["metadata"]["source_dialog_id"]: x for x in read_rows(path)
                }
            old = reuse_sources[path][md["source_dialog_id"]]
            if (
                sha256_json([m, indices]) != evidence["exact_messages_mask_sha256"]
                or m != old["messages"]
                or indices != old["supervision"]["message_indices"]
            ):
                raise ValueError("Reused dialogue changed")
            for key in ("input_ids_sha256", "labels_sha256", "total_tokens", "assistant_tokens"):
                if md["render_token_counts"][key] != old["metadata"]["render_token_counts"][key]:
                    raise ValueError("Fresh render differs from reused evidence")

        if audit_curriculum_dates(m):
            raise ValueError(f"Invalid visible dates: {md['source_dialog_id']}")
        if indices != [i for i, x in enumerate(m) if x["role"] == "assistant"]:
            raise ValueError("Assistant mask mismatch")
        tc = md["render_token_counts"]
        if tc["total_tokens"] > 24576 or tc["assistant_tokens"] <= 0:
            raise ValueError("Token bounds")
        if tc["mode"] != "actual_local_tokenizer_complete_dialogue_no_truncation":
            raise ValueError("Nonactual token counts")
        for i, x in enumerate(m):
            for c in x.get("tool_calls", []):
                name = c["function"]["name"]
                if name.startswith("update_") or name in (
                    "book_reservation",
                    "cancel_reservation",
                    "send_certificate",
                ):
                    write_counts[name] += 1
                    if (
                        i == 0
                        or m[i - 1]["role"] != "user"
                        or not m[i - 1]["content"].lower().startswith("yes")
                    ):
                        missing_confirmation.append(
                            dict(source_id=md["source_dialog_id"], index=i, name=name)
                        )
                if name == "book_reservation":
                    receipt = json.loads(m[i + 1]["content"])
                    newids[receipt["reservation_id"]] += 1
        labels.append(
            dict(
                source_id=md["source_dialog_id"],
                group_id=md["entity_group"],
                group_status="reviewed",
                label_status="reviewed",
                label_review_scope="authored_branch_contract_review_not_independent_per_dialogue_semantic_review",
                difficulty={"foundation": "easy", "constraints": "medium", "strategy": "hard"}[
                    md["curriculum_bucket"]
                ],
                difficulty_basis="task_constraints",
                needs_replanning=md["construction_kind"] == "book_date_replan",
                needs_dependent_subgoals=md["curriculum_bucket"] == "strategy",
                locked_split=md["split"],
            )
        )
    if missing_confirmation:
        raise ValueError(f"Missing immediate explicit confirmations: {missing_confirmation}")
    group_manifest = build_split(
        labels, seed=42, validation_ratio=60 / 460, protected_groups=lock["protected_users"]
    )
    if group_manifest["held"]:
        raise ValueError(group_manifest["held"])
    for dest in merged:
        if set(group_manifest["splits"][dest]) != {
            r["metadata"]["source_dialog_id"] for r in merged[dest]
        }:
            raise ValueError("Locked split changed")
    group_manifest["split_scope"] = lock["split_scope"]
    group_manifest["label_scope"] = (
        "reviewed authored branch contracts, not 460 independently human-read dialogues"
    )
    group_manifest["bucket_note"] = (
        "Group buckets in this splitter are max difficulty per user; materialized curriculum buckets remain per-row authored task constraints."
    )
    write("group_split_manifest.json", group_manifest)
    for dest, values in merged.items():
        write_rows(f"{dest}.jsonl", values)
    cumulative = []
    stages = {}
    for phase, stage in zip(phases, ("A100", "B250", "C400")):
        cumulative += rows["train"][phase]
        name = f"{stage}_train.jsonl"
        write_rows(name, cumulative)
        stages[stage] = dict(
            path=str(output / name),
            rows=len(cumulative),
            sha256=sha256_file(output / name),
            assistant_tokens=sum(
                r["metadata"]["render_token_counts"]["assistant_tokens"] for r in cumulative
            ),
            total_tokens=sum(
                r["metadata"]["render_token_counts"]["total_tokens"] for r in cumulative
            ),
        )

    def quantiles(values):
        values = sorted(values)
        return {
            **{
                f"p{int(q * 100)}": values[max(0, math.ceil(q * len(values)) - 1)]
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "max": max(values),
            "sum": sum(values),
        }

    stats = {}
    for dest, groups in rows.items():
        stats[dest] = {}
        for phase, values in groups.items():
            toolchains = Counter(
                tuple(c["function"]["name"] for m in r["messages"] for c in m.get("tool_calls", []))
                for r in values
            )
            stats[dest][phase] = dict(
                rows=len(values),
                families=dict(Counter(r["metadata"]["construction_kind"] for r in values)),
                unique_users=len(users(values)),
                unique_source_reservations=len(source_rids(values)),
                unique_parameter_signatures=len(
                    {sha256_json(r["metadata"]["parameter_signature"]) for r in values}
                ),
                unique_tool_sequence_structures=len(toolchains),
                planning_scope=dict(
                    Counter(
                        "natural_goal_search_and_itinerary_choice"
                        if r["metadata"]["construction_kind"] in BOOK_KINDS
                        else "specified_target_dependent_execution"
                        if phase == "strategy"
                        else "single_goal"
                        for r in values
                    )
                ),
                tool_sequence_counts=[dict(tools=list(k), count=v) for k, v in toolchains.items()],
                user_reuse_max=max(Counter(r["metadata"]["entity_group"] for r in values).values()),
                token_counts=quantiles(
                    [r["metadata"]["render_token_counts"]["total_tokens"] for r in values]
                ),
                assistant_token_counts=quantiles(
                    [r["metadata"]["render_token_counts"]["assistant_tokens"] for r in values]
                ),
                demonstration_scope=dict(
                    Counter(
                        r["metadata"].get(
                            "strategy_demonstration_scope", "single_goal_authored_contract"
                        )
                        for r in values
                    )
                ),
            )
    summaries = {
        phase: json.loads((output / f"{phase}_summary.json").read_text()) for phase in phases
    }
    rejected = [e for s in summaries.values() for e in s["errors"]]
    report = dict(
        version=output.name,
        status="cpu_verified_full_dialogue_candidates_not_training_authorization",
        counts=dict(train=len(merged["train"]), validation=len(merged["validation"])),
        stats=stats,
        cumulative=stages,
        split_scope=lock["split_scope"],
        isolation=dict(
            user_collisions=0,
            source_reservation_collisions=0,
            exact_message_mask_collisions=0,
            historical_old100_repair72_exact_collisions=0,
            protected_users=len(lock["protected_users"]),
            historical_exposure_users=len(set(lock["historical_training_users"]) | formal_rl_users),
            formal_rl_reference_users=sorted(formal_rl_users),
            formal_rl_manifest_sha256=sha256_file(formal_rl_manifest),
            formal_rl_dev_collisions=0,
            generated_reservation_id_counts=dict(newids),
            generated_id_scope="Native allocator reuses HATHAT in independent DB clones; created entities are clone/user-local, not globally disjoint literal IDs",
            shared_database_hashes_not_db_disjoint=True,
        ),
        checks=dict(
            full_native_receipts=len(allrows),
            independent_native_replay_evidence=len(allrows),
            native_replay_reused_from_exact_current_session_candidates=sum(
                "exact_candidate_reuse" in r["metadata"] for r in allrows
            ),
            new_native_replay_this_materialization=sum(
                "exact_candidate_reuse" not in r["metadata"] for r in allrows
            ),
            independent_full_expected_database_equal=len(allrows),
            actual_local_tokenized=len(allrows),
            max_context=24576,
            no_receipt_truncation=True,
            assistant_only_mask=len(allrows),
            explicit_write_confirmation_counts=dict(write_counts),
        ),
        rejected_attempts=len(rejected),
        oversize_rejections=sum(
            "max_length" in e["error"] or "tokens" in e["error"] for e in rejected
        ),
        rejected=rejected,
        review_scope="Generator branch contracts inspected and native integration tested; not independent per-record human semantic review",
        diversity_scope="Distinct authored branch and goal-parameter signatures; shared templates, no semantic-family independence claim",
        original_source_task_solutions_claimed=0,
        gpu_hours=0,
        paid_api_calls=0,
        training_started=False,
        dependency_sha256={
            p: sha256_file(p)
            for p in [
                "tau3_grpo/data/grounded_gap_pilot.py",
                "tau3_grpo/data/decision_repair.py",
                "tau3_grpo/data/sft_policy_checks.py",
                "tau3_grpo/training/sft/dataset.py",
                "tau3_grpo/models/qwen35_template.py",
                "tau2-bench/src/tau2/domains/airline/tools.py",
                "tau2-bench/data/tau2/domains/airline/policy.md",
                "models/Qwen3.5-4B/tokenizer.json",
                "models/Qwen3.5-4B/tokenizer_config.json",
                "models/Qwen3.5-4B/chat_template.jinja",
                "configs/envs/tool_config.yaml",
            ]
            if Path(p).exists()
        },
        generation_code_sha256=sorted({r["metadata"]["generation_code_sha256"] for r in allrows}),
        input_sha256={
            str(output / f"{p}_{d}.jsonl"): sha256_file(output / f"{p}_{d}.jsonl")
            for p in phases
            for d in merged
        },
    )
    write("summary.json", report)
    (audit_output / "final_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            dict(
                counts=report["counts"],
                rejections=report["rejected_attempts"],
                cumulative={k: v["rows"] for k, v in stages.items()},
            )
        )
    )
    return report


def build_cabin_then_baggage(entry, rid):
    """Two-write pipeline probe; second price uses the first write's receipt.

    This authors a new subtask on a training DB, not a solution to its source task.
    It deliberately stays outside the existing single-action generation KINDS.
    """
    adapted = adapt_record(ArealTaskRecord.model_validate(entry.task))
    if entry.split != "train" or adapted.db_file_hash != entry.db_hash:
        raise ValueError("Expected unchanged training source DB")
    b = Builder(adapted, entry, rid, "cabin_then_baggage")
    db = b.before
    r = db["reservations"][rid]
    uid = r["user_id"]
    u = db["users"][uid]
    n = len(r["passengers"])
    if not feasible(db, rid, "cabin_basic_upgrade") or r["nonfree_baggages"] != 0:
        raise ValueError("Probe needs eligible basic upgrade without previously paid bags")
    cabin = "economy"
    card = sorted(k for k, v in u["payment_methods"].items() if v["source"] == "credit_card")[-1]
    free = ALLOWANCE[u["membership"]][cabin] * n
    if free <= ALLOWANCE[u["membership"]][r["cabin"]] * n or free + 1 <= r["total_baggages"]:
        raise ValueError("Probe needs a changed allowance and additional bags")
    target = free + 1
    b.expected = copy.deepcopy(db)
    er = b.expected["reservations"][rid]
    b.say(
        "user",
        f"My user ID is {uid}. First upgrade all passengers on reservation {rid} to economy, keeping every flight and date. After the upgrade succeeds, set the reservation to {target} checked bags in total, using the new cabin allowance. Quote and confirm each charge separately. Keep passengers and insurance unchanged.",
    )
    observed = b.tool("get_reservation_details", {"reservation_id": rid})
    profile = b.tool("get_user_details", {"user_id": uid})
    prices = []
    for f in observed["flights"]:
        options = b.tool(
            "search_direct_flight",
            {"origin": f["origin"], "destination": f["destination"], "date": f["date"]},
        )
        matched = next(x for x in options if x["flight_number"] == f["flight_number"])
        if matched["available_seats"][cabin] < n:
            raise ValueError("Insufficient seats")
        prices.append(matched["prices"][cabin])
    delta = (sum(prices) - sum(f["price"] for f in observed["flights"])) * n
    b.tool(
        "calculate",
        {"expression": f"({sum(prices)}-{sum(f['price'] for f in observed['flights'])})*{n}"},
    )
    b.say(
        "assistant",
        f"All existing flights have economy seats for {n} passengers. A cabin-only upgrade is permitted before departure. The additional airfare is ${delta}, charged to stored credit card {card}. Flights, dates, passengers, insurance and current bags will remain unchanged during this first step. May I upgrade reservation {rid}?",
    )
    b.say(
        "user",
        f"Yes, upgrade to economy on the same flights and charge ${delta} to {card}. Then quote the bags using the updated allowance.",
    )
    er["cabin"] = cabin
    for f, p in zip(er["flights"], prices):
        f["price"] = p
    er["payment_history"].append({"payment_id": card, "amount": delta})
    upgraded = b.tool(
        "update_reservation_flights",
        {
            "reservation_id": rid,
            "cabin": cabin,
            "flights": [
                {"flight_number": f["flight_number"], "date": f["date"]}
                for f in observed["flights"]
            ],
            "payment_id": card,
        },
    )
    # Derive the second action from the actual new state, not the old cabin.
    new_free = ALLOWANCE[profile["membership"]][upgraded["cabin"]] * len(upgraded["passengers"])
    paid = max(0, target - new_free)
    cost = 50 * (paid - upgraded["nonfree_baggages"])
    b.tool("calculate", {"expression": f"({paid}-{upgraded['nonfree_baggages']})*50"})
    b.say(
        "assistant",
        f"The upgrade succeeded. Your {profile['membership']} membership with the updated {upgraded['cabin']} cabin allows {new_free} free checked bags across {n} passengers. Setting the reservation to {target} bags means {paid} paid bag; with {upgraded['nonfree_baggages']} previously paid bags, the additional baggage charge is ${cost}. May I set that baggage total and charge {card}?",
    )
    b.say(
        "user", f"Yes, set {target} bags in total and charge the additional ${cost} using {card}."
    )
    er["total_baggages"] = target
    er["nonfree_baggages"] = paid
    if cost:
        er["payment_history"].append({"payment_id": card, "amount": cost})
    b.tool(
        "update_reservation_baggages",
        {
            "reservation_id": rid,
            "total_baggages": target,
            "nonfree_baggages": paid,
            "payment_id": card,
        },
    )
    b.say(
        "assistant",
        f"Reservation {rid} is now economy with {target} checked bags in total, including {paid} paid bag. The upgrade charge was ${delta} and the additional baggage charge was ${cost}. Flights, dates, passengers and insurance remain unchanged.",
    )
    row = b.finish(
        "Upgrade cabin first, then recompute baggage allowance from updated cabin and add requested bags.",
        [
            "two_separate_explicit_write_confirmations",
            "second_allowance_derived_from_first_native_receipt",
            "full_expected_database_and_protected_fields_checked",
            "independent_native_replay",
            "new_authored_subtask_not_original_source_task_solution",
        ],
    )
    row["metadata"]["difficulty"] = {
        "level": "pending_semantic_review",
        "basis": "two dependent writes; pipeline probe only",
    }
    return row


def user_ids(rows):
    """Visible entity exposure includes native receipts, not just call arguments."""
    ids = set()

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("user_id"), str):
                ids.add(value["user_id"])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for row in rows:
        for m in row["messages"]:
            for c in m.get("tool_calls") or []:
                a = c.get("function", c).get("arguments", {})
                if isinstance(a, str):
                    a = json.loads(a)
                visit(a)
            if m.get("role") == "tool":
                try:
                    visit(json.loads(m.get("content", "")))
                except (ValueError, TypeError):
                    pass
    return ids


def choose_flight(db, r, *, book=False):
    # Single segment reduces ambiguity; existing source supplies complex bookings.
    if len(r["flights"]) != 1 or r["flight_type"] != "one_way":
        return None
    old = r["flights"][0]
    cabin = "economy" if book else r["cabin"]
    n = len(r["passengers"])
    options = []
    for fid, f in db["flights"].items():
        if (f["origin"], f["destination"]) != (r["origin"], r["destination"]):
            continue
        for date, state in f["dates"].items():
            if date <= max("2024-05-15", old["date"]) or state.get("status") != "available":
                continue
            if state["available_seats"][cabin] < n:
                continue
            options.append((date, state["prices"][cabin], fid))
    return min(options) if options else None


def suitable(db, rid, kind):
    if kind in BASE_KINDS:
        return feasible(db, rid, kind)
    r = db["reservations"].get(rid)
    if not r or r.get("status") == "cancelled" or not r["passengers"]:
        return False
    if any(
        f["flight_number"] not in db["flights"]
        or f["date"] not in db["flights"][f["flight_number"]]["dates"]
        for f in r["flights"]
    ):
        return False
    u = db["users"][r["user_id"]]
    future = all(f["date"] > "2024-05-15" for f in r["flights"])
    cards = any(p["source"] == "credit_card" for p in u["payment_methods"].values())
    if kind == "passenger_correction":
        return future
    if kind == "flight_change":
        return (
            future and cards and r["cabin"] != "basic_economy" and choose_flight(db, r) is not None
        )
    if kind == "flight_refusal":
        return future and r["cabin"] == "basic_economy"
    if kind == "book_direct":
        return (
            future
            and cards
            and len(r["passengers"]) <= 5
            and choose_flight(db, r, book=True) is not None
        )
    if kind == "transfer_flown":
        return any(
            db["flights"][f["flight_number"]]["dates"][f["date"]].get("status")
            in ("flying", "landed")
            for f in r["flights"]
        )
    if kind == "connection_lookup":
        return future and len(r["flights"]) == 2 and r["flight_type"] == "one_way"
    return future


def build_extra(entry, rid, kind):
    adapted = adapt_record(ArealTaskRecord.model_validate(entry.task))
    if adapted.db_file_hash != entry.db_hash:
        raise ValueError("Source DB identity changed")
    b = Builder(adapted, entry, rid, kind)
    db = b.before
    r = db["reservations"][rid]
    uid = r["user_id"]
    u = db["users"][uid]
    b.expected = copy.deepcopy(db)
    er = b.expected["reservations"][rid]
    n = len(r["passengers"])
    cards = sorted(k for k, v in u["payment_methods"].items() if v["source"] == "credit_card")
    checks = ["real_tools", "independent_replay", "explicit_target_and_protected_state_checked"]
    if kind == "passenger_correction":
        old = r["passengers"][0]
        new = {**old, "first_name": old["first_name"] + "a"}
        reason = (
            f"Correct one passenger first name on {rid}, preserving count and all other details."
        )
        b.say(
            "user",
            f"My user ID is {uid}. On reservation {rid}, please correct passenger {old['first_name']} {old['last_name']}, born {old['dob']}: the first name must be {new['first_name']}. Keep their last name, birth date, every other passenger and all travel details unchanged.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        b.say(
            "assistant",
            f"I found {n} passengers. I will change only the first name of {old['first_name']} {old['last_name']}, born {old['dob']}, to {new['first_name']}. Passenger count, other passengers, flights and payment details will stay unchanged. May I proceed?",
        )
        b.say("user", "Yes, make only that first-name correction.")
        er["passengers"][0] = new
        b.tool(
            "update_reservation_passengers", {"reservation_id": rid, "passengers": er["passengers"]}
        )
        b.say(
            "assistant",
            f"The first name is now {new['first_name']}. The other passenger details and the reservation are otherwise unchanged.",
        )
        checks += [
            "explicit_identity_from_user",
            "preserved_passenger_multiset_except_requested_field",
        ]
    elif kind == "flight_refusal":
        reason = f"Request a date change without changing basic-economy cabin; accept policy refusal and preserve {rid}."
        b.say(
            "user",
            f"My user ID is {uid}. For reservation {rid}, can I move my flights to a later date while keeping basic economy? I do not want a cabin upgrade, cancellation or a new booking. If the policy forbids it, explain and leave my booking unchanged.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        b.say(
            "assistant",
            "This reservation is basic economy. The policy does not allow changing its flights. A cabin-only change is a separate permitted action before departure, but you declined an upgrade. I have left your reservation unchanged.",
        )
        b.say("user", "Understood; keep my original booking. That is all.")
        b.say(
            "assistant",
            "Your original booking remains unchanged. No cancellation or new booking was made.",
        )
        checks += ["correct_refusal", "no_unrequested_workaround"]
    elif kind == "transfer_flown":
        reason = f"Cancellation assistance for already-flown reservation {rid}; transfer with no cancellation."
        b.say(
            "user",
            f"My user ID is {uid}. I would like to cancel reservation {rid} because my plans changed. Please check whether that is still possible. If this is outside your capabilities, please transfer me.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        for f in r["flights"]:
            b.tool("get_flight_status", {"flight_number": f["flight_number"], "date": f["date"]})
        b.say(
            "assistant",
            "At least part of this reservation has already flown. I cannot cancel it with these tools under the policy, so I will transfer you for assistance.",
        )
        b.tool(
            "transfer_to_human_agents",
            {
                "summary": f"User {uid} requests cancellation of {rid} for changed plans; at least one segment has already flown. No cancellation or refund performed."
            },
        )
        b.say("assistant", "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.")
        checks += [
            "verified_flown_status",
            "no_illegal_cancellation",
            "transfer_tool_before_required_message",
        ]
    elif kind == "airport_lookup":
        reason = "Ask for airport codes for a city pair before choosing a flight; do not book."
        names = {
            "SFO": "San Francisco",
            "JFK": "New York",
            "LAX": "Los Angeles",
            "ORD": "Chicago",
            "DFW": "Dallas",
            "DEN": "Denver",
            "SEA": "Seattle",
            "ATL": "Atlanta",
            "MIA": "Miami",
            "BOS": "Boston",
            "PHX": "Phoenix",
            "IAH": "Houston",
            "LAS": "Las Vegas",
            "MCO": "Orlando",
            "EWR": "Newark",
            "CLT": "Charlotte",
            "MSP": "Minneapolis",
            "DTW": "Detroit",
            "PHL": "Philadelphia",
            "LGA": "LaGuardia",
        }
        a, z = r["origin"], r["destination"]
        b.say(
            "user",
            f"Which airport codes does your service list for {names[a]} and {names[z]}? I only need the codes; do not search for tickets or create a booking.",
        )
        listing = b.tool("list_all_airports", {})
        if not all(any(x["iata"] == code for x in listing) for code in (a, z)):
            raise ValueError("Airport code missing")
        b.say(
            "assistant",
            f"{names[a]} is listed as {a}, and {names[z]} as {z}. No booking has been made.",
        )
        checks += ["read_only_fact_grounding"]
    elif kind == "connection_lookup":
        date = r["flights"][0]["date"]
        origin = r["origin"]
        destination = r["destination"]
        reason = f"Inspect one-stop options from {origin} to {destination} on {date}; no booking."
        b.say(
            "user",
            f"Please check one-stop flights from {origin} to {destination} on {date} for {n} passengers in economy. Tell me one available itinerary and its total fare. I am only checking options; do not book anything.",
        )
        options = b.tool(
            "search_onestop_flight", {"origin": origin, "destination": destination, "date": date}
        )
        options = [o for o in options if all(f["available_seats"]["economy"] >= n for f in o)]
        if not options:
            raise ValueError("No verified one-stop option")
        one = min(options, key=lambda o: (sum(f["prices"]["economy"] for f in o), str(o)))
        total = sum(f["prices"]["economy"] for f in one) * n
        b.tool(
            "calculate",
            {"expression": f"({one[0]['prices']['economy']}+{one[1]['prices']['economy']})*{n}"},
        )
        b.say(
            "assistant",
            f"One available economy itinerary is {one[0]['flight_number']} on {one[0]['date']}, {origin} to {one[0]['destination']}, then {one[1]['flight_number']} on {one[1]['date']} to {destination}. The combined fare is ${total} for {n} passengers, before any optional extras. I have not booked it.",
        )
        checks += ["native_connection_results", "seats_checked", "no_unrequested_booking"]
    elif kind in ("flight_change", "book_direct"):
        book = kind == "book_direct"
        date, price, fid = choose_flight(db, r, book=book)
        cabin = "economy" if book else r["cabin"]
        card = cards[-1]
        _ = db["flights"][fid]
        old = r["flights"][0]
        total = price * n
        if book:
            reason = f"Book one-way {r['origin']} to {r['destination']} on {date}, specified flight {fid}, no baggage or insurance."
            b.say(
                "user",
                f"My user ID is {uid}. I want a NEW one-way economy booking from {r['origin']} to {r['destination']} on {date}, flight {fid}, for these passengers: {json.dumps(r['passengers'])}. I need no checked bags. Leave all existing reservations unchanged.",
            )
            b.tool("get_user_details", {"user_id": uid})
            options = b.tool(
                "search_direct_flight",
                {"origin": r["origin"], "destination": r["destination"], "date": date},
            )
            if not any(f["flight_number"] == fid for f in options):
                raise ValueError("Selected flight missing")
            b.say(
                "assistant",
                "Would you like travel insurance? It costs $30 per passenger and covers cancellation for health or weather reasons.",
            )
            b.say("user", f"No insurance. Use my stored credit card {card}.")
            b.tool("calculate", {"expression": f"{price}*{n}"})
            b.say(
                "assistant",
                f"Please confirm the new one-way economy booking: {fid} on {date}, {r['origin']} to {r['destination']}, {n} passengers {json.dumps(r['passengers'])}, no checked bags and no insurance. The total is ${total}, paid using {card}. Shall I book it?",
            )
            b.say("user", "Yes, book exactly those details at that total.")
            args = dict(
                user_id=uid,
                origin=r["origin"],
                destination=r["destination"],
                flight_type="one_way",
                cabin=cabin,
                flights=[dict(flight_number=fid, date=date)],
                passengers=r["passengers"],
                payment_methods=[dict(payment_id=card, amount=total)],
                total_baggages=0,
                nonfree_baggages=0,
                insurance="no",
            )
            receipt = b.tool("book_reservation", args)
            newid = receipt["reservation_id"]
            if newid in db["reservations"]:
                raise ValueError("New reservation ID overwrote old state")
            b.expected["reservations"][newid] = dict(
                reservation_id=newid,
                user_id=uid,
                origin=r["origin"],
                destination=r["destination"],
                flight_type="one_way",
                cabin=cabin,
                flights=[
                    dict(
                        flight_number=fid,
                        date=date,
                        origin=r["origin"],
                        destination=r["destination"],
                        price=price,
                    )
                ],
                passengers=copy.deepcopy(r["passengers"]),
                payment_history=[dict(payment_id=card, amount=total)],
                created_at="2024-05-15T15:00:00",
                total_baggages=0,
                nonfree_baggages=0,
                insurance="no",
                status=None,
            )
            b.expected["users"][uid]["reservations"].append(newid)
            b.expected["flights"][fid]["dates"][date]["available_seats"][cabin] -= n
            b.say(
                "assistant",
                f"Booked successfully: reservation {newid}, flight {fid} on {date}. The ${total} payment used {card}. There is no insurance or checked baggage, and your existing reservations are unchanged.",
            )
            checks += ["all_booking_fields_confirmed", "new_reservation_payment_and_seats_verified"]
        else:
            delta = total - old["price"] * n
            reason = f"Change only {rid} to specified flight {fid} on {date}, same route/cabin/passengers, using {card}."
            b.say(
                "user",
                f"My user ID is {uid}. Please change reservation {rid} to flight {fid} on {date}. Keep the same route, cabin, passengers, bags and insurance. I want to use {card} for the fare difference. Check the price before updating.",
            )
            b.tool("get_reservation_details", {"reservation_id": rid})
            b.tool("get_user_details", {"user_id": uid})
            b.tool(
                "search_direct_flight",
                {"origin": r["origin"], "destination": r["destination"], "date": date},
            )
            b.tool("calculate", {"expression": f"({price}-{old['price']})*{n}"})
            description = (
                f"an additional charge of ${delta}" if delta >= 0 else f"a refund of ${-delta}"
            )
            b.say(
                "assistant",
                f"The requested flight {fid} on {date} is available in {cabin} for all {n} passengers. The old airfare was ${old['price'] * n}, and the new airfare is ${total}, giving {description} using {card}. All other booking details stay unchanged. May I update reservation {rid}?",
            )
            b.say(
                "user",
                f"Yes, change to {fid} on {date} with that fare difference and payment method.",
            )
            er["flights"] = [
                dict(
                    flight_number=fid,
                    date=date,
                    origin=r["origin"],
                    destination=r["destination"],
                    price=price,
                )
            ]
            if delta:
                er["payment_history"].append(dict(payment_id=card, amount=delta))
            b.tool(
                "update_reservation_flights",
                dict(
                    reservation_id=rid,
                    cabin=cabin,
                    flights=[dict(flight_number=fid, date=date)],
                    payment_id=card,
                ),
            )
            b.say(
                "assistant",
                f"Reservation {rid} is now on {fid} on {date}. The update produced {description} using {card}; the other booking details are unchanged.",
            )
            checks += [
                "single_flight_target_verified",
                "fare_delta_from_observed_old_price",
                "profile_payment_verified",
            ]
    else:
        raise ValueError(kind)
    return b.finish(reason, checks)


def prepare(output):
    old_train = user_ids(read_rows("data/sft/staged_v2_A100_20260925/train.jsonl"))
    old_dev = user_ids(read_rows("data/sft/staged_v2_A100_20260925/validation.jsonl"))
    selection_users = set()
    for e in read_manifest("data/manifests/areal_airline_selection_seed42.jsonl"):
        record = ArealTaskRecord.model_validate(e.task)
        db = raw_database(str(record.resolve_db_path(AREAL_DB_ROOT)))
        text = json.dumps(record.user_scenario) + json.dumps(record.evaluation_criteria)
        selection_users.update(uid for uid in db["users"] if uid in text)
    protected = old_dev | selection_users
    # Exclude active formal RL reference entities from the development split.
    for e in read_manifest(
        "data/manifests/rl_curriculum50_20260912/areal_airline_train_seed42.jsonl"
    ):
        raw = raw_database(
            str(ArealTaskRecord.model_validate(e.task).resolve_db_path(AREAL_DB_ROOT))
        )
        protected.update(
            raw["reservations"][rid]["user_id"]
            for rid in task_feature(e.task)["reservation_ids"]
            if rid in raw["reservations"]
        )
        protected.update(
            a["arguments"]["user_id"]
            for a in e.task["evaluation_criteria"].get("actions", [])
            if a.get("arguments", {}).get("user_id")
        )
    pool = {}
    for entry in read_manifest("data/manifests/areal_airline_train_seed42.jsonl"):
        if entry.split != "train" or entry.task.get("initial_state"):
            continue
        record = ArealTaskRecord.model_validate(entry.task)
        db = raw_database(str(record.resolve_db_path(AREAL_DB_ROOT)))
        for rid in task_feature(entry.task)["reservation_ids"]:
            r = db["reservations"].get(rid)
            if not r or r["user_id"] in protected:
                continue
            pool.setdefault(
                (entry.db_hash, rid),
                dict(task_id=entry.task_id, rid=rid, user_id=r["user_id"], db_hash=entry.db_hash),
            )
    entries = {
        e.task_id: e for e in read_manifest("data/manifests/areal_airline_train_seed42.jsonl")
    }
    eligible = {}
    for kind in KINDS:
        items = []
        for x in pool.values():
            e = entries[x["task_id"]]
            db = raw_database(
                str(ArealTaskRecord.model_validate(e.task).resolve_db_path(AREAL_DB_ROOT))
            )
            if suitable(db, x["rid"], kind):
                items.append(x)
        eligible[kind] = sorted(
            items, key=lambda x: sha256_json([42, kind, x["user_id"], x["rid"]])
        )
    # Freeze one unseen-user probe per branch BEFORE choosing training variants.
    dev = []
    dev_users = set()
    used = set()
    for kind in sorted(
        KINDS,
        key=lambda k: len({x["user_id"] for x in eligible[k] if x["user_id"] not in old_train}),
    ):
        options = [
            x
            for x in eligible[kind]
            if x["user_id"] not in old_train and x["user_id"] not in dev_users
        ]
        if not options:
            raise ValueError("Insufficient distinct unseen-user probe family: " + kind)
        x = options[0]
        dev.append(dict(kind=kind, split="validation", **x))
        dev_users.add(x["user_id"])
        used.add(x["rid"])
    train = []
    counts = Counter()
    for kind in sorted(KINDS, key=lambda k: len(eligible[k])):
        selected_users = set()
        for x in sorted(
            eligible[kind], key=lambda x: (counts[x["user_id"]], sha256_json([42, kind, x["rid"]]))
        ):
            if x["user_id"] in dev_users | selected_users or x["rid"] in used:
                continue
            train.append(dict(kind=kind, split="train", **x))
            used.add(x["rid"])
            selected_users.add(x["user_id"])
            counts[x["user_id"]] += 1
            if len(selected_users) == 3:
                break
        if not selected_users:
            raise ValueError("No protected training family for: " + kind)
    plan = dict(
        version="grounded_entity_split_v1",
        requested_train_per_kind=3,
        actual_train_counts=dict(Counter(x["kind"] for x in train)),
        train=train,
        validation=dev,
        heldout_users=sorted(dev_users),
        protected_users=sorted(protected),
        historical_train_users=sorted(old_train),
        source_sha256=sha256_file(__file__),
        base_builder_sha256=sha256_file(Path(__file__).with_name("grounded_gap_pilot.py")),
        final_or_reserve_read=False,
        notes=[
            "Entity split; templates are shared, not independent task-family generalization.",
            "New source records using heldout users must be excluded from training before freeze.",
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    return plan, entries


def run(args):
    import yaml

    plan, entries = prepare(args.output)
    schemas = [
        x["tool_schema"]
        for x in yaml.safe_load(Path("configs/envs/tool_config.yaml").read_text())["tools"]
    ]
    results = []
    errors = []
    for item in plan["train"] + plan["validation"]:
        try:
            builder = build_case if item["kind"] in BASE_KINDS else build_extra
            row = builder(entries[item["task_id"]], item["rid"], item["kind"])
            if row is None:
                raise ValueError("Frozen selection failed live feasibility")
            row["metadata"].update(
                source_dialog_id=f"repair_{item['kind']}_{item['task_id']}_{item['rid']}",
                split=item["split"],
                entity_group=item["user_id"],
                construction_kind=item["kind"],
                plan_sha256=sha256_file(args.output / "plan.json"),
                status="executed_candidate_needs_final_review",
                difficulty={"level": "unrated", "basis": "branch rubric pending semantic review"},
            )
            failures = audit_tool_calls(row, schemas) + ordered_tool_receipts(row)
            if failures:
                raise ValueError(failures)
            with (args.output / (item["split"] + ".jsonl")).open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(item)
        except Exception as exc:
            errors.append(dict(**item, error=str(exc)[:500]))
        print(
            json.dumps(
                {
                    "done": len(results),
                    "errors": len(errors),
                    "kind": item["kind"],
                    "split": item["split"],
                }
            ),
            flush=True,
        )
    report = dict(
        planned=len(plan["train"]) + len(plan["validation"]),
        completed=len(results),
        splits=dict(Counter(x["split"] for x in results)),
        errors=errors,
        counts=dict(Counter(x["kind"] for x in results)),
        gpu_hours=0,
        new_api_calls=0,
        training_frozen=False,
        input_plan_sha256=sha256_file(args.output / "plan.json"),
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--curriculum-phase", choices=["foundation", "constraints", "strategy"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--curriculum-finalize", action="store_true")
    p.add_argument("--reuse-curriculum", type=Path)
    args = p.parse_args()
    finalize_curriculum(args.output) if args.curriculum_finalize else run_curriculum(
        args
    ) if args.curriculum_phase else run(args)
