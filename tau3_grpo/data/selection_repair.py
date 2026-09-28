"""Version exposed selection tasks against their own DB and written policy.

No model trajectories, rewards, final tasks, or training examples are inputs.
The output is a derived development set, not an unchanged upstream benchmark.
Task/reference consistency and live communication grading have separate gates.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from tau3_grpo.analysis.prepare_outcome_contract import (
    action_policy_flags,
    repair_known_reference_actions,
)
from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.evaluation.outcome_contract import execute_actions, outcome_hash, runtime_identity
from tau3_grpo.paths import AREAL_DB_ROOT
from tau3_grpo.utils.hashing import sha256_file, sha256_json

VERSION = "selection60_consistency_repair_v1"
DATES = [f"2024-05-{i:02}" for i in range(16, 21)]
WRITE_NAMES = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
}


def action(name, **arguments):
    return dict(name=name, arguments=arguments)


def candidates(
    db, origin, destination, dates, cabin, passengers, lo="00:00:00", hi="23:59:59", flight=None
):
    out = []
    for fid, f in db["flights"].items():
        if (f["origin"], f["destination"]) != (origin, destination) or (flight and fid != flight):
            continue
        for date in dates:
            s = f["dates"].get(date)
            if not s or s["status"] != "available" or s["available_seats"][cabin] < passengers:
                continue
            if not lo <= f["scheduled_departure_time_est"] <= hi:
                continue
            out.append(
                dict(
                    flight_number=fid,
                    date=date,
                    price=s["prices"][cabin],
                    departure=f["scheduled_departure_time_est"],
                )
            )
    return out


def choose(options, rank):
    if not options:
        return None

    def key(x):
        if rank == "cheapest":
            primary = (x["price"], x["date"], x["departure"])
        elif rank == "premium":
            primary = (-x["price"], x["date"], x["departure"])
        elif rank == "latest":
            primary = (x["date"], -int(x["departure"].replace(":", "")))
        else:
            primary = (x["date"], x["departure"])
        return (*primary, x["flight_number"])

    return min(options, key=key)


def fs(flights):
    return [dict(flight_number=f["flight_number"], date=f["date"]) for f in flights]


def lookup(uid, ids):
    return [action("get_user_details", user_id=uid)] + [
        action("get_reservation_details", reservation_id=rid) for rid in ids
    ]


def searches(origin, destination, dates):
    return [
        action("search_direct_flight", origin=origin, destination=destination, date=d)
        for d in dates
    ]


def set_scene(record, db, uid, reason, body):
    user = db["users"][uid]
    record.description = {"purpose": reason}
    record.user_scenario["instructions"] = {
        "domain": "airline",
        "known_info": json.dumps(
            dict(
                user_id=uid,
                name=user["name"],
                membership=user["membership"],
                payment_methods=list(user["payment_methods"]),
                existing_reservations=user["reservations"],
            ),
            ensure_ascii=False,
        ),
        "reason_for_call": reason,
        "task_instructions": body.strip()
        + """\n\nCOMMON USER BEHAVIOR:
The simulated current date/time is May 15, 2024, 15:00 EST. Use the exact dates stated above.
Provide your user ID when asked. Do not disclose reservation IDs or precise facts before asked
unless the task explicitly says to start with them. Answer clarification questions consistently.
Before any write, wait for the agent to describe the specific change, fare difference and payment;
say yes only when these satisfy your stated goals. Do not invent new passenger identities or goals.
Accept a supported policy refusal; never turn an assistant's unsupported claim into a new request.
Do not ask for a human transfer unless this task explicitly requests one. After all goals are addressed,
end the conversation. Do not force the agent to use a particular tool or recite tool names.
""",
    }


def patch_record(record, db):
    r = copy.deepcopy(record)
    tid = int(r.id.split("_")[-1])
    notes = []
    checks = []
    facts = {}
    actions = copy.deepcopy(r.evaluation_criteria.get("actions") or [])
    ins = r.user_scenario["instructions"]

    def scene(uid, reason, body):
        set_scene(r, db, uid, reason, body)

    def note(message):
        notes.append(message)

    # A shared reservation appears in several independently authored tasks. Keep
    # their different requested dates and priorities; never choose from scores.
    if tid in (14, 23, 126, 691, 1106, 651, 864):
        booking = db["reservations"]["HKEG34"]
        uid = booking["user_id"]
        n = len(booking["passengers"])
        dates = {
            14: ["2024-05-28"],
            23: ["2024-05-20"],
            126: ["2024-05-17"],
            691: ["2024-05-18"],
            1106: ["2024-05-16", "2024-05-18"],
            651: DATES,
            864: DATES,
        }[tid]
        cabin = "economy" if tid in (651, 864) else "business"
        rank = "cheapest" if tid == 651 else ("latest" if tid in (23, 864) else "premium")
        lo, hi = ("00:00:00", "23:59:59") if tid in (651, 1106, 864) else ("08:00:00", "21:00:00")
        bag_count = 1 if tid in (651, 864) else 2
        options = candidates(db, "DEN", "LAS", dates, cabin, n, lo, hi)
        selected = choose(options, rank)
        card = "credit_card_1955700" if tid == 126 else "credit_card_4421486"
        assert card in db["users"][uid]["payment_methods"]
        budget = (
            "Net additional cost must not exceed $200; a refund satisfies this cap."
            if tid in (14, 23, 126, 691)
            else "Explain any fare difference before seeking confirmation."
        )
        ranking = {
            "cheapest": "Choose the lowest total economy airfare across the specified dates.",
            "premium": "By premium/best I mean the highest quoted business airfare among available flights on the specified dates.",
            "latest": "Use the earliest date with a qualifying flight, then the latest departure on that date.",
        }[rank]
        fallback = (
            "If no business option meets the cap, ask about economy at the same dates and time window; I accept the best qualifying economy option within the cap."
            if tid in (23, 126, 691)
            else ""
        )
        body = f"""You are Mia Li, calling about reservation HKEG34. It is a one-way DEN to LAS business reservation on May 27.
The booked passenger is Amelia Ahmed; you are the account holder. Do not change the passenger name or date of birth.
Move this reservation to one direct flight on {", ".join(dates)}, in {cabin}, departing between {lo} and {hi}.
These exact dates take priority over vague words such as tomorrow or later. There is no return flight.
{ranking} Break fare ties by earlier date, then earlier departure; break remaining ties by flight number.
{budget} {fallback}
Use {card} for both a charge or a refund after I confirm it; do not switch cards merely because the original booking used another card.
After a successful flight change, ask to add {bag_count} checked bag(s) to the existing total and confirm the allowance and any cost.
Keep all other reservation details unchanged, including insurance and passenger identities.
If no flight satisfies the date, time and budget constraints, explain this and leave the entire reservation, including baggage, unchanged.
I may initially believe Gold status waives fees, but accept the actual policy and a correctly explained fare difference.
I do not insist on a named flight. The route, dates, time window and ranking above determine my choice."""
        scene(
            uid,
            f"Change HKEG34 on {', '.join(dates)} using {rank} priority, then add {bag_count} bag(s) if changed.",
            body,
        )
        actions = lookup(uid, ["HKEG34"]) + searches("DEN", "LAS", dates)
        if selected:
            delta = (selected["price"] - sum(f["price"] for f in booking["flights"])) * n
            if tid in (14, 23, 126, 691) and delta > 200:
                raise ValueError("Budget fallback must be explicitly compiled before approval")
            actions += [
                action(
                    "update_reservation_flights",
                    reservation_id="HKEG34",
                    cabin=cabin,
                    flights=fs([selected]),
                    payment_id=card,
                ),
                action(
                    "update_reservation_baggages",
                    reservation_id="HKEG34",
                    total_baggages=booking["total_baggages"] + bag_count,
                    nonfree_baggages=0,
                    payment_id=card,
                ),
            ]
        facts = dict(
            dates=dates, clock=[lo, hi], rank=rank, selected=selected, requested_bags=bag_count
        )
        checks = [
            "Correct date/time and price ranking; preserve the booked passenger, not the account holder.",
            "Explain no feasible flight and perform no writes if the hard constraints have no solution.",
        ]
        note(
            "Resolve inconsistent relative dates, incorrect named flights, one-way/return confusion, cabin and payment priority; align the question time window with the legitimate reference departure (afternoon is acceptable for 864)."
        )

    elif tid in (1143, 925, 216, 630, 899, 490, 976):
        booking = db["reservations"]["GCZ58I"]
        uid = booking["user_id"]
        n = len(booking["passengers"])
        assert (
            n == 2
            and booking["flights"][1]["origin"] == "LAX"
            and booking["flights"][1]["destination"] == "SFO"
        )
        modifying = tid in (490, 976)
        dates = DATES if tid == 490 else DATES[:3]
        body = """You are Lucas Sanchez. Ask about reservation GCZ58I: a business round trip with two passengers.
The May 11 SFO to LAX segment HAT273 was cancelled. The May 15 return HAT249 is LAX to SFO and is delayed.
Ask why the disruption occurred and explicitly request compensation. The tools may provide only status,
not an operational cause. Accept an honest explanation that the cause is unavailable; do not require a fabricated reason.
Request policy-eligible travel certificates and accept the policy amount after it is explained.
Do not treat a goodwill certificate as a cash refund or permission to cancel an unrelated booking."""
        actions = lookup(uid, ["GCZ58I"]) + [
            action("get_flight_status", flight_number=f["flight_number"], date=f["date"])
            for f in booking["flights"]
        ]
        if modifying:
            selected = choose(candidates(db, "LAX", "SFO", dates, "business", n), "premium")
            assert selected
            body += f"""\nMove ONLY the delayed return from LAX to SFO to a direct business flight on {", ".join(dates)}.
Keep the original cancelled outbound entry, passengers, bags and insurance unchanged; do not cancel/rebook the whole reservation.
By premium I mean the highest quoted business airfare with seats for both passengers, ties by earlier date then departure.
Initially say you think there are three passengers; when the agent verifies the two actual passengers, accept the correction.
Initially prefer certificate_6221489. If told certificates cannot pay for an existing-reservation change,
explicitly choose stored gift_card_6216249 for the fare difference or refund and confirm the amount.
Request eligible compensation for the cancellation and for the delay AFTER the return has been successfully changed."""
            actions += searches("LAX", "SFO", dates) + [
                action(
                    "update_reservation_flights",
                    reservation_id="GCZ58I",
                    cabin="business",
                    flights=fs([booking["flights"][0], selected]),
                    payment_id="gift_card_6216249",
                )
            ]
            amount = 150 * n
            facts = dict(
                selected=selected, dates=dates, compensation=amount, delay_after_change=True
            )
        else:
            body += "\nDo not change or cancel any flight. Accept an explanation that delay compensation needs a completed change/cancellation; request only the cancelled-flight certificate for this call."
            amount = 100 * n
            facts = dict(compensation=amount, no_booking_change=True)
        actions.append(action("send_certificate", user_id=uid, amount=amount))
        scene(
            uid,
            "Explain verified flight disruptions and issue eligible compensation"
            + (
                " after changing only the delayed return."
                if modifying
                else " without changing the reservation."
            ),
            body,
        )
        checks = [
            "Explain verified cancellation/delay status without inventing an operational cause.",
            "Explain compensation eligibility and amount; do not pay a delayed-flight amount before a successful change/cancellation.",
        ]
        note(
            "Correct return direction, compensation conditions/amount/date, and certificate-to-stored-gift-card fallback; remove contradictory flight names and unsupported cause requirement."
        )

    elif tid == 548:
        booking = db["reservations"]["MM67S8"]
        uid = booking["user_id"]
        n = len(booking["passengers"])
        selected = choose(candidates(db, "CLT", "BOS", DATES, "business", n), "earliest")
        assert selected
        scene(
            uid,
            "Request cancellation for changed plans; if ineligible, keep the old booking and buy a separate earlier business trip.",
            """You are Chen Hernandez. Ask to cancel MM67S8, your May21 CLT to BOS basic-economy booking, because a meeting moved earlier.
Accept that insurance does not cover this change-of-plans reason. If cancellation is ineligible, explicitly agree to keep
MM67S8 unchanged and pay for a SEPARATE new one-way business reservation CLT to BOS, earliest available direct departure
from May16 through May20 (earliest date, then earliest time). This is an additional booking, not a refund or modification
of MM67S8. The new passengers are Chen Hernandez born July19 1965 and Juan Davis born April26 1970.
Request zero checked bags and no travel insurance for the new reservation. Confirm its complete price before paying
with credit_card_8453507. Do not add bags then remove them. Do not invent a health or weather cancellation reason.""",
        )
        actions = [a for a in actions if a["name"] != "update_reservation_baggages"]
        next(a for a in actions if a["name"] == "book_reservation")["arguments"][
            "total_baggages"
        ] = 0
        facts = dict(selected=selected, cancelled=False, new_booking=True, bags_at_booking=0)
        checks = [
            "Explain cancellation ineligibility and get explicit consent for a separate paid booking while the old one remains active."
        ]
        note(
            "Align the question with a separate new business booking; replace illegal four-to-zero baggage reduction with zero bags at booking."
        )

    elif tid == 393:
        uid = "raj_young_5c9945"
        selected = choose(
            candidates(db, "LGA", "CLT", ["2024-05-16"], "economy", 1, hi="06:59:59"), "earliest"
        )
        assert (
            selected
            and selected["flight_number"] == "OG3IHSH"
            and selected["departure"] == "06:00:00"
        )
        scene(
            uid,
            "Book the earliest direct LGA to CLT economy departure before 07:00 on May16.",
            """You are Raj Young, born November2 1978. Ask for the earliest available direct one-way economy flight
LGA to CLT on May16, 2024, departing strictly BEFORE 07:00. A 06:00 departure is acceptable. Request travel insurance
and zero checked bags, pay with credit_card_1000001 after confirming the total. Do not change existing reservations.
If no flight satisfies the stated requirements, accept an explanation and no new booking.""",
        )
        facts = dict(
            selected=selected, departure_deadline="07:00:00", old_departure_deadline="05:00:00"
        )
        checks = ["Quote and obtain consent for the flight, insurance and full price."]
        note(
            "User-authorized question correction: before05:00 becomes before07:00, making the original06:00 reference booking valid."
        )

    elif tid == 293:
        uid = "ivan_carter_f83877"
        owned = {rid: x for rid, x in db["reservations"].items() if x["user_id"] == uid}
        now = datetime.fromisoformat("2024-05-15T15:00:00")
        cancel = [
            rid
            for rid, x in owned.items()
            if any(f["date"] == "2024-05-18" for f in x["flights"])
            and 0 <= (now - datetime.fromisoformat(x["created_at"])).total_seconds() <= 86400
        ]
        assert cancel == ["A92BCE"] or set(cancel) == {"A92BCE"}
        scene(
            uid,
            "Cancel only May18 departures booked within the last24 hours.",
            """You are Ivan Carter and do not remember reservation IDs. Your only request is to cancel every trip
DEPARTING ON MAY18 that was booked within the last24 hours, because your plans changed. The current time is May15,2024
at15:00 EST. Ask the agent to inspect your bookings and identify the exact eligible set, then confirm cancellation.
Do not cancel trips departing May19 or bookings outside that creation-time window. You have no baggage, passenger
replacement, or new-booking request in this call. Accept refunds to the original payment methods.""",
        )
        facts = dict(
            cancel_ids=sorted(cancel),
            scope_narrowed=True,
            removed_unresolvable_subgoals=["IAH-SFO baggage", "family passenger replacement"],
        )
        checks = [
            "Identify bookings using both the May18 departure filter and the24-hour creation-time filter before confirmation."
        ]
        note(
            "Narrow the question to May18 and remove nonexistent extra target trips so the original single cancellation is complete; this changes task scope/difficulty."
        )

    elif tid == 1073:
        uid = "ella_garcia_8275b1"
        scene(
            uid,
            "Check adult-only bookings, cancel the identified May22 business trip, and request human review of a different uninsured family trip.",
            """You are Ella Garcia and do not know reservation IDs. First ask whether all your May22 afternoon
flights (departing12:00–18:00) list adults only. If they already do, accept no passenger update. Never change passenger count.
Separately, explicitly request cancellation of your May22 DEN to MIA BUSINESS trip departing17:10 with Ella and Carlos,
because your plans changed. Confirm that exact booking and its refund; do not cancel the May22 economy trip at13:45.
Then ask to cancel the May20 DEN to MIA economy family trip with Ella, Carlos and Liam due to illness. You initially
believe it was insured. Accept the database correction and explanation that its lack of insurance makes cancellation
ineligible. Then explicitly request transfer to a human to discuss alternatives for this May20 family reservation.
Do not claim that transfer guarantees a refund, and do not change this family reservation.""",
        )
        facts = dict(cancel_ids=["9F989F"], refused_reservation="9814DE", explicit_transfer=True)
        checks = [
            "Explain already-adult passenger lists; distinguish the requested business cancellation from the ineligible family cancellation.",
            "Obtain cancellation confirmation and transfer only after the explicit human-help request.",
        ]
        note(
            "Add the distinct business-trip cancellation and explicit human-review request to the question, matching the legitimate original reference."
        )

    elif tid == 33:
        uid = "noah_lopez_2532"
        scene(
            uid,
            "Verify a suspected passenger surname typo against the requested passport spelling.",
            """You are Noah Lopez, account noah_lopez_2532, calling about passenger Chen Johansson on reservation FPJKQM.
You think you previously saw the surname Johannson. Your requested passport spelling is Johansson and Chen's DOB is
August 16, 1988. Ask the agent to verify and correct ONLY the surname if it differs. If it is already Johansson,
accept that no correction is needed. Do not request a DOB change or replace Chen with the account holder.""",
        )
        actions = lookup(uid, ["FPJKQM"])
        facts = dict(no_writes=True, passenger_dob="1988-08-16")
        checks = [
            "State the current surname already matches Johansson; do not claim to have corrected a nonexistent typo."
        ]
        note(
            "Resolve wrong DOB and forced-success instructions for a surname that already matches."
        )

    elif tid == 435:
        uid = "timothy_muller_f69e49"
        scene(
            uid,
            "Identify the insured late-May trip and cancel it due to illness after clarification.",
            """You are Timothy Muller. You are ill and want to cancel the insured late-May trip but initially misremember
the destination as Florida. You do not know the reservation ID. When the agent reports there is no matching Florida trip
and offers actual reservation details, clarify that you mean the PHX to DTW trip on May 24, booked with travel insurance.
Confirm that exact booking after the agent identifies it. Cancellation reason remains illness. Do not cancel the
uninsured LAX to EWR trips or the unrelated business trip. Accept the refund to the original payment methods.""",
        )
        actions = lookup(uid, db["users"][uid]["reservations"]) + [
            action("cancel_reservation", reservation_id="501B40")
        ]
        facts = dict(cancel_ids=["501B40"], clarified_route=["PHX", "DTW"])
        checks = [
            "Obtain clarification identifying the insured PHX–DTW booking before cancellation."
        ]
        note(
            "Preserve mistaken-memory disambiguation while allowing correction to the actual insured route."
        )

    elif tid == 363:
        booking = db["reservations"]["OR3ZU0"]
        uid = booking["user_id"]
        card = "credit_card_4465695"
        assert card in db["users"][uid]["payment_methods"]
        selected = [
            choose(
                candidates(db, o, d, [date], "basic_economy", len(booking["passengers"])),
                "earliest",
            )
            for o, d, date in [("CLT", "DEN", "2024-05-26"), ("DEN", "CLT", "2024-05-27")]
        ]
        assert all(selected)
        scene(
            uid,
            "After a passenger-removal refusal, downgrade both travelers to the earliest available direct flights on the same dates.",
            f"""You are Isabella Khan. Initially request removal of one passenger from OR3ZU0 and their fare refund.
If passenger removal is prohibited, accept keeping BOTH passengers and choose basic economy on the earliest available
direct CLT to DEN departure on May26 and earliest available direct DEN to CLT return on May27, with seats for both.
Changing the flight numbers is acceptable; dates, route, passengers, bags and insurance stay unchanged. For departures
at the same time, prefer the lower fare, then flight number. Ask for the fare difference before consenting.
Explicitly select stored card {card} to receive the refund or pay a difference. It need not be the original payment card.""",
        )
        facts = dict(selected=selected, payment_id=card, passenger_count=len(booking["passengers"]))
        checks = [
            "Explain passenger-count restriction and obtain consent for both travelers changing cabin/flights and the selected refund card."
        ]
        note(
            "Align earliest same-date departures and explicit refund-card choice with the original reference; remove conflicting same-flight/original-payment conditions."
        )

    elif tid == 466:
        booking = db["reservations"]["K5V7FX"]
        uid = booking["user_id"]
        dates = ["2024-05-21", "2024-05-29"]
        selected = []
        for origin, dest, date in [("MIA", "DEN", dates[0]), ("DEN", "MIA", dates[1])]:
            selected.append(
                choose(
                    candidates(
                        db, origin, dest, [date], "basic_economy", len(booking["passengers"])
                    ),
                    "latest",
                )
            )
        assert all(selected)
        card = "credit_card_3520382"
        assert card in db["users"][uid]["payment_methods"]
        scene(
            uid,
            "After a passenger-removal refusal, choose latest same-date departures for both directions in basic economy.",
            f"""You are Omar Lee. Initially ask to remove one traveler from your MIA to DEN round trip and reveal K5V7FX after clarification.
If passenger removal is prohibited, accept keeping both passengers and request basic economy on the latest available
MIA to DEN departure on May21 and the latest DEN to MIA return on May29. Do not change the dates or confuse the two directions.
Keep bags and insurance unchanged. Ask for the net fare difference and explicitly choose stored credit card {card}
for the refund or charge. You authorize this card even though the original payment was a gift card.""",
        )
        actions = (
            lookup(uid, ["K5V7FX"])
            + searches("MIA", "DEN", [dates[0]])
            + searches("DEN", "MIA", [dates[1]])
            + [
                action(
                    "update_reservation_flights",
                    reservation_id="K5V7FX",
                    cabin="basic_economy",
                    flights=fs(selected),
                    payment_id=card,
                )
            ]
        )
        facts = dict(selected=selected, explicit_refund_card=card)
        checks = [
            "Explain passenger-count refusal, fare difference and the explicitly selected credit-card refund."
        ]
        note(
            "Resolve both-legs-same-direction error, undefined date range, and explicitly choose the reference Mastercard instead of requiring the original gift card."
        )

    elif tid in (740, 792):
        rid = "P2YRA6" if tid == 740 else "SANMNF"
        booking = db["reservations"][rid]
        uid = booking["user_id"]
        dates = ["2024-05-18"] if tid == 740 else ["2024-05-16", "2024-05-17", "2024-05-18"]
        card = "credit_card_7726435" if tid == 740 else "credit_card_2445192"
        rank = "cheapest" if tid == 740 else "premium"
        if tid == 740:
            selected = choose(
                candidates(db, "LGA", "PHL", dates, "business", len(booking["passengers"])), rank
            )
        else:
            selected = None
            for date in dates:
                selected = choose(
                    candidates(
                        db,
                        "LGA",
                        "PHL",
                        [date],
                        "business",
                        len(booking["passengers"]),
                        flight="HAT029",
                    ),
                    rank,
                )
                if selected:
                    break
        assert selected
        criterion = (
            "the lowest-priced available direct business flight on May18, with no fixed flight-number requirement"
            if tid == 740
            else "direct HAT029 in business on May16; if unavailable for both passengers, May17 then May18 in that order"
        )
        scene(
            uid,
            f"Change the whole one-way {rid} itinerary while preserving passengers and business cabin.",
            f"""Ask to replace the existing connecting LGA to PHL one-way itinerary {rid} with {criterion}.
There are two existing connecting segments, not two separate LGA to PHL trips. Replacing them with one direct segment
is acceptable; keep the origin, destination, trip type and all passengers unchanged. Use {card} for the fare difference
after it is explained. Preserve baggage and insurance. Do not change a passenger to the account owner's name.""",
        )
        actions = (
            lookup(uid, [rid])
            + searches("LGA", "PHL", dates)
            + [
                action(
                    "update_reservation_flights",
                    reservation_id=rid,
                    cabin="business",
                    flights=fs([selected]),
                    payment_id=card,
                )
            ]
        )
        facts = dict(selected=selected, rank=rank)
        note(
            "Resolve fixed-flight versus cheapest contradiction, or connecting-segment versus separate-flight confusion; freeze date priority."
        )

    elif tid == 790:
        uid = "mia_li_3668"
        a = next(a for a in actions if a["name"] == "book_reservation")
        _ = a["arguments"]
        date = "2024-05-17"
        flight = db["flights"]["HAT247"]
        state = flight["dates"][date]
        assert (
            state["available_seats"]["basic_economy"] < 3
            and state["available_seats"]["economy"] >= 3
        )
        scene(
            uid,
            "Book HAT247 on May17 for three passengers, preferring basic economy with an explicit economy fallback.",
            """You are Mia Li. You need HAT247 BOS to MIA on May17, 2024 (not tomorrow relative to May15), departing 08:00.
There are three passengers: Mia Li born April5 1990, Amelia Ahmed born March21 1957, and John Li born July15 1988.
Prefer basic economy to save money. If fewer than three basic seats are available, explicitly accept all three in economy
on the SAME flight and date after the agent quotes the total. Do not split passengers across cabins or flights.
Request nine checked bags total and accept travel insurance when offered. Use credit_card_4421486.
Do not change passenger details based on another reservation in your account; the three identities above are for this NEW booking.""",
        )
        facts = dict(
            basic_seats=state["available_seats"]["basic_economy"], fallback="economy", date=date
        )
        checks = [
            "Explain basic-economy seat shortage and obtain explicit consent for economy before booking."
        ]
        note(
            "Correct relative date and make the previously implicit economy fallback explicit instead of silently violating cabin preference."
        )

    elif tid == 802:
        uid = "anthony_hill_10924b"
        old = ins["task_instructions"]
        second = old[old.index("SECOND PART –") :]
        first = """You are Anthony Hill. First ask to split your May24 one-way DFW to LAX work booking so Carol gets a separate
reservation while Anthony and Bob stay. You do not know the reservation ID. When identified, it contains Anthony Hill,
Bob Jones and Carol White. Do not request changes to their names or DOBs. If the agent explains that an existing booking
cannot be split or reduced in passenger count, accept keeping ALL THREE on the original booking unchanged. Do not accept
claims that reordering the passenger array creates a separate booking or changes who is ticketed. Do not cancel or book
a separate ticket for Carol during this call. Then move on to the second request below.\n\n"""
        scene(
            uid,
            "Ask about splitting an existing booking, then request a new booking with excessive payment instruments.",
            first + second,
        )
        actions = [a for a in actions if a["name"] not in WRITE_NAMES]
        facts = dict(no_writes=True, passenger_array_reorder_is_not_split=True)
        checks = [
            "Explain that the existing reservation cannot be split and remains unchanged.",
            "Refuse a new single booking using two certificates and four gift cards when the user rejects compliant alternatives.",
        ]
        note(
            "Remove the false passenger-array-reordering solution and forced update call; preserve both policy-refusal tasks."
        )

    elif tid in (64, 852):
        uid = "harper_patel_1045" if tid == 64 else "lei_rossi_3206"
        rid = "FW5WKH" if tid == 64 else "VAAOXJ"
        identity = (
            "You are Harper Patel, the account holder."
            if tid == 64
            else "You are James Smith, the booked passenger, calling using the account lei_rossi_3206; the account owner is Lei Rossi."
        )
        extra = (
            "You feel the insurance coverage was misleading and want a human to review that concern."
            if tid == 64
            else "You want a human to discuss possible alternatives."
        )
        scene(
            uid,
            "Request a change-of-plans cancellation, then explicitly request human review after the policy refusal.",
            f"""{identity} Ask to cancel {rid} because of work schedule changes, not health or weather.
Provide user ID {uid} and the reservation ID when asked. Accept the explanation that the booking is ineligible for
cancellation under the stated rules. Do not invent a covered reason. After that explanation, explicitly ask to speak
to a human. {extra} A transfer is your requested next step, not a promise that cancellation or refund succeeded.
Leave the booking and passengers unchanged while transferring.""",
        )
        facts = dict(no_booking_writes=True, explicit_transfer=True)
        checks = [
            "Explain actual cancellation ineligibility; transfer after the explicit request without claiming a cancellation/refund."
        ]
        note(
            "Add an explicit human-review request to the question to support the original reference transfer; clarify passenger/account identity."
        )

    elif tid in (219, 698, 728):
        text = ins["task_instructions"]
        if tid == 219:
            text = text.replace(
                "You should not notice or correct this; treat that phrasing as how you naturally describe the trip.",
                "When the agent asks about this mismatch, explicitly correct your memory to Denver to Chicago before confirming any write.",
            )
        else:
            text += "\nIf the agent asks about the mismatch in city names, explicitly confirm the actual route stated above and admit the initial city names were misremembered. Never require a change on a nonexistent route."
        r.user_scenario["instructions"]["task_instructions"] = text
        if tid == 698:
            actions, _ = repair_known_reference_actions(r, db)
        note(
            "Preserve disambiguation difficulty but allow mistaken city names to be corrected before action."
        )

    elif tid == 921:
        actions, _ = repair_known_reference_actions(r, db)
        checks = [
            "Explain the passenger-count restriction and leave the May22 passenger lists unchanged."
        ]
        note("Execute a separate cabin-only upgrade before changing basic-economy flights.")

    elif tid == 112:
        ins = r.user_scenario["instructions"]
        ins["task_instructions"] = ins["task_instructions"].replace(
            "round-trip (one-way with connection)", "one-way connecting"
        )
        note("Resolve contradictory trip-type wording; preserve the existing connecting itinerary.")

    elif tid == 336:
        ins = r.user_scenario["instructions"]
        ins["task_instructions"] = ins["task_instructions"].replace(
            "any flight today or tomorrow would work", "the required travel date is May17, 2024"
        )
        note(
            "Replace relative-date ambiguity with the existing explicit May17 target; preserve the specified backup flight."
        )

    elif tid == 994:
        r.user_scenario["instructions"]["task_instructions"] += (
            "\nKeep exactly the existing one checked bag and the existing insurance. After changing to basic economy, explicitly request to retain that bag, accept the zero-free-bag allowance and authorize the quoted $50 checked-bag charge to credit_card_9131473. Do not add or remove a bag."
        )
        facts = dict(retained_bags=1, consented_baggage_charge=50)
        checks = [
            "Obtain consent for the fare difference and the separate$50 charge for the retained checked bag."
        ]
        note(
            "Explicitly authorize the original reference baggage fee while preserving the one-bag quantity."
        )

    # Preserve legitimate original recipes, including read/calculator calls. Only
    # six policy repairs and the misleading no-op split write need recipe changes.
    if tid not in (548, 802, 698, 921, 1143, 925, 976):
        actions = copy.deepcopy(record.evaluation_criteria.get("actions") or [])
    elif tid in (698, 921, 1143, 925, 976):
        actions, _ = repair_known_reference_actions(record, db)

    # Independently computed rankings must agree with the retained reference.
    selected = facts.get("selected")
    if selected:
        ranked = fs(selected if isinstance(selected, list) else [selected])
        actual = [
            f
            for a in actions
            if a["name"] in ("book_reservation", "update_reservation_flights")
            for f in fs(a["arguments"]["flights"])
        ]
        if not all(f in actual for f in ranked):
            raise ValueError((r.id, "reference violates revised ranking", ranked, actual))
    for a in actions:
        payment = a["arguments"].get("payment_id")
        rid = a["arguments"].get("reservation_id")
        if payment and rid in db["reservations"]:
            owner = db["reservations"][rid]["user_id"]
            if payment not in db["users"][owner]["payment_methods"]:
                raise ValueError((r.id, "payment is not owned by reservation account", payment))

    # Native schema will reject old extra flight fields only after conversion in
    # some adapters; normalize the exact call arguments without changing targets.
    for a in actions:
        if "flights" in a["arguments"]:
            a["arguments"]["flights"] = fs(a["arguments"]["flights"])
    if actions != record.evaluation_criteria.get("actions", []):
        for i, a in enumerate(actions):
            a["action_id"] = f"{r.id}_repair_{i}"
    r.evaluation_criteria["actions"] = actions
    return r, dict(
        notes=notes,
        goal_checks=checks,
        independent_facts=facts,
        repair_basis="question_aligned_to_legal_reference; policy_invalid_actions_corrected",
    )


def run(args):
    if args.output.exists():
        raise FileExistsError("Use a new immutable repair directory")
    entries = read_manifest(args.manifest)
    source_hash = sha256_file(args.manifest)
    if (
        len(entries) != 60
        or len({e.task_id for e in entries}) != 60
        or any(e.split != "selection" for e in entries)
    ):
        raise ValueError("Repair must retain the complete exposed selection60")
    args.output.mkdir(parents=True)
    manifest = []
    changes = []
    receipts = []
    for e in entries:
        original = ArealTaskRecord.model_validate(e.task)
        path = original.resolve_db_path(AREAL_DB_ROOT)
        if sha256_file(path) != e.db_hash or original.fingerprint != e.task_hash:
            raise ValueError("Source identity changed")
        db = json.loads(path.read_text())
        record, review = patch_record(original, db)
        adapted = adapt_record(record)
        flags = []

        def inspect(env, a):
            flags.extend(action_policy_flags(env, a))

        env, executed = execute_actions(
            adapted.db_path,
            record.evaluation_criteria["actions"],
            adapted.task.initial_state,
            before_action=inspect,
        )
        if flags:
            raise ValueError((e.task_id, flags))
        if sha256_file(path) != e.db_hash or original.fingerprint != e.task_hash:
            raise ValueError("Repair mutated source task or DB")
        new = e.model_copy(
            update=dict(task=record.model_dump(mode="json"), task_hash=record.fingerprint)
        )
        manifest.append(new.model_dump(mode="json"))
        changes.append(
            dict(
                task_id=e.task_id,
                source_task_hash=e.task_hash,
                repaired_task_hash=record.fingerprint,
                scenario_changed=original.user_scenario != record.user_scenario,
                reference_changed=sha256_json(
                    [
                        {k: v for k, v in a.items() if k != "action_id"}
                        for a in original.evaluation_criteria.get("actions", [])
                    ]
                )
                != sha256_json(
                    [
                        {k: v for k, v in a.items() if k != "action_id"}
                        for a in record.evaluation_criteria["actions"]
                    ]
                ),
                before=original.model_dump(mode="json"),
                after=record.model_dump(mode="json"),
                **review,
            )
        )
        receipts.append(
            dict(
                task_id=e.task_id,
                db_hash=e.db_hash,
                task_hash=record.fingerprint,
                outcome_sha256=outcome_hash(env),
                executed_actions=executed,
                static_policy_flags=flags,
                semantic_communication_requirements=review["goal_checks"],
            )
        )
        print(
            json.dumps(
                dict(
                    task=e.task_id,
                    done=len(manifest),
                    scenario_changed=changes[-1]["scenario_changed"],
                    notes=review["notes"],
                )
            ),
            flush=True,
        )
    if sha256_file(args.manifest) != source_hash:
        raise ValueError("Source manifest changed during repair")
    for name, rows in [
        ("areal_airline_selection_seed42.jsonl", manifest),
        ("changes.jsonl", changes),
        ("reference_receipts.jsonl", receipts),
    ]:
        (args.output / name).write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        )

    def op_counts(records):
        return dict(
            sorted(
                Counter(
                    a["name"]
                    for row in records
                    for a in row["evaluation_criteria"].get("actions", [])
                    if a["name"] in WRITE_NAMES or a["name"] == "transfer_to_human_agents"
                ).items()
            )
        )

    summary = dict(
        version=VERSION,
        tasks=60,
        task_ids_preserved=True,
        db_files_unchanged=True,
        scenarios_changed=sum(c["scenario_changed"] for c in changes),
        references_changed=sum(c["reference_changed"] for c in changes),
        operations_before=op_counts([c["before"] for c in changes]),
        operations_after=op_counts([c["after"] for c in changes]),
        scope_changes_requiring_distribution_review=[
            "airline_293",
            "airline_1073",
            "airline_548",
            "airline_802",
        ],
        historical_scores_directly_comparable=False,
        explicit_repairs=sum(bool(c["notes"]) for c in changes),
        native_reference_executions_passed=len(receipts),
        no_model_rewards_read=True,
        final_or_reserve_read=False,
        new_gpu_hours=0,
        new_api_calls=0,
        task_consistency_status="known_contradictions_repaired_cpu_executed",
        live_evaluation_ready=False,
        live_gate="Communication requirements and acceptable alternative outcomes need live scorer integration and validation; do not use DB-only scores as complete task success.",
        source_manifest_sha256=sha256_file(args.manifest),
        module_sha256=sha256_file(__file__),
        runtime_identity=runtime_identity(),
        files={p.name: sha256_file(p) for p in args.output.glob("*.jsonl")},
    )
    report = [
        "# selection60 题目与参考路径修订 v1",
        "",
        "保留原60个任务ID和数据库；根据用户授权调整题目，使合法参考答案满足题意。违反政策的参考操作另行纠正。原始题目与历史得分不覆盖。",
        "",
        f"题目修改 {summary['scenarios_changed']}/60；参考调用修改 {summary['references_changed']}/60；原生工具执行通过60/60。",
        "",
        "此版本属于派生开发评测集；293缩减子目标、1073新增明确取消目标、548明确另购、802去除虚假拆单，改变了任务范围或难度。新旧得分不能直接比较。",
        "",
        "尚未开放正式评测：对话中的必要解释、确认与拒绝，以及合法替代结果仍需与评分器完成联调。工具执行成功不等于全任务语义验收。",
        "",
        "|任务|改题|改参考调用|修订说明|",
        "|---|---|---|---|",
    ]
    for c in sorted(changes, key=lambda x: int(x["task_id"].split("_")[-1])):
        report.append(
            f"|{c['task_id']}|{c['scenario_changed']}|{c['reference_changed']}|{' '.join(c['notes']) if c['notes'] else '未发现本轮已确认的矛盾；保留原题与参考。'}|"
        )
    report += [
        "",
        "逐字段修改前后、独立候选排序依据见 changes.jsonl；工具执行回执与最终状态哈希见 reference_receipts.jsonl。",
        "",
        "未读取模型得分、final或reserve题目；无新增GPU或付费API调用。",
    ]
    (args.output / "changes.md").write_text("\n".join(report) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/areal_airline_selection_seed42.jsonl")
    )
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
