"""Training-only decision repair dialogues with native tools and exact DB targets.

Reuses the protected grounded cohort and adds observed-price, identity, and policy
branches. No test answers, invented tool observations, paid APIs or GPUs.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from tau3_grpo.data.grounded_gap_pilot import Builder, build_case, raw_database
from tau3_grpo.data.grounded_repair import BASE_KINDS, build_extra, task_feature
from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.data.sft_expansion import audit_tool_calls
from tau3_grpo.data.staged_sft import ordered_tool_receipts
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.paths import AREAL_DB_ROOT
from tau3_grpo.utils.hashing import sha256_file, sha256_json

KINDS = (
    "historical_upgrade",
    "downgrade_refund",
    "multileg_cabin",
    "card_tail_binding",
    "cancel_within24",
    "cancel_after24",
    "cause_unavailable",
    "profile_identity",
)
NOW = datetime(2024, 5, 15, 15)


def cabin_option(db, r, kind):
    n = len(r["passengers"])
    states = [db["flights"][f["flight_number"]]["dates"][f["date"]] for f in r["flights"]]
    if not all(f["date"] > "2024-05-15" for f in r["flights"]):
        return None
    if any(
        s.get("status") != "available" or "prices" not in s or "available_seats" not in s
        for s in states
    ):
        return None
    old = sum(f["price"] for f in r["flights"]) * n
    current = sum(s["prices"][r["cabin"]] for s in states) * n
    if kind == "downgrade_refund":
        if r["cabin"] != "business":
            return None
        target = "economy"
    else:
        if r["cabin"] == "business":
            return None
        target = "economy" if r["cabin"] == "basic_economy" else "business"
    if any(s.get("status") != "available" or s["available_seats"][target] < n for s in states):
        return None
    new = sum(s["prices"][target] for s in states) * n
    if kind == "downgrade_refund" and new >= old:
        return None
    if kind != "downgrade_refund" and new <= old:
        return None
    if kind == "historical_upgrade" and current == old:
        return None
    if kind == "multileg_cabin" and (len(states) < 2 or n < 2):
        return None
    return target, old, new, current


def eligible(db, rid, kind):
    r = db["reservations"].get(rid)
    if not r or r.get("status") == "cancelled" or not r["passengers"]:
        return False
    if r["user_id"] not in db["users"]:
        return False
    if any(
        f["flight_number"] not in db["flights"]
        or f["date"] not in db["flights"][f["flight_number"]]["dates"]
        for f in r["flights"]
    ):
        return False
    u = db["users"][r["user_id"]]
    cards = [v for v in u["payment_methods"].values() if v["source"] == "credit_card"]
    if kind in KINDS[:4]:
        return (
            bool(cards)
            and cabin_option(db, r, kind) is not None
            and (
                kind != "card_tail_binding"
                or any(
                    v.get("last_four")
                    and sum(x.get("last_four") == v["last_four"] for x in cards) == 1
                    for v in cards
                )
            )
        )
    if kind.startswith("cancel_"):
        age = (NOW - datetime.fromisoformat(r["created_at"])).total_seconds()
        if age < 0 or r["cabin"] == "business" or r["insurance"] != "no":
            return False
        if not all(
            f["date"] > "2024-05-15"
            and db["flights"][f["flight_number"]]["dates"][f["date"]]["status"] == "available"
            for f in r["flights"]
        ):
            return False
        return sum(p["amount"] for p in r["payment_history"]) > 0 and (
            (age <= 86400) if kind == "cancel_within24" else age > 86400
        )
    if kind == "cause_unavailable":
        return any(
            db["flights"][f["flight_number"]]["dates"][f["date"]]["status"]
            in ("cancelled", "delayed")
            for f in r["flights"]
        )
    if kind == "profile_identity":
        return bool(u.get("dob")) and any(
            p["first_name"] == u["name"]["first_name"] and p["last_name"] == u["name"]["last_name"]
            for p in r["passengers"]
        )
    return False


def build_decision(entry, rid, kind):
    adapted = adapt_record(ArealTaskRecord.model_validate(entry.task))
    if adapted.db_file_hash != entry.db_hash:
        raise ValueError("DB identity changed")
    b = Builder(adapted, entry, rid, kind)
    db = b.before
    if not eligible(db, rid, kind):
        raise ValueError("Invalid frozen candidate")
    b.expected = copy.deepcopy(db)
    r = db["reservations"][rid]
    uid = r["user_id"]
    u = db["users"][uid]
    er = b.expected["reservations"][rid]
    n = len(r["passengers"])
    checks = [
        "visible_prefix_grounding",
        "native_tools",
        "independent_replay",
        "exact_protected_DB_state",
    ]
    facts = {}
    if kind in KINDS[:4]:
        target, old, new, current = cabin_option(db, r, kind)
        delta = new - old
        cards = sorted(k for k, v in u["payment_methods"].items() if v["source"] == "credit_card")
        card = (
            next(
                k
                for k in cards
                if u["payment_methods"][k].get("last_four")
                and sum(
                    v.get("last_four") == u["payment_methods"][k]["last_four"]
                    for v in u["payment_methods"].values()
                )
                == 1
            )
            if kind == "card_tail_binding"
            else cards[-1]
        )
        method = (
            f"my {u['payment_methods'][card]['brand']} ending in {u['payment_methods'][card]['last_four']}"
            if kind == "card_tail_binding"
            else f"my stored card {card}"
        )
        b.say(
            "user",
            f"My user ID is {uid}. For reservation {rid}, change only the cabin to {target} for all passengers on all existing flights. Keep dates, flights, passengers, bags and insurance unchanged. Use {method}. Tell me the precise fare difference before doing anything.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        profile = b.tool("get_user_details", {"user_id": uid})
        assert card in profile["payment_methods"]
        for f in r["flights"]:
            options = b.tool(
                "search_direct_flight", {k: f[k] for k in ("origin", "destination", "date")}
            )
            observed = next(v for v in options if v["flight_number"] == f["flight_number"])
            assert (
                observed["prices"][target]
                == db["flights"][f["flight_number"]]["dates"][f["date"]]["prices"][target]
            )
        result = b.tool("calculate", {"expression": f"({new // n}-{old // n})*{n}"})
        assert float(result) == delta
        direction = f"an additional charge of ${delta}" if delta > 0 else f"a refund of ${-delta}"
        b.say(
            "assistant",
            f"The stored reservation airfare is ${old} for {n} passengers across {len(r['flights'])} flight segment(s). The current fare in {target} is ${new} total. New fare minus stored fare is {delta}, so this is {direction}. The price currently advertised for the old cabin is not the amount you originally paid. Your chosen card is verified in your profile as {card}. All flight numbers, dates, passengers, bags and insurance will remain unchanged. Shall I make this cabin-only change at that amount?",
        )
        b.say(
            "user",
            f"Yes, change the entire reservation to {target} with {direction} using that verified card. Keep the other details unchanged.",
        )
        er["cabin"] = target
        for f in er["flights"]:
            f["price"] = db["flights"][f["flight_number"]]["dates"][f["date"]]["prices"][target]
        er["payment_history"].append({"payment_id": card, "amount": delta})
        response = b.tool(
            "update_reservation_flights",
            {
                "reservation_id": rid,
                "cabin": target,
                "flights": [{k: f[k] for k in ("flight_number", "date")} for f in r["flights"]],
                "payment_id": card,
            },
        )
        assert response["payment_history"][-1] == {"payment_id": card, "amount": delta}
        b.say(
            "assistant",
            f"The update succeeded. The receipt confirms {direction} using {card}, and {target} applies to every passenger and flight. No flight, date, passenger, baggage or insurance change was made.",
        )
        checks += [
            "historical_fare_not_current_old_cabin_quote",
            "all_segments_times_passengers",
            "exact_signed_delta_before_yes",
            "profile_card_binding",
            "receipt_matches_quote",
        ]
        facts = dict(
            old_total=old,
            new_total=new,
            current_old_cabin_total=current,
            net_delta=delta,
            passengers=n,
            segments=len(r["flights"]),
        )
    elif kind.startswith("cancel_"):
        age = (NOW - datetime.fromisoformat(r["created_at"])).total_seconds()
        hours, mins = divmod(int(age // 60), 60)
        b.say(
            "user",
            f"My user ID is {uid}. My plans changed. Can I cancel reservation {rid}? Please check its creation timestamp against the policy clock, 2024-05-15 15:00:00 EST. Cancel only if permitted; otherwise explain and keep it unchanged.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        for f in r["flights"]:
            b.tool("get_flight_status", {k: f[k] for k in ("flight_number", "date")})
        timing = f"It was created at {r['created_at']}; at 2024-05-15 15:00:00 EST its age is {hours} hours {mins} minutes."
        if kind == "cancel_within24":
            refund = sum(x["amount"] for x in r["payment_history"])
            b.say(
                "assistant",
                f"{timing} This is within 24 hours, and no segment has flown, so cancellation for changed plans is allowed. I will cancel the whole booking and refund ${refund} to the original payment method(s), retaining each original payment destination. Would you like me to proceed?",
            )
            b.say("user", f"Yes, cancel {rid} and refund ${refund} to the original methods.")
            er["status"] = "cancelled"
            er["payment_history"] += [
                dict(payment_id=p["payment_id"], amount=-p["amount"]) for p in r["payment_history"]
            ]
            b.tool("cancel_reservation", {"reservation_id": rid})
            b.say(
                "assistant",
                f"Cancellation succeeded. The refund is ${refund} to the original payment method(s), within 5 to 7 business days.",
            )
        else:
            b.say(
                "assistant",
                f"{timing} It is outside the 24-hour window. The booking is not business class, has no insurance, and no flight is airline-cancelled. A change-of-plans cancellation is therefore not permitted. I have left the booking unchanged.",
            )
            b.say("user", "Understood. Leave it unchanged; that answers my question.")
            b.say("assistant", "No cancellation or refund was performed.")
        facts = dict(age_seconds=age, within24=kind == "cancel_within24")
        checks += [
            "timestamp_arithmetic",
            "policy_disjunction",
            "no_guessed_eligibility",
            "original_refund_methods",
        ]
    elif kind == "cause_unavailable":
        f = next(
            f
            for f in r["flights"]
            if db["flights"][f["flight_number"]]["dates"][f["date"]]["status"]
            in ("cancelled", "delayed")
        )
        b.say(
            "user",
            f"My user ID is {uid}, reservation {rid}. What is the status of flight {f['flight_number']} on {f['date']}, and why did the disruption happen? I only want information; do not change my reservation or issue compensation.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        status = b.tool("get_flight_status", {k: f[k] for k in ("flight_number", "date")})
        b.say(
            "assistant",
            f"The flight-status tool reports: {status}. It does not provide an operational cause, so I cannot determine why this disruption occurred from the available information. I have not changed the reservation or issued compensation.",
        )
        checks += [
            "answer_direct_why_question",
            "no_invented_weather_or_operations",
            "no_unrequested_write",
        ]
    elif kind == "profile_identity":
        person = next(
            p
            for p in r["passengers"]
            if p["first_name"] == u["name"]["first_name"]
            and p["last_name"] == u["name"]["last_name"]
        )
        b.say(
            "user",
            f"My user ID is {uid}. For reservation {rid}, please check the date of birth recorded for me, {person['first_name']} {person['last_name']}, against my account profile. My correct date of birth is on my profile. Do not guess it or change any booking details; just tell me whether the two records agree.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        profile = b.tool("get_user_details", {"user_id": uid})
        agree = "agree" if person["dob"] == profile["dob"] else "differ"
        b.say(
            "assistant",
            f"Your profile date of birth is {profile['dob']}; the reservation records {person['dob']} for {person['first_name']} {person['last_name']}. The two records {agree}. I have not changed either record.",
        )
        checks += ["query_profile_for_missing_DOB", "identity_binding", "read_only_scope"]
    row = b.finish(f"Training-only {kind}: {rid}", checks)
    row["metadata"]["repair_kind"] = kind
    row["metadata"]["verified_facts"] = facts
    row["metadata"]["difficulty"] = {
        "level": "unrated",
        "basis": "no_calibrated_intrinsic_difficulty_claim",
    }
    return row


def run(output):
    import yaml

    olddir = Path("data/sft/grounded_repair_20260926_v1")
    old = json.loads((olddir / "plan.json").read_text())
    entries = {
        e.task_id: e for e in read_manifest("data/manifests/areal_airline_train_seed42.jsonl")
    }
    protected = set(old["protected_users"])
    oldtrain = set(old["historical_train_users"])
    devusers = set(old["heldout_users"])
    used = {x["rid"] for split in ("train", "validation") for x in old[split]}
    trainusers = {x["user_id"] for x in old["train"]}
    pool = {}
    for e in entries.values():
        if e.split != "train":
            raise ValueError("Training manifest violation")
        if e.task.get("initial_state"):
            continue
        db = raw_database(
            str(ArealTaskRecord.model_validate(e.task).resolve_db_path(AREAL_DB_ROOT))
        )
        for rid in task_feature(e.task)["reservation_ids"]:
            r = db["reservations"].get(rid)
            if not r or r["user_id"] in protected or rid in used:
                continue
            pool.setdefault(
                (e.db_hash, rid),
                dict(task_id=e.task_id, rid=rid, user_id=r["user_id"], db_hash=e.db_hash),
            )
    pools = {}
    for kind in KINDS:
        pools[kind] = []
        for x in pool.values():
            e = entries[x["task_id"]]
            db = raw_database(
                str(ArealTaskRecord.model_validate(e.task).resolve_db_path(AREAL_DB_ROOT))
            )
            if eligible(db, x["rid"], kind):
                pools[kind].append(x)
    additions = []
    shortfalls = []
    # Hold out one new user per branch before training selection; don't reuse old SFT users.
    for kind in sorted(KINDS, key=lambda k: len(pools[k])):
        options = [
            x
            for x in pools[kind]
            if x["user_id"] not in oldtrain | trainusers | devusers and x["rid"] not in used
        ]
        if not options:
            shortfalls.append(dict(kind=kind, split="validation", requested=1, actual=0))
            continue
        x = min(options, key=lambda x: sha256_json([42, kind, x]))
        devusers.add(x["user_id"])
        used.add(x["rid"])
        additions.append(dict(kind=kind, split="validation", **x))
    counts = Counter(x["user_id"] for x in old["train"])
    for kind in sorted(KINDS, key=lambda k: len(pools[k])):
        picked = set()
        for x in sorted(
            pools[kind], key=lambda x: (counts[x["user_id"]], sha256_json([42, kind, x]))
        ):
            if x["user_id"] in devusers | picked or x["rid"] in used:
                continue
            additions.append(dict(kind=kind, split="train", **x))
            used.add(x["rid"])
            picked.add(x["user_id"])
            counts[x["user_id"]] += 1
            if len(picked) == 4:
                break
        if len(picked) < 4:
            shortfalls.append(dict(kind=kind, split="train", requested=4, actual=len(picked)))
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(
        version="native_decision_repair_v1",
        source_plan_sha256=sha256_file(olddir / "plan.json"),
        source_code_sha256=sha256_file(__file__),
        additions=additions,
        shortfalls=shortfalls,
        protected_users=sorted(protected),
        validation_users=sorted(devusers),
        seed=42,
        final_read=False,
        reserve_read=False,
        new_api_calls=0,
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = {"train": [], "validation": []}
    errors = []
    schemas = [
        x["tool_schema"]
        for x in yaml.safe_load(Path("configs/envs/tool_config.yaml").read_text())["tools"]
    ]
    items = [dict(x, existing=True) for split in rows for x in old[split]] + [
        dict(x, existing=False) for x in additions
    ]
    for item in items:
        entry = entries[item["task_id"]]
        kind = item["kind"]
        rid = item["rid"]
        try:
            if item["existing"]:
                row = (
                    build_case(entry, rid, kind)
                    if kind in BASE_KINDS
                    else build_extra(entry, rid, kind)
                )
            else:
                row = build_decision(entry, rid, kind)
            if row is None:
                raise ValueError("Builder returned no dialogue")
            failures = audit_tool_calls(row, schemas) + ordered_tool_receipts(row)
            if failures:
                raise ValueError(failures)
            row["metadata"].update(
                repair_kind=kind,
                split=item["split"],
                status="native_replayed_repair_candidate",
                source_user_id=item["user_id"],
            )
            rows[item["split"]].append(row)
        except Exception as exc:
            errors.append(dict(item, error=str(exc)[:500]))
        print(
            json.dumps(dict(done=sum(map(len, rows.values())), errors=len(errors), kind=kind)),
            flush=True,
        )
    for split, values in rows.items():
        (output / (split + ".jsonl")).write_text(
            "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values)
        )
    trainuids = {r["metadata"]["source_user_id"] for r in rows["train"]}
    devuids = {r["metadata"]["source_user_id"] for r in rows["validation"]}
    assert not trainuids & (devuids | protected)
    report = dict(
        counts={k: len(v) for k, v in rows.items()},
        errors=errors,
        shortfalls=shortfalls,
        kinds={s: dict(Counter(r["metadata"]["repair_kind"] for r in v)) for s, v in rows.items()},
        user_overlap=[],
        replayed_independent_DB=True,
        new_api_calls=0,
        limitations=[
            "Authored templates, not live policy rollouts or independent semantic-family test.",
            "Original AReaL SFT source trajectories are not included in this repair-only round.",
            "The validation set is for held-out entity diagnostic loss, not task success rate.",
        ],
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise ValueError("Data construction errors; inspect summary, do not train")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
