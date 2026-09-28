"""Small authored training-only pilot for gaps absent from source dialogues.

Every tool result is executed by the real pinned Airline environment. These are
explicitly authored sub-tasks on existing TRAIN databases, not fabricated AReaL
source trajectories, LLM simulator runs, or evidence of an SFT accuracy gain.
"""

from __future__ import annotations

import argparse
import copy
import json
from functools import lru_cache
from pathlib import Path

from tau3_grpo.analysis.capability_distribution import sft_feature, task_feature
from tau3_grpo.analysis.prepare_outcome_contract import duplicate_cancellation_allowed
from tau3_grpo.analysis.sft_coldstart_audit import audit_record
from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.data.sft_expansion import audit_tool_calls
from tau3_grpo.data.sft_policy_checks import ALLOWANCE, audit_baggage_allowances
from tau3_grpo.data.staged_sft import ordered_tool_receipts
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.envs.tau2_bridge import message_models
from tau3_grpo.evaluation.outcome_contract import execute_actions, fresh_environment, outcome_hash
from tau3_grpo.paths import AREAL_DB_ROOT
from tau3_grpo.prompts import build_system_prompt, prompt_provenance
from tau3_grpo.utils.hashing import sha256_file, sha256_text


def read_rows(path):
    return [
        json.loads(lint_item)
        for lint_item in Path(path).read_text().splitlines()
        if lint_item.strip()
    ]


class Builder:
    def __init__(self, adapted, source, rid, kind):
        self.adapted = adapted
        self.source = source
        self.kind = kind
        self.rid = rid
        self.env = fresh_environment(adapted.db_path, adapted.task.initial_state)
        self.messages = [{"role": "system", "content": build_system_prompt()}]
        self.receipts = []
        self.actions = []
        self.before = self.env.tools.db.model_dump(mode="json")
        self.expected = None

    def say(self, role, content):
        self.messages.append({"role": role, "content": content})

    def tool(self, name, args):
        cid = f"gap_{len(self.actions):03d}"
        call = message_models()["ToolCall"](
            id=cid, name=name, arguments=args, requestor="assistant"
        )
        response = self.env.get_response(call)
        if response.error:
            raise ValueError(f"Native tool failed: {name}: {response.content}")
        self.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
                ],
            }
        )
        self.messages.append(
            {"role": "tool", "tool_call_id": cid, "name": name, "content": response.content}
        )
        self.receipts.append(
            {"name": name, "arguments": args, "response_sha256": sha256_text(response.content)}
        )
        self.actions.append({"name": name, "arguments": args})
        try:
            return json.loads(response.content)
        except ValueError:
            return response.content

    def finish(self, reason, semantic_checks):
        # Fresh reference execution; all responses in the training record remain native receipts.
        reference, _ = execute_actions(
            self.adapted.db_path, self.actions, self.adapted.task.initial_state
        )
        if outcome_hash(reference) != outcome_hash(self.env):
            raise ValueError("Independent replay mismatch")
        if self.expected is None or self.expected != self.env.tools.db.model_dump(mode="json"):
            raise ValueError("Authored target or protected-state predicate failed")
        idx = [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]
        row = {
            "messages": self.messages,
            "supervision": {
                "version": "approved_assistant_v1",
                "message_indices": idx,
                "basis": "authored_policy_checked_and_native_tool_executed",
            },
            "metadata": {
                **prompt_provenance(),
                "source": "areal_train_db_authored_subtask",
                "source_dialog_id": f"gap_{self.kind}_{self.source.task_id}",
                "source_task_id": self.source.task_id,
                "source_task_hash": self.source.task_hash,
                "source_db_hash": self.source.db_hash,
                "source_db_path": self.source.db_path,
                "source_family": self.rid,
                "reason_for_call": reason,
                "verification": "native_receipts_and_independent_reference_execution",
                "executions": self.receipts,
                "semantic_checks": semantic_checks,
                "status": "executed_pilot_candidate_not_frozen_training_set",
                "construction": "authored_full_dialogue_not_llm_generated_or_online_simulation",
                "difficulty": {
                    "level": "foundation_candidate",
                    "basis": "one_explicit_goal_one_reservation_known_ids_no_replanning",
                },
                "dialogue_hash": sha256_text(
                    json.dumps(self.messages, sort_keys=True, separators=(",", ":"))
                ),
                "outcome_sha256": outcome_hash(self.env),
            },
        }
        checks = audit_record(row)
        if any(x["status"] == "contradiction" for x in checks["findings"]):
            raise ValueError(checks["findings"])
        baggage = audit_baggage_allowances(self.messages)
        if any(x["status"] != "satisfied" for x in baggage):
            raise ValueError(baggage)
        row["metadata"]["prefix_audit"] = checks
        row["metadata"]["baggage_checks"] = baggage
        return row


def build_case(entry, rid, kind):
    record = ArealTaskRecord.model_validate(entry.task)
    adapted = adapt_record(record)
    if adapted.db_file_hash != entry.db_hash:
        raise ValueError("Source DB changed")
    b = Builder(adapted, entry, rid, kind)
    db = b.env.tools.db.model_dump(mode="json")
    r = db["reservations"][rid]
    u = db["users"][r["user_id"]]
    uid = r["user_id"]
    n = len(r["passengers"])
    cards = [k for k, v in u["payment_methods"].items() if v["source"] == "credit_card"]
    common = [
        "user_id_provided_before_write",
        "reservation_only_scope",
        "explicit_details_then_yes",
        "native_tool_responses",
        "explicit_expected_state_and_protected_fields_verified",
    ]
    b.expected = copy.deepcopy(db)
    expected_r = b.expected["reservations"][rid]
    if kind.startswith("baggage"):
        if not cards:
            return None
        free = ALLOWANCE[u["membership"]][r["cabin"]] * n
        if r["nonfree_baggages"] != max(0, r["total_baggages"] - free):
            return None
        target = (
            r["total_baggages"] + 1
            if kind == "baggage_free"
            else max(r["total_baggages"] + 1, free + 1)
        )
        paid = max(0, target - free)
        cost = 50 * (paid - r["nonfree_baggages"])
        if kind == "baggage_free" and cost != 0:
            return None
        card = cards[-1]
        reason = f"Add checked bags to reservation {rid} to reach {target} bags total, keeping all other booking details unchanged."
        b.say(
            "user",
            f"My user ID is {uid}. For reservation {rid}, please increase my checked bags to {target} in total. Keep everything else unchanged. Please check the allowance and cost first.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        b.tool("get_user_details", {"user_id": uid})
        b.say(
            "assistant",
            f"Reservation {rid} currently has {r['total_baggages']} checked bags. With {n} passenger(s), {u['membership']} membership and {r['cabin']} cabin, the free allowance is {free} bags. Changing to {target} total bags means {paid} paid bags and an additional charge of ${cost}. I can use your stored credit card {card}. May I make this baggage-only update?",
        )
        b.say(
            "user",
            f"Yes, update reservation {rid} to {target} total bags and use {card} for the ${cost} charge. Do not change the passengers or flights.",
        )
        expected_r.update(total_baggages=target, nonfree_baggages=paid)
        if cost:
            expected_r["payment_history"].append({"payment_id": card, "amount": cost})
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
            f"Reservation {rid} now has {target} checked bags in total, of which {paid} are paid bags. The additional baggage charge is ${cost}. The flights and passengers are unchanged.",
        )
        common += [
            "allowance_per_passenger_verified",
            "incremental_paid_bags_not_total_bags_charged",
            "payment_read_from_profile",
        ]
    elif kind.startswith("cabin"):
        source_cabin, target = (
            ("basic_economy", "economy")
            if kind == "cabin_basic_upgrade"
            else ("economy", "business")
        )
        if r["cabin"] != source_cabin or not cards:
            return None
        if any(f["date"] <= "2024-05-15" for f in r["flights"]):
            return None
        states = [
            db["flights"][f["flight_number"]]["dates"].get(f["date"], {}) for f in r["flights"]
        ]
        if any(s.get("status") != "available" or s["available_seats"][target] < n for s in states):
            return None
        delta = (
            sum(s["prices"][target] for s in states) - sum(f["price"] for f in r["flights"])
        ) * n
        if delta <= 0:
            return None
        card = cards[-1]
        flights = [{k: f[k] for k in ("flight_number", "date")} for f in r["flights"]]
        reason = f"Upgrade reservation {rid} from {source_cabin} to {target} without changing any flight or passenger."
        b.say(
            "user",
            f"My user ID is {uid}. Can you upgrade reservation {rid} to {target}? Keep the exact same flights, dates, passengers and baggage. Tell me the fare difference before making changes.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        b.tool("get_user_details", {"user_id": uid})
        for f in r["flights"]:
            b.tool(
                "search_direct_flight",
                {"origin": f["origin"], "destination": f["destination"], "date": f["date"]},
            )
        expression = f"({sum(s['prices'][target] for s in states)} - {sum(f['price'] for f in r['flights'])}) * {n}"
        b.tool("calculate", {"expression": expression})
        b.say(
            "assistant",
            f"The {target} cabin is available on every existing flight. A cabin-only change is allowed before departure, including from basic economy. Upgrading all {n} passenger(s) on the same flights and dates costs an additional ${delta}. I can charge your stored credit card {card}. All other booking details stay unchanged. May I proceed?",
        )
        b.say(
            "user",
            f"Yes, upgrade the whole reservation {rid} to {target} on the same flights and pay ${delta} using {card}.",
        )
        expected_r["cabin"] = target
        for f, state in zip(expected_r["flights"], states):
            f["price"] = state["prices"][target]
        expected_r["payment_history"].append({"payment_id": card, "amount": delta})
        b.tool(
            "update_reservation_flights",
            {"reservation_id": rid, "cabin": target, "flights": flights, "payment_id": card},
        )
        b.say(
            "assistant",
            f"Reservation {rid} is now in {target} on all existing flights. The fare difference charged was ${delta}; dates, route, passengers and baggage are unchanged.",
        )
        common += [
            "basic_cabin_change_distinguished_from_flight_change",
            "all_segments_same_cabin",
            "stored_payment",
            "price_delta_verified",
        ]
    elif kind.startswith("cancel"):
        if any(f["date"] <= "2024-05-15" for f in r["flights"]):
            return None
        allowed = duplicate_cancellation_allowed(r, db)
        if allowed != (kind == "cancel_allowed"):
            return None
        if kind == "cancel_denied" and r["insurance"] != "no":
            return None
        reason = f"Check whether reservation {rid} can be cancelled for change of plans; cancel only if eligible."
        b.say(
            "user",
            f"My user ID is {uid}. My plans changed. Please check whether I can cancel reservation {rid}. If the policy does not allow it, keep my booking as it is; I do not want a transfer.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        if allowed:
            if r["cabin"] != "business":
                return None
            refund = sum(p["amount"] for p in r["payment_history"])
            if refund <= 0:
                return None
            b.say(
                "assistant",
                f"Reservation {rid} is a business-class booking and none of its flights has departed, so cancellation for a change of plans is allowed. I will cancel the entire reservation and refund ${refund} to the original payment method(s), usually within 5 to 7 business days. May I proceed?",
            )
            b.say("user", f"Yes, cancel reservation {rid} and refund the original payment methods.")
            expected_r["status"] = "cancelled"
            expected_r["payment_history"] += [
                dict(payment_id=p["payment_id"], amount=-p["amount"]) for p in r["payment_history"]
            ]
            b.tool("cancel_reservation", {"reservation_id": rid})
            b.say(
                "assistant",
                f"Reservation {rid} has been cancelled. The ${refund} refund goes to the original payment method(s) within 5 to 7 business days.",
            )
        else:
            for f in r["flights"]:
                b.tool(
                    "get_flight_status", {"flight_number": f["flight_number"], "date": f["date"]}
                )
            b.say(
                "assistant",
                f"Reservation {rid} was booked more than 24 hours ago, is not business class, has no travel insurance, and its flights are not cancelled by the airline. A change-of-plans cancellation is not allowed under this policy. I have left the reservation unchanged as you requested.",
            )
            b.say("user", "Understood. Please leave everything unchanged. That is all I needed.")
            b.say(
                "assistant", "Your booking remains unchanged. No cancellation or refund was made."
            )
        common += [
            "cancellation_policy_checked",
            "original_refund_destination" if allowed else "correct_refusal_and_zero_db_writes",
        ]
    else:
        cancelled = [
            f
            for f in r["flights"]
            if db["flights"][f["flight_number"]]["dates"].get(f["date"], {}).get("status")
            == "cancelled"
        ]
        if not cancelled or not (
            u["membership"] in ("silver", "gold")
            or r["insurance"] == "yes"
            or r["cabin"] == "business"
        ):
            return None
        f = cancelled[0]
        amount = 100 * n
        reason = f"Request the policy travel certificate for an airline-cancelled flight on reservation {rid}; no other booking changes."
        b.say(
            "user",
            f"My user ID is {uid}. I am complaining about the cancellation of flight {f['flight_number']} on {f['date']}, reservation {rid}. Please check if I qualify for a travel certificate. I am not asking you to modify or cancel the reservation.",
        )
        b.tool("get_reservation_details", {"reservation_id": rid})
        b.tool("get_flight_status", {"flight_number": f["flight_number"], "date": f["date"]})
        b.tool("get_user_details", {"user_id": uid})
        b.say(
            "assistant",
            f"I verified that flight {f['flight_number']} on {f['date']} was cancelled by the airline. Your {u['membership']} membership, {r['cabin']} cabin and insurance status ({r['insurance']}) meet the compensation eligibility rule. The cancelled-flight certificate is $100 per passenger, so ${amount} for {n} passenger(s). May I issue that certificate to your account without changing the reservation?",
        )
        b.say("user", f"Yes, issue the ${amount} certificate. Keep the reservation unchanged.")
        b.tool("send_certificate", {"user_id": uid, "amount": amount})
        after_methods = b.env.tools.db.model_dump(mode="json")["users"][uid]["payment_methods"]
        new_keys = set(after_methods) - set(u["payment_methods"])
        if len(new_keys) != 1:
            raise ValueError("Expected exactly one new certificate")
        key = new_keys.pop()
        cert = after_methods[key]
        if cert != {"id": key, "amount": amount, "source": "certificate"}:
            raise ValueError("Certificate amount/type mismatch")
        b.expected["users"][uid]["payment_methods"][key] = cert
        b.say(
            "assistant",
            f"The ${amount} travel certificate has been issued to your account. The reservation itself is unchanged.",
        )
        common += [
            "user_requested_compensation",
            "cancelled_status_verified",
            "eligibility_and_100_per_passenger",
            "no_unsolicited_reservation_modification",
        ]
    return b.finish(reason, common)


@lru_cache(maxsize=4)
def raw_database(path):
    return json.loads(Path(path).read_text())


def feasible(db, rid, kind):
    """Cheap raw-DB prefilter; live checks are repeated before creating a row."""
    r = db["reservations"].get(rid)
    if not r or r.get("status") == "cancelled" or not r["passengers"]:
        return False
    if r["user_id"] not in db["users"] or any(
        f["flight_number"] not in db["flights"] for f in r["flights"]
    ):
        return False
    u = db["users"][r["user_id"]]
    n = len(r["passengers"])
    cards = [k for k, v in u["payment_methods"].items() if v["source"] == "credit_card"]
    states = [db["flights"][f["flight_number"]]["dates"].get(f["date"], {}) for f in r["flights"]]
    if not all(states):
        return False
    if kind.startswith("baggage"):
        free = ALLOWANCE[u["membership"]][r["cabin"]] * n
        return (
            bool(cards)
            and r["nonfree_baggages"] == max(0, r["total_baggages"] - free)
            and (kind != "baggage_free" or r["total_baggages"] < free)
        )
    if kind.startswith("cabin"):
        source, target = (
            ("basic_economy", "economy")
            if kind == "cabin_basic_upgrade"
            else ("economy", "business")
        )
        return (
            bool(cards)
            and r["cabin"] == source
            and all(f["date"] > "2024-05-15" for f in r["flights"])
            and all(
                s.get("status") == "available" and s["available_seats"][target] >= n for s in states
            )
            and sum(s["prices"][target] for s in states) > sum(f["price"] for f in r["flights"])
        )
    if kind.startswith("cancel"):
        if any(f["date"] <= "2024-05-15" for f in r["flights"]):
            return False
        allowed = duplicate_cancellation_allowed(r, db)
        return (
            (
                allowed
                and r["cabin"] == "business"
                and sum(p["amount"] for p in r["payment_history"]) > 0
            )
            if kind == "cancel_allowed"
            else (not allowed and r["insurance"] == "no")
        )
    return any(s.get("status") == "cancelled" for s in states) and (
        u["membership"] in ("silver", "gold") or r["insurance"] == "yes" or r["cabin"] == "business"
    )


def run(args):
    import yaml

    if args.output.exists():
        raise FileExistsError("Immutable pilot output already exists")
    args.output.mkdir(parents=True)
    (args.output / "executed_source.py").write_text(Path(__file__).read_text())
    entries = read_manifest("data/manifests/areal_airline_train_seed42.jsonl")
    blocked = set()
    for e in read_manifest("data/manifests/areal_airline_selection_seed42.jsonl"):
        blocked.update(task_feature(e.task)["reservation_ids"])
    for r in read_rows("data/sft/staged_v2_candidates_20260925/offline_dev.jsonl"):
        blocked.update(sft_feature(r)["reservation_ids"])
    # Final examples are never opened by this generator. Preserve prior frozen source protections.
    used = set()
    rows = []
    rejected = []
    kinds = [
        "baggage_free",
        "baggage_paid",
        "cabin_basic_upgrade",
        "cabin_economy_upgrade",
        "cancel_allowed",
        "cancel_denied",
        "compensation_1",
        "compensation_2",
    ]
    schemas = [
        x["tool_schema"]
        for x in yaml.safe_load(Path("configs/envs/tool_config.yaml").read_text())["tools"]
    ]
    for kind in kinds:
        for entry in entries:
            if entry.split != "train":
                raise ValueError("Non-training task reached generator")
            ids = task_feature(entry.task)["reservation_ids"]
            if set(ids) & blocked:
                continue
            for rid in ids:
                if rid in used:
                    continue
                record = ArealTaskRecord.model_validate(entry.task)
                if entry.task.get("initial_state"):
                    continue  # Authored raw-state prefilter cannot interpret initialization actions.
                raw = raw_database(str(record.resolve_db_path(AREAL_DB_ROOT)))
                if not feasible(raw, rid, kind):
                    continue
                try:
                    row = build_case(entry, rid, kind)
                except Exception as exc:
                    rejected.append(
                        {
                            "source_task": entry.task_id,
                            "reservation": rid,
                            "kind": kind,
                            "error": str(exc)[:300],
                        }
                    )
                    continue
                if row is None:
                    continue
                errors = audit_tool_calls(row, schemas) + ordered_tool_receipts(row)
                if errors:
                    raise ValueError(errors)
                rows.append(row)
                used.add(rid)
                with (args.output / "candidates.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                break
            else:
                continue
            break
    (args.output / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    report = {
        "planned": len(kinds),
        "completed": len(rows),
        "completed_kinds": [r["metadata"]["source_dialog_id"] for r in rows],
        "rejections": rejected,
        "source_train_manifest_sha256": sha256_file(
            "data/manifests/areal_airline_train_seed42.jsonl"
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "final_or_reserve_content_read": False,
        "new_llm_calls": 0,
        "gpu_hours": 0,
        "training_started": False,
        "limitation": "Small deterministic authored pilot. Semantic family independence from final is not established; no generalization or SFT benefit claim.",
        "candidate_sha256": sha256_file(args.output / "candidates.jsonl"),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
