"""Versioned, executable outcome sets. No LLM scores or permissive field masks.

Each accepted outcome is produced by executing a reviewed action sequence on a
fresh original database. Only passenger list order is ignored; identity, DOB,
duplicates, money, unrelated reservations and flight ordering remain exact.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from tau3_grpo.envs.adapter import build_environment, load_flight_db
from tau3_grpo.envs.tau2_bridge import message_models
from tau3_grpo.utils.hashing import sha256_file, sha256_json

VERSION = "airline_outcome_contract_v1"
BEHAVIOR_VERSION = "airline_outcome_contract_v2"
DUAL_VERSION = "airline_outcome_contract_v3_dual"
DUAL_RULE = {
    "version": "airline_dual_success_v1",
    "primary": "equivalent_compliant",
    "secondary": "reference_compliant",
    "reference_selection": "original_reference_in_repaired_manifest",
    "shared_gates": [
        "scorable_termination",
        "non_db_reward_components",
        "path_policy",
        "required_tools_order",
        "communication",
        "prior_consent",
    ],
    "order_policy": "No trace matching; only explicit business dependencies are ordered.",
    "state_policy": "Passenger multiset preserves duplicates; other fields remain exact.",
    "unresolved_policy": "Same planned denominator; withhold both aggregate rates.",
}
USER_SCOPE = """
USER SIMULATION SCOPE (evaluation protocol airline_outcome_contract_v1):
Keep the original requested goals, passenger identities and fixed facts throughout the conversation.
The account holder is not necessarily a booked passenger. A different passenger name is not an
invitation to replace that passenger. Do not invent an additional change to a name, DOB, baggage,
flight, cabin, insurance or payment preference that the original scenario did not request.
Use only the original scenario's facts when answering identity questions. Accept policy-required
refusals and explicitly allowed alternatives, without claiming the original request was completed.
If the assistant asserts an account fact inconsistent with your scenario, ask it to verify with tools;
do not turn an unsupported assistant assertion into a new personal fact. Respect any explicit
prohibition on requesting transfers or supervisors even when the assistant suggests one. An actual
assistant-initiated transfer may end the interaction; do not initiate a new escalation request.
These rules constrain your simulated behavior, not the airline agent's tools or policy.
""".strip()


def canonical_state(state):
    state = copy.deepcopy(state)
    for reservation in state.get("reservations", {}).values():
        if isinstance(reservation.get("passengers"), list):
            reservation["passengers"] = sorted(reservation["passengers"], key=sha256_json)
    return state


def outcome_hash(environment):
    return sha256_json(
        {
            "agent_db": canonical_state(environment.tools.db.model_dump(mode="json")),
            "user_db": environment.get_user_db_hash(),
        }
    )


def fresh_environment(db_path, initial=None):
    env = build_environment(load_flight_db(db_path))
    if initial:
        env.set_state(
            initialization_data=initial.initialization_data,
            initialization_actions=initial.initialization_actions,
            message_history=initial.message_history or [],
            strict=True,
        )
    return env


def execute_actions(db_path, actions, initial=None, *, before_action=None):
    env = fresh_environment(db_path, initial)
    receipts = []
    for i, action in enumerate(actions):
        if before_action:
            before_action(env, action)
        call = message_models()["ToolCall"](
            id=f"reference_{i}",
            name=action["name"],
            requestor=action.get("requestor", "assistant"),
            arguments=action["arguments"],
        )
        response = env.get_response(call)
        if response.error:
            raise ValueError(f"Reference execution failed at {i}: {response.content}")
        receipts.append(
            {
                "name": action["name"],
                "arguments": action["arguments"],
                "response_sha256": sha256_json(response.content),
            }
        )
    return env, receipts


def action_policy_flags(env, action):
    name, args = action["name"], action["arguments"]
    if name not in (
        "update_reservation_flights",
        "update_reservation_passengers",
        "update_reservation_baggages",
    ):
        return []
    db = env.tools.db.model_dump(mode="json")
    r = db["reservations"].get(args.get("reservation_id"))
    out = []
    if r and name == "update_reservation_flights":

        def keys(fs):
            return [(f["flight_number"], f["date"]) for f in fs]

        if r["cabin"] == "basic_economy" and keys(r["flights"]) != keys(args["flights"]):
            out.append("basic_economy_flights_changed_before_separate_upgrade")
    if (
        r
        and name == "update_reservation_passengers"
        and len(r["passengers"]) != len(args["passengers"])
    ):
        out.append("passenger_count_changed")
    if r and name == "update_reservation_baggages" and args["total_baggages"] < r["total_baggages"]:
        out.append("baggage_removed")
    return out


def replay_outcome(db_path, messages, initial=None, *, with_policy=False):
    models = message_models()
    roles = {
        k.lower().replace("message", ""): v for k, v in models.items() if k.endswith("Message")
    }
    env = fresh_environment(db_path)
    flags = []
    original_response = env.get_response

    def observe(call):
        try:
            before = action_policy_flags(env, {"name": call.name, "arguments": call.arguments})
        except (KeyError, TypeError):
            before = []  # Malformed calls are rejected by the native tool.
        response = original_response(call)
        if not response.error:
            flags.extend(dict(call_id=call.id, violation=x) for x in before)
        return response

    if with_policy:
        env.get_response = observe
    flattened = []
    for m in messages:
        flattened.extend(m["tool_messages"] if isinstance(m.get("tool_messages"), list) else [m])
    env.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=[roles[m["role"]].model_validate(m) for m in flattened],
        strict=True,
    )
    return (
        {"outcome_sha256": outcome_hash(env), "policy_violations": flags}
        if with_policy
        else outcome_hash(env)
    )


def runtime_identity():
    """Bind accepted outcomes to policy, native tools, adapter and scorer code."""
    from tau3_grpo.envs import adapter
    from tau3_grpo.envs.tau2_bridge import import_tau2
    from tau3_grpo.evaluation import communication_contract, communication_facts
    from tau3_grpo.prompts import prompt_provenance
    from tau3_grpo.tracking import judge_budget

    modules = {
        name: import_tau2(name)
        for name in (
            "tau2.domains.airline.tools",
            "tau2.domains.airline.data_model",
            "tau2.environment.environment",
        )
    }
    return {
        "prompt": prompt_provenance(),
        "native_sources": {name: sha256_file(module.__file__) for name, module in modules.items()},
        "adapter_sha256": sha256_file(adapter.__file__),
        "scorer_sha256": sha256_file(__file__),
        "communication_sha256": sha256_file(communication_contract.__file__),
        "communication_facts_sha256": sha256_file(communication_facts.__file__),
        "judge_budget_sha256": sha256_file(judge_budget.__file__),
    }


def load_bundle(path, entries, *, require_ready=True):
    bundle = json.loads(Path(path).read_text())
    if bundle.get("version") not in (VERSION, BEHAVIOR_VERSION, DUAL_VERSION) or bundle.get(
        "user_scope_sha256"
    ) != sha256_json(USER_SCOPE):
        raise ValueError("Unknown outcome contract or simulator instructions")
    if bundle.get("runtime_identity") != runtime_identity():
        raise ValueError("Outcome contract policy/tool/scorer identity changed")
    by_id = {e.task_id: e for e in entries}
    if set(by_id) != set(bundle["tasks"]):
        raise ValueError("Contract must cover the exact complete task set; no selective omission")
    if bundle["version"] == DUAL_VERSION and bundle.get("dual_rule") != DUAL_RULE:
        raise ValueError("Unknown dual metric definition")
    for task_id, entry in by_id.items():
        item = bundle["tasks"][task_id]
        if (
            item["task_hash"] != entry.task_hash
            or item["db_hash"] != entry.db_hash
            or item["record_sha256"] != sha256_json(entry.task)
        ):
            raise ValueError(f"Stale contract for {task_id}")
        if bundle["version"] in (BEHAVIOR_VERSION, DUAL_VERSION):
            from tau3_grpo.evaluation.communication_contract import SYSTEM
            from tau3_grpo.evaluation.communication_contract import VERSION as CV

            c = item.get("communication", {})
            if (
                c.get("version") != CV
                or c.get("system_sha256") != sha256_json(SYSTEM)
                or not c.get("requirements")
            ):
                raise ValueError("Missing or stale communication predicates")
            if require_ready and not bundle.get("behavior_validation", {}).get("passed"):
                raise ValueError("Communication calibration is pending")
        if bundle["version"] == DUAL_VERSION:
            references = [
                x for x in item["accepted_outcomes"] if x["variant"] == "original_reference"
            ]
            if (
                len(references) != 1
                or item.get("strict_outcome_sha256") != references[0]["outcome_sha256"]
                or item.get("dual_metric_version") != DUAL_RULE["version"]
            ):
                raise ValueError("Missing or ambiguous frozen reference outcome")
        if require_ready and item["status"] != "approved":
            raise ValueError(f"Task quality gate blocked: {task_id}: {item['status']}")
        if item["status"] == "approved" and not item.get("accepted_outcomes"):
            raise ValueError(f"No executable accepted outcomes: {task_id}")
    return bundle


def score_simulation(
    simulation, *, db_path, task, contract, behavior=None, check_communication=True
):
    if sha256_file(db_path) != contract["db_hash"]:
        raise ValueError("Live database file changed after contract freeze")
    raw = simulation.model_dump(mode="json")
    info = simulation.reward_info
    old = float(info.reward)
    reason = getattr(simulation.termination_reason, "value", str(simulation.termination_reason))
    from tau3_grpo.evaluation.eligibility import SCORABLE_TERMINATIONS

    breakdown = {
        getattr(k, "value", str(k)): float(v) for k, v in (info.reward_breakdown or {}).items()
    }
    basis = [getattr(k, "value", str(k)) for k in info.reward_basis or []]
    state = None
    match = None
    policy_violations = []
    if reason not in SCORABLE_TERMINATIONS:
        reward = 0.0
    else:
        if contract.get("communication"):
            replay = replay_outcome(db_path, raw["messages"], task.initial_state, with_policy=True)
            state = replay["outcome_sha256"]
            policy_violations = replay["policy_violations"]
        else:
            state = replay_outcome(db_path, raw["messages"], task.initial_state)
        match = state in {x["outcome_sha256"] for x in contract["accepted_outcomes"]}
        if "DB" not in {x.upper() for x in basis}:
            raise ValueError("Outcome contract requires a DB-based task")
        reward = 1.0
        for name in basis:
            if name.upper() == "DB":
                reward *= float(match)
            elif name in breakdown:
                reward *= breakdown[name]
            else:
                raise ValueError(f"Missing non-DB reward component {name}")
    if contract.get("communication") and check_communication and reason in SCORABLE_TERMINATIONS:
        from tau3_grpo.evaluation.communication_contract import (
            VERSION as CV,
        )
        from tau3_grpo.evaluation.communication_contract import (
            CommunicationUnresolved,
            requirements,
            visible_events,
        )

        events = visible_events(raw["messages"])
        if (
            not behavior
            or behavior.get("version") != CV
            or type(behavior.get("passed")) is not bool
            or behavior.get("messages_sha256") != sha256_json(events)
            or behavior.get("requirements_sha256") != sha256_json(requirements(events, contract))
            or behavior.get("contract_sha256") != sha256_json(contract["communication"])
        ):
            raise CommunicationUnresolved("Missing or stale communication receipt")
        reward *= float(behavior["passed"])
        reward *= float(not policy_violations)
    result = {
        "version": BEHAVIOR_VERSION if contract.get("communication") else VERSION,
        "legacy_reward": old,
        "reward": reward,
        "behavior": behavior,
        "action_policy_violations": policy_violations,
        "outcome_match": match,
        "actual_outcome_sha256": state,
        "passenger_order": "multiset_preserve_duplicates",
        "other_state_fields": "exact",
        "simulator_rule_compliance": "not_certified_by_outcome_match",
    }
    if contract.get("dual_metric_version"):
        if contract["dual_metric_version"] != DUAL_RULE["version"] or not check_communication:
            raise ValueError("Dual metrics require the frozen full compliance gates")
        reference = contract.get("strict_outcome_sha256")
        if not reference or reference not in {
            x["outcome_sha256"] for x in contract["accepted_outcomes"]
        }:
            raise ValueError("Reference must be a member of accepted outcomes")
        reference_match = state == reference if state is not None else None
        from tau3_grpo.evaluation.metrics import is_benchmark_success

        equivalent_success = is_benchmark_success(reward)
        result.update(
            version=DUAL_VERSION,
            dual_metric_version=DUAL_RULE["version"],
            reference_outcome_match=reference_match,
            matched_variants=[
                x["variant"] for x in contract["accepted_outcomes"] if x["outcome_sha256"] == state
            ],
            reference_compliant_reward=float(equivalent_success and reference_match is True),
            equivalent_compliant_reward=float(equivalent_success),
            non_db_components={k: v for k, v in breakdown.items() if k.upper() != "DB"},
            reward=float(equivalent_success),
        )
    return result
