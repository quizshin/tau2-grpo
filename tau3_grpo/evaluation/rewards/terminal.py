"""Versioned RL terminal selection; native evaluator results remain auditable.

The passenger protocol replaces only the DB component with a complete-state
comparison that ignores passenger permutation. It is not the broader compliant
outcome contract and adds no policy, communication, or partial-credit rewards.
"""
from __future__ import annotations

import math
from dataclasses import replace

from tau3_grpo.evaluation.verifier import SCORABLE_TERMINATIONS, classify_failure

NATIVE = "tau2_native_v1"
PASSENGER_MULTISET = "airline_passenger_multiset_v1"
PROTOCOLS = (NATIVE, PASSENGER_MULTISET)


def validate_protocol(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown terminal reward protocol: {protocol}")
    return protocol


def select_terminal_reward(session, native, protocol=NATIVE):
    """Select a training terminal reward after the strict native verification.

    The saved SimulationRun and the input TerminalReward are never modified.
    Native components and the selection receipt travel in verifier_info_json.
    Gold actions execute on an independent database; invalid references fail
    closed rather than assigning a guessed training label.
    """
    validate_protocol(protocol)
    if protocol == NATIVE:
        return native
    receipt = {
        "protocol": protocol,
        "native_reward": native.reward,
        "native_breakdown": dict(native.reward_breakdown),
        "native_basis": list(native.reward_basis),
        "native_failure_category": native.failure_category.value,
        "scored": native.scored,
        "applied": False,
    }
    reward, breakdown = native.reward, dict(native.reward_breakdown)
    if native.scored and native.termination_reason in SCORABLE_TERMINATIONS and "DB" in native.reward_basis:
        from tau3_grpo.evaluation.outcome_contract import execute_actions, outcome_hash

        if any(key not in breakdown or not math.isfinite(breakdown[key]) for key in native.reward_basis):
            raise ValueError("Missing or nonfinite native reward components")
        if session.environment.get_db_hash() != native.db_hash:
            raise ValueError("Session state changed after native verification")
        task = session.adapted.task
        actions = [action.model_dump(mode="json") for action in task.evaluation_criteria.actions or []]
        reference, _ = execute_actions(session.adapted.db_path, actions, task.initial_state)
        predicted_hash, reference_hash = outcome_hash(session.environment), outcome_hash(reference)
        breakdown["DB"] = float(predicted_hash == reference_hash)
        reward = math.prod(breakdown[key] for key in native.reward_basis)
        receipt.update(applied=True, predicted_outcome_hash=predicted_hash,
                       reference_outcome_hash=reference_hash)
    receipt.update(selected_reward=reward, selected_breakdown=breakdown)
    return replace(native, reward=reward, reward_breakdown=breakdown,
                   failure_category=(classify_failure(reward, native.termination_reason)
                                     if reward != native.reward else native.failure_category),
                   info={**native.info, "terminal_reward_selection": receipt})
