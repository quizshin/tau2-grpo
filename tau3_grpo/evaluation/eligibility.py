"""Shared execution validity, separate from algorithm-specific dynamic filtering.

v1 preserves capped trajectories as zero-outcome candidates. Infrastructure
failures abort a training batch and leave evaluation trials unresolved. There is
no automatic replacement sampling or retry that silently changes the budget.
"""
from __future__ import annotations

import math

POLICY_VERSION = "tau3_execution_eligibility_v1"
SCORABLE_TERMINATIONS = frozenset({"agent_stop", "user_stop"})
INFRASTRUCTURE_TERMINATIONS = frozenset({"infrastructure_error", "unexpected_error"})
LIMIT_TERMINATIONS = frozenset({"max_steps", "timeout", "context_window_exceeded", "length"})
FALLBACK_TERMINATIONS = LIMIT_TERMINATIONS | {"too_many_errors", "agent_error", "user_error"}


def execution_eligibility(reason, *, reward=None, exception=False):
    reason = getattr(reason, "value", reason)
    infra = exception or reason in INFRASTRUCTURE_TERMINATIONS
    if reward is not None and (not isinstance(reward, (int, float)) or not math.isfinite(reward)):
        raise ValueError("Execution outcome must be finite")
    if reason not in SCORABLE_TERMINATIONS and reward not in (None, 0, 0.0):
        raise ValueError("Unscored termination cannot carry a positive official outcome")
    known = reason in SCORABLE_TERMINATIONS or reason in FALLBACK_TERMINATIONS
    category = ("infrastructure_error" if infra else "officially_scored" if reason in SCORABLE_TERMINATIONS
                else "budget_limit" if reason in LIMIT_TERMINATIONS else "unscored_failure" if known
                else "unknown_termination")
    eligible = known and not infra
    return {"policy": POLICY_VERSION, "category": category,
            "officially_scored": reason in SCORABLE_TERMINATIONS and not infra,
            "training_candidate_eligible": eligible,
            "evaluation_trial_complete": eligible,
            "training_action": "retain_candidate" if eligible else "abort_batch",
            "evaluation_action": "record_trial" if eligible else "record_unresolved_error",
            "automatic_retries": 0, "replacement_sampling": False,
            "explicit_retry_allowed": infra,
            "dynamic_filter_applied": False}
