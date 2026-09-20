import pytest

from tau3_grpo.evaluation.eligibility import execution_eligibility


@pytest.mark.parametrize("reason", ["agent_stop", "user_stop"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_scored_success_and_failure_are_both_valid_candidates(reason, reward):
    p = execution_eligibility(reason, reward=reward)
    assert p["officially_scored"] and p["training_candidate_eligible"]
    assert p["evaluation_trial_complete"] and not p["dynamic_filter_applied"]


@pytest.mark.parametrize("reason", ["max_steps", "timeout", "context_window_exceeded", "length"])
def test_limits_keep_fixed_candidate_budget_without_claiming_scored(reason):
    p = execution_eligibility(reason, reward=0)
    assert p["training_candidate_eligible"] and not p["officially_scored"]
    assert p["evaluation_trial_complete"]
    assert p["automatic_retries"] == 0 and not p["replacement_sampling"]


@pytest.mark.parametrize("reason", ["infrastructure_error", "unexpected_error"])
def test_infrastructure_does_not_become_a_negative_training_example(reason):
    p = execution_eligibility(reason, reward=0)
    assert not p["training_candidate_eligible"] and not p["evaluation_trial_complete"]
    assert p["training_action"] == "abort_batch" and p["explicit_retry_allowed"]


def test_exception_and_unknown_reason_fail_closed():
    assert not execution_eligibility("user_stop", exception=True)["training_candidate_eligible"]
    assert not execution_eligibility("new_unregistered_reason")["training_candidate_eligible"]
    with pytest.raises(ValueError):
        execution_eligibility("max_steps", reward=1)
    with pytest.raises(ValueError):
        execution_eligibility("user_stop", reward=float("nan"))
