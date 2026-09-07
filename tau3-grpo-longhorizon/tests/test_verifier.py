"""Verifier bridge: failure categories and the SimulationRun contract.

The reward itself comes from tau2's `evaluate_simulation`; these tests pin the
project-side logic that surrounds it, in particular the distinction between "the
agent finished and was wrong" and "the agent never finished", which the upstream
reward of 0.0 conflates.
"""

from __future__ import annotations

import pytest

from tau3_grpo.env.verifier import (
    SCORABLE_TERMINATIONS,
    FailureCategory,
    TerminalReward,
    classify_failure,
)


def test_positive_reward_has_no_failure():
    assert classify_failure(1.0, "agent_stop") is FailureCategory.NONE
    assert classify_failure(0.5, "user_stop") is FailureCategory.NONE


def test_finished_but_wrong_is_wrong_outcome():
    assert classify_failure(0.0, "agent_stop") is FailureCategory.WRONG_OUTCOME
    assert classify_failure(0.0, "user_stop") is FailureCategory.WRONG_OUTCOME


@pytest.mark.parametrize("reason", ["max_steps", "timeout"])
def test_turn_limit_is_distinguished_from_wrong_outcome(reason):
    assert classify_failure(0.0, reason) is FailureCategory.TURN_LIMIT


def test_too_many_errors_maps_to_tool_errors():
    assert classify_failure(0.0, "too_many_errors") is FailureCategory.TOOL_ERRORS


@pytest.mark.parametrize("reason", ["agent_error", "context_window_exceeded"])
def test_agent_error_category(reason):
    assert classify_failure(0.0, reason) is FailureCategory.AGENT_ERROR


@pytest.mark.parametrize("reason", ["infrastructure_error", "unexpected_error"])
def test_infrastructure_category(reason):
    assert classify_failure(0.0, reason) is FailureCategory.INFRASTRUCTURE


def test_user_error_maps_to_transfer():
    assert classify_failure(0.0, "user_error") is FailureCategory.USER_TRANSFER


def test_finished_with_tool_errors_is_attributed_to_tools():
    assert (
        classify_failure(0.0, "agent_stop", tool_error_count=3) is FailureCategory.TOOL_ERRORS
    )


def test_unknown_termination_is_unfinished():
    assert classify_failure(0.0, "something_new") is FailureCategory.UNFINISHED


def test_scorable_terminations_match_upstream():
    # Upstream returns reward 0.0 without evaluating for anything else.
    assert SCORABLE_TERMINATIONS == {"agent_stop", "user_stop"}


def test_terminal_reward_solved_flag():
    reward = TerminalReward(
        task_id="airline_1",
        reward=0.0,
        termination_reason="agent_stop",
        failure_category=FailureCategory.WRONG_OUTCOME,
    )
    assert reward.solved is False
    assert TerminalReward(
        task_id="airline_1",
        reward=1.0,
        termination_reason="agent_stop",
        failure_category=FailureCategory.NONE,
    ).solved


def test_terminal_reward_serialises_everything_reports_need():
    reward = TerminalReward(
        task_id="airline_1",
        reward=0.5,
        termination_reason="agent_stop",
        failure_category=FailureCategory.NONE,
        reward_breakdown={"db": 1.0, "communicate": 0.5},
        reward_basis=["db", "communicate"],
        db_hash="after",
        initial_db_hash="before",
        trajectory={"session_id": "s1"},
    )
    payload = reward.to_dict()
    assert payload["reward_breakdown"] == {"db": 1.0, "communicate": 0.5}
    assert payload["reward_basis"] == ["db", "communicate"]
    assert payload["db_hash"] == "after"
    assert payload["initial_db_hash"] == "before"
    assert payload["failure_category"] == "none"
    assert payload["trajectory"]["session_id"] == "s1"


@pytest.mark.tau3
def test_build_simulation_run_matches_tau2_model(requires_tau2):
    from tau3_grpo.env.verifier import build_simulation_run

    run = build_simulation_run(
        task_id="airline_1",
        messages=[],
        termination_reason="agent_stop",
        session_id="s1",
        duration=1.5,
        seed=42,
    )
    assert run.task_id == "airline_1"
    assert run.id == "s1"
    assert run.duration == 1.5
    assert run.seed == 42
    assert run.termination_reason == "agent_stop"


@pytest.mark.tau3
def test_evaluation_type_all_exists(requires_tau2):
    from tau3_grpo.tau2_bridge import evaluator

    assert evaluator().EvaluationType.ALL.value == "all"


@pytest.mark.tau3
def test_termination_reason_enum_covers_our_categories(requires_tau2):
    from tau3_grpo.tau2_bridge import simulation_models

    values = {item.value for item in simulation_models()["TerminationReason"]}
    for reason in (
        "agent_stop",
        "user_stop",
        "max_steps",
        "too_many_errors",
        "agent_error",
        "user_error",
        "infrastructure_error",
        "context_window_exceeded",
        "unexpected_error",
        "timeout",
    ):
        assert reason in values


def test_verify_trajectory_replays_record_specific_db(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tau3_grpo.env import verifier as module

    db_path = tmp_path / "record-db.json"
    db_path.write_text("{}", encoding="utf-8")
    replay_db = object()
    captured = {}

    class _Evaluator:
        class EvaluationType:
            ALL = "all"

        @staticmethod
        def evaluate_simulation(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                reward=1.0,
                reward_breakdown={},
                reward_basis=[],
                info={},
            )

    monkeypatch.setattr(module, "evaluator", lambda: _Evaluator)
    monkeypatch.setattr(module, "load_flight_db", lambda path: replay_db)
    monkeypatch.setattr(
        module,
        "build_simulation_run",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    session = SimpleNamespace(
        task_id="airline_1",
        messages=[],
        session_id="s1",
        seed=42,
        adapted=SimpleNamespace(task=object(), db_path=db_path),
        initial_db_hash="before",
        db_hash=lambda: "after",
        metadata=lambda: {},
    )

    reward = module.verify_trajectory(session, termination_reason="agent_stop")
    assert reward.reward == 1.0
    assert captured["env_kwargs"] == {"db": replay_db}
