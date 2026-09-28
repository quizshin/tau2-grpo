import pytest

from tau3_grpo.evaluation.outcome_contract import (
    USER_SCOPE,
    VERSION,
    canonical_state,
    load_bundle,
    runtime_identity,
)
from tau3_grpo.utils.hashing import sha256_json


def test_passenger_order_is_not_identity_or_count_or_flight_order():
    a = {
        "reservations": {"R": {"passengers": [{"name": "A"}, {"name": "B"}], "flights": ["X", "Y"]}}
    }
    b = {
        "reservations": {"R": {"passengers": [{"name": "B"}, {"name": "A"}], "flights": ["X", "Y"]}}
    }
    assert canonical_state(a) == canonical_state(b)
    b["reservations"]["R"]["passengers"].append({"name": "A"})
    assert canonical_state(a) != canonical_state(b)
    assert a["reservations"]["R"]["passengers"][0] == {"name": "A"}
    b["reservations"]["R"]["passengers"].pop()
    b["reservations"]["R"]["flights"].reverse()
    assert canonical_state(a) != canonical_state(b)


def test_contract_rejects_subset_stale_or_pending(tmp_path):
    import json
    from types import SimpleNamespace

    entry = SimpleNamespace(task_id="T", task_hash="t", db_hash="d", task={"id": "T"})
    item = dict(
        task_hash="t",
        db_hash="d",
        record_sha256=sha256_json(entry.task),
        status="pending",
        accepted_outcomes=[],
    )
    b = dict(
        version=VERSION,
        user_scope_sha256=sha256_json(USER_SCOPE),
        tasks={"T": item},
        runtime_identity=runtime_identity(),
    )
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="quality gate"):
        load_bundle(p, [entry])
    with pytest.raises(ValueError, match="complete task set"):
        load_bundle(p, [])
    item["status"] = "approved"
    item["accepted_outcomes"] = [{"outcome_sha256": "x"}]
    p.write_text(json.dumps(b))
    assert load_bundle(p, [entry])
    b["runtime_identity"]["scorer_sha256"] = "stale"
    p.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="identity changed"):
        load_bundle(p, [entry])
    b["runtime_identity"] = runtime_identity()
    p.write_text(json.dumps(b))
    entry.task = {"id": "modified"}
    with pytest.raises(ValueError, match="Stale"):
        load_bundle(p, [entry])


@pytest.mark.parametrize(
    "communication,reason,expected",
    [(0.0, "user_stop", 0.0), (1.0, "user_stop", 1.0), (1.0, "max_steps", 0.0)],
)
def test_outcome_match_does_not_bypass_communication_or_termination(
    monkeypatch, tmp_path, communication, reason, expected
):
    from types import SimpleNamespace

    from tau3_grpo.evaluation import outcome_contract as oc

    db = tmp_path / "db.json"
    db.write_text("{}")
    monkeypatch.setattr(oc, "replay_outcome", lambda *a: "expected")
    simulation = SimpleNamespace(
        model_dump=lambda **k: {"messages": []},
        reward_info=SimpleNamespace(
            reward=0.0,
            reward_breakdown={"DB": 0.0, "COMMUNICATE": communication},
            reward_basis=["DB", "COMMUNICATE"],
        ),
        termination_reason=SimpleNamespace(value=reason),
    )
    result = oc.score_simulation(
        simulation,
        db_path=db,
        task=SimpleNamespace(initial_state=None),
        contract={
            "db_hash": oc.sha256_file(db),
            "accepted_outcomes": [{"outcome_sha256": "expected"}],
        },
    )
    assert result["reward"] == expected
    assert simulation.reward_info.reward == 0.0


def test_new_contract_cannot_compare_to_legacy_or_other_contract(tmp_path):
    from test_evaluation_compare import artifact, change

    from tau3_grpo.evaluation.compare import compare_evaluations

    base, treat = artifact(tmp_path / "base"), artifact(tmp_path / "treat")
    change(treat, lambda d: d["provenance"].update(quality_bundle_sha256="v2"))
    assert not compare_evaluations(base, treat)["comparable"]
    change(base, lambda d: d["provenance"].update(quality_bundle_sha256="other"))
    assert not compare_evaluations(base, treat)["comparable"]
    change(base, lambda d: d["provenance"].update(quality_bundle_sha256="v2"))
    assert compare_evaluations(base, treat)["comparable"]


def test_duplicate_cancellation_respects_24_hours_and_already_flown():
    from tau3_grpo.analysis.prepare_outcome_contract import duplicate_cancellation_allowed

    r = {
        "cabin": "economy",
        "created_at": "2024-05-15T10:00:00",
        "flights": [{"flight_number": "F", "date": "2024-05-25"}],
    }
    db = {"flights": {"F": {"dates": {"2024-05-25": {"status": "available"}}}}}
    assert duplicate_cancellation_allowed(r, db)
    r["created_at"] = "2024-05-10T10:00:00"
    assert not duplicate_cancellation_allowed(r, db)
    r["cabin"] = "business"
    assert duplicate_cancellation_allowed(r, db)
    db["flights"]["F"]["dates"]["2024-05-25"]["status"] = "landed"
    assert not duplicate_cancellation_allowed(r, db)


def test_reference_receipt_uses_native_serializer_and_errors(monkeypatch):
    from types import SimpleNamespace

    from tau3_grpo.evaluation import outcome_contract as oc

    monkeypatch.setattr(
        oc, "message_models", lambda: {"ToolCall": lambda **k: SimpleNamespace(**k)}
    )
    response = SimpleNamespace(error=False, content='[{"nested_flight": {"date": "2024-05-25"}}]')
    env = SimpleNamespace(get_response=lambda call: response)
    monkeypatch.setattr(oc, "fresh_environment", lambda *a: env)
    _, receipts = oc.execute_actions("db", [{"name": "search_direct_flight", "arguments": {}}])
    assert receipts[0]["response_sha256"] == sha256_json(response.content)
    response.error = True
    with pytest.raises(ValueError, match="Reference execution failed"):
        oc.execute_actions("db", [{"name": "search_direct_flight", "arguments": {}}])


def test_reference_repair_uses_separate_upgrade_without_mutating_original():
    import copy
    from types import SimpleNamespace

    from tau3_grpo.analysis.prepare_outcome_contract import repair_known_reference_actions

    action = dict(
        name="update_reservation_flights",
        arguments=dict(
            reservation_id="R",
            cabin="economy",
            flights=[dict(flight_number="NEW", date="2024-05-18")],
            payment_id="CARD",
        ),
    )
    original = copy.deepcopy(action)
    record = SimpleNamespace(id="airline_698", evaluation_criteria={"actions": [action]})
    db = {
        "reservations": {
            "R": dict(cabin="basic_economy", flights=[dict(flight_number="OLD", date="2024-05-18")])
        }
    }
    actions, changes = repair_known_reference_actions(record, db)
    assert actions[0]["arguments"]["flights"][0]["flight_number"] == "OLD"
    assert actions[1] == original and action == original
    assert changes and len(actions) == 2
    db["reservations"]["R"]["cabin"] = "business"
    with pytest.raises(ValueError, match="no longer applies"):
        repair_known_reference_actions(record, db)


def test_compensation_repair_does_not_pay_delayed_extra_without_change():
    from types import SimpleNamespace

    from tau3_grpo.analysis.prepare_outcome_contract import repair_known_reference_actions

    record = SimpleNamespace(
        id="airline_925",
        evaluation_criteria={
            "actions": [
                dict(
                    name="get_flight_status",
                    arguments=dict(flight_number="HAT273", date="2024-05-15"),
                ),
                dict(name="send_certificate", arguments=dict(user_id="U", amount=300)),
            ]
        },
    )
    db = {
        "reservations": {
            "GCZ58I": dict(
                passengers=[{}, {}], flights=[dict(flight_number="HAT273", date="2024-05-11")]
            )
        },
        "flights": {"HAT273": {"dates": {"2024-05-11": {"status": "cancelled"}}}},
    }
    actions, changes = repair_known_reference_actions(record, db)
    assert actions[0]["arguments"]["date"] == "2024-05-11"
    assert actions[1]["arguments"]["amount"] == 200 and len(changes) == 2
    assert record.evaluation_criteria["actions"][1]["arguments"]["amount"] == 300


def test_question_repair_aligns_deadline_with_legal_gold_without_mutation():
    import copy
    from types import SimpleNamespace

    from tau3_grpo.data.selection_repair import patch_record

    gold = dict(
        name="book_reservation",
        arguments=dict(flights=[dict(flight_number="OG3IHSH", date="2024-05-16")]),
    )
    record = SimpleNamespace(
        id="airline_393",
        description={"purpose": "before05:00"},
        user_scenario={"instructions": {"task_instructions": "before05:00"}},
        evaluation_criteria={"actions": [gold]},
    )
    db = {
        "users": {
            "raj_young_5c9945": dict(
                name="Raj", membership="silver", payment_methods={}, reservations=[]
            )
        },
        "flights": {
            "OG3IHSH": dict(
                origin="LGA",
                destination="CLT",
                scheduled_departure_time_est="06:00:00",
                dates={
                    "2024-05-16": dict(
                        status="available", available_seats={"economy": 1}, prices={"economy": 240}
                    )
                },
            )
        },
    }
    before = copy.deepcopy(record.__dict__)
    db_before = copy.deepcopy(db)
    revised, review = patch_record(record, db)
    assert record.__dict__ == before and db == db_before
    assert revised.evaluation_criteria["actions"] == [gold]
    assert "BEFORE 07:00" in revised.user_scenario["instructions"]["task_instructions"]
    assert "07:00" in revised.description["purpose"]
    db["flights"]["OG3IHSH"]["dates"]["2024-05-16"]["available_seats"]["economy"] = 0
    with pytest.raises(AssertionError):
        patch_record(record, db)


def test_rank_candidates_filters_seats_status_dates_and_clock():
    from tau3_grpo.data.selection_repair import candidates, choose

    def flight(clock, seats=2, status="available"):
        return dict(
            origin="A",
            destination="B",
            scheduled_departure_time_est=clock,
            dates={
                "2024-05-16": dict(
                    status=status, available_seats={"economy": seats}, prices={"economy": 10}
                )
            },
        )

    db = {
        "flights": {
            "early": flight("04:00:00"),
            "sold": flight("12:00:00", 1),
            "cancelled": flight("13:00:00", status="cancelled"),
            "ok": flight("14:00:00"),
        }
    }
    opts = candidates(db, "A", "B", ["2024-05-16"], "economy", 2, lo="08:00:00", hi="21:00:00")
    assert [x["flight_number"] for x in opts] == ["ok"]
    assert choose(opts, "latest")["departure"] == "14:00:00"
    assert candidates(db, "A", "B", ["2024-05-16"], "economy", 2, lo="18:00:00") == []


def behavior_fixture():
    from tau3_grpo.evaluation.communication_contract import requirements, visible_events

    c = {
        "communication": {
            "requirements": ["Explain completion."],
            "required_successful_tools": [],
            "action_order": [],
        }
    }
    events = visible_events(
        [
            {"role": "assistant", "content": "Cancel R and refund $100 to your original card?"},
            {"role": "user", "content": "Yes, cancel R."},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "name": "cancel_reservation", "arguments": {"reservation_id": "R"}}
                ],
            },
            {"role": "tool", "id": "c1", "content": '{"status":"cancelled"}', "error": False},
            {"role": "assistant", "content": "R has been cancelled."},
        ]
    )
    p = {
        "checks": [
            {
                "id": "goal_0",
                "status": "pass",
                "reason_code": "evidence",
                "reason": "Explains receipt",
                "evidence": [{"event_id": "m0004", "quote": "R has been cancelled."}],
            },
            {
                "id": "consent_c1",
                "status": "pass",
                "reason_code": "evidence",
                "reason": "Confirmed before write",
                "evidence": [
                    {
                        "event_id": "m0000",
                        "quote": "Cancel R and refund $100 to your original card?",
                    },
                    {"event_id": "m0001", "quote": "Yes, cancel R."},
                ],
            },
        ],
        "simulator_scope": {"status": "pass", "evidence": [], "reason": "No drift"},
    }
    p["checks"].append(
        dict(
            id="answer_coverage",
            status="pass",
            reason_code="evidence",
            reason="Request answered",
            evidence=[dict(event_id="m0004", quote="R has been cancelled.")],
        )
    )
    p["checks"].append(
        dict(
            id="time_window_accuracy",
            status="pass",
            reason_code="evidence",
            reason="No time-window claim",
            evidence=[dict(event_id="m0004", quote="R has been cancelled.")],
        )
    )
    return c, events, requirements(events, c), p


def test_behavior_quotes_consent_and_native_tool_ids():
    from tau3_grpo.evaluation.communication_contract import structural_checks, validate_packet

    c, e, r, p = behavior_fixture()
    assert validate_packet(p, e, r)["passed"]
    c["communication"]["required_successful_tools"] = ["cancel_reservation"]
    assert structural_checks(e, c)["passed"]
    e[3]["error"] = True
    assert not structural_checks(e, c)["passed"]


def test_behavior_forged_quote_and_posthoc_consent_fail_closed():
    from tau3_grpo.evaluation.communication_contract import CommunicationUnresolved, validate_packet

    c, e, r, p = behavior_fixture()
    p["checks"][0]["evidence"][0]["quote"] = "invented"
    with pytest.raises(CommunicationUnresolved, match="quote"):
        validate_packet(p, e, r)
    c, e, r, p = behavior_fixture()
    r[1]["call"]["event_index"] = 1
    with pytest.raises(CommunicationUnresolved, match="precede|assistant language"):
        validate_packet(p, e, r)


def test_behavior_unknown_and_simulator_drift_are_not_model_zero():
    from tau3_grpo.evaluation.communication_contract import CommunicationUnresolved, validate_packet

    c, e, r, p = behavior_fixture()
    p["checks"][0].update(status="unknown", reason_code="ambiguous")
    with pytest.raises(CommunicationUnresolved, match="Unknown"):
        validate_packet(p, e, r)
    c, e, r, p = behavior_fixture()
    p["simulator_scope"].update(
        status="fail", evidence=[{"event_id": "m0001", "quote": "Yes, cancel R."}]
    )
    with pytest.raises(CommunicationUnresolved, match="Simulator"):
        validate_packet(p, e, r)


def test_behavior_empty_goodbye_is_a_failure_not_successful_noop():
    from tau3_grpo.evaluation.communication_contract import validate_packet

    events = [dict(event_id="m0000", role="assistant", content="Goodbye.")]
    req = [dict(id="refusal", criterion="Explain cancellation ineligibility.")]
    p = {
        "checks": [
            dict(
                id="refusal",
                status="fail",
                reason_code="missing",
                reason="No refusal explanation.",
                evidence=[],
            )
        ],
        "simulator_scope": dict(status="pass", reason="No drift", evidence=[]),
    }
    assert validate_packet(p, events, req)["passed"] is False


def test_required_transfer_and_compensation_order_use_executed_calls():
    from tau3_grpo.evaluation.communication_contract import structural_checks, visible_events

    contract = {
        "communication": {
            "required_successful_tools": ["transfer_to_human_agents"],
            "action_order": [],
        }
    }
    assert not structural_checks(
        visible_events([{"role": "assistant", "content": "You are transferred."}]), contract
    )["passed"]
    calls = [
        {"id": "cert", "name": "send_certificate", "arguments": {}},
        {"id": "cancel", "name": "cancel_reservation", "arguments": {"reservation_id": "R"}},
    ]
    events = visible_events(
        [
            dict(role="assistant", tool_calls=calls),
            dict(
                role="tool",
                tool_messages=[
                    dict(role="tool", id="cert", content="ok", error=False),
                    dict(role="tool", id="cancel", content="ok", error=False),
                ],
            ),
        ]
    )
    contract["communication"] = {
        "required_successful_tools": [],
        "action_order": [
            dict(
                before="cancel_reservation",
                before_arguments={"reservation_id": "R"},
                after="send_certificate",
            )
        ],
    }
    assert not structural_checks(events, contract)["passed"]
    events[0]["tool_calls"].reverse()
    assert structural_checks(events, contract)["passed"]


def test_budget_common_module_keeps_existing_ledger_implementation():
    from tau3_grpo.analysis.rubric_pilot import Budget as old
    from tau3_grpo.tracking.judge_budget import Budget as new

    assert old is new


def test_tool_call_field_is_valid_evidence_but_not_user_consent():
    from tau3_grpo.evaluation.communication_contract import CommunicationUnresolved, validate_packet

    c, e, r, p = behavior_fixture()
    p["checks"][1]["evidence"].append({"event_id": "m0002", "quote": "cancel_reservation"})
    assert validate_packet(p, e, r)["passed"]
    p["checks"][1]["evidence"] = [{"event_id": "m0002", "quote": "cancel_reservation"}]
    with pytest.raises(CommunicationUnresolved, match="precede|assistant language"):
        validate_packet(p, e, r)


def test_failed_behavior_cannot_be_overridden_by_correct_database(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tau3_grpo.evaluation import outcome_contract as oc
    from tau3_grpo.evaluation.communication_contract import (
        VERSION as CV,
    )
    from tau3_grpo.evaluation.communication_contract import (
        CommunicationUnresolved,
        requirements,
    )

    c, e, r, p = behavior_fixture()
    db = tmp_path / "db"
    db.write_text("{}")
    c.update(db_hash=oc.sha256_file(db), accepted_outcomes=[{"outcome_sha256": "right"}])
    monkeypatch.setattr(
        oc, "replay_outcome", lambda *a, **k: dict(outcome_sha256="right", policy_violations=[])
    )
    sim = SimpleNamespace(
        model_dump=lambda **k: {"messages": e},
        termination_reason=SimpleNamespace(value="user_stop"),
        reward_info=SimpleNamespace(
            reward=1,
            reward_breakdown={"DB": 1, "COMMUNICATE": 1},
            reward_basis=["DB", "COMMUNICATE"],
        ),
    )
    receipt = dict(
        version=CV,
        passed=False,
        messages_sha256=sha256_json(e),
        requirements_sha256=sha256_json(requirements(e, c)),
        contract_sha256=sha256_json(c["communication"]),
    )
    assert (
        oc.score_simulation(
            sim, db_path=db, task=SimpleNamespace(initial_state=None), contract=c, behavior=receipt
        )["reward"]
        == 0
    )
    receipt["passed"] = True
    assert (
        oc.score_simulation(
            sim, db_path=db, task=SimpleNamespace(initial_state=None), contract=c, behavior=receipt
        )["reward"]
        == 1
    )
    receipt["messages_sha256"] = "stale"
    with pytest.raises(CommunicationUnresolved, match="stale"):
        oc.score_simulation(
            sim, db_path=db, task=SimpleNamespace(initial_state=None), contract=c, behavior=receipt
        )


def test_policy_violation_cannot_be_hidden_by_equal_final_state(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tau3_grpo.evaluation import outcome_contract as oc
    from tau3_grpo.evaluation.communication_contract import VERSION as CV
    from tau3_grpo.evaluation.communication_contract import requirements

    c, e, r, p = behavior_fixture()
    db = tmp_path / "db"
    db.write_text("{}")
    c.update(db_hash=oc.sha256_file(db), accepted_outcomes=[{"outcome_sha256": "right"}])
    monkeypatch.setattr(
        oc,
        "replay_outcome",
        lambda *a, **k: dict(
            outcome_sha256="right", policy_violations=[{"violation": "baggage_removed"}]
        ),
    )
    sim = SimpleNamespace(
        model_dump=lambda **k: {"messages": e},
        termination_reason=SimpleNamespace(value="user_stop"),
        reward_info=SimpleNamespace(reward=1, reward_breakdown={"DB": 1}, reward_basis=["DB"]),
    )
    receipt = dict(
        version=CV,
        passed=True,
        messages_sha256=sha256_json(e),
        requirements_sha256=sha256_json(requirements(e, c)),
        contract_sha256=sha256_json(c["communication"]),
    )
    result = oc.score_simulation(
        sim, db_path=db, task=SimpleNamespace(initial_state=None), contract=c, behavior=receipt
    )
    assert result["reward"] == 0 and result["action_policy_violations"]


def test_tool_observation_alone_is_not_an_assistant_explanation():
    from tau3_grpo.evaluation.communication_contract import CommunicationUnresolved, validate_packet

    c, e, r, p = behavior_fixture()
    p["checks"][0]["evidence"] = [{"event_id": "m0003", "quote": "cancelled"}]
    with pytest.raises(CommunicationUnresolved, match="assistant language"):
        validate_packet(p, e, r)


def test_unavailable_intermediate_cabin_is_not_an_accepted_two_step_path():
    from types import SimpleNamespace

    from tau3_grpo.analysis.prepare_outcome_contract import variants

    r = dict(cabin="business", passengers=[{}], flights=[dict(flight_number="OLD", date="D")])
    db = {
        "reservations": {"R": r},
        "flights": {
            "OLD": {"dates": {"D": {"available_seats": {"economy": 2, "business": 2}}}},
            "NEW": {
                "dates": {
                    "D": {"status": "available", "available_seats": {"economy": 2, "business": 0}}
                }
            },
        },
    }
    rec = SimpleNamespace(
        id="unrelated",
        evaluation_criteria={
            "actions": [
                dict(
                    name="update_reservation_flights",
                    arguments=dict(
                        reservation_id="R",
                        cabin="economy",
                        flights=[dict(flight_number="NEW", date="D")],
                        payment_id="P",
                    ),
                )
            ]
        },
    )
    opts = variants(rec, db)
    assert not any("flight_then_cabin" in name for name, _ in opts)
    assert any("cabin_then_flight" in name for name, _ in opts)


def test_smoke_only_contract_cannot_start_formal_evaluation(monkeypatch, tmp_path):
    from tau3_grpo.evaluation import outcome_contract as oc
    from tau3_grpo.evaluation import runtime as rt

    monkeypatch.setattr(oc, "load_bundle", lambda *a, **k: {"approval_scope": "smoke_only"})
    monkeypatch.setattr(
        rt, "_selection_jobs", lambda *a, **k: pytest.fail("must stop before schedule/calls")
    )
    with pytest.raises(ValueError, match="smoke only"):
        rt.run_evaluation(
            spec=rt.EvalSpec(target="selection", trials=4, quality_bundle="dummy"),
            policy=rt.Endpoint("p", "http://p"),
            user=rt.Endpoint("u", "http://u"),
            output_dir=tmp_path,
        )
