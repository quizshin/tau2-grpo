import json

import pytest

from tau3_grpo.analysis.capability_distribution import confirmation_screen, features, summarize


def test_final_aggregation_does_not_export_task_text_ids_or_arguments():
    secret = "PRIVATE_RESERVATION_ID"
    row = features(
        "Private passenger needs one bag",
        [("update_reservation_baggages", {"reservation_id": secret})],
    )
    out = summarize([row], "reference_actions_not_necessary_minimum")
    encoded = json.dumps(out)
    assert secret not in encoded and "Private passenger" not in encoded
    assert "reservation_ids" not in encoded
    assert out["tool_presence"]["update_reservation_baggages"] == 1
    assert out["tool_coverage_is_necessity"] is False


def test_source_difficulty_proxy_does_not_use_teacher_length():
    reason = "Move only the return flight after 5pm without changing the cabin"
    short = features(reason, [])
    long = features(reason, [("get_user_details", {"user_id": "U"})] * 100)
    assert short["intent_proxy_bin"] == long["intent_proxy_bin"]
    assert short["tool_calls"] != long["tool_calls"]


def test_missing_confirmation_exchange_is_flagged_without_inventing_consent():
    call = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "cancel_reservation", "arguments": {"reservation_id": "R"}}}
        ],
    }
    assert confirmation_screen([{"role": "user", "content": "Cancel R"}, call])
    # Only a necessary-condition screen, not a semantic approval validator.
    assert not confirmation_screen(
        [
            {"role": "assistant", "content": "May I cancel R?"},
            {"role": "user", "content": "yes"},
            call,
        ]
    )


def test_grounded_pilot_rejects_missing_flights_and_policy_ineligible_cancel():
    from tau3_grpo.data.grounded_gap_pilot import feasible

    db = {
        "users": {"U": {"membership": "regular", "payment_methods": {}}},
        "reservations": {
            "R": {
                "user_id": "U",
                "cabin": "economy",
                "insurance": "no",
                "created_at": "2024-05-10T10:00:00",
                "passengers": [{}],
                "payment_history": [{"amount": 100}],
                "flights": [{"flight_number": "F", "date": "2024-05-25"}],
            }
        },
        "flights": {},
    }
    assert not feasible(db, "R", "cancel_allowed")
    db["flights"]["F"] = {"dates": {"2024-05-25": {"status": "available"}}}
    assert not feasible(db, "R", "cancel_allowed")
    assert feasible(db, "R", "cancel_denied")
    db["reservations"]["R"]["cabin"] = "business"
    assert feasible(db, "R", "cancel_allowed")
    assert not feasible(db, "R", "cancel_denied")
    db["reservations"]["R"]["status"] = "cancelled"
    assert not feasible(db, "R", "cancel_allowed")


def test_quality_review_requires_evidence_and_no_contradictory_eligibility():
    import copy

    import pytest

    from tau3_grpo.analysis.quality_review import AXES, validate

    events = [{"event_id": "m000", "content": "Cancel my reservation only if allowed."}]
    packet = {
        "verdict": "eligible_candidate",
        "checks": [
            dict(area=a, status="satisfied", refs=["m000"])
            for a in ("scope", "evidence", "policy", "arithmetic", "completion")
        ],
        "issues": [],
        "uncertainties": [],
        "difficulty": {a: dict(level=1, refs=["m000"]) for a in AXES},
    }
    validate(packet, events)
    bad = copy.deepcopy(packet)
    bad["checks"][0]["status"] = "unknown"
    with pytest.raises(ValueError, match="conflicts"):
        validate(bad, events)
    bad = copy.deepcopy(packet)
    bad["verdict"] = "hold"
    bad["issues"] = [dict(refs=["m000"], quote="invented quote", severity="major")]
    with pytest.raises(ValueError, match="quotation"):
        validate(bad, events)


def test_repair_selection_rejects_changed_evidence_and_returned_dev_user(tmp_path):
    import copy

    import pytest

    from tau3_grpo.analysis.rubric_pilot import visible_events
    from tau3_grpo.data.finalize_staged_sft import adjudicated_sources, visible_user_ids
    from tau3_grpo.utils.hashing import sha256_file, sha256_json

    row = dict(
        messages=[dict(role="tool", content=json.dumps({"user_id": "HELDOUT"}))],
        metadata={"source_dialog_id": "test"},
        supervision={"basis": "original"},
    )
    assert visible_user_ids(row) == {"HELDOUT"}
    records = tmp_path / "records"
    records.mkdir()
    path = records / "test.json"
    digest = sha256_json(visible_events(row["messages"]))
    path.write_text(json.dumps(dict(evidence_sha256=digest, review={"difficulty": {}})))
    decision = dict(
        id="test",
        decision="include_experimental",
        review_evidence_sha256=digest,
        review_record_sha256=sha256_file(path),
    )
    accepted, held = adjudicated_sources([copy.deepcopy(row)], tmp_path, [decision], {"HELDOUT"})
    assert not accepted and held[0]["reason"] == "heldout_entity"
    changed = copy.deepcopy(row)
    changed["messages"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        adjudicated_sources([changed], tmp_path, [decision], set())
    with pytest.raises(ValueError, match="exact unique"):
        adjudicated_sources([row], tmp_path, [], set())


def test_decision_repair_filters_nonavailable_fares_and_time_boundary():
    import copy

    from tau3_grpo.data.decision_repair import eligible

    db = {
        "users": {
            "u": {
                "payment_methods": {"card": {"source": "credit_card"}},
                "name": {"first_name": "A", "last_name": "B"},
                "dob": "1990-01-01",
            }
        },
        "reservations": {
            "r": {
                "user_id": "u",
                "status": None,
                "passengers": [{"first_name": "A", "last_name": "B"}],
                "cabin": "economy",
                "insurance": "no",
                "created_at": "2024-05-14T15:00:00",
                "payment_history": [{"amount": 100}],
                "flights": [{"flight_number": "F", "date": "2024-05-17", "price": 100}],
            }
        },
        "flights": {
            "F": {
                "dates": {
                    "2024-05-17": {
                        "status": "available",
                        "prices": {"economy": 120, "business": 300},
                        "available_seats": {"economy": 2, "business": 2},
                    }
                }
            }
        },
    }
    assert eligible(db, "r", "cancel_within24")
    assert not eligible(db, "r", "cancel_after24")
    db["reservations"]["r"]["created_at"] = "2024-05-14T14:59:00"
    assert eligible(db, "r", "cancel_after24")
    db["reservations"]["r"]["created_at"] = "2024-05-15T15:01:00"
    assert not eligible(db, "r", "cancel_within24")
    assert eligible(db, "r", "historical_upgrade")
    cancelled = copy.deepcopy(db)
    cancelled["flights"]["F"]["dates"]["2024-05-17"] = {"status": "cancelled"}
    assert not eligible(cancelled, "r", "historical_upgrade")


def test_source_inventory_preserves_target_positions_and_reason_join(tmp_path):
    from tau3_grpo.analysis.sft_source_inventory import inventory

    prefix = [{'role': 'system', 'content': 'policy'}, {'role': 'user', 'content': 'Check account'}]
    first_answer = {'role': 'assistant', 'content': 'Please provide your user ID.'}
    second_prefix = prefix + [first_answer, {'role': 'user', 'content': 'U'}]
    answer = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'function': {'name': 'get_user_details', 'arguments': {'user_id': 'U'}}}]}
    source = []
    for turn, messages, reply in [(0, prefix, first_answer), (1, second_prefix, answer)]:
        source.append({'metadata': {'source_dialog_id': 'airline_dialog_1', 'turn_index': turn,
                                   'correct': 1, 'reward': 1.0, 'reason_for_call': 'Check account'},
                       'messages': messages, 'answer': reply})
    (tmp_path / 'tau2_sft_train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in source))
    (tmp_path / 'tau2_rl_train.jsonl').write_text(json.dumps({
        'id': 'airline_1', 'user_scenario': {'instructions': {'domain': 'airline',
                                                           'reason_for_call': 'Check account'}}}) + '\n')
    result = inventory(tmp_path)
    assert result['summary']['prefix_mismatch_rows'] == 0
    assert result['summary']['answer_mismatch_rows'] == 0
    assert result['records'][0]['provided_target_message_indices'] == [2, 4]
    assert result['records'][0]['exact_reason_rl_ids'] == ['airline_1']
    # Source flags remain source flags, never frozen semantic acceptance.
    assert 'quality_accepted' not in result['records'][0]


@pytest.mark.parametrize('module,option', [
    ('tau3_grpo.data.rl_curriculum_screen', '--out'),
    ('tau3_grpo.data.rl_curriculum_extend', '--output'),
    ('tau3_grpo.analysis.sft_source_inventory', '--output'),
])
def test_promoted_commands_reject_existing_outputs_before_reading_sources(module, option, tmp_path):
    import importlib

    from pytest import raises

    existing = tmp_path / 'frozen'
    existing.mkdir()
    with raises(FileExistsError, match='new|fresh'):
        importlib.import_module(module).main([option, str(existing)])
    assert list(existing.iterdir()) == []
