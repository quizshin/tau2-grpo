from copy import deepcopy

import pytest

from tau3_grpo.evaluation.process_reward import (
    deep_equal,
    execution_arguments,
    match_arguments,
    score_turns,
)


def event(name="cancel_reservation", arguments=None, error=False):
    return {"name": name, "arguments": arguments or {"reservation_id": "A"}, "error": error}


def turn(calls, k=0):
    return {
        "schema": "tau3_turn_v1",
        "turn_index": k,
        "token_span": [k, k + 1],
        "tool_calls": calls,
        "parsed": True,
        "truncated": False,
    }


GOLD = {
    "name": "cancel_reservation",
    "arguments": {"reservation_id": "A"},
    "requestor": "assistant",
}


def test_strict_deep_equal_and_required_keys():
    assert deep_equal({"b": [1], "a": "x"}, {"a": "x", "b": [1]})
    assert not deep_equal([1, 2], [2, 1])
    assert not deep_equal(True, 1)
    assert not deep_equal("123", 123)
    assert not deep_equal({}, {"a": None})
    assert not match_arguments({"name": GOLD["name"], "arguments": {}}, GOLD)[0]
    assert not match_arguments(
        {"name": GOLD["name"], "arguments": {}}, {**GOLD, "compare_args": ["reservation_id"]}
    )[0]


def test_audit_is_zero_but_candidates_recorded_without_mutating_input():
    records = [turn([event(), event(error=True)])]
    saved = deepcopy(records)
    p = score_turns(records, [GOLD], ["ACTION"])
    assert p["turn_rewards"] == [0]
    assert p["turn_records"][0]["candidate_reward"] == pytest.approx(0.9)
    assert records == saved


def test_conservative_gold_requires_action_basis_and_success():
    records = [turn([event(), event(), event(error=True)])]
    p = score_turns(records, [GOLD], ["ACTION"], {"mode": "conservative"})
    assert p["turn_rewards"] == pytest.approx([0.9])
    assert p["turn_records"][0]["reward_types"] == ["gold_exact", "duplicate", "error"]
    p = score_turns(records, [GOLD], ["DB"], {"mode": "conservative"})
    assert p["turn_rewards"] == pytest.approx([-0.1])


def test_retry_and_multiple_reference_occurrences():
    p = score_turns(
        [turn([event(error=True), event(), event()])],
        [GOLD, GOLD],
        ["ACTION"],
        {"mode": "conservative"},
    )
    assert p["turn_rewards"] == pytest.approx([1.9])
    assert [c["matched_gold_index"] for c in p["turn_records"][0]["tool_calls"]] == [None, 0, 1]


def test_read_message_non_gold_write_and_unknown_neutral():
    records = [
        turn(
            [
                event("get_user_details"),
                event(arguments={"reservation_id": "B"}),
                event("future_tool"),
            ]
        ),
        turn([], 1),
    ]
    p = score_turns(records, [GOLD], ["ACTION"], {"mode": "conservative"})
    assert p["turn_rewards"] == [0, 0]
    assert p["turn_records"][0]["reward_types"] == ["read_only", "state_change", "unknown"]
    assert p["turn_records"][1]["reward_types"] == ["message"]


def test_soft_match_never_pays_and_user_actions_not_rewarded():
    gold = {**GOLD, "arguments": {"reservation_id": "A", "other": 1}}
    p = score_turns([turn([event()])], [gold], ["ACTION"], {"mode": "conservative"})
    assert p["turn_rewards"] == [0] and p["turn_records"][0]["reward_types"] == ["soft_match"]
    p = score_turns(
        [turn([event()])], [{**GOLD, "requestor": "user"}], ["ACTION"], {"mode": "conservative"}
    )
    assert p["turn_rewards"] == [0]


def test_settings_and_error_metadata_are_validated():
    with pytest.raises(ValueError):
        score_turns([], [], [], {"mode": "typo"})
    with pytest.raises(ValueError):
        score_turns([], [], [], {"weights": {"gold_exact": float("nan")}})
    with pytest.raises(ValueError):
        score_turns([turn([{"name": "x", "arguments": {}}])], [], [])


V2 = {"mode": "reference_write", "version": "v2"}


def test_v2_success_gate_reference_writes_only_and_total_budget():
    gold = [GOLD, {**GOLD, "arguments": {"reservation_id": "B"}},
            {"name": "get_user_details", "arguments": {"user_id": "x"}},
            {"name": "transfer_to_human_agents", "arguments": {"summary": "x"}}]
    calls = [event(g["name"], g["arguments"]) for g in gold]
    records = [turn(calls + [calls[0]])]
    success = score_turns(records, gold, ["DB", "COMMUNICATE"], V2, official_outcome=1)
    assert success["turn_rewards"] == [1]
    assert [e["reward"] for e in success["turn_records"][0]["tool_calls"]] == [.5, .5, 0, 0, 0]
    assert success["reference_write_count"] == 2
    assert score_turns(records, gold, ["DB"], V2, official_outcome=0)["turn_rewards"] == [0]
    assert score_turns(records, gold, ["ACTION"], V2, official_outcome=1)["turn_rewards"] == [0]


def test_v2_wrong_full_arguments_and_partial_matches_never_pay():
    gold = {**GOLD, "arguments": {"reservation_id": "A", "payment_id": "right"},
            "compare_args": ["reservation_id"]}
    records = [turn([event(arguments={"reservation_id": "A", "payment_id": "wrong"})])]
    assert score_turns(records, [gold], ["DB"], V2, official_outcome=1)["turn_rewards"] == [0]


def test_v2_error_retry_read_classification_and_missing_outcome():
    records = [turn([event(error=True), event(), event(), event("get_flight_status")])]
    p = score_turns(records, [GOLD], ["DB"], V2, official_outcome=1)
    assert p["turn_rewards"] == pytest.approx([.9])
    assert p["turn_records"][0]["tool_calls"][-1]["reward_type"] == "read_only"
    failed = score_turns(records, [GOLD], ["DB"], V2, official_outcome=0)
    assert failed["turn_rewards"] == [-.1]
    with pytest.raises(ValueError, match="official_outcome"):
        score_turns(records, [GOLD], ["DB"], V2)
    with pytest.raises(ValueError, match="unsupported"):
        score_turns(records, [GOLD], ["DB"], {"mode": "reference_write"}, official_outcome=1)
    with pytest.raises(ValueError, match="budget"):
        score_turns(records, [GOLD], ["DB"], {**V2, "weights": {"gold_exact": 2}}, official_outcome=1)


V3 = {"mode": "reference_write", "version": "v3"}


@pytest.mark.parametrize("outcome", [0, 1])
def test_v3_partial_progress_without_terminal_success_and_shared_dicts(outcome):
    call = event()
    records = [turn([call, call, event(error=True)])]
    gold = [GOLD, {**GOLD, "arguments": {"reservation_id": "B"}}]
    p = score_turns(records, gold, ["DB"], V3, official_outcome=outcome)
    assert [c['reward'] for c in p['turn_records'][0]['tool_calls']] == [.5, 0, -.1]
    assert p['turn_rewards'] == pytest.approx([.4])
    assert 'reward' not in call
    assert p['official_outcome'] == outcome
    assert score_turns(records, gold, ["ACTION"], V3, official_outcome=outcome)['turn_rewards'] == [-.1]


def test_v3_execution_equivalence_and_critical_argument_mismatches(requires_tau2):
    gold = {'name': 'update_reservation_flights', 'arguments': {
        'reservation_id': 'A', 'cabin': 'economy', 'payment_id': 'CC1',
        'flights': [{'flight_number': 'F1', 'date': '2024-05-17'},
                    {'flight_number': 'F2', 'date': '2024-05-18'}]}}
    same = deepcopy(gold['arguments'])
    same['flights'][0].update(price=999, origin='SFO', extra=None)
    records = [turn([event(gold['name'], same), event(gold['name'], gold['arguments'])])]
    p = score_turns(records, [gold], ['DB'], V3, official_outcome=0)
    assert [e['reward'] for e in p['turn_records'][0]['tool_calls']] == [1, 0]
    assert p['turn_records'][0]['tool_calls'][0]['effective_arguments'] == gold['arguments']
    for field, value in [('reservation_id', 'B'), ('cabin', 'business'), ('payment_id', 'CC2'),
                         ('flights', list(reversed(same['flights']))), ('unknown_top_level', 1)]:
        wrong = {**same, field: value}
        result = score_turns([turn([event(gold['name'], wrong)])], [gold], ['DB'], V3, official_outcome=1)
        assert result['turn_rewards'] == [0], field
    wrong = deepcopy(same)
    del wrong['flights'][0]['date']
    assert score_turns([turn([event(gold['name'], wrong)])], [gold], ['DB'], V3,
                       official_outcome=1)['turn_rewards'] == [0]
    assert score_turns([turn([event(gold['name'], same, error=True)])], [gold], ['DB'], V3,
                       official_outcome=1)['turn_rewards'] == [-.1]


def test_v3_booking_models_preserve_effective_payment_and_passenger_values(requires_tau2):
    args = {'flights': [{'flight_number': 'F', 'date': '2024-05-17', 'price': 1}],
            'passengers': [{'first_name': 'A', 'last_name': 'B', 'dob': '1990-01-01', 'extra': 'x'}],
            'payment_methods': [{'payment_id': 'CC', 'amount': '350', 'brand': 'visa'}],
            'user_id': '001'}
    canonical = execution_arguments('book_reservation', args)
    assert canonical['payment_methods'] == [{'payment_id': 'CC', 'amount': 350}]
    assert canonical['user_id'] == '001'
    assert 'extra' not in canonical['passengers'][0]
    changed = deepcopy(args)
    changed['payment_methods'][0]['amount'] = 351
    assert not deep_equal(execution_arguments('book_reservation', changed), canonical)
    changed = deepcopy(args)
    changed['passengers'][0]['dob'] = '1990-01-02'
    assert not deep_equal(execution_arguments('book_reservation', changed), canonical)


def test_v3_passenger_list_order_is_execution_significant(requires_tau2):
    """The pinned Airline tool stores passenger lists in caller order."""
    args = {
        'reservation_id': 'R',
        'passengers': [
            {'first_name': 'Chloe', 'last_name': 'A', 'dob': '1990-01-01'},
            {'first_name': 'Daniel', 'last_name': 'B', 'dob': '1991-01-01'},
        ],
    }
    canonical = execution_arguments('update_reservation_passengers', args)
    reordered = deepcopy(args)
    reordered['passengers'].reverse()
    assert execution_arguments('update_reservation_passengers', reordered) != canonical


def test_v3_canonical_match_has_same_real_tool_effect(requires_tau2):
    import json

    from tau3_grpo.envs.adapter import build_environment, load_flight_db
    from tau3_grpo.envs.tau2_bridge import message_models
    from tau3_grpo.paths import CODE_ROOT, DATA_ROOT

    manifest = CODE_ROOT / 'results/analysis/rl_curriculum50_20260912/manifests/areal_airline_train_seed42.jsonl'
    if not manifest.is_file():
        pytest.skip('formal training manifest unavailable')
    entry = next(x for x in map(json.loads, manifest.read_text().splitlines()) if x['task_id'] == 'airline_803')
    db_path = DATA_ROOT / 'raw/areal_tau2' / entry['db_path']
    if not db_path.is_file():
        pytest.skip('formal task database unavailable')
    gold = entry['task']['evaluation_criteria']['actions']
    hashes = []
    tool_call = message_models()['ToolCall']
    for add_extra in (False, True):
        env = build_environment(load_flight_db(db_path))
        records = []
        for k, action in enumerate(gold):
            args = deepcopy(action['arguments'])
            if add_extra and action['name'] == 'update_reservation_flights':
                for flight in args['flights']:
                    flight['price'] = 123456
            call = tool_call(id=str(k), name=action['name'], arguments=args)
            response = env.get_response(call)
            assert not response.error
            records.append(turn([event(call.name, call.arguments)], k))
        hashes.append(env.get_db_hash())
        p = score_turns(records, gold, ['DB'], V3, official_outcome=0)
        assert sum(p['turn_rewards']) == pytest.approx(1)
    assert hashes[0] == hashes[1]
