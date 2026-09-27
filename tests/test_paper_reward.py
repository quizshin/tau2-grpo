from copy import deepcopy

import pytest

from tau3_grpo.evaluation.paper_reward import match_arguments, normalize
from tau3_grpo.evaluation.process_reward import reward_settings, score_turns

PAPER = {"mode": "paper", "version": "paper_v1"}
GOLD = {"name": "cancel_reservation", "arguments": {"reservation_id": "A"}}


def call(name="cancel_reservation", args=None, error=False):
    return {"name": name, "arguments": {"reservation_id": "A"} if args is None else args,
            "error": error}


def turn(calls, k=0):
    return {"schema": "tau3_turn_v1", "turn_index": k, "token_span": [2*k, 2*k+2],
            "tool_calls": calls}


def score(turns, gold=None, config=None, outcome=0):
    return score_turns(turns, [GOLD] if gold is None else gold, ["DB"], config or PAPER,
                       official_outcome=outcome)


def test_paper_recursive_normalization_and_legacy_untouched():
    from tau3_grpo.evaluation.process_reward import deep_equal

    left = {"n": "001.00", "empty": None, "items": [{"a": "2", "blank": ""}, {"a": 1}]}
    right = {"n": 1, "items": [{"a": 1}, {"a": 2}]}
    assert normalize(left) == normalize(right)
    assert not deep_equal(left, right)
    assert normalize(False) != normalize(0)
    assert normalize({"false": False, "zero": 0}) != normalize({})
    assert normalize([1, 2]) != normalize([2, 1])
    assert normalize("123456789012345678901234567890") == normalize(123456789012345678901234567890)
    assert normalize([{"a": "1.0", "b": 2}, {"a": 1, "b": 1}]) == normalize(
        [{"a": "1.00", "b": 1}, {"a": 1, "b": 2}])
    with pytest.raises(ValueError, match="non-finite"):
        normalize(float("nan"))


def test_extra_cancellations_are_penalized_without_claiming_negative_turn_advantage():
    turns = [turn([call(), call(args={"reservation_id": "B"}), call(args={"reservation_id": "C"})])]
    result = score(turns)
    assert result["turn_rewards"] == pytest.approx([.8])
    assert [e["reward"] for e in result["turn_records"][0]["tool_calls"]] == [1, -.1, -.1]
    old = score(turns, config={"mode": "reference_write", "version": "v3"})
    assert old["turn_rewards"] == [1]
    # The turn remains positive: reward penalties alone are not a correctness guarantee.


def test_gold_reads_pay_and_multiple_gold_writes_have_no_budget_cap():
    read = {"name": "get_user_details", "arguments": {"user_id": "U"}}
    other = {**GOLD, "arguments": {"reservation_id": "B"}}
    turns = [turn([call(read["name"], read["arguments"]), call(), call(args=other["arguments"])])]
    assert score(turns, [read, GOLD, other])["turn_rewards"] == [3]


def test_soft_formula_and_constant_calibration_mode():
    gold = {"name": "update_reservation_baggages", "arguments": {"reservation_id": "A", "total": 2}}
    event = call(gold["name"], {"reservation_id": "A", "total": 3})
    assert match_arguments(event, gold) == (False, .5)
    assert score([turn([event])], [gold])["turn_rewards"] == [.75]
    recipe = {**PAPER, "weights": {"soft_match": .2}, "paper_options": {"soft_scoring": "constant"}}
    assert score([turn([event])], [gold], recipe)["turn_rewards"] == [.2]


def test_flight_status_is_read_only_with_gold_error_and_duplicate_precedence():
    event = call("get_flight_status", {"flight_number": "AA100", "date": "2026-09-17"})
    records = [turn([event, event, {**event, "error": True}])]
    result = score(records, gold=[])
    assert result["turn_records"][0]["reward_types"] == ["read_only", "duplicate", "error"]
    assert result["turn_rewards"] == pytest.approx([-.3])
    gold = {"name": event["name"], "arguments": event["arguments"]}
    assert score([turn([event])], gold=[gold])["turn_rewards"] == [1]
    # Preserve historical audit semantics while closing the paper tier coverage gap.
    old = score([turn([event])], gold=[], config={"mode": "audit", "version": "v1"})
    assert old["turn_records"][0]["reward_types"] == ["unknown"]


def test_full_arguments_not_compare_args_and_missing_keys():
    gold = {**GOLD, "arguments": {"reservation_id": "A", "amount": 10}, "compare_args": ["reservation_id"]}
    exact, ratio = match_arguments(call(), gold)
    assert not exact and ratio == .5
    assert match_arguments(call(args={}), GOLD) == (False, 0)


def test_duplicates_successful_calls_retry_and_repeated_gold_occurrences():
    events = [call(error=True), call(), call(), call(args={"reservation_id": "B"}),
              call(args={"reservation_id": "B"})]
    result = score([turn(events)])
    assert result["turn_records"][0]["reward_types"] == ["error", "gold_exact", "duplicate", "state_change", "duplicate"]
    assert result["turn_rewards"] == pytest.approx([.4])
    result = score([turn([call(), call(), call()])], [GOLD, GOLD])
    assert result["turn_rewards"] == pytest.approx([1.8])


def test_aggregation_message_weights_aliasing_and_roundtrip():
    event = call()
    records = [turn([event, event]), turn([], 1)]
    original = deepcopy(records)
    config = {**PAPER, "weights": {"message": .1}, "paper_options": {"aggregation": "mean"}}
    result = score(records, config=config)
    assert result["turn_rewards"] == pytest.approx([.4, .1])
    assert records == original
    assert score(result["turn_records"], config=config) == result


@pytest.mark.parametrize("config", [
    {"mode": "paper", "version": "v3"},
    {**PAPER, "paper_options": {"aggregation": "max"}},
    {**PAPER, "paper_options": {"invented": True}},
    {"mode": "conservative", "paper_options": {}},
])
def test_invalid_version_or_options(config):
    with pytest.raises(ValueError):
        reward_settings(config)


def test_outcome_and_error_required():
    with pytest.raises(ValueError, match="official_outcome"):
        score([turn([call()])], outcome=None)
    event = call()
    event.pop("error")
    with pytest.raises(ValueError, match="execution error"):
        score([turn([event])])
