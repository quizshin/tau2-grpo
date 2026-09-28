import pytest

from tau3_grpo.analysis.kimi_sft_review import diff, validate, validate_receipt


def packet(status="pass", decision="accept"):
    return {
        "conditions": [
            {
                "id": "intent.requirements.0",
                "status": status,
                "evidence": "messages[2] confirms amount",
            }
        ],
        "hard_rejects": [],
        "decision": decision,
    }


def test_unknown_and_hard_reject_cannot_enter_accepted_data():
    expected = [{"id": "intent.requirements.0"}]
    assert not validate(packet("unknown", "reject_or_hold"), expected)
    with pytest.raises(ValueError):
        validate(packet("unknown", "accept"), expected)
    p = packet()
    p["hard_rejects"] = ["unconfirmed write"]
    with pytest.raises(ValueError):
        validate(p, expected)


def test_complete_condition_coverage_and_evidence_required():
    p = packet()
    p["conditions"][0]["evidence"] = ""
    with pytest.raises(ValueError):
        validate(p, [{"id": "intent.requirements.0"}])
    with pytest.raises(ValueError):
        validate(packet(), [{"id": "different"}])
    with pytest.raises(ValueError):
        validate(packet(), [{"id": "intent.requirements.0"}, {"id": "other"}])


def test_diff_preserves_unrelated_state_mutations_and_deletions():
    before = {
        "users": {"target": {"balance": 30}, "other": {"balance": 50}},
        "orders": {"deleted": {}},
    }
    after = {"users": {"target": {"balance": 20}, "other": {"balance": 0}}, "orders": {}}
    changes = diff(before, after)
    assert {x["path"] for x in changes} == {
        "/users/target/balance",
        "/users/other/balance",
        "/orders/deleted",
    }


def test_receipt_requires_exact_model_and_nonnegative_integer_usage():
    valid = {"response_model": "Kimi-K3", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    assert validate_receipt(valid) == valid["usage"]
    with pytest.raises(ValueError):
        validate_receipt({**valid, "response_model": "other-model"})
    with pytest.raises(ValueError):
        validate_receipt({"response_model": "Kimi-K3", "usage": {"total_tokens": 15}})
