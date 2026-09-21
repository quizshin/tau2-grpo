import pytest

from tau3_grpo.data.sft_expansion import audit_tool_calls, coverage, select_expansion
from tau3_grpo.training.sft.train import training_schedule


def record(key, reason, tool):
    return {"metadata": {"source_dialog_id": key, "reason_for_call": reason},
            "messages": [{"role": "assistant", "tool_calls": [{"name": tool, "arguments": {}}]},
                         {"role": "tool", "name": tool, "content": "ok"}]}


def test_nested_expansion_preserves_anchors_and_fills_missing_tools():
    anchor = record("old", "find my reservation", "lookup")
    candidates = [record("duplicate", "Find my reservation!", "lookup"),
                  record("rare", "calculate a fare difference", "calculate"),
                  record("extra", "book a family vacation", "lookup")]
    result = select_expansion([anchor], candidates, size=3,
                              required_tools={"lookup", "calculate"})
    assert result[0] is anchor
    assert {r["metadata"]["source_dialog_id"] for r in result} == {"old", "rare", "extra"}
    assert coverage(result) == {"lookup": 2, "calculate": 1}
    assert result == select_expansion([anchor], list(reversed(candidates)), size=3,
                                      required_tools={"lookup", "calculate"})


def test_expansion_refuses_missing_tool_or_insufficient_distinct_data():
    anchor = record("old", "find a reservation", "lookup")
    with pytest.raises(ValueError, match="No eligible"):
        select_expansion([anchor], [], size=2, required_tools={"calculate"})
    with pytest.raises(ValueError, match="Cannot meet"):
        select_expansion([anchor], [record("same", "Find a reservation!", "lookup")],
                         size=2, required_tools={"lookup"})


def test_tool_audit_checks_arguments_and_response_pairing():
    schemas = [{"function": {"name": "lookup", "parameters": {"type": "object",
                "properties": {"id": {"type": "string"}}, "required": ["id"]}}}]
    example = record("one", "reason", "lookup")
    assert "arguments" in audit_tool_calls(example, schemas)[0]
    example["messages"][0]["tool_calls"][0]["arguments"] = {"id": "abc"}
    assert audit_tool_calls(example, schemas) == []
    example["messages"][1]["name"] = "wrong"
    assert any("unmatched" in issue for issue in audit_tool_calls(example, schemas))


def test_sft_budget_generalizes_sizes_without_silently_changing_old_budget():
    assert training_schedule({"num_epochs": 5}, 45, 1, 8) == (5, -1, 30)
    assert training_schedule({"num_epochs": 5, "max_steps": 30}, 200, 1, 8) == (5, 30, 30)
    with pytest.raises(ValueError, match="125 updates"):
        training_schedule({"num_epochs": 5}, 200, 1, 8)
    with pytest.raises(ValueError, match="effective batch"):
        training_schedule({"num_epochs": 5}, 45, 1, 4)
    with pytest.raises(ValueError, match="positive"):
        training_schedule({"num_epochs": 5, "max_steps": 0}, 45, 1, 8)
