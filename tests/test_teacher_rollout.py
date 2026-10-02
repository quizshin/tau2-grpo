"""Fake model only; real native airline DB and tools, no paid requests."""

import asyncio
import copy
import json

import pytest

from tau3_grpo.analysis.teacher_review import replay_candidate
from tau3_grpo.data.teacher_rollout import IncrementBudget, generate_candidate
from tau3_grpo.envs.adapter import load_default_flight_db
from tau3_grpo.utils.hashing import sha256_file


@pytest.fixture
def task(tmp_path):
    db = load_default_flight_db()
    path = tmp_path / "db.json"
    path.write_text(json.dumps(db.model_dump(mode="json")))
    uid = next(iter(db.users))
    return dict(
        task_id="pilot1",
        split="train",
        db_path=str(path),
        db_hash=sha256_file(path),
        initial_user_text=f"My user ID is {uid}. Please check my account.",
        user_instructions="PRIVATE_SCENARIO_SENTINEL: ask about account and end when answered.",
        evaluation_criteria={"gold": "GOLD_SENTINEL"},
        source_old_assistant="OLD_ASSISTANT_SENTINEL",
    )


def test_native_multi_call_and_private_separation(task, tmp_path):
    seen = []

    async def model(output, budget, cid, system, events, payload, **kwargs):
        seen.append(copy.deepcopy((cid, system, events, payload)))
        if "_teacher_000" in cid:
            return dict(
                content="",
                stop=False,
                tool_calls=[
                    {
                        "name": "get_user_details",
                        "arguments": {"user_id": task["initial_user_text"].split()[4].rstrip(".")},
                    },
                    {"name": "calculate", "arguments": {"expression": "2 + 3"}},
                ],
            )
        if "_teacher_" in cid:
            return dict(content="Your account was retrieved.", tool_calls=[], stop=False)
        return dict(content="Thank you.", stop=True)

    result = asyncio.run(
        generate_candidate(
            task,
            db_root=tmp_path,
            output=tmp_path / "run",
            candidate_id="x",
            budget=None,
            model_call=model,
        )
    )
    assert result["status"] == "user_stop" and not result["metadata"]["quality_accepted"]
    tools = [m for m in result["messages"] if m["role"] == "tool"]
    assert len(tools) == 2 and tools[0]["tool_call_id"] != tools[1]["tool_call_id"]
    assert json.loads(tools[1]["content"]) == 5
    assert all(not e["native_receipt"]["error"] for e in result["executions"])
    assert replay_candidate(result, tmp_path)["passed"]
    changed = copy.deepcopy(result)
    changed["final_db"]["users"] = {}
    with pytest.raises(ValueError, match="Final complete state mismatch"):
        replay_candidate(changed, tmp_path)
    for cid, system, events, payload in seen:
        request = json.dumps([system, events, payload])
        assert "GOLD_SENTINEL" not in request and "OLD_ASSISTANT_SENTINEL" not in request
        if "_teacher_" in cid:
            assert "PRIVATE_SCENARIO_SENTINEL" not in request
        else:
            assert "PRIVATE_SCENARIO_SENTINEL" in request and "tool_call_id" not in request


def test_native_error_does_not_skip_later_call_and_failure_saved(task, tmp_path):
    n = 0

    async def model(*args, **kwargs):
        nonlocal n
        n += 1
        if n == 1:
            return dict(
                content="",
                stop=False,
                tool_calls=[
                    {"name": "get_user_details", "arguments": {"user_id": "does_not_exist"}},
                    {"name": "calculate", "arguments": {"expression": "4*7"}},
                ],
            )
        raise RuntimeError("uncertain paid failure")

    with pytest.raises(RuntimeError, match="uncertain"):
        asyncio.run(
            generate_candidate(
                task,
                db_root=tmp_path,
                output=tmp_path / "run",
                candidate_id="x",
                budget=None,
                model_call=model,
            )
        )
    saved = json.loads((tmp_path / "run/candidates/x.json").read_text())
    assert n == 2 and saved["status"] == "failed"
    assert len(saved["executions"]) == 2
    assert saved["executions"][0]["native_receipt"]["error"]
    assert json.loads(saved["executions"][1]["native_receipt"]["content"]) == 28
    assert saved["final_db_hash"] == saved["initial_db_hash"]


def test_teacher_cannot_claim_termination(task, tmp_path):
    async def model(*args, **kwargs):
        return dict(content="Success.", tool_calls=[], stop=True)

    with pytest.raises(ValueError, match="cannot stop"):
        asyncio.run(
            generate_candidate(
                task,
                db_root=tmp_path,
                output=tmp_path / "run",
                candidate_id="x",
                budget=None,
                model_call=model,
            )
        )
    assert json.loads((tmp_path / "run/candidates/x.json").read_text())["status"] == "failed"


def test_incremental_budget_retains_unknown_usage(tmp_path):
    budget = IncrementBudget(
        tmp_path / "budget.json", max_calls=5, base_accounted=0, increment_cny=0.05
    )
    row = budget.reserve("a", "hello", {}, 100)
    budget.settle(row, {}, "failed")
    assert budget.accounted == row["reserved_cny"]
    with pytest.raises(ValueError, match="incremental"):
        budget.reserve("b", "x" * 30000, {}, 100)


def test_run_freezes_shared_budget_and_code(task, tmp_path, monkeypatch):
    import argparse

    from tau3_grpo.data import teacher_rollout as module
    from tau3_grpo.tracking.judge_budget import Budget

    ledger = tmp_path / "shared/budget.json"
    ledger.parent.mkdir()
    Budget(ledger, max_calls=100)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(task) + "\n")

    async def fake_candidate(*args, **kwargs):
        assert kwargs["budget"].path == ledger
        return {}

    monkeypatch.setattr(module, "generate_candidate", fake_candidate)
    args = argparse.Namespace(
        tasks=tasks,
        db_root=tmp_path,
        output=tmp_path / "run",
        budget=ledger,
        max_calls=2000,
        increment_cny=10,
        attempts=1,
        max_turns=40,
        max_tokens=4096,
    )
    asyncio.run(module.run(args))
    saved = json.loads((args.output / "run.json").read_text())
    assert saved["status"] == "complete_candidates_only"
    assert len(saved["candidate_ids"]) == 1
    assert saved["code_files"]["data/teacher_rollout.py"]
    assert (args.output / "code_snapshot/data/teacher_rollout.py").exists()
    assert saved["base_accounted_cny"] == 0
    assert saved["started_at_utc"] <= saved["finished_at_utc"]


@pytest.mark.parametrize(
    "packet,defaults",
    [
        ({"content": "Hello"}, {"tool_calls": [], "stop": False}),
        ({"content": "Hello", "stop": False}, {"tool_calls": []}),
        ({"content": "Hello", "tool_calls": []}, {"stop": False}),
    ],
)
def test_text_envelope_defaults_are_recorded(task, tmp_path, packet, defaults):
    raw = copy.deepcopy(packet)

    async def model(output, budget, cid, *args, **kwargs):
        return packet if "_teacher_" in cid else {"content": "Goodbye", "stop": True}

    result = asyncio.run(
        generate_candidate(
            task,
            db_root=tmp_path,
            output=tmp_path / "run",
            candidate_id="x",
            budget=None,
            model_call=model,
        )
    )
    action = result["teacher_actions"][0]
    assert result["status"] == "user_stop"
    assert packet == raw == action["raw_packet"]
    assert action["normalization"]["inserted_fields"] == defaults
    assert action["normalized_packet"] == {"content": "Hello", "tool_calls": [], "stop": False}
    assert replay_candidate(result, tmp_path)["passed"]


@pytest.mark.parametrize(
    "packet",
    [
        {"content": "Hello", "extra": "not allowed"},
        {"content": "Hello", "stop": True},
        {"stop": False, "tool_calls": []},
        {"content": "", "tool_calls": []},
        {"content": "Hello", "tool_calls": "invalid"},
        {"content": "Hello", "tool_calls": [{"name": "calculate", "arguments": {}}]},
        {"content": "", "tool_calls": [{"name": "calculate", "arguments": {}}]},
    ],
)
def test_invalid_teacher_envelope_keeps_raw_failure(task, tmp_path, packet):
    async def model(*args, **kwargs):
        return packet

    with pytest.raises(ValueError):
        asyncio.run(
            generate_candidate(
                task,
                db_root=tmp_path,
                output=tmp_path / "run",
                candidate_id="x",
                budget=None,
                model_call=model,
            )
        )
    result = json.loads((tmp_path / "run/candidates/x.json").read_text())
    assert result["status"] == "failed"
    assert result["teacher_actions"][0]["raw_packet"] == packet
    assert result["executions"] == []


def test_native_write_snapshots_real_owned_db_and_rejects_legacy(task, tmp_path):
    from tau3_grpo.analysis.teacher_review import reconstruct_legacy_snapshot
    from tau3_grpo.utils.hashing import sha256_json

    initial = json.loads(open(task["db_path"]).read())
    rid, reservation = next(iter(initial["reservations"].items()))
    passengers = copy.deepcopy(reservation["passengers"])
    passengers[0]["first_name"] = "Corrected"

    async def model(output, budget, cid, *args, **kwargs):
        if "_teacher_000" in cid:
            return {
                "content": "",
                "stop": False,
                "tool_calls": [
                    {
                        "name": "update_reservation_passengers",
                        "arguments": {"reservation_id": rid, "passengers": passengers},
                    }
                ],
            }
        if "_teacher_" in cid:
            return {"content": "Updated.", "tool_calls": [], "stop": False}
        return {"content": "Thanks.", "stop": True}

    result = asyncio.run(
        generate_candidate(
            task,
            db_root=tmp_path,
            output=tmp_path / "run",
            candidate_id="x",
            budget=None,
            model_call=model,
        )
    )
    assert result["initial_db"] == initial
    assert result["final_db"] != result["initial_db"]
    assert result["final_db"]["reservations"][rid]["passengers"] == passengers
    assert result["initial_db_hash"] != result["final_db_hash"]
    assert (
        replay_candidate(result, tmp_path)["complete_state_verified_against"]
        == "native_environment_owned_db"
    )
    assert sha256_file(task["db_path"]) == task["db_hash"]
    legacy = copy.deepcopy(result)
    legacy["final_db"] = copy.deepcopy(initial)
    before = sha256_json(legacy)
    with pytest.raises(ValueError, match="legacy snapshot is untrusted"):
        replay_candidate(legacy, tmp_path)
    fixed = reconstruct_legacy_snapshot(legacy, tmp_path, source_file_sha256="a" * 64)
    assert sha256_json(legacy) == before
    assert fixed["final_db"] == result["final_db"]
    assert not fixed["snapshot_reconstruction"]["original_final_snapshot_matched"]
    assert not fixed["metadata"]["quality_accepted"]
    assert replay_candidate(fixed, tmp_path)["passed"]
    bad = copy.deepcopy(legacy)
    bad["executions"][0]["native_receipt"]["content"] = "fabricated"
    with pytest.raises(ValueError, match="receipt mismatch"):
        reconstruct_legacy_snapshot(bad, tmp_path, source_file_sha256="b" * 64)


def test_mixed_text_tools_order_and_customer_visibility(task, tmp_path):
    seen = []

    async def model(output, budget, cid, system, events, payload, **kwargs):
        seen.append(copy.deepcopy((cid, payload)))
        if "_teacher_000" in cid:
            assert "airline_teacher_visible_text_multicall_v2" in system
            assert "Do not send a user-facing reply and tool calls in the same turn" not in system
            return {
                "content": "I will check these calculations.",
                "stop": False,
                "tool_calls": [
                    {"name": "calculate", "arguments": {"expression": "2+3"}},
                    {"name": "calculate", "arguments": {"expression": "4+5"}},
                ],
            }
        if "_user_000" in cid:
            disk = json.loads((output / "candidates/x.json").read_text())
            assert [m["role"] for m in disk["messages"]] == [
                "system",
                "user",
                "assistant",
                "tool",
                "tool",
            ]
            assert len(disk["executions"]) == 2
            assert payload["conversation"][-1] == {
                "role": "assistant",
                "content": "I will check these calculations.",
            }
            assert "tool_calls" not in json.dumps(payload) and "tool_call_id" not in json.dumps(
                payload
            )
            return {"content": "Please tell me the results.", "stop": False}
        if "_teacher_001" in cid:
            assert [m["role"] for m in payload["messages"]] == [
                "user",
                "assistant",
                "tool",
                "tool",
                "user",
            ]
            assert json.loads(payload["messages"][2]["content"]) == 5
            assert json.loads(payload["messages"][3]["content"]) == 9
            return {"content": "The results are 5 and 9.", "stop": False, "tool_calls": []}
        return {"content": "Thanks.", "stop": True}

    result = asyncio.run(
        generate_candidate(
            task,
            db_root=tmp_path,
            output=tmp_path / "run",
            candidate_id="x",
            budget=None,
            model_call=model,
        )
    )
    assert result["status"] == "user_stop"
    assert [m["role"] for m in result["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
        "user",
    ]
    assert result["messages"][2]["content"] == "I will check these calculations."
    assert result["metadata"]["tool_protocol"] == "airline_teacher_visible_text_multicall_v2"
    assert replay_candidate(result, tmp_path)["passed"]
    assert not result["metadata"]["quality_accepted"]


def test_teacher_history_projection_preserves_facts_removes_transport():
    from tau3_grpo.data.teacher_rollout import teacher_visible_messages

    original = [
        {
            "role": "assistant",
            "content": "Checking",
            "tool_calls": [
                {
                    "id": "wire_123",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": '{"expression":"2+3"}'},
                }
            ],
        },
        {
            "role": "tool",
            "name": "calculate",
            "content": "5",
            "error": False,
            "tool_call_id": "wire_123",
        },
    ]
    before = copy.deepcopy(original)
    projected = teacher_visible_messages(original)
    assert projected == [
        {
            "role": "assistant",
            "content": "Checking",
            "tool_calls": [{"name": "calculate", "arguments": {"expression": "2+3"}}],
        },
        {"role": "tool", "name": "calculate", "content": "5", "error": False},
    ]
    assert original == before
    assert "wire_123" not in json.dumps(projected)


def test_known_format_failures_never_treat_unknown_or_caps_as_safe(tmp_path):
    from types import SimpleNamespace

    from tau3_grpo.data.teacher_rollout import known_format_failure

    target = tmp_path / "candidate.json"
    target.write_text(json.dumps({"status": "failed", "api_request_ids": ["r"]}))
    call = {
        "request_id": "r",
        "status": "failed",
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        "accounting": "usage_at_time_cache_prices_not_invoice",
    }
    budget = SimpleNamespace(state={"calls": [call]})
    assert known_format_failure(ValueError("Invalid teacher action fields"), target, budget)
    call["accounting"] = "unknown_usage_full_reservation_retained"
    assert not known_format_failure(ValueError(), target, budget)
    call["accounting"] = "usage_at_time_cache_prices_not_invoice"
    call["status"] = "reserved"
    assert not known_format_failure(ValueError(), target, budget)
    call["status"] = "failed"
    call["usage"] = {}
    assert not known_format_failure(ValueError(), target, budget)
    budget.state["calls"] = []
    assert not known_format_failure(ValueError("Budget cap: not sent"), target, budget)
    assert not known_format_failure(RuntimeError("unexpected implementation error"), target, budget)


def test_exact_response_format_echo_is_traceable_not_general_key_repair(task, tmp_path):
    from tau3_grpo.data.teacher_rollout import normalize_teacher_action

    raw = {"type": "json_object", "content": "Hello", "tool_calls": [], "stop": False}

    async def model(output, budget, cid, *args, **kwargs):
        return raw if "_teacher_" in cid else {"content": "Thanks", "stop": True}

    result = asyncio.run(
        generate_candidate(
            task,
            db_root=tmp_path,
            output=tmp_path / "run",
            candidate_id="x",
            budget=None,
            model_call=model,
        )
    )
    action = result["teacher_actions"][0]
    assert action["raw_packet"] == raw
    assert action["normalization"]["removed_transport_fields"] == {"type": "json_object"}
    assert action["normalized_packet"] == {"content": "Hello", "tool_calls": [], "stop": False}
    assert result["status"] == "user_stop"
    with pytest.raises(ValueError, match="fields"):
        normalize_teacher_action(
            {"type": "invented", "content": "Hello", "tool_calls": [], "stop": False}
        )
    with pytest.raises(ValueError, match="fields"):
        normalize_teacher_action(dict(raw, unrecognized="still forbidden"))


@pytest.mark.parametrize('name,split,limit', [
    ('teacher_rollout', 'train', 100.0),
    ('teacher_rollout_validation', 'validation', 100.0),
    ('teacher_rollout_validation_v4', 'validation', 150.0),
    ('teacher_rollout_validation_v4_retry', 'validation', 150.0),
])
def test_legacy_entries_preserve_defaults_and_snapshot_common_code(
    name, split, limit, task, tmp_path, monkeypatch,
):
    import importlib

    from tau3_grpo.data import teacher_rollout as core
    from tau3_grpo.tracking.judge_budget import Budget

    module = importlib.import_module('tau3_grpo.data.' + name)
    task['split'] = split
    tasks = tmp_path / 'tasks.jsonl'
    tasks.write_text(json.dumps(task) + '\n')
    ledger = tmp_path / 'budget.json'
    Budget(ledger, limit=limit, max_calls=10)
    output = tmp_path / 'run'

    async def fake_candidate(*args, **kwargs):
        assert kwargs['split'] == split
        assert kwargs['budget'].state['limit_cny'] == limit
        return {}

    monkeypatch.setattr(core, 'generate_candidate', fake_candidate)
    module.main(['--tasks', str(tasks), '--db-root', str(tmp_path), '--output', str(output),
                 '--budget', str(ledger), '--max-calls', '20'])
    saved = json.loads((output / 'run.json').read_text())
    assert saved['task_split'] == split and saved['total_limit_cny'] == limit
    assert saved['code_files']['data/teacher_rollout.py']
    if name != 'teacher_rollout':
        assert saved['code_files'][f'data/{name}.py']
    assert saved['status'] == 'complete_candidates_only'


@pytest.mark.parametrize('name', [
    'teacher_rollout', 'teacher_rollout_validation',
    'teacher_rollout_validation_v4', 'teacher_rollout_validation_v4_retry',
])
def test_legacy_entries_reject_other_split_before_model_call(name, task, tmp_path):
    import importlib

    module = importlib.import_module('tau3_grpo.data.' + name)
    task['split'] = 'validation' if name == 'teacher_rollout' else 'train'

    async def must_not_call(*args, **kwargs):
        raise AssertionError('Wrong split reached model')

    with pytest.raises(ValueError, match='explicitly frozen'):
        asyncio.run(module.generate_candidate(task, db_root=tmp_path, output=tmp_path / 'run',
                                             candidate_id='x', budget=None, model_call=must_not_call))


def test_budget_limit_requires_explicit_choice_and_cannot_change_existing_ledger(tmp_path):
    from tau3_grpo.data.teacher_rollout import build_parser
    from tau3_grpo.tracking.judge_budget import Budget

    parser = build_parser()
    args = parser.parse_args(['--tasks', 't', '--db-root', 'd', '--output', 'o', '--budget', 'b',
                              '--max-calls', '10'])
    assert args.split == 'train' and args.limit_cny == 100.0
    ledger = tmp_path / 'budget.json'
    Budget(ledger, limit=100.0)
    with pytest.raises(ValueError, match='Cannot change a run budget'):
        IncrementBudget(ledger, max_calls=10, base_accounted=0, increment_cny=10, limit_cny=150.0)
