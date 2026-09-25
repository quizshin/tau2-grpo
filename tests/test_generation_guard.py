"""CPU checks of real guard helpers and engine-stream failure boundaries."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tau3_grpo.integrations.verl.generation_guard import consume_guarded_generation
from tau3_grpo.integrations.verl.token_budget import (
    generation_request,
    generation_result,
    observation,
)
from tau3_grpo.models.generation_guard import GenerationGuard, detect_repetition, sampling_penalty
from tau3_grpo.models.token_budget import default_budget
from tau3_grpo.training.rl.update_guard import (
    HarnessUpdateHalt,
    UpdateGuard,
    batch_health,
    enforce_update_guard,
    guard_decision,
    validate_guard_configuration,
)

SMALL = GenerationGuard(mode="abort", min_tokens=8, repeats=4, min_text_chars=16)


@pytest.mark.parametrize("validation,value", [(False, 1.2), (True, 1.3)])
def test_sampling_override_and_inheritance(validation, value):
    assert sampling_penalty({"repetition_penalty": 1.2, "val_kwargs": {"repetition_penalty": 1.3}},
                            validation=validation) == value
    assert sampling_penalty({"repetition_penalty": 1.2, "val_kwargs": {"repetition_penalty": None}},
                            validation=True) == 1.2
    assert sampling_penalty({}) == 1.0


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "1.2"])
def test_invalid_sampling_penalty(value):
    with pytest.raises(ValueError):
        sampling_penalty({"repetition_penalty": value})


def test_repeated_symbols_and_multi_token_patterns_but_not_short_lists_or_tools():
    assert detect_repetition([1, 2] * 8, "✅ " * 16, SMALL)["kind"] == "token_cycle"
    assert detect_repetition(list(range(32)), "✅ " * 32, SMALL)["kind"] == "text_cycle"
    assert detect_repetition(list(range(32)), "✓" * 32, SMALL)
    assert detect_repetition(list(range(32)), "Flight A ✅\nFlight B ✅\nFlight C ✅", SMALL) is None
    assert detect_repetition([1] * 32, '<tool_call><function=x><parameter=p>' + 'x' * 32, SMALL) is None
    assert detect_repetition([1] * 32, 'x' * 32, replace(SMALL, mode="off")) is None


def chunk(ids, finish=None, probs=True):
    return SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(ids), finish_reason=finish,
        logprobs=[{t: SimpleNamespace(logprob=-0.2)} for t in ids] if probs else None)])


async def stream(outputs):
    for output in outputs:
        await asyncio.sleep(0)
        yield output


def consume(outputs, *, config=SMALL, ack=True):
    calls = []

    async def cancel():
        calls.append(True)
        return {"aborted": ack}

    async def run():
        result = await consume_guarded_generation(stream(outputs), tokenizer=SimpleNamespace(
            decode=lambda ids, **kw: "x" * len(ids)), config=config, cancel=cancel, require_logprobs=True)
        return result, calls

    return asyncio.run(run())


def test_empty_abort_receipt_preserves_actual_prefix_and_logprobs():
    (output, event), calls = consume([chunk([1] * 8), chunk([], "abort")])
    assert calls == [True] and output.outputs[0].token_ids == [1] * 8
    assert len(output.outputs[0].logprobs) == 8
    assert output.outputs[0].finish_reason == "abort" and event["cancelled"]


def test_natural_finish_and_cancel_race_keep_attested_final_tokens():
    (output, event), calls = consume([chunk([1] * 8), chunk([1] * 8 + [2], "stop")])
    assert calls == [True] and output.outputs[0].token_ids[-1] == 2
    assert event["cancel_requested"] and not event["cancelled"]
    (_, event), calls = consume([chunk([1] * 8, "stop")])
    assert not calls and not event["cancelled"]


def test_observe_keeps_stream_and_does_not_cancel():
    (_, event), calls = consume([chunk([1] * 8), chunk([1] * 16, "length")], config=replace(SMALL, mode="observe"))
    assert not calls and event["final_tokens"] == 16


@pytest.mark.parametrize("outputs", [
    [chunk([1, 2]), chunk([3, 4], "stop")],
    [chunk([1] * 8, probs=False)],
    [chunk([1], "abort")],
    [chunk([1])],
])
def test_bad_stream_is_execution_failure_not_trainable_repetition(outputs):
    with pytest.raises((ValueError, RuntimeError)):
        consume(outputs)


def test_cancel_failure_does_not_claim_successful_stop():
    with pytest.raises(RuntimeError, match="not acknowledged"):
        consume([chunk([1] * 8), chunk([], "abort")], ack=False)


def test_concurrent_streams_do_not_share_guard_or_token_state():
    async def run():
        async def one(token):
            async def cancel():
                return {"aborted": True}
            output, _ = await consume_guarded_generation(stream([chunk([token] * 8), chunk([], "abort")]),
                tokenizer=SimpleNamespace(decode=lambda ids, **kw: "x" * len(ids)), config=SMALL,
                cancel=cancel, require_logprobs=True)
            return output.outputs[0].token_ids
        return await asyncio.gather(one(1), one(2))
    assert asyncio.run(run()) == [[1] * 8, [2] * 8]


def budget_objects(prompt=5000, appended=0):
    loop = SimpleNamespace(token_budget=default_budget(), tokenizer=SimpleNamespace(decode=lambda ids, **kw: "x"))
    data = SimpleNamespace(prompt_ids=[1] * prompt, response_mask=[1] * appended, token_receipts=[], extra_fields={})
    return loop, data


def test_budget_binding_and_guard_reason_do_not_change_official_labels():
    loop, data = budget_objects()
    assert generation_request(loop, data) == 1024
    assert data.token_receipts[-1]["binding_limits"] == ["per_turn_limit"]
    output = SimpleNamespace(token_ids=[2] * 1024, extra_fields={"finish_reason": "length"})
    assert generation_result(loop, data, output) == "context_window_exceeded"
    assert data.extra_fields["termination_detail"] == "per_turn_limit"
    loop, data = budget_objects(24570, 16378)
    assert generation_request(loop, data) == 6
    assert set(data.token_receipts[-1]["binding_limits"]) == {"context_budget", "response_budget"}
    assert not observation(loop, data, [2] * 7, kind="tool")
    assert data.extra_fields["termination_detail"] == "observation_budget"
    loop, data = budget_objects()
    generation_request(loop, data)
    output.extra_fields = {"finish_reason": "abort", "generation_guard": {
        "version": SMALL.version, "mode": "abort", "cancelled": True}}
    assert generation_result(loop, data, output) == "agent_error"
    assert data.extra_fields["termination_detail"] == "repetition_detected"


def facts(index, repeat=False, length=False):
    return json.dumps({"schema": "tau3_trajectory_facts_v1", "identity": {"trajectory_id": str(index)},
        "turns": [{"generation_guard": {"kind": "token_cycle"} if repeat else None,
                   "finish_reason": "length" if length else "stop"}], "terminal": {"reward": 0.0}})


def test_guard_counts_candidates_not_padding_or_filtered_tokens_and_requires_streak():
    health = batch_health({"trajectory_facts_json": [facts(0, True), facts(1), None],
                          "tau3_is_padding": [False, False, True]})
    assert health["candidates"] == 2 and health["repetition_fraction"] == .5
    config = UpdateGuard(mode="halt", repetition_fraction=.5, consecutive_steps=2)
    first = guard_decision(health, config, step=24)
    assert not first["halt"]
    assert guard_decision(health, config, step=25, previous=first)["halt"]
    assert not guard_decision(health, config, step=26, previous=first)["halt"]


def test_all_failed_but_no_repetition_is_not_filtered_or_halted(tmp_path):
    batch = SimpleNamespace(non_tensor_batch={"trajectory_facts_json": [facts(i) for i in range(8)]})
    state, _ = enforce_update_guard(batch, {"mode": "halt", "repetition_fraction": .1}, step=1, output_dir=tmp_path)
    assert not state["halt"] and not list(tmp_path.iterdir())


def test_halt_persists_pending_batch_and_redacted_config_not_checkpoint(tmp_path):
    class Batch:
        non_tensor_batch = {"trajectory_facts_json": [facts(0, True)]}

        def save_to_disk(self, path):
            from pathlib import Path
            Path(path).write_bytes(b"pending batch")

    with pytest.raises(HarnessUpdateHalt):
        enforce_update_guard(Batch(), {"mode": "halt", "repetition_fraction": .1}, step=25,
                             output_dir=tmp_path, resolved_config={"api_key": "test-secret"})
    receipt = json.loads(next(tmp_path.rglob("fault.json")).read_text())
    assert receipt["pending_step"] == 25 and receipt["last_completed_outer_step"] == 24
    assert not receipt["is_resumable_checkpoint"] and not receipt["optimizer_update_started_for_pending_batch"]
    assert receipt["resolved_config"]["api_key"] == "[REDACTED]"


def test_guard_configuration_rejects_missing_generation_telemetry():
    config = {"tau3_token_protocol": "tau3_token_budget_v1",
              "actor_rollout_ref": {"rollout": {"name": "vllm", "multi_turn": {"enable": True},
                                                "calculate_log_probs": True}},
              "tau3_update_guard": {"mode": "halt", "repetition_fraction": .1}}
    with pytest.raises(ValueError, match="telemetry"):
        validate_guard_configuration(config)
    config["actor_rollout_ref"]["rollout"]["generation_guard"] = {"mode": "observe"}
    validate_guard_configuration(config)
    config["actor_rollout_ref"]["rollout"]["calculate_log_probs"] = False
    with pytest.raises(ValueError, match="sampled-token logprobs"):
        validate_guard_configuration(config)


def test_trainer_guard_precedes_worker_updates_and_pool_allocation():
    import ast

    from tau3_grpo.paths import CODE_ROOT

    tree = ast.parse((CODE_ROOT / "verl/verl/trainer/ppo/ray_trainer.py").read_text())
    methods = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    calls = [(getattr(n.func, "id", getattr(n.func, "attr", None)), n.lineno)
             for n in ast.walk(methods["fit"]) if isinstance(n, ast.Call)]
    guard = next(line for name, line in calls if name == "enforce_update_guard")
    assert all(guard < line for name, line in calls if name in {"_update_actor", "_update_critic"})
    calls = [(getattr(n.func, "id", getattr(n.func, "attr", None)), n.lineno)
             for n in ast.walk(methods["init_workers"]) if isinstance(n, ast.Call)]
    assert next(line for name, line in calls if name == "validate_guard_configuration") < next(
        line for name, line in calls if name == "create_resource_pool")


def test_single_request_abort_uses_public_engine_api_without_resetting_shared_cache():
    import ast
    import logging
    from typing import Any

    from tau3_grpo.paths import CODE_ROOT

    path = CODE_ROOT / "verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMHttpServer")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "abort_request")
    namespace = {"Any": Any, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    calls = []

    async def abort(request_id):
        calls.append(request_id)

    async def clear():
        pytest.fail("shared prefix cache reset")

    server = SimpleNamespace(engine=SimpleNamespace(abort=abort), clear_kv_cache=clear)
    result = asyncio.run(namespace["abort_request"](server, "external-id", reset_prefix_cache=False))
    assert result == {"aborted": True, "request_id": "external-id"}
    assert calls == ["external-id"]

    async def broken(request_id):
        raise RuntimeError("engine unavailable")

    server.engine.abort = broken
    result = asyncio.run(namespace["abort_request"](server, "external-id", reset_prefix_cache=False))
    assert not result["aborted"] and "engine unavailable" in result["error"]
