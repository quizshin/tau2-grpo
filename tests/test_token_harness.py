"""Native training vs independent token transport, with no model or GPU calls."""
import asyncio
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau3_grpo.models.token_budget import complete_tool_envelopes, default_budget
from tau3_grpo.paths import CODE_ROOT


@pytest.fixture(scope="module")
def tokenizer():
    from tau3_grpo.evaluation.token_runtime import load_tokenizer

    extra = [Path(p) for p in os.environ.get("TAU3_TEST_TOKENIZERS", "").split(os.pathsep) if p]
    for path in (*extra, CODE_ROOT / "results/analysis/harness_inputs_20260920/tokenizer",
                 CODE_ROOT / "checkpoints/sft-merged/new-off"):
        if (path / "tokenizer.json").exists():
            return load_tokenizer(path)
    pytest.skip("Pinned Qwen tokenizer files absent; no downloads during tests")


@pytest.fixture
def entry():
    from tau3_grpo.data.manifest import read_manifest
    from tau3_grpo.paths import MANIFEST_ROOT

    path = MANIFEST_ROOT / "areal_airline_selection_seed42.jsonl"
    if not path.exists():
        pytest.skip("Pinned selection manifest absent")
    return read_manifest(path)[0]


def test_cold_concurrent_native_loop_initialization(tokenizer, tmp_path):
    """A fresh interpreter exposes the partial-module race hidden by test order."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent('''
        import asyncio, sys, threading, time
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from tau3_grpo.envs import adapter
        from tau3_grpo.evaluation.runtime import Endpoint
        from tau3_grpo.evaluation.token_runtime import load_tokenizer, make_loop
        from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

        assert "tau3_grpo.envs.tools" not in sys.modules
        original = adapter.airline_tool_schemas
        def slow_schema():
            time.sleep(0.25)
            return original()
        adapter.airline_tool_schemas = slow_schema
        tokenizer = load_tokenizer(sys.argv[1])
        barrier = threading.Barrier(4)
        def construct(index):
            directory = Path(sys.argv[2]) / str(index)
            directory.mkdir()
            barrier.wait()
            loop, _ = asyncio.run(make_loop(tokenizer=tokenizer, manager=object(),
                user=Endpoint("user", "http://unused"), directory=directory))
            assert isinstance(loop, ToolAgentLoop)
            assert len(loop.tools) == 14
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(construct, range(4)))
    ''')
    result = subprocess.run([sys.executable, "-c", script, tokenizer.name_or_path, str(tmp_path)],
                            capture_output=True, text=True, errors="replace", timeout=180,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert result.returncode == 0, result.stdout + result.stderr


def xml(expression="1+1"):
    return f"<tool_call><function=calculate><parameter=expression>{expression}</parameter></function></tool_call>"


def test_budget_counts_observations_and_allows_exact_fit():
    budget = default_budget()
    assert budget.allowance(5200 + 16000, 16000) == 384
    assert budget.fits(5200 + 16000, 16000, 384)
    assert not budget.fits(5200 + 16000, 16000, 385)
    assert budget.allowance(5200 + 16384, 16384) == 0
    assert budget.allowance(24575, 16000) == 1
    with pytest.raises(ValueError):
        budget.allowance(1, 2)


@pytest.mark.parametrize("text,complete", [
    (xml(), True), (xml() + xml("2+2"), True), (xml()[:-12], False),
    (xml() + "<tool_call><function=cancel_reservation>", False),
    ("Fare <", True), (xml() + "<tool_", False), (xml() + "<", False),
    (xml() + "<tool_call", False), (xml() + "<tool_call><function=", False),
    ("<tool_call><function=x><parameter=p>v</function></tool_call>", False),
])
def test_partial_xml_batch_cannot_dispatch(text, complete):
    assert complete_tool_envelopes(text) is complete


class ScriptedManager:
    def __init__(self, tokenizer, texts, *, finish="stop"):
        self.tokens = [tokenizer.encode(t + ("<|im_end|>" if finish == "stop" else ""),
                                       add_special_tokens=False) for t in texts]
        self.finish = finish
        self.requests = []

    async def generate(self, *, prompt_ids, sampling_params, **kwargs):
        from verl.workers.rollout.replica import TokenOutput

        index = len(self.requests)
        self.requests.append((list(prompt_ids), dict(sampling_params)))
        ids = self.tokens[index]
        return TokenOutput(token_ids=ids, log_probs=[-0.2] * len(ids), stop_reason="completed",
                           extra_fields={"finish_reason": self.finish})

    def transport(self, payload):
        index = len(self.requests)
        self.requests.append((list(payload["prompt"]), deepcopy(payload)))
        ids = self.tokens[index]
        return {"model": payload["model"], "choices": [{"token_ids": ids,
                "prompt_token_ids": payload["prompt"], "finish_reason": self.finish,
                "logprobs": {"tokens": [f"token_id:{t}" for t in ids],
                             "token_logprobs": [-0.2] * len(ids)}}],
                "usage": {"prompt_tokens": len(payload["prompt"]), "completion_tokens": len(ids)}}


async def train(tokenizer, entry, manager, user, directory, *, estimator="grpo", budget=None,
                sampling_identity=None, reward_config=None):
    from tau3_grpo.data.parquet_builder import build_row
    from tau3_grpo.evaluation.token_runtime import make_loop
    from tau3_grpo.paths import TAU2_BENCH_ROOT

    loop, handler = await make_loop(tokenizer=tokenizer, manager=manager, user=user,
                                    directory=directory, estimator=estimator, evaluation=False)
    if budget:
        loop.token_budget = budget
    if reward_config is not None:
        from tau3_grpo.evaluation.process_reward import reward_settings

        loop.process_reward_config = reward_settings(reward_config)
    row = build_row(entry, policy=(TAU2_BENCH_ROOT / "data/tau2/domains/airline/policy.md").read_text(),
                    split="selection", seed=42)
    output = await loop.run({"temperature": 0.7, "top_p": 1.0, "top_k": -1,
                            "repetition_penalty": 1.0, "logprobs": True},
                            raw_prompt=row["prompt"], extra_info=row["extra_info"],
                            **({'tau3_sampling_identity': sampling_identity} if sampling_identity else {}))
    return output


@pytest.mark.parametrize("estimator", ["grpo", "tau_gigpo", "mt_gtpo"])
def test_real_token_multiturn_inputs_match_across_transports(tokenizer, entry, tmp_path, monkeypatch, estimator):
    from tau2.data_model.message import AssistantMessage
    from tau2.user import user_simulator

    from tau3_grpo.evaluation.runtime import Endpoint
    from tau3_grpo.evaluation.token_runtime import run_one
    from tau3_grpo.models.token_endpoint import TokenEndpoint

    texts = [xml() + xml("2+2"), "Let me check.", xml("3+3"), "Done."]
    users = []

    def user_generate(**kwargs):
        users.append(deepcopy(kwargs))
        return AssistantMessage(role="assistant", content="Continue." if len(users) % 2 else "###STOP###")

    monkeypatch.setattr(user_simulator, "generate", user_generate)
    policy, user = Endpoint("policy", "http://unused"), Endpoint("user", "http://unused")
    native = ScriptedManager(tokenizer, texts)
    output = asyncio.run(train(tokenizer, entry, native, user, tmp_path, estimator=estimator))
    transport = ScriptedManager(tokenizer, texts)
    simulation = run_one(entry=entry, policy=policy, user=user, seed=42, trial=0, tokenizer=tokenizer,
                         manager=TokenEndpoint(policy, transport=transport.transport))
    assert len(native.requests) == len(transport.requests) == 4
    assert [r[0] for r in native.requests] == [r[0] for r in transport.requests]
    assert [r[1]["max_tokens"] for r in native.requests] == [r[1]["max_tokens"] for r in transport.requests]
    native_receipt = json.loads(output.extra_fields["token_protocol_json"])
    assert native_receipt == simulation.info["token_protocol"]
    assert simulation.termination_reason.value == "user_stop"
    assert output.reward_score == simulation.reward_info.reward
    assert [u["max_tokens"] for u in users] == [1024] * 4
    # Reusing the shared loop must still reach real tools and the native verifier.
    assert [m.content for m in simulation.messages if m.role == "tool"] == ["2.0", "4.0", "6.0"]
    from tau3_grpo.envs.registry import SESSIONS

    assert SESSIONS.active_count() == 0


@pytest.mark.parametrize("mode", ["complete_tool_exact", "partial_batch", "text_length", "stop_overflow", "user_overflow"])
def test_native_budget_boundaries(tokenizer, entry, tmp_path, monkeypatch, mode):
    from tau2.data_model.message import AssistantMessage
    from tau2.user import user_simulator

    from tau3_grpo.evaluation.runtime import Endpoint

    users = []

    def reply(**kwargs):
        users.append(kwargs)
        return AssistantMessage(role="assistant", content="Continue." if mode == "user_overflow" else "###STOP###")

    monkeypatch.setattr(user_simulator, "generate", reply)
    text = {"complete_tool_exact": xml(), "partial_batch": xml() + "<tool_call><function=calculate>",
            "text_length": "unfinished answer", "stop_overflow": "Done.", "user_overflow": "Done."}[mode]
    manager = ScriptedManager(tokenizer, [text], finish="stop" if mode in {"stop_overflow", "user_overflow"} else "length")
    budget = replace(default_budget(), response=len(manager.tokens[0]))
    output = asyncio.run(train(tokenizer, entry, manager, Endpoint("user", "http://unused"),
                               tmp_path, budget=budget))
    receipt = json.loads(output.extra_fields["token_protocol_json"])
    facts = json.loads(output.extra_fields["trajectory_facts_json"])
    if mode == "stop_overflow":
        assert receipt["termination_reason"] == "user_stop" and len(users) == 1
        assert receipt["receipts"][-1]["terminal"] and not receipt["receipts"][-1]["retained"]
    elif mode == "user_overflow":
        assert receipt["termination_reason"] == "context_window_exceeded" and len(users) == 1
        assert not receipt["receipts"][-1]["retained"]
    else:
        assert receipt["termination_reason"] == "context_window_exceeded" and not users
    assert len(output.response_ids) == len(output.response_mask) == budget.response
    assert facts
    if mode == "complete_tool_exact":
        assert receipt["receipts"][-1]["kind"] == "tool"
        assert not receipt["receipts"][-1]["retained"]
    if mode == "partial_batch":
        assert not any(r["kind"] == "tool" for r in receipt["receipts"])


@pytest.mark.parametrize("corrupt", ["ids", "prompt", "logprobs", "finish", "usage"])
def test_token_transport_fails_closed(corrupt):
    from tau3_grpo.models.token_endpoint import TokenEndpoint

    response = {"model": "policy", "choices": [{"token_ids": [3], "prompt_token_ids": [1, 2],
        "finish_reason": "stop", "logprobs": {"tokens": ["token_id:3"], "token_logprobs": [-0.2]}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
    choice = response["choices"][0]
    if corrupt == "ids":
        choice.pop("token_ids")
    elif corrupt == "prompt":
        choice["prompt_token_ids"] = [1, 99]
    elif corrupt == "logprobs":
        choice["logprobs"]["token_logprobs"] = [float("nan")]
    elif corrupt == "finish":
        choice["finish_reason"] = "abort"
    else:
        response["usage"]["completion_tokens"] = 2
    endpoint = TokenEndpoint(SimpleNamespace(model="policy"), transport=lambda _: response)
    with pytest.raises(ValueError):
        asyncio.run(endpoint.generate(request_id="test", prompt_ids=[1, 2],
                                       sampling_params={"temperature": .7, "max_tokens": 10}))


def test_token_v4_final_guard_before_task_load(monkeypatch, capsys):
    from tau3_grpo.evaluation import run
    from tau3_grpo.evaluation.harness import TOKENS_V4

    monkeypatch.setattr(run, "official_airline_task_ids", lambda: pytest.fail("loaded final"))
    assert run.main(["--target", "tau3-final", "--checkpoint", "/missing",
                     "--harness-protocol", TOKENS_V4, "--dry-run"]) == 2


@pytest.mark.parametrize("arm", ["e0", "e2", "mt_gtpo"])
def test_token_profile_reaches_training_launcher(arm):
    from tau3_grpo.launch import prepare

    command, env, _ = prepare("rl", CODE_ROOT / "configs/train/rl/formal50_token_v4.yaml", arm,
                              42, [], {"TAU3_ROOT": str(CODE_ROOT.parent)})
    assert "+tau3_token_protocol=tau3_token_budget_v1" in command
    assert env["TOOL_SCHEMA_VERSION"] == "tau3_full_schema_v2"


@pytest.mark.parametrize("estimator", ["grpo", "tau_gigpo", "mt_gtpo"])
def test_formal_token_option_resolves_hydra(estimator, tmp_path, monkeypatch):
    import shlex
    import subprocess

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from tau3_grpo.training.rl import runner

    for key, value in {"TAU3_ROOT": tmp_path, "TAU3_RUN_ROOT": tmp_path / "runs",
                       "TAU3_MODEL_ROOT": tmp_path / "models", "TAU3_ENV_FILE": tmp_path / "absent"}.items():
        monkeypatch.setenv(key, str(value))
    command, env, snapshot = runner.resolve(tmp_path / "run", estimator=estimator,
                                           token_protocol="tau3_token_budget_v1")
    output = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN="1"), text=True)
    args = shlex.split(output.splitlines()[-1])
    with initialize_config_dir(config_dir=str(CODE_ROOT / "verl/verl/trainer/config"), version_base=None):
        config = OmegaConf.to_container(compose(config_name="ppo_trainer",
            overrides=args[args.index("tau3_grpo.training.rl.train") + 1:]), resolve=True)
    assert config["algorithm"]["adv_estimator"] == estimator
    assert config["tau3_token_protocol"] == "tau3_token_budget_v1"
    rollout = config["actor_rollout_ref"]["rollout"]
    assert (rollout["prompt_length"], rollout["response_length"], rollout["max_model_len"]) == (8192, 16384, 24576)
    assert env["TOOL_SCHEMA_VERSION"] == "tau3_full_schema_v2"
    assert env["TAU3_USER_MAX_MODEL_LEN"] == "16384"
    assert snapshot["token_protocol"]["version"] == "tau3_token_budget_v1"


@pytest.mark.parametrize("old,new", [(None, "tau3_token_budget_v1"), ("tau3_token_budget_v1", None)])
def test_resume_cannot_switch_token_protocol(tmp_path, old, new, monkeypatch):
    from tau3_grpo.training.rl import runner

    for key, value in {"TAU3_ROOT": tmp_path, "TAU3_RUN_ROOT": tmp_path / "runs",
                       "TAU3_MODEL_ROOT": tmp_path / "models", "TAU3_ENV_FILE": tmp_path / "absent"}.items():
        monkeypatch.setenv(key, str(value))
    (tmp_path / "resolved-hydra.yaml").write_text(json.dumps({"tau3_token_protocol": old}))
    with pytest.raises(ValueError, match="cannot change the token/budget protocol"):
        runner.resolve(tmp_path, estimator="grpo", resume_from=tmp_path / "global_step_10",
                       token_protocol=new)


@pytest.mark.parametrize("corrupt", [False, True])
def test_full_v4_runtime_concurrent_persistence_and_failures(tokenizer, entry, tmp_path, monkeypatch, corrupt):
    from tau2.data_model.message import AssistantMessage
    from tau2.user import user_simulator

    from tau3_grpo.envs.registry import SESSIONS
    from tau3_grpo.evaluation import token_runtime
    from tau3_grpo.evaluation.harness import TOKENS_V4
    from tau3_grpo.evaluation.rescore import rescore_run
    from tau3_grpo.evaluation.runtime import Endpoint, EvalSpec, run_evaluation
    from tau3_grpo.models.token_endpoint import TokenEndpoint

    checks, requests = [], []
    monkeypatch.setattr(token_runtime, "check_context", lambda endpoint, limit: checks.append((endpoint.model, limit)))
    monkeypatch.setattr(user_simulator, "generate", lambda **kw: AssistantMessage(role="assistant", content="###STOP###"))

    def post(self, payload):
        assert self.timeout == 600
        requests.append(payload)
        ids = tokenizer.encode("Done.<|im_end|>", add_special_tokens=False)
        choice = {"token_ids": ids, "prompt_token_ids": payload["prompt"], "finish_reason": "stop",
                  "logprobs": {"tokens": [f"token_id:{t}" for t in ids], "token_logprobs": [-.2] * len(ids)}}
        if corrupt:
            choice.pop("token_ids")
        return {"model": payload["model"], "choices": [choice],
                "usage": {"prompt_tokens": len(payload["prompt"]), "completion_tokens": len(ids)}}

    monkeypatch.setattr(TokenEndpoint, "_post", post)
    result = run_evaluation(spec=EvalSpec(target="selection", trials=2, max_concurrency=2,
            harness_protocol=TOKENS_V4, tokenizer_path=str(tokenizer.name_or_path),
            token_request_timeout=600),
        policy=Endpoint("policy", "http://unused/v1"), user=Endpoint("user", "http://unused/v1"),
        output_dir=tmp_path, selection_entries=[entry])
    assert checks == [("policy", 24576), ("user", 16384)]
    assert len(requests) == 2  # Failed transports do not silently retry.
    assert len({r["seed"] for r in requests}) == 2
    assert result["failed_trajectories"] == (2 if corrupt else 0)
    assert result["completed_trajectories"] == (0 if corrupt else 2)
    assert result == rescore_run(tmp_path)
    assert SESSIONS.active_count() == 0
    if corrupt:
        assert result["metrics"] is None
        assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 2
    else:
        rows = [json.loads(line) for line in (tmp_path / "trajectories.jsonl").read_text().splitlines()]
        assert [row["trial"] for row in rows] == [0, 1]
        assert all(row["simulation"]["info"]["token_protocol"]["receipts"] for row in rows)
        assert len(result["provenance"]["tokenizer_files_sha256"]) == 3


def test_real_http_token_transport():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from tau3_grpo.evaluation.runtime import Endpoint
    from tau3_grpo.models.token_endpoint import TokenEndpoint

    captured = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append((self.path, payload))
            response = {"model": "policy", "choices": [{"prompt_token_ids": payload["prompt"],
                "token_ids": [3], "finish_reason": "stop", "logprobs": {
                    "tokens": ["token_id:3"], "token_logprobs": [-.2]}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = TokenEndpoint(Endpoint("policy", f"http://127.0.0.1:{server.server_port}/v1"))
        output = asyncio.run(endpoint.generate(request_id="r", prompt_ids=[1, 2],
            sampling_params={"temperature": .7, "max_tokens": 10}))
        assert output.token_ids == [3]
        path, payload = captured[0]
        assert path == "/v1/completions" and payload["prompt"] == [1, 2]
        assert payload["return_token_ids"] and not payload["add_special_tokens"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("problem", ["user_context", "initial_context", "malformed_call"])
def test_token_boundary_cleanup(tokenizer, entry, tmp_path, monkeypatch, problem):
    from litellm import ContextWindowExceededError
    from tau2.user import user_simulator

    from tau3_grpo.envs.registry import SESSIONS
    from tau3_grpo.evaluation.runtime import Endpoint

    def fail_user(**kwargs):
        raise ContextWindowExceededError(message="context too long", model="user", llm_provider="openai")

    monkeypatch.setattr(user_simulator, "generate", fail_user)
    text = "Done." if problem != "malformed_call" else "<tool_call></tool_call>"
    manager = ScriptedManager(tokenizer, [text])
    budget = replace(default_budget(), prompt=1) if problem == "initial_context" else None
    operation = train(tokenizer, entry, manager, Endpoint("user", "http://unused"), tmp_path, budget=budget)
    if problem == "initial_context":
        with pytest.raises(ValueError, match="Initial prompt exceeds"):
            asyncio.run(operation)
        assert not manager.requests
    else:
        output = asyncio.run(operation)
        receipt = json.loads(output.extra_fields["token_protocol_json"])
        assert receipt["termination_reason"] == ("agent_error" if problem == "malformed_call" else "context_window_exceeded")
    assert SESSIONS.active_count() == 0


@pytest.mark.parametrize("partial", [False, True])
def test_exact_limit_write_and_partial_batch_match_transports(tokenizer, entry, tmp_path, partial):
    from tau3_grpo.data.schema import ArealTaskRecord
    from tau3_grpo.envs.adapter import adapt_record, load_flight_db
    from tau3_grpo.evaluation.runtime import Endpoint
    from tau3_grpo.models.token_endpoint import TokenEndpoint

    adapted = adapt_record(ArealTaskRecord.model_validate(entry.task))
    database = load_flight_db(adapted.db_path)
    reservation = next(iter(database.reservations))
    original = database.model_dump_json()
    text = ("<tool_call><function=cancel_reservation><parameter=reservation_id>"
            + reservation + "</parameter></function></tool_call>")
    if partial:
        text += "<tool_call"
    native = ScriptedManager(tokenizer, [text], finish="length")
    transport = ScriptedManager(tokenizer, [text], finish="length")
    budget = replace(default_budget(), response=len(native.tokens[0]))
    user = Endpoint("user", "http://unused")
    left = asyncio.run(train(tokenizer, entry, native, user, tmp_path, budget=budget))
    right = asyncio.run(train(tokenizer, entry, TokenEndpoint(Endpoint("policy", "http://unused"),
        transport=transport.transport), user, tmp_path, budget=budget))
    assert left.extra_fields["token_protocol_json"] == right.extra_fields["token_protocol_json"]
    assert [r[0] for r in native.requests] == [r[0] for r in transport.requests]
    for output in (left, right):
        reward = output.extra_fields["reward_extra_info"]
        assert (reward["initial_db_hash"] == reward["db_hash"]) is partial
        assert len(output.response_ids) == budget.response
    assert load_flight_db(adapted.db_path).model_dump_json() == original


def test_token_protocol_rejects_different_control_envelope(tokenizer, tmp_path):
    from tau3_grpo.evaluation.runtime import Endpoint
    from tau3_grpo.evaluation.token_runtime import make_loop
    from tau3_grpo.integrations.verl.token_budget import configure

    async def check():
        loop, _ = await make_loop(tokenizer=tokenizer, manager=ScriptedManager(tokenizer, []),
                                  user=Endpoint("user", "http://unused"), directory=tmp_path)
        loop.max_user_turns = 16
        with pytest.raises(ValueError, match="Turn/observation limits"):
            configure(loop)

    asyncio.run(check())


@pytest.mark.parametrize('estimator', ['grpo', 'tau_gigpo', 'mt_gtpo'])
def test_call_attribution_is_published_without_changing_native_outputs(tokenizer, entry, tmp_path, monkeypatch, estimator):
    from tau2.data_model.message import AssistantMessage
    from tau2.user import user_simulator

    from tau3_grpo.evaluation.runtime import Endpoint

    monkeypatch.setattr(user_simulator, 'generate', lambda **kwargs:
                        AssistantMessage(role='assistant', content='###STOP###'))
    texts = [xml('1+1') + xml('1/0') + xml('2+2'), 'Done.']
    user = Endpoint('user', 'http://unused')
    monkeypatch.setenv('TAU3_RECORD_CALL_ATTRIBUTION', '0')
    before = asyncio.run(train(tokenizer, entry, ScriptedManager(tokenizer, texts), user,
                               tmp_path, estimator=estimator))
    monkeypatch.setenv('TAU3_RECORD_CALL_ATTRIBUTION', '1')
    after = asyncio.run(train(tokenizer, entry, ScriptedManager(tokenizer, texts), user,
                              tmp_path, estimator=estimator))
    assert before.prompt_ids == after.prompt_ids
    assert before.response_ids == after.response_ids
    assert before.response_mask == after.response_mask
    assert before.response_logprobs == after.response_logprobs
    assert before.reward_score == after.reward_score
    facts = json.loads(after.extra_fields['trajectory_facts_json'])
    before_facts = json.loads(before.extra_fields['trajectory_facts_json'])
    assert 'call_attribution' not in before_facts['turns'][0]
    first, second = facts['turns']
    attr = first['call_attribution']
    assert attr['eligible_for_call_credit']
    assert [c['error'] for c in attr['calls']] == [False, True, False]
    assert [c['call_id'] for c in attr['calls']] == [c['id'] for c in first['tool_calls']]
    assert len({c['call_id'] for c in attr['calls']}) == 3
    assert not second['call_attribution']['calls']
    for turn in facts['turns']:
        a, b = turn['call_attribution']['retained_span']
        assert turn['call_attribution']['emitted_token_ids'] == after.response_ids[a:b]
    if estimator == 'mt_gtpo':
        a = json.loads(before.extra_fields['process_reward_json'])
        b = json.loads(after.extra_fields['process_reward_json'])
        for key in ['turn_rewards', 'turn_spans']:
            assert a[key] == b[key]
        assert a.get('official_outcome') == b.get('official_outcome')
        assert all('call_attribution' not in t for t in b['turn_records'])


@pytest.mark.parametrize('mode', ['call_local_v1', 'call_residual_v1'])
def test_native_generated_calls_reach_call_local_trainer(tokenizer, entry, tmp_path, monkeypatch, mode):
    import numpy as np
    import torch
    from tau2.data_model.message import AssistantMessage
    from tau2.user import user_simulator

    from tau3_grpo.evaluation.runtime import Endpoint
    from tau3_grpo.integrations.verl.mt_gtpo import compute_mt_gtpo_verl

    monkeypatch.setenv('TAU3_RECORD_CALL_ATTRIBUTION', '1')
    monkeypatch.setattr(user_simulator, 'generate', lambda **kw: AssistantMessage(role='assistant', content='###STOP###'))
    reward_config = {'mode': 'paper', 'version': 'paper_env_split_v4'}
    outputs = [asyncio.run(train(tokenizer, entry, ScriptedManager(tokenizer, [text, 'Done.']),
        Endpoint('user', 'http://unused'), tmp_path, estimator='mt_gtpo', reward_config=reward_config,
        sampling_identity={'sample_group_uid': 'native-group', 'trial': i, 'seed': 42}))
        for i, text in enumerate([xml('1/0') + xml('1+1'), xml('1+1')])]
    length = max(len(o.response_ids) for o in outputs)
    responses, mask, rewards = (torch.zeros((2, length), dtype=dtype) for dtype in (torch.long, torch.long, torch.float32))
    for i, output in enumerate(outputs):
        n = len(output.response_ids)
        responses[i, :n] = torch.tensor(output.response_ids)
        mask[i, :n] = torch.tensor(output.response_mask)
        rewards[i, -1] = output.reward_score
    metadata = {key: np.asarray([o.extra_fields[key] for o in outputs], dtype=object)
                for key in ['process_reward_json', 'trajectory_facts_json']}
    config = {'process_reward': reward_config, 'mt_gtpo': {'credit_mode': mode}}
    before = mask.clone()
    advantages, _ = compute_mt_gtpo_verl(rewards, mask, ['native-group']*2, config, metadata,
                                        batch={'responses': responses})
    assert torch.equal(mask, before)
    assert not advantages[mask == 0].any()
    record = json.loads(metadata['mt_gtpo_replay_json'][0])
    assert len(record['call_credit'][0]['call_advantages']) == 2
    assert record['call_credit'][0]['call_advantages'][0] < record['call_credit'][0]['call_advantages'][1]
