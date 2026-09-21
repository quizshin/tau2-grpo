"""Live evaluator orchestration without making endpoint calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tau3_grpo.evaluation import runtime as eval_runtime
from tau3_grpo.evaluation.rescore import main as rescore_main
from tau3_grpo.evaluation.rescore import rescore_run
from tau3_grpo.evaluation.runtime import Endpoint, EvalSpec, run_evaluation


@pytest.fixture(autouse=True)
def isolated_provenance(monkeypatch):
    # These tests replace real task construction; source/data capture has its own integration tests.
    monkeypatch.setattr(eval_runtime, "evaluation_provenance", lambda jobs: {"provenance_schema": "test_fixture"})


def test_endpoint_uses_openai_compatible_litellm_prefix():
    endpoint = Endpoint(model="Qwen/Test", base_url="http://localhost:8000/v1")
    assert endpoint.litellm_model == "openai/Qwen/Test"
    assert endpoint.llm_args()["api_base"] == "http://localhost:8000/v1"


def test_endpoint_preserves_explicit_provider_prefix():
    endpoint = Endpoint(model="openai/local-model", base_url="http://localhost/v1")
    assert endpoint.litellm_model == "openai/local-model"


def test_run_evaluation_writes_trajectory_and_summary(monkeypatch, tmp_path):
    jobs = [
        {"task_id": "t1", "trial": 0, "seed": 42, "task": object(), "db_path": None},
        {"task_id": "t1", "trial": 1, "seed": 43, "task": object(), "db_path": None},
    ]
    monkeypatch.setattr(eval_runtime, "_selection_jobs", lambda entries, trials, seed: jobs)

    def fake_run_one(**kwargs):
        seed = kwargs["seed"]
        return SimpleNamespace(
            reward_info=SimpleNamespace(reward=1.0 if seed == 42 else 0.0),
            termination_reason=SimpleNamespace(value="user_stop"),
            model_dump=lambda mode: {"id": f"simulation-{seed}"},
        )

    monkeypatch.setattr(eval_runtime, "_run_one", fake_run_one)
    summary = run_evaluation(
        spec=EvalSpec(target="selection", trials=2, max_concurrency=2, include_pass_hat=True),
        policy=Endpoint("policy", "http://policy/v1"),
        user=Endpoint("user", "http://user/v1", temperature=1.0),
        output_dir=tmp_path,
        provenance={"checkpoint": "exact-merged", "checkpoint_hash": "verified-hash"},
    )

    assert summary["planned_trajectories"] == 2
    assert summary["completed_trajectories"] == 2
    assert summary["failed_trajectories"] == 0
    assert summary["mean_reward"] == 0.5
    assert summary["metrics"] == {"pass@1": 0.5, "pass@2": 1.0, "pass^1": 0.5, "pass^2": 0.0}
    assert rescore_run(tmp_path) == summary
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert summary["provenance"]["checkpoint_hash"] == "verified-hash"
    assert "api_key" not in (tmp_path / "run.json").read_text()
    rescored = tmp_path / "rescored.json"
    assert rescore_main(["--run-dir", str(tmp_path), "--output", str(rescored)]) == 0
    assert json.loads(rescored.read_text()) == summary
    assert rescore_main(["--run-dir", str(tmp_path), "--output", str(rescored)]) == 2
    assert json.loads(rescored.read_text()) == summary
    rows = [json.loads(line) for line in (tmp_path / "trajectories.jsonl").read_text().splitlines()]
    assert [row["trial"] for row in rows] == [0, 1]


def test_run_evaluation_records_individual_failures(monkeypatch, tmp_path):
    jobs = [
        {"task_id": "t1", "trial": 0, "seed": 42, "task": object(), "db_path": None}
    ]
    monkeypatch.setattr(eval_runtime, "_selection_jobs", lambda entries, trials, seed: jobs)
    monkeypatch.setattr(
        eval_runtime,
        "_run_one",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("endpoint unavailable")),
    )
    summary = run_evaluation(
        spec=EvalSpec(target="selection", trials=1, max_concurrency=1),
        policy=Endpoint("policy", "http://policy/v1"),
        user=Endpoint("user", "http://user/v1"),
        output_dir=tmp_path,
    )
    assert summary["failed_trajectories"] == 1
    assert summary["metrics"] is None
    assert not summary["metrics_valid"]
    assert rescore_run(tmp_path) == summary
    error = json.loads((tmp_path / "errors.jsonl").read_text().strip())
    assert error["error_type"] == "RuntimeError"


@pytest.mark.parametrize("reason", ["infrastructure_error", "unexpected_error"])
def test_official_infrastructure_error_is_not_a_scored_model_failure(monkeypatch, tmp_path, reason):
    jobs = [{"task_id": "t1", "trial": 0, "seed": 42, "task": object(), "db_path": None}]
    monkeypatch.setattr(eval_runtime, "_selection_jobs", lambda *args: jobs)
    monkeypatch.setattr(eval_runtime, "_run_one", lambda **kwargs: SimpleNamespace(
        reward_info=SimpleNamespace(reward=0),
        termination_reason=SimpleNamespace(value=reason),
        model_dump=lambda mode: {"termination_reason": reason},
    ))
    summary = run_evaluation(
        spec=EvalSpec(target="selection", trials=1),
        policy=Endpoint("policy", "http://policy"), user=Endpoint("user", "http://user"),
        output_dir=tmp_path,
    )
    assert summary["completed_trajectories"] == 0
    assert summary["failed_trajectories"] == 1
    assert summary["metrics"] is None
    assert rescore_main(["--run-dir", str(tmp_path)]) == 1
    # Even if interruption leaves no trajectory/error files, planned tasks remain.
    (tmp_path / "errors.jsonl").unlink()
    (tmp_path / "trajectories.jsonl").unlink()
    assert rescore_run(tmp_path)["missing_trajectories"] == 1


def test_existing_output_is_rejected_before_any_endpoint_call(monkeypatch, tmp_path):
    jobs = [{"task_id": "t1", "trial": 0, "seed": 42, "task": object(), "db_path": None}]
    monkeypatch.setattr(eval_runtime, "_selection_jobs", lambda *args: jobs)
    monkeypatch.setattr(eval_runtime, "_run_one", lambda **kwargs: pytest.fail("endpoint called"))
    (tmp_path / "summary.json").write_text("preserve me")
    with pytest.raises(ValueError, match="already exists"):
        run_evaluation(
            spec=EvalSpec(target="selection", trials=1),
            policy=Endpoint("policy", "http://policy"), user=Endpoint("user", "http://user"),
            output_dir=tmp_path,
        )
    assert (tmp_path / "summary.json").read_text() == "preserve me"


@pytest.mark.parametrize("side,expected", [
    ("left", "abcd...(truncated)"),
    ("right", "(truncated)...ghij"),
    ("middle", "ab...(truncated)...ij"),
])
def test_observation_projection_preserves_historical_marker_semantics(side, expected):
    from tau3_grpo.envs.observations import project_tool_text

    text, receipt = project_tool_text("abcdefghij", 4, side)
    assert text == expected
    assert receipt["raw_chars"] == 10 and receipt["visible_chars"] == len(expected)
    assert receipt["raw_sha256"] != receipt["visible_sha256"]
    assert project_tool_text("abcd", 4, side)[0] == "abcd"


def test_policy_projection_never_mutates_raw_tool_messages_or_prior_history(monkeypatch):
    import tau2.agent.llm_agent as native
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage

    from tau3_grpo.envs.agent import MultiCallAirlineAgent

    requests = []

    def generate(**kwargs):
        requests.append(kwargs["messages"])
        return AssistantMessage(role="assistant", content="done")

    monkeypatch.setattr(native, "generate", generate)
    agent = MultiCallAirlineAgent(tools=[], domain_policy="policy", llm="test", project_observations=True)
    state = agent.get_init_state()
    raw = ToolMessage(id="long", role="tool", requestor="assistant", content="a" * 40000 + "b" * 40000)
    batch = MultiToolMessage(role="tool", tool_messages=[raw])
    _, state = agent.generate_next_message(batch, state)
    _, state = agent.generate_next_message(UserMessage(role="user", content="continue"), state)
    assert raw.content == state.messages[0].content == "a" * 40000 + "b" * 40000
    assert len(agent.observation_receipts) == 1
    receipt = agent.observation_receipts[0]
    assert receipt["truncated"] and receipt["visible_chars"] == 65553
    for request in requests:
        visible = next(m for m in request if isinstance(m, ToolMessage))
        assert visible.id == raw.id and len(visible.content) == 65553
        assert visible.content == "a" * 32768 + "...(truncated)..." + "b" * 32768


def test_v2_metadata_rejects_inapplicable_legacy_budget_overrides():
    from tau3_grpo.evaluation.harness import CONTROL_V2, protocol_metadata

    with pytest.raises(ValueError, match="do not override"):
        protocol_metadata(CONTROL_V2, max_steps=32)
    with pytest.raises(ValueError, match="Unknown"):
        protocol_metadata("typo")


def test_single_execution_error_does_not_cancel_other_planned_trials(monkeypatch, tmp_path):
    jobs = [{'task_id': 'task', 'trial': i, 'seed': 42 + i, 'task': object(), 'db_path': None}
            for i in range(4)]
    monkeypatch.setattr(eval_runtime, '_selection_jobs', lambda *args: jobs)
    called = []

    def run_one(**kwargs):
        seed = kwargs['seed']
        called.append(seed)
        if seed == 42:
            raise RuntimeError('isolated request failure')
        return SimpleNamespace(
            reward_info=SimpleNamespace(reward=1.0),
            termination_reason=SimpleNamespace(value='user_stop'),
            model_dump=lambda mode: {'id': str(seed)},
        )

    monkeypatch.setattr(eval_runtime, '_run_one', run_one)
    summary = run_evaluation(
        spec=EvalSpec(target='selection', trials=4, max_concurrency=1),
        policy=Endpoint('policy', 'http://unused'), user=Endpoint('user', 'http://unused'),
        output_dir=tmp_path,
    )
    assert called == [42, 43, 44, 45]  # Includes later trials, but no retry/replacement.
    assert summary['completed_trajectories'] == 3
    assert summary['failed_trajectories'] == 1 and summary['missing_trajectories'] == 0
    assert not summary['metrics_valid'] and summary['metrics'] is None
    error = json.loads((tmp_path / 'errors.jsonl').read_text())
    assert 'RuntimeError: isolated request failure' in error['traceback']
