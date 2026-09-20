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
