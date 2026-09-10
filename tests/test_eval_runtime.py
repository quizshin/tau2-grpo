"""Live evaluator orchestration without making endpoint calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

from tau3_grpo.evaluation import runtime as eval_runtime
from tau3_grpo.evaluation.runtime import Endpoint, EvalSpec, run_evaluation


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
        spec=EvalSpec(target="selection", trials=2, max_concurrency=2),
        policy=Endpoint("policy", "http://policy/v1"),
        user=Endpoint("user", "http://user/v1", temperature=1.0),
        output_dir=tmp_path,
    )

    assert summary["planned_trajectories"] == 2
    assert summary["completed_trajectories"] == 2
    assert summary["failed_trajectories"] == 0
    assert summary["mean_reward"] == 0.5
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
    error = json.loads((tmp_path / "errors.jsonl").read_text().strip())
    assert error["error_type"] == "RuntimeError"
