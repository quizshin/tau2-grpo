"""Live selection and official-final evaluation against served checkpoints.

The policy and user simulator are OpenAI-compatible endpoints. AReaL selection
tasks are run against their record-specific FlightDB; official τ³ tasks use the
pinned Airline ``base`` split from tau2-bench. The official evaluator, rather
than a project-side reward reimplementation, scores every trajectory.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tau3_grpo.data.manifest import ManifestEntry
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.env.adapter import (
    AIRLINE_DOMAIN,
    adapt_record,
    build_environment,
    load_default_flight_db,
    load_flight_db,
)


@dataclass(frozen=True)
class Endpoint:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    temperature: float = 0.4

    @property
    def litellm_model(self) -> str:
        if self.model.startswith(("openai/", "hosted_vllm/")):
            return self.model
        return f"openai/{self.model}"

    def llm_args(self) -> dict[str, Any]:
        return {
            "api_base": self.base_url,
            "api_key": self.api_key,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class EvalSpec:
    target: str
    trials: int = 4
    seed: int = 42
    max_steps: int = 30
    max_errors: int = 10
    max_concurrency: int = 16


def _run_one(
    *,
    task: Any,
    db_path: Path | None,
    policy: Endpoint,
    user: Endpoint,
    seed: int,
    max_steps: int,
    max_errors: int,
) -> Any:
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_agent, build_user
    from tau2.runner.simulation import run_simulation

    if db_path is None:
        environment = build_environment(load_default_flight_db())
        replay_db = load_default_flight_db()
    else:
        environment = build_environment(load_flight_db(db_path))
        replay_db = load_flight_db(db_path)

    agent = build_agent(
        "llm_agent",
        environment,
        llm=policy.litellm_model,
        llm_args=policy.llm_args(),
        task=task,
    )
    simulated_user = build_user(
        "user_simulator",
        environment,
        task,
        llm=user.litellm_model,
        llm_args=user.llm_args(),
    )
    orchestrator = Orchestrator(
        domain=AIRLINE_DOMAIN,
        agent=agent,
        user=simulated_user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
    )
    return run_simulation(
        orchestrator,
        evaluation_type=EvaluationType.ALL,
        env_kwargs={"db": replay_db},
    )


def _selection_jobs(entries: Sequence[ManifestEntry], trials: int, seed: int):
    for entry in entries:
        if entry.task is None:
            raise ValueError(f"selection entry {entry.task_id} has no pinned record")
        record = ArealTaskRecord.model_validate(entry.task)
        adapted = adapt_record(record)
        for trial in range(trials):
            yield {
                "task_id": entry.task_id,
                "trial": trial,
                "seed": seed + trial,
                "task": adapted.task,
                "db_path": adapted.db_path,
            }


def _official_jobs(trials: int, seed: int):
    from tau2.runner.helpers import load_tasks

    tasks = load_tasks(task_set_name="airline", task_split_name="base")
    if len(tasks) != 50:
        raise ValueError(f"official τ³ Airline base must contain 50 tasks, got {len(tasks)}")
    for task in tasks:
        for trial in range(trials):
            yield {
                "task_id": task.id,
                "trial": trial,
                "seed": seed + trial,
                "task": task,
                "db_path": None,
            }


def run_evaluation(
    *,
    spec: EvalSpec,
    policy: Endpoint,
    user: Endpoint,
    output_dir: str | Path,
    selection_entries: Sequence[ManifestEntry] = (),
) -> dict[str, Any]:
    """Run all task/trial pairs, persist full trajectories, and return a summary."""

    if spec.trials <= 0 or spec.max_concurrency <= 0:
        raise ValueError("trials and max_concurrency must be positive")
    if spec.target == "selection":
        jobs = list(_selection_jobs(selection_entries, spec.trials, spec.seed))
    elif spec.target == "tau3-final":
        jobs = list(_official_jobs(spec.trials, spec.seed))
    else:
        raise ValueError(f"unknown evaluation target: {spec.target}")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=spec.max_concurrency) as pool:
        futures = {
            pool.submit(
                _run_one,
                task=job["task"],
                db_path=job["db_path"],
                policy=policy,
                user=user,
                seed=job["seed"],
                max_steps=spec.max_steps,
                max_errors=spec.max_errors,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                simulation = future.result()
                reward = float(simulation.reward_info.reward)
                results.append(
                    {
                        "task_id": job["task_id"],
                        "trial": job["trial"],
                        "seed": job["seed"],
                        "reward": reward,
                        "termination_reason": simulation.termination_reason.value,
                        "simulation": simulation.model_dump(mode="json"),
                    }
                )
            except Exception as exc:  # retain failures without losing completed trials
                errors.append(
                    {
                        "task_id": job["task_id"],
                        "trial": job["trial"],
                        "seed": job["seed"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    results.sort(key=lambda item: (item["task_id"], item["trial"]))
    errors.sort(key=lambda item: (item["task_id"], item["trial"]))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    with (out / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, sort_keys=True) + "\n")

    rewards = [item["reward"] for item in results]
    summary = {
        "target": spec.target,
        "tasks": len({job["task_id"] for job in jobs}),
        "trials_per_task": spec.trials,
        "planned_trajectories": len(jobs),
        "completed_trajectories": len(results),
        "failed_trajectories": len(errors),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "solve_rate": sum(value > 0.0 for value in rewards) / len(rewards) if rewards else 0.0,
        "policy_model": policy.model,
        "user_model": user.model,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
