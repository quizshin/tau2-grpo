"""Live selection and official-final evaluation against served checkpoints.

The policy and user simulator are OpenAI-compatible endpoints. AReaL selection
tasks are run against their record-specific FlightDB; official τ³ tasks use the
pinned Airline ``base`` split from tau2-bench. The official evaluator, rather
than a project-side reward reimplementation, scores every trajectory.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tau3_grpo.data.manifest import TAU3_REVISION, ManifestEntry
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.adapter import (
    AIRLINE_DOMAIN,
    adapt_record,
    build_environment,
    load_default_flight_db,
    load_flight_db,
)
from tau3_grpo.evaluation.scoring import resolve_ks, summarize_trials
from tau3_grpo.prompts import prompt_provenance


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
    ks: tuple[int, ...] | None = None
    include_pass_hat: bool = False


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
    from tau2.runner.build import build_user
    from tau2.runner.simulation import run_simulation

    from tau3_grpo.envs.agent import MultiCallAirlineAgent

    if db_path is None:
        environment = build_environment(load_default_flight_db())
        replay_db = load_default_flight_db()
    else:
        environment = build_environment(load_flight_db(db_path))
        replay_db = load_flight_db(db_path)

    agent = MultiCallAirlineAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=policy.litellm_model,
        llm_args=policy.llm_args(),
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
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all task/trial pairs, persist full trajectories, and return a summary."""

    if spec.max_steps <= 0 or spec.max_errors <= 0 or spec.max_concurrency <= 0:
        raise ValueError("max_steps, max_errors and max_concurrency must be positive")
    ks = resolve_ks(spec.trials, spec.ks)
    if spec.target == "selection":
        jobs = list(_selection_jobs(selection_entries, spec.trials, spec.seed))
    elif spec.target == "tau3-final":
        jobs = list(_official_jobs(spec.trials, spec.seed))
    else:
        raise ValueError(f"unknown evaluation target: {spec.target}")

    planned = [{key: job[key] for key in ("task_id", "trial", "seed")} for job in jobs]
    # Validate the full schedule before any paid endpoint calls.
    summarize_trials(planned=planned, results=[], errors=[], trials=spec.trials, ks=ks)
    metadata = {
        "schema_version": 1,
        "benchmark_revision": TAU3_REVISION,
        "spec": {**asdict(spec), "ks": list(ks)},
        "planned": planned,
        "endpoints": {
            name: {"model": endpoint.model, "temperature": endpoint.temperature}
            for name, endpoint in (("policy", policy), ("user", user))
        },
        "provenance": {**(provenance or {}), **prompt_provenance()},
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "trajectories.jsonl", "errors.jsonl", "summary.json"):
        if (out / name).exists():
            raise ValueError(f"evaluation output already exists: {out / name}; use a new directory")
    with (out / "run.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with (
        (out / "trajectories.jsonl").open("x", encoding="utf-8") as trajectory_file,
        (out / "errors.jsonl").open("x", encoding="utf-8") as error_file,
        ThreadPoolExecutor(max_workers=spec.max_concurrency) as pool,
    ):
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
            identity = {key: job[key] for key in ("task_id", "trial", "seed")}
            scored = False
            try:
                simulation = future.result()
                if simulation.termination_reason.value == "infrastructure_error":
                    row = {
                        **identity,
                        "error_type": "InfrastructureError",
                        "error": "official simulation reported infrastructure_error",
                        "simulation": simulation.model_dump(mode="json"),
                    }
                else:
                    reward = float(simulation.reward_info.reward)
                    if not math.isfinite(reward):
                        raise ValueError("simulation reward must be finite")
                    row = {
                        **identity,
                        "reward": reward,
                        "termination_reason": simulation.termination_reason.value,
                        "simulation": simulation.model_dump(mode="json"),
                    }
                    scored = True
            except Exception as exc:  # retain failures without losing completed trials
                row = {**identity, "error_type": type(exc).__name__, "error": str(exc)}
            # Disk/serialization failures must propagate, not duplicate a trial
            # into both success and error files.
            handle = trajectory_file if scored else error_file
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            (results if scored else errors).append(row)

    results.sort(key=lambda item: (item["task_id"], item["trial"]))
    errors.sort(key=lambda item: (item["task_id"], item["trial"]))
    for name, rows in (("trajectories.jsonl", results), ("errors.jsonl", errors)):
        temporary = out / f"{name}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(out / name)

    summary = {
        "target": spec.target,
        **summarize_trials(
            planned=planned, results=results, errors=errors, trials=spec.trials,
            ks=ks, include_pass_hat=spec.include_pass_hat,
        ),
        "policy_model": policy.model,
        "user_model": user.model,
        "benchmark_revision": TAU3_REVISION,
        "provenance": metadata["provenance"],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
