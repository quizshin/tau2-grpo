"""Live selection and official-final evaluation against served checkpoints.

The policy and user simulator are OpenAI-compatible endpoints. AReaL selection
tasks are run against their record-specific FlightDB; official τ³ tasks use the
pinned Airline ``base`` split from tau2-bench. The official evaluator, rather
than a project-side reward reimplementation, scores every trajectory.
"""

from __future__ import annotations

import json
import math
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tau3_grpo.data.manifest import TAU3_REVISION, ManifestEntry
from tau3_grpo.data.opening import initial_user_message as task_opening
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.adapter import (
    AIRLINE_DOMAIN,
    adapt_record,
    build_environment,
    load_default_flight_db,
    load_flight_db,
)
from tau3_grpo.evaluation.eligibility import execution_eligibility
from tau3_grpo.evaluation.harness import (
    CONTROL_V2,
    INPUTS_V3,
    LEGACY,
    TOKENS_V4,
    protocol_metadata,
    request_args,
    validate_opening,
    validate_target,
)
from tau3_grpo.evaluation.provenance import evaluation_provenance
from tau3_grpo.evaluation.scoring import resolve_ks, summarize_trials
from tau3_grpo.prompts import prompt_provenance


@dataclass(frozen=True)
class Endpoint:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    temperature: float = 0.7
    repetition_penalty: float | None = None

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
    harness_protocol: str = LEGACY
    tokenizer_path: str | None = None
    token_request_timeout: float = 120
    quality_bundle: str | None = None
    recovery_input: str | None = None
    user_protocol: str = "scope_v1"

    def __post_init__(self):
        import math

        if not math.isfinite(self.token_request_timeout) or self.token_request_timeout <= 0:
            raise ValueError("token_request_timeout must be finite and positive")


def _run_one(
    *,
    task: Any,
    db_path: Path | None,
    policy: Endpoint,
    user: Endpoint,
    seed: int,
    max_steps: int,
    max_errors: int,
    harness_protocol: str = LEGACY,
    initial_user_message: str | None = None,
    outcome_contract: dict | None = None,
    behavior_output: str | None = None,
    behavior_budget_directory: str | None = None,
    completed_rollout_path: str | None = None,
    user_protocol: str = "scope_v1",
) -> Any:
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.build import build_user
    from tau2.runner.simulation import run_simulation

    from tau3_grpo.envs.agent import MultiCallAirlineAgent

    protocol = protocol_metadata(harness_protocol, max_steps=max_steps, max_errors=max_errors)
    policy_args = request_args(harness_protocol, policy, role="policy")
    user_args = request_args(harness_protocol, user, role="user")
    if harness_protocol == INPUTS_V3:
        if db_path is None:
            raise ValueError("train_inputs_v3 requires a pinned AReaL task database")
        validate_opening(task, initial_user_message)
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
        llm_args=policy_args,
        project_observations=harness_protocol in (CONTROL_V2, INPUTS_V3),
    )
    user_task = task
    if outcome_contract is not None:
        from tau3_grpo.envs.simulator_scope import scope_text
        USER_SCOPE = scope_text(user_protocol)
        user_task = task.model_copy(deep=True)
        if isinstance(user_task.user_scenario.instructions, str):
            user_task.user_scenario.instructions += '\n\n' + USER_SCOPE
        else:
            user_task.user_scenario.instructions.task_instructions += '\n\n' + USER_SCOPE
    simulated_user = build_user(
        "user_simulator",
        environment,
        user_task,
        llm=user.litellm_model,
        llm_args=user_args,
    )
    from tau3_grpo.envs.orchestrator import (
        EvaluationOrchestrator,
        TrainingControlOrchestrator,
        TrainingInputOrchestrator,
    )

    orchestrator_cls = {
        LEGACY: EvaluationOrchestrator, CONTROL_V2: TrainingControlOrchestrator,
        INPUTS_V3: TrainingInputOrchestrator,
    }[harness_protocol]
    orchestrator = orchestrator_cls(
        domain=AIRLINE_DOMAIN,
        agent=agent,
        user=simulated_user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        **({"initial_user_message": initial_user_message} if harness_protocol == INPUTS_V3 else {}),
    )
    if completed_rollout_path:
        import time
        started_path = Path(completed_rollout_path).with_suffix(".started.json")
        started_path.parent.mkdir(parents=True, exist_ok=True)
        with started_path.open("x") as handle:
            json.dump(dict(task_id=task.id, seed=seed, started_at=time.time()), handle)
    try:
        simulation = run_simulation(
            orchestrator,
            evaluation_type=EvaluationType.ALL,
            env_kwargs={"db": replay_db},
        )
    except Exception as error:
        error.evaluation_evidence = orchestrator.failure_snapshot()
        error.evaluation_evidence["harness_protocol"] = protocol
        raise
    simulation.info = {**(simulation.info or {}), "harness_protocol": protocol,
                       "tool_observation_receipts": getattr(agent, "observation_receipts", [])}
    if completed_rollout_path:
        # Complete native trajectory must be durable BEFORE any paid judge call.
        import os
        with Path(completed_rollout_path).open("x") as handle:
            json.dump(simulation.model_dump(mode="json"), handle)
            handle.flush(); os.fsync(handle.fileno())
    if outcome_contract is not None:
        from tau3_grpo.evaluation.outcome_contract import score_simulation
        behavior=None
        if outcome_contract.get('communication'):
            from tau3_grpo.evaluation.communication_contract import judge_communication
            from tau3_grpo.evaluation.eligibility import SCORABLE_TERMINATIONS
            from tau3_grpo.prompts import build_system_prompt
            if simulation.termination_reason.value in SCORABLE_TERMINATIONS:
                try:
                    if not behavior_output or not behavior_budget_directory:
                        raise ValueError('Behavior scoring requires an output and shared budget directory')
                    behavior=judge_communication(simulation.model_dump(mode='json')['messages'],
                        scenario=task.user_scenario.model_dump(mode='json'),policy=build_system_prompt(),
                        contract=outcome_contract,output=behavior_output,budget_directory=behavior_budget_directory)
                except Exception as error:
                    # Keep the completed rollout so a judge error never forces GPU resampling.
                    error.evaluation_evidence={'stage':'communication_scoring',
                        'simulation':simulation.model_dump(mode='json'),'resampling_required':False}
                    raise
        simulation.info['outcome_contract'] = score_simulation(
            simulation, db_path=db_path, task=task, contract=outcome_contract,behavior=behavior)
    if harness_protocol == INPUTS_V3:
        from tau3_grpo.evaluation.provenance import digest

        simulation.info["initial_user_message_sha256"] = digest(initial_user_message)
    return simulation


def _selection_jobs(entries: Sequence[ManifestEntry], trials: int, seed: int):
    for entry in entries:
        if entry.task is None:
            raise ValueError(f"selection entry {entry.task_id} has no pinned record")
        record = ArealTaskRecord.model_validate(entry.task)
        adapted = adapt_record(record)
        for trial in range(trials):
            yield {
                "entry": entry,
                "task_id": entry.task_id,
                "trial": trial,
                "seed": seed + trial,
                "task": adapted.task,
                "db_path": adapted.db_path,
                "initial_user_message": task_opening(entry.task),
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
    if policy.repetition_penalty is not None:
        from tau3_grpo.models.generation_guard import repetition_penalty

        penalty = repetition_penalty(policy.repetition_penalty)
        if penalty != 1.0 and spec.harness_protocol != TOKENS_V4:
            raise ValueError("Non-default repetition penalty requires token-v4 evaluation")
    protocol = protocol_metadata(spec.harness_protocol, max_steps=spec.max_steps, max_errors=spec.max_errors)
    validate_target(spec.harness_protocol, spec.target)
    request_args(spec.harness_protocol, policy, role="policy")
    request_args(spec.harness_protocol, user, role="user")
    from tau3_grpo.envs.simulator_scope import scope_text
    scope_text(spec.user_protocol)
    if spec.user_protocol != "scope_v1" and (spec.target != "selection" or not spec.quality_bundle or spec.harness_protocol != LEGACY):
        raise ValueError("User protocol v2 requires selection legacy with a frozen quality bundle")
    ks = resolve_ks(spec.trials, spec.ks)
    quality = None
    if spec.quality_bundle:
        if spec.target != 'selection' or spec.harness_protocol != LEGACY:
            raise ValueError('Outcome contract v1 currently supports selection with legacy control only')
        from tau3_grpo.evaluation.outcome_contract import load_bundle
        quality = load_bundle(spec.quality_bundle, selection_entries)
        if quality.get('approval_scope')=='smoke_only' and spec.trials!=1:
            raise ValueError('This contract is approved for one-trial pipeline smoke only; formal evaluation is blocked')
    if spec.target == "selection":
        jobs = list(_selection_jobs(selection_entries, spec.trials, spec.seed))
    elif spec.target == "tau3-final":
        jobs = list(_official_jobs(spec.trials, spec.seed))
    else:
        raise ValueError(f"unknown evaluation target: {spec.target}")

    if spec.harness_protocol in (INPUTS_V3, TOKENS_V4):
        for job in jobs:
            validate_opening(job["task"], job.get("initial_user_message"))
            if job["db_path"] is None:
                raise ValueError("train_inputs_v3 requires a pinned AReaL task database")
    else:
        # Legacy/v2 generate their own opening. Do not attest the parquet opening
        # as an input to a run that never consumes it.
        jobs = [{key: value for key, value in job.items() if key != "initial_user_message"}
                for job in jobs]

    token_runtime = None
    tokenizer = None
    token_provenance = {}
    policy_penalty = None
    if spec.harness_protocol == TOKENS_V4:
        from tau3_grpo.evaluation import token_runtime
        from tau3_grpo.models.generation_guard import repetition_penalty
        from tau3_grpo.models.token_budget import default_budget

        policy_penalty = repetition_penalty(1.0 if policy.repetition_penalty is None else policy.repetition_penalty)

        if not spec.tokenizer_path:
            raise ValueError("token_v4 requires the served checkpoint tokenizer_path")
        tokenizer = token_runtime.load_tokenizer(spec.tokenizer_path)
        token_runtime.prepare_runtime()
        token_provenance["tokenizer_files_sha256"] = token_runtime.tokenizer_identity(spec.tokenizer_path)
        budget = default_budget()
        token_runtime.check_context(policy, budget.context)
        token_runtime.check_context(user, budget.user_context)

    planned = [{key: job[key] for key in ("task_id", "trial", "seed")} for job in jobs]
    # Validate the full schedule before any paid endpoint calls.
    summarize_trials(planned=planned, results=[], errors=[], trials=spec.trials, ks=ks)
    metadata = {
        "schema_version": 1,
        "benchmark_revision": TAU3_REVISION,
        "spec": {**asdict(spec), "ks": list(ks)},
        "planned": planned,
        "endpoints": {
            name: {"model": endpoint.model, "temperature": endpoint.temperature,
                   **({"repetition_penalty": policy_penalty} if name == "policy" else {})}
            for name, endpoint in (("policy", policy), ("user", user))
        },
        "provenance": {**(provenance or {}), **token_provenance, **prompt_provenance(), **evaluation_provenance(jobs),
                       "harness_protocol": protocol},
    }
    if quality is not None:
        from tau3_grpo.utils.hashing import sha256_json
        metadata['provenance']['quality_bundle_sha256'] = sha256_json(quality)
        metadata['provenance']['outcome_contract_version'] = quality['version']
        metadata['provenance']['user_scope_sha256'] = sha256_json(scope_text(spec.user_protocol))
        metadata['provenance']['user_protocol'] = spec.user_protocol
    recovered = []
    if spec.recovery_input:
        if quality is None:
            raise ValueError("Recovery requires a frozen quality bundle")
        from tau3_grpo.evaluation.recovery import validate_recovery
        recovered, recovery_provenance = validate_recovery(spec.recovery_input, metadata, quality)
        metadata["provenance"]["saved_rollout_recovery"] = recovery_provenance
    completed_keys = {(r["task_id"], r["trial"], r["seed"]) for r in recovered}
    pending_jobs = [j for j in jobs if (j["task_id"], j["trial"], j["seed"]) not in completed_keys]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "trajectories.jsonl", "errors.jsonl", "summary.json"):
        if (out / name).exists():
            raise ValueError(f"evaluation output already exists: {out / name}; use a new directory")
    with (out / "run.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    results: list[dict[str, Any]] = list(recovered)
    errors: list[dict[str, Any]] = []
    with (
        (out / "trajectories.jsonl").open("x", encoding="utf-8") as trajectory_file,
        (out / "errors.jsonl").open("x", encoding="utf-8") as error_file,
        ThreadPoolExecutor(max_workers=spec.max_concurrency) as pool,
    ):
        for row in recovered:
            trajectory_file.write(json.dumps(row, sort_keys=True) + "\n")
        trajectory_file.flush()
        futures = {
            (pool.submit(token_runtime.run_one, entry=job["entry"], policy=policy, user=user,
                         seed=job["seed"], trial=job["trial"], tokenizer=tokenizer,
                         request_timeout=spec.token_request_timeout)
             if token_runtime is not None else pool.submit(
                _run_one,
                task=job["task"],
                db_path=job["db_path"],
                policy=policy,
                user=user,
                seed=job["seed"],
                max_steps=spec.max_steps,
                max_errors=spec.max_errors,
                harness_protocol=spec.harness_protocol,
                completed_rollout_path=str(out/"completed_rollouts"/f"{job['task_id']}_{job['trial']}_{job['seed']}.json"),
                user_protocol=spec.user_protocol,
                **({'outcome_contract': quality['tasks'][job['task_id']],
                    'behavior_output':str(out/'behavior'),
                    'behavior_budget_directory':quality.get('budget_directory')}
                   if quality else {}),
                **({"initial_user_message": job["initial_user_message"]}
                   if spec.harness_protocol == INPUTS_V3 else {}),
            )): job
            for job in pending_jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            identity = {key: job[key] for key in ("task_id", "trial", "seed")}
            scored = False
            try:
                simulation = future.result()
                reason = simulation.termination_reason.value
                eligibility = execution_eligibility(reason)
                if not eligibility["evaluation_trial_complete"]:
                    row = {
                        **identity,
                        "error_type": "InfrastructureError",
                        "error": f"official simulation reported unresolved {reason}",
                        "execution_eligibility": eligibility,
                        "simulation": simulation.model_dump(mode="json"),
                    }
                else:
                    reward = float(simulation.reward_info.reward)
                    if quality is not None:
                        reward = float(simulation.info['outcome_contract']['reward'])
                    if not math.isfinite(reward):
                        raise ValueError("simulation reward must be finite")
                    row = {
                        **identity,
                        "reward": reward,
                        "termination_reason": simulation.termination_reason.value,
                        "execution_eligibility": execution_eligibility(reason, reward=reward),
                        "simulation": simulation.model_dump(mode="json"),
                    }
                    scored = True
            except Exception as exc:  # retain failures without losing completed trials
                row = {**identity, "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc(),
                       "execution_evidence": getattr(exc, "evaluation_evidence", None),
                       "execution_eligibility": execution_eligibility(None, exception=True)}
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
    if quality is not None and quality.get('dual_rule'):
        from tau3_grpo.evaluation.dual_metrics import summarize_dual
        summary['dual_metrics'] = summarize_dual(
            planned=planned, results=results, errors=errors, trials=spec.trials,
            ks=ks, include_pass_hat=spec.include_pass_hat)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
