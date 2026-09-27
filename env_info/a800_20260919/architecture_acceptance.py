"""Bounded, explicitly authorized GPU engineering acceptance, not formal scores.

Prepare is CPU-only. execute-a starts the shared three-hour budget and runs four
real training-harness validation trajectories per estimator without updating.
All generated inputs and overrides are retained under the supplied run root.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

import psutil
import requests
import yaml

from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.parquet_builder import build_rows, write_parquet
from tau3_grpo.data.trajectory import trajectory_facts
from tau3_grpo.envs.adapter import airline_policy
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.tracking.rl_continuity import atomic_json
from tau3_grpo.tracking.swanlab import load_tracking_env
from tau3_grpo.training.rl.runner import resolve, snapshot_source, validate_inputs
from tau3_grpo.training.services import launch_process, stop_process


ESTIMATORS = ("grpo", "tau_gigpo", "mt_gtpo")
REMOVED_ENV = ("TAU3_STOP_REQUEST_PATH", "TAU3_BUDGET_STATE_PATH", "TAU3_BUDGET_INTERVAL")


def ensure_port_available(port):
    """Match server socket reuse semantics without accepting a live listener."""
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    load_tracking_env()
    plans = {}
    for estimator in ESTIMATORS:
        result = root / ("a-" + estimator)
        result.mkdir()
        command, env, snapshot = resolve(result, estimator=estimator, updates=20)
        report = validate_inputs(env, result, 20, "v3", estimator)
        entries = read_manifest(Path(env["TRAIN_MANIFEST_DIR"]) / "areal_airline_train_seed42.jsonl")[:2]
        parquet = result / "engineering-two-tasks.parquet"
        write_parquet(build_rows(entries, policy=airline_policy(), split="train",
                                 anchor_mode="structured", seed=42), parquet)
        # Explicit engineering exceptions, appended after all formal defaults.
        command = [part for part in command if not any(
            f"runtime_env.env_vars.{key}=" in part for key in REMOVED_ENV)]
        overrides = ["trainer.total_training_steps=2", "trainer.val_before_train=true",
                     "trainer.val_only=true", "trainer.save_freq=-1", "trainer.test_freq=-1",
                     "trainer.logger=[console]", "actor_rollout_ref.rollout.val_kwargs.n=2",
                     f"data.val_files={parquet}", "data.val_batch_size=2",
                     "++ray_kwargs.ray_init.runtime_env.env_vars.TAU3_SWANLAB_CONTINUITY='0'"]
        overrides += ["++ray_kwargs.ray_init.runtime_env.env_vars.VERL_QWEN35_FULL_WEIGHT_AUDIT='1'"]
        env["VERL_QWEN35_FULL_WEIGHT_AUDIT"] = "1"
        command += overrides
        env.update(TOTAL_UPDATES="2", TRAIN_PARQUET=str(result / "train_schedule.parquet"),
                   TAU3_SWANLAB_CONTINUITY="0", TRAINER_LOGGERS="[console]",
                   SWANLAB_EXPERIMENT_NAME=f"engineering-A-{estimator}")
        for key in REMOVED_ENV:
            env.pop(key, None)
        dry = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN="1"),
                                      cwd=CODE_ROOT, text=True)
        native = shlex.split(dry.splitlines()[-1])
        hydra = subprocess.check_output(native + ["--cfg", "job"],
                                       env=dict(env, CUDA_VISIBLE_DEVICES=""), cwd=CODE_ROOT, text=True)
        config = yaml.safe_load(hydra)
        correction = config["algorithm"]["rollout_correction"]
        assert correction["rollout_is"] is None and correction["rollout_rs"] is None
        assert correction["bypass_mode"] is False
        assert config["trainer"]["val_only"] and config["trainer"]["save_freq"] == -1
        assert config["actor_rollout_ref"]["rollout"]["val_kwargs"]["n"] == 2
        assert config["algorithm"]["adv_estimator"] == estimator
        (result / "resolved-hydra.yaml").write_text(hydra)
        (result / "resolved-command.txt").write_text(dry)
        atomic_json(result / "launch.json", snapshot)
        atomic_json(result / "engineering-overrides.json", {
            "kind": "engineering_not_formal", "phase": "A", "overrides": overrides,
            "removed_formal_environment": list(REMOVED_ENV), "task_ids": [e.task_id for e in entries],
            "trajectories": 4, "parameter_updates": 0, "formal_inputs_preflight": report})
        plans[estimator] = {"command": command, "environment_changes": {
            key: value for key, value in env.items()
            if key not in os.environ or os.environ[key] != value}, "remove_environment": list(REMOVED_ENV)}
    atomic_json(root / "phase-a-plan.json", plans)
    atomic_json(root / "authorization.json", {
        "authorized_date": "2026-09-19", "purpose": "bounded architecture GPU engineering acceptance",
        "wall_seconds": 10800, "gpus": 5, "gpu_hours_upper_bound": 15,
        "max_phase_a_trajectories": 12, "max_training_candidates": 576,
        "max_independent_evaluation_trajectories": 8, "automatic_retry": False,
        "formal_training_authorized": False, "final50_authorized": False})
    snapshot_source(root / "source")
    print(json.dumps({"prepared": str(root), "gpu_started": False}), flush=True)


class Supervisor:
    def __init__(self, root):
        self.root = root
        self.children = []
        self.descendants = {}
        self.process_descendants = {}
        self.deadline = json.loads((root / "budget.json").read_text())["deadline_unix"]

    def observe(self):
        for process in self.children:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    self.descendants[child.pid] = child.create_time()
                    self.process_descendants.setdefault(process.pid, {})[child.pid] = child.create_time()
            except psutil.NoSuchProcess:
                pass
        if time.time() >= self.deadline - 30:
            raise TimeoutError("Configured budget reached; reserve 30 seconds for cleanup")

    def launch(self, command, env, name):
        self.observe()
        process = launch_process(command, cwd=CODE_ROOT, env=env,
                                 log_path=self.root / f"{name}.log",
                                 pid_path=self.root / f"{name}.pid")
        self.children.append(process)
        return process

    def wait(self, process):
        while process.poll() is None:
            self.observe()
            time.sleep(1)
        self.observe()
        if process.returncode:
            raise RuntimeError(f"Owned process {process.pid} exited {process.returncode}")

    def stop(self, process):
        self.observe()
        stop_process(process, timeout=5)
        self._stop_descendants(self.process_descendants.get(process.pid, {}))

    @staticmethod
    def _stop_descendants(descendants):
        remaining = []
        for pid, birth in descendants.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == birth and process.is_running():
                    process.terminate()
                    remaining.append(process)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(remaining, timeout=3)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=3)

    def close(self):
        # Ray workers may detach from the session. Kill only descendants whose
        # birth times were observed while this controller owned their parent.
        for process in reversed(self.children):
            stop_process(process, timeout=2)
        self._stop_descendants(self.descendants)


def audit_interfaces(records, result, trials):
    """Verify live metadata and every active text weight, without inferring seeds."""
    groups = {}
    turns = generated = 0
    for row in records:
        facts = json.loads(row["trajectory_facts_json"])
        identity, tokens = facts["identity"], facts["tokens"]
        uid = identity["sample_group_uid"]
        assert uid and identity["seed"] == 42
        assert identity["seed_semantics"] == "configured_data_seed"
        assert identity["generation_request_seed"] is None
        group = groups.setdefault(uid, {"task": identity["task_id"], "trials": []})
        assert group["task"] == identity["task_id"]
        group["trials"].append(identity["trial"])
        assert facts["capabilities"]["complete_response_logprobs"]
        probs, mask = tokens["response_logprobs"], tokens["response_mask"]
        assert len(probs) == len(mask) == len(tokens["response_ids"])
        for probability, selected in zip(probs, mask, strict=True):
            assert math.isfinite(probability)
            assert probability <= 1e-6 if selected else probability == 0.0
            generated += int(selected)
        eligibility = facts["terminal"]["execution_eligibility"]
        assert eligibility["policy"] == "tau3_execution_eligibility_v1"
        assert eligibility["training_candidate_eligible"]
        for turn in facts["turns"]:
            assert turn["generated_logprob_mode"] == "processed_logprobs"
            assert turn["generated_logprobs_available"]
            assert isinstance(turn["generation_engine_seed"], int)
            assert turn["generation_sampling_parameters"] is not None
            turns += 1
    assert all(sorted(g["trials"]) == list(range(trials)) for g in groups.values())
    reports = [json.loads(p.read_text()) for p in (result / "weight-audits").glob("full-weight-audit-*.json")]
    assert reports and all(r["all_active_parameters_match"] for r in reports)
    assert all(not r["missing_active_parameters"] for r in reports)
    states = {tuple((p["destination"], p["actual_sha256"]) for p in r["parameters"]) for r in reports}
    return {"trajectories": len(records), "actual_groups": len(groups), "trials_per_group": trials,
            "turns": turns, "generated_tokens": generated,
            "logprob_semantics": "actual processed generation distribution; not old/ref logprobs",
            "weight_audits": len(reports), "distinct_text_parameter_states": len(states),
            "active_parameter_names": sorted({r["active_parameter_names"] for r in reports}),
            "all_active_text_parameters_match": True, "vision_parameters_verified": False}


def audit_phase_a(result, estimator):
    rows = [json.loads(line) for line in (result / "validation/0.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    ids, tasks, tools, multi, raw_turns = set(), {}, 0, 0, 0
    for row in rows:
        facts = json.loads(row["trajectory_facts_json"])
        identity, tokens = facts["identity"], facts["tokens"]
        assert facts["schema"] == "tau3_trajectory_facts_v1" and identity["trajectory_id"] not in ids
        ids.add(identity["trajectory_id"])
        task = identity["task_id"]
        tasks[task] = tasks.get(task, 0) + 1
        rebuilt = trajectory_facts(request_id=identity["trajectory_id"], task_id=task,
                                   turns=facts["turns"], response_ids=tokens["response_ids"],
                                   response_mask=tokens["response_mask"],
                                   response_logprobs=tokens["response_logprobs"], terminal=facts["terminal"],
                                   sample_group_uid=identity.get("sample_group_uid"),
                                   trial=identity.get("trial"), seed=identity.get("seed"))
        rebuilt["identity"].update(identity)
        assert rebuilt == facts
        assert facts["turns"] and all(t["finish_reason"] is not None for t in facts["turns"])
        assert facts["terminal"]["initial_db_hash"] and facts["terminal"]["db_hash"]
        assert facts["terminal"]["scored"] == facts["terminal"]["execution_eligibility"]["officially_scored"]
        assert row["score"] == facts["terminal"]["reward"]
        assert ("process_reward_json" in row) == (estimator == "mt_gtpo")
        for turn in facts["turns"]:
            raw_turns += 1
            calls = turn["tool_calls"]
            tools += len(calls)
            multi += int(len(calls) > 1)
            for call in calls:
                assert "raw_arguments" in call and "observation" in call and "error" in call
    assert len(tasks) == 2 and set(tasks.values()) == {2}
    assert not list(result.glob("global_step_*")) and not list((result / "update-batches").glob("*.pkl"))
    return {"trajectories": len(rows), "task_trials": tasks, "raw_finish_turns": raw_turns,
            "tool_calls": tools, "multi_tool_turns": multi, "parameter_updates": 0,
            "multi_tool_live_coverage": bool(multi),
            "interfaces": audit_interfaces(rows, result, trials=2),
            "db_isolation_limit": "separate identities and DB facts; exhaustive isolation covered by CPU harness tests"}


def execute_a(root):
    plans = json.loads((root / "phase-a-plan.json").read_text())
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if active.strip():
        raise RuntimeError("GPU occupied; refusing to start")
    ensure_port_available(8100)
    if (root / "budget.json").exists():
        raise ValueError("Existing attempt: no automatic restart or budget reset")
    started = time.time()
    atomic_json(root / "budget.json", {"started_unix": started, "deadline_unix": started + 10800})
    supervisor = Supervisor(root)
    load_tracking_env()
    simenv = dict(os.environ, TAU3_USER_CUDA_DEVICES="4", TAU3_USER_MAX_NUM_SEQS="16",
                  TAU3_USER_MAX_MODEL_LEN="16384", TAU3_USER_GPU_MEMORY_UTILIZATION=".65",
                  TAU3_USER_ENFORCE_EAGER="1", TAU3_USER_PORT="8100")
    summary = {"status": "running", "phase": "A", "estimators": {}}
    atomic_json(root / "phase-a-status.json", summary)
    try:
        simulator = supervisor.launch(["bash", str(CODE_ROOT / "scripts/serve/simulator_qwen38.sh")],
                                      simenv, "a-simulator")
        startup_deadline = min(started + 1200, supervisor.deadline - 30)
        while True:
            supervisor.observe()
            if simulator.poll() is not None:
                raise RuntimeError("Simulator exited during startup")
            try:
                if requests.get("http://127.0.0.1:8100/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() > startup_deadline:
                raise TimeoutError("Simulator startup exceeded 20 minutes")
            time.sleep(1)
        for estimator, plan in plans.items():
            env = dict(os.environ, **plan["environment_changes"])
            for key in plan["remove_environment"]:
                env.pop(key, None)
            summary["active_estimator"] = estimator
            atomic_json(root / "phase-a-status.json", summary)
            train = supervisor.launch(plan["command"], env, "a-" + estimator)
            supervisor.wait(train)
            supervisor.stop(train)
            summary["estimators"][estimator] = audit_phase_a(root / ("a-" + estimator), estimator)
            atomic_json(root / "phase-a-status.json", summary)
        summary["status"] = "passed"
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        supervisor.close()
        summary["finished_unix"] = time.time()
        atomic_json(root / "phase-a-status.json", summary)
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute-a"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"Acceptance interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    (prepare if args.mode == "prepare" else execute_a)(args.root.resolve())


if __name__ == "__main__":
    main()
