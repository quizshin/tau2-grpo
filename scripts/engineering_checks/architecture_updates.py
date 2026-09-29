"""Phases B/C of the authorized engineering run; shares phase A's deadline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

import requests
import yaml

from architecture_acceptance import REMOVED_ENV, Supervisor, ensure_port_available
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.tracking.rl_continuity import atomic_json
from tau3_grpo.tracking.swanlab import load_tracking_env
from tau3_grpo.training.rl.checkpoints import validate_checkpoint
from tau3_grpo.training.rl.runner import resolve, snapshot_source, validate_inputs


ARMS = (("grpo", False), ("tau_gigpo", False), ("mt_gtpo", False), ("mt_gtpo", True))


def prepare(root, arms=ARMS):
    if (root / "phase-b-plan.json").exists():
        raise ValueError("Existing phase B plan; refusing to overwrite")
    load_tracking_env()
    plans = {}
    for estimator, filtering in arms:
        label = f"b-{estimator}-df{int(filtering)}"
        result = root / label
        result.mkdir()
        command, env, snapshot = resolve(result, updates=20, estimator=estimator,
                                          dynamic_filter=filtering, reward_version="v3")
        report = validate_inputs(env, result, 20, "v3", estimator)
        command = [part for part in command if not any(
            f"runtime_env.env_vars.{key}=" in part for key in REMOVED_ENV)]
        overrides = ["trainer.total_training_steps=2", "trainer.val_before_train=false",
                     "trainer.val_only=false", "trainer.save_freq=1", "trainer.test_freq=-1",
                     "trainer.logger=[console,swanlab]",
                     f"++ray_kwargs.ray_init.runtime_env.env_vars.TAU3_GRPO_SIGNAL_AUDIT_DIR={result / 'signal-audit'}"]
        overrides += ["++ray_kwargs.ray_init.runtime_env.env_vars.VERL_QWEN35_FULL_WEIGHT_AUDIT='1'"]
        env["VERL_QWEN35_FULL_WEIGHT_AUDIT"] = "1"
        command += overrides
        env.update(TOTAL_UPDATES="2", TRAIN_PARQUET=str(result / "train_schedule.parquet"),
                   TRAINER_LOGGERS="[console,swanlab]", SWANLAB_MODE="online",
                   TAU3_GRPO_SIGNAL_AUDIT_DIR=str(result / "signal-audit"),
                   SWANLAB_EXPERIMENT_NAME=f"engineering-{label}-2steps")
        for key in REMOVED_ENV:
            env.pop(key, None)
        dry = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN="1"), cwd=CODE_ROOT, text=True)
        native = shlex.split(dry.splitlines()[-1])
        hydra = subprocess.check_output(native + ["--cfg", "job"],
                                       env=dict(env, CUDA_VISIBLE_DEVICES=""), cwd=CODE_ROOT, text=True)
        config = yaml.safe_load(hydra)
        correction = config["algorithm"]["rollout_correction"]
        assert correction["rollout_is"] is None and correction["rollout_rs"] is None
        assert correction["bypass_mode"] is False
        assert config["trainer"]["total_training_steps"] == 2
        assert config["trainer"]["save_freq"] == 1 and config["trainer"]["test_freq"] == -1
        assert config["algorithm"]["dynamic_filter"]["enable"] == filtering
        assert config["actor_rollout_ref"]["actor"]["optim"]["lr_scheduler_type"] == "constant"
        (result / "resolved-hydra.yaml").write_text(hydra)
        (result / "resolved-command.txt").write_text(dry)
        atomic_json(result / "launch.json", snapshot)
        atomic_json(result / "engineering-overrides.json", {
            "phase": "B", "kind": "engineering_not_formal", "overrides": overrides,
            "removed_formal_environment": list(REMOVED_ENV), "max_candidates": 128,
            "task_schedule": "first two batches of the unchanged formal20 schedule",
            "formal_inputs_preflight": report})
        plans[label] = {"command": command, "estimator": estimator, "filtering": filtering,
                        "environment_changes": {k: v for k, v in env.items()
                                                if k not in os.environ or os.environ[k] != v}}
    atomic_json(root / "phase-b-plan.json", plans)
    snapshot_source(root / "source-bc")
    print(json.dumps({"prepared": list(plans), "gpu_started": False}), flush=True)


def single_arm_plan(root):
    """Fail before service startup if the separately approved scope has drifted."""
    plans = json.loads((root / "phase-b-plan.json").read_text())
    if set(plans) != {"b-tau_gigpo-df0"}:
        raise ValueError("Single-arm authorization covers only GiGPO DF off")
    label, plan = next(iter(plans.items()))
    if plan["estimator"] != "tau_gigpo" or plan["filtering"] is not False:
        raise ValueError("Single-arm algorithm mismatch")
    config = yaml.safe_load((root / label / "resolved-hydra.yaml").read_text())
    if (config["algorithm"]["adv_estimator"] != "tau_gigpo"
            or config["algorithm"]["dynamic_filter"]["enable"] is not False
            or config["trainer"]["total_training_steps"] != 2
            or config["trainer"]["save_freq"] != 1
            or config["trainer"]["test_freq"] != -1
            or config["data"]["train_batch_size"] != 8
            or config["actor_rollout_ref"]["rollout"]["n"] != 8):
        raise ValueError("Single-arm resolved protocol mismatch")
    return label, plan


def check_boundary_once(rows, checked, remaining_seconds):
    """A completed-step decision must not be re-applied inside the next step."""
    if not rows or rows[-1]["step"] in checked:
        return False
    step = rows[-1]["step"]
    checked.add(step)
    return step == 1 and remaining_seconds < 1263


def prepare_single(root, prior):
    """Prepare this approved 90-minute GiGPO-only attempt; no GPU operations."""
    prior = prior.resolve()
    phase_a = prior / "phase-a-status.json"
    if json.loads(phase_a.read_text())["status"] != "passed":
        raise ValueError("Prior phase A did not pass")
    # Changes to the actual harness/algorithm require new compatibility review.
    old_hashes = json.loads((prior / "source-bc/source-sha256.json").read_text())
    prefixes = ("tau3_grpo/algorithms/", "tau3_grpo/integrations/", "tau3_grpo/envs/",
                "tau3_grpo/evaluation/", "tau3_grpo/data/", "verl/verl/", "tau2-bench/src/")
    checked = {name: digest for name, digest in old_hashes.items() if name.startswith(prefixes)}
    changed = [name for name, digest in checked.items()
               if not (CODE_ROOT / name).is_file()
               or hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError(f"Prior A runtime source changed: {changed}")
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "authorization.json", {
        "authorized_date": "2026-09-19", "purpose": "GiGPO two-update engineering acceptance",
        "estimator": "tau_gigpo", "dynamic_filter": False, "max_training_candidates": 128,
        "wall_seconds": 5400, "gpus": 5, "gpu_hours_upper_bound": 7.5,
        "automatic_retry": False, "resume_authorized": False, "evaluation_authorized": False,
        "prior_phase_a": str(phase_a),
        "prior_phase_a_sha256": hashlib.sha256(phase_a.read_bytes()).hexdigest(),
        "runtime_source_files_unchanged": len(checked),
        "launcher_cleanup_evidence": "results/maintenance/architecture-cleanup-1789815395/command-comparison.json"})
    prepare(root, arms=(("tau_gigpo", False),))
    single_arm_plan(root)


def execute_single(root):
    """Execute one arm, then release GPUs before the CPU/cloud evidence audit."""
    label, plan = single_arm_plan(root)
    authorization = json.loads((root / "authorization.json").read_text())
    if authorization["wall_seconds"] != 5400 or authorization["max_training_candidates"] != 128:
        raise ValueError("Unexpected authorization budget")
    if (root / "budget.json").exists() or (root / "phase-b-status.json").exists():
        raise ValueError("Existing single-arm attempt: no automatic retry or budget reset")
    if shutil.disk_usage(root).free < 110 * 2**30:
        raise RuntimeError("Less than 110 GiB free; preserve old checkpoints")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if active.strip():
        raise RuntimeError("GPU occupied; refusing to start")
    ensure_port_available(8100)
    load_tracking_env()
    started = time.time()
    atomic_json(root / "budget.json", {"started_unix": started, "deadline_unix": started + 5400})
    supervisor = Supervisor(root)
    status = {"status": "running", "phase": "B", "active_arm": label, "started_unix": started}
    atomic_json(root / "phase-b-status.json", status)
    # Also interrupt blocking service calls; allow one minute to stop owned processes.
    previous_alarm = signal.signal(signal.SIGALRM, lambda signum, frame: (
        _raise_budget_timeout()))
    signal.setitimer(signal.ITIMER_REAL, 5340)
    try:
        simenv = dict(os.environ, TAU3_USER_CUDA_DEVICES="4", TAU3_USER_PORT="8100",
                      TAU3_USER_MAX_NUM_SEQS="16", TAU3_USER_MAX_MODEL_LEN="16384",
                      TAU3_USER_GPU_MEMORY_UTILIZATION=".65", TAU3_USER_ENFORCE_EAGER="1")
        simulator = supervisor.launch(["bash", str(CODE_ROOT / "scripts/serve/simulator_qwen38.sh")],
                                      simenv, "b-simulator")
        startup_deadline = started + 1200
        while True:
            supervisor.observe()
            if simulator.poll() is not None:
                raise RuntimeError("Simulator startup failed")
            try:
                if requests.get("http://127.0.0.1:8100/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() >= startup_deadline:
                raise TimeoutError("Simulator startup exceeded 20 minutes")
            time.sleep(1)
        env = dict(os.environ, **plan["environment_changes"])
        for key in REMOVED_ENV:
            env.pop(key, None)
        process = supervisor.launch(plan["command"], env, label)
        status["trainer_started_unix"] = time.time()
        atomic_json(root / "phase-b-status.json", status)
        result = root / label
        checked_boundaries = set()
        while process.poll() is None:
            supervisor.observe()
            # Metrics are published after a complete save boundary. Do not begin
            # another long step when even the prior measured fast step cannot fit.
            metrics_path = result / "metrics.jsonl"
            if metrics_path.exists():
                lines = metrics_path.read_text().splitlines()
                try:
                    rows = [json.loads(line) for line in lines if line.strip()]
                except json.JSONDecodeError:
                    rows = []  # A writer may be appending the final JSONL record.
                if check_boundary_once(rows, checked_boundaries, supervisor.deadline - time.time()):
                    validate_checkpoint(result, 1)
                    status.update(status="partial_budget_boundary_stop", checkpoint_step=1,
                                  reason="less than observed fastest GRPO step + cleanup remains")
                    break
            time.sleep(1)
        else:
            if process.returncode:
                raise RuntimeError(f"Owned trainer exited {process.returncode}")
            status["status"] = "training_completed_audit_pending"
    except BaseException as exc:
        status.update(status="budget_timeout" if isinstance(exc, TimeoutError) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        supervisor.close()
        status["gpu_finished_unix"] = time.time()
        status["elapsed_seconds"] = status["gpu_finished_unix"] - started
        status["five_card_reservation_gpu_hours_upper_bound"] = status["elapsed_seconds"] * 5 / 3600
        status["gpu_compute_processes_after_cleanup"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).splitlines()
        atomic_json(root / "phase-b-status.json", status)
    if status["status"] == "training_completed_audit_pending":
        try:
            status["result"] = audit_update(root / label, "tau_gigpo", 2)
            status["status"] = "passed_with_mapping_scope_limit"
        except BaseException as exc:
            status.update(status="audit_failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "phase-b-status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "passed_with_mapping_scope_limit" else 1


def _raise_budget_timeout():
    raise TimeoutError("90-minute budget: stopping with one minute reserved for cleanup")


def audit_update(result, estimator, expected_step):
    import torch
    from verl import DataProto

    validate_checkpoint(result, expected_step)
    rows = [json.loads(line) for line in (result / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, expected_step + 1))
    steps = []
    for row in rows:
        metrics = row["metrics"]
        for key, value in metrics.items():
            if key.startswith("actor/") and isinstance(value, (int, float)):
                assert math.isfinite(value), f"Nonfinite {key} at step {row['step']}"
        rollout = result / f"rollouts/{row['step']}.jsonl"
        records = [json.loads(line) for line in rollout.read_text().splitlines()]
        assert len(records) == 64
        assert all("trajectory_facts_json" in record for record in records)
        skipped = bool(metrics.get("actor/skipped_no_effective_tokens", 0))
        packet = result / f"update-batches/update_{row['step']:06d}.pkl"
        nonzero = None
        if not skipped:
            data = DataProto.load_from_disk(str(packet))
            assert data.meta_info["tau3_estimator_diagnostics"]["estimator"] == estimator
            assert torch.isfinite(data.batch["advantages"]).all()
            nonzero = int(torch.count_nonzero(data.batch["advantages"] * data.batch["response_mask"]))
        steps.append({"step": row["step"], "candidates": 64, "skipped_actor": skipped,
                      "effective_nonzero_advantage_tokens": nonzero,
                      "grad_norm": metrics.get("actor/grad_norm")})
    audits = [json.loads(p.read_text()) for p in (result / "weight-audits").glob("worker-audit-*.json")]
    assert audits and all(a["all_conv_match"] for a in audits)
    conv_states = {tuple(sorted((c["key"], c["sha256"]) for c in a["conv"])) for a in audits}
    informative = any(s["effective_nonzero_advantage_tokens"] and s["grad_norm"]
                      and s["grad_norm"] > 0 for s in steps)
    changed = len(conv_states) > 1
    if informative:
        assert changed, "Informative batch but no observed convolution parameter change"
    import swanlab
    tracking = json.loads((result / "swanlab-run.json").read_text())
    values = swanlab.Api().run(tracking["run_path"]).metrics(keys=["trainer/global_step"], all=True)
    points = [(int(p["step"]), int(p["value"])) for r in values.get("list", []) for p in r.get("metrics", [])]
    assert sorted(points) == [(s, s) for s in range(1, expected_step + 1)], points
    return {"steps": steps, "checkpoint_step": expected_step, "cloud_steps": points,
            "swanlab_run_id": tracking["run_id"], "swanlab_url": tracking["url"],
            "informative_update_observed": informative and changed,
            "weight_transfer": {"scope": "all GDN conv1d tensors at TP1; not all model parameters",
                                "audits": len(audits), "exact_matches": True,
                                "distinct_parameter_states": len(conv_states)},
            "full_parameter_mapping_verified": False}


def execute_bc(root):
    if json.loads((root / "phase-a-status.json").read_text())["status"] != "passed":
        raise ValueError("Phase A has not passed")
    if (root / "phase-bc-status.json").exists():
        raise ValueError("Existing B/C attempt: no automatic retry")
    load_tracking_env()
    supervisor = Supervisor(root)
    plans = json.loads((root / "phase-b-plan.json").read_text())
    status = {"status": "running", "phase": "B", "arms": {}}
    atomic_json(root / "phase-bc-status.json", status)
    try:
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if active.strip():
            raise RuntimeError("Unexpected GPU process before B")
        simenv = dict(os.environ, TAU3_USER_CUDA_DEVICES="4", TAU3_USER_PORT="8100",
                      TAU3_USER_MAX_NUM_SEQS="16", TAU3_USER_MAX_MODEL_LEN="16384",
                      TAU3_USER_GPU_MEMORY_UTILIZATION=".65", TAU3_USER_ENFORCE_EAGER="1")
        simulator = supervisor.launch(["bash", str(CODE_ROOT / "scripts/serve/simulator_qwen38.sh")],
                                      simenv, "bc-simulator")
        startup_deadline = min(time.time() + 1200, supervisor.deadline - 30)
        while True:
            supervisor.observe()
            if simulator.poll() is not None:
                raise RuntimeError("B/C simulator startup failed")
            try:
                if requests.get("http://127.0.0.1:8100/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() >= startup_deadline:
                raise TimeoutError("B/C simulator startup timeout")
            time.sleep(1)
        for label, plan in plans.items():
            if shutil.disk_usage(root).free < 110 * 2**30:
                raise RuntimeError("Less than 110 GiB free for complete-checkpoint rotation; preserve all existing results")
            status["active_arm"] = label
            atomic_json(root / "phase-bc-status.json", status)
            env = dict(os.environ, **plan["environment_changes"])
            for key in REMOVED_ENV:
                env.pop(key, None)
            process = supervisor.launch(plan["command"], env, label)
            supervisor.wait(process)
            supervisor.stop(process)
            supervisor.observe()
            result = audit_update(root / label, plan["estimator"], 2)
            status["arms"][label] = result
            atomic_json(root / "phase-bc-status.json", status)
        chosen = next((label for label, result in status["arms"].items()
                       if result["informative_update_observed"]), None)
        if chosen is None:
            raise RuntimeError("No informative completed update available; C not validated, no extra sampling")
        if shutil.disk_usage(root).free < 60 * 2**30:
            raise RuntimeError("Insufficient space for resumed checkpoint rotation")
        plan, result = plans[chosen], root / chosen
        checkpoint = validate_checkpoint(result, 2)
        original_run = status["arms"][chosen]["swanlab_run_id"]
        env = dict(os.environ, **plan["environment_changes"])
        for key in REMOVED_ENV:
            env.pop(key, None)
        command = plan["command"] + ["trainer.total_training_steps=3", "trainer.resume_mode=resume_path",
                                     f"trainer.resume_from_path={checkpoint}"]
        dry = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN="1"), cwd=CODE_ROOT, text=True)
        (result / "resume-command.txt").write_text(dry)
        status.update(phase="C", resume_arm=chosen)
        atomic_json(root / "phase-bc-status.json", status)
        process = supervisor.launch(command, env, "c-resume")
        supervisor.wait(process)
        supervisor.stop(process)
        resumed = audit_update(result, plan["estimator"], 3)
        assert resumed["swanlab_run_id"] == original_run
        tracking = json.loads((result / "swanlab-run.json").read_text())
        assert tracking["sessions"][-1]["restored_step"] == 2
        assert tracking["sessions"][-1]["first_update"] == 3
        status.update(status="passed_with_mapping_scope_limit", continuation=resumed)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        supervisor.close()
        status["finished_unix"] = time.time()
        atomic_json(root / "phase-bc-status.json", status)
        print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute-bc", "prepare-single", "execute-single"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"Acceptance interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if args.mode == "prepare-single":
        if args.prior_root is None:
            parser.error("prepare-single requires --prior-root")
        prepare_single(args.root.resolve(), args.prior_root)
    elif args.mode == "execute-single":
        raise SystemExit(execute_single(args.root.resolve()))
    else:
        (prepare if args.mode == "prepare" else execute_bc)(args.root.resolve())


if __name__ == "__main__":
    main()
