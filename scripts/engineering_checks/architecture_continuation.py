"""Run one approved engineering update/resume stage without repeating phase B.

The original launch plan is retained. Each attempt owns a new bounded supervisor,
while resume keeps the training directory, checkpoint and SwanLab identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import requests
from architecture_acceptance import REMOVED_ENV, Supervisor, audit_interfaces, ensure_port_available
from architecture_updates import audit_update

from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.tracking.rl_continuity import atomic_json
from tau3_grpo.tracking.swanlab import load_tracking_env
from tau3_grpo.training.rl.checkpoints import validate_checkpoint


def resume_command(plan, checkpoint, step):
    if step != 2:
        raise ValueError("This engineering continuation resumes only completed step 2")
    return plan["command"] + ["trainer.total_training_steps=3",
                              "trainer.resume_mode=resume_path",
                              f"trainer.resume_from_path={checkpoint}"]


def validate_before_resume(result):
    checkpoint = validate_checkpoint(result, 2)
    rows = [json.loads(line) for line in (result / "metrics.jsonl").read_text().splitlines()]
    if [row["step"] for row in rows] != [1, 2]:
        raise ValueError("Resume requires exactly two previously logged updates")
    tracking = json.loads((result / "swanlab-run.json").read_text())
    if tracking["last_logged_step"] != 2:
        raise ValueError("Tracking is not at the saved boundary")
    return checkpoint, tracking


def restore_log_evidence(text):
    # Every rank must actually enter all four checkpoint load branches.
    missing = []
    for kind, prefix in (("model", "model"), ("optimizer", "optim"),
                         ("rng", "extra_state"), ("lr_scheduler", "extra_state")):
        for rank in range(4):
            if not any(f"Loaded {kind} from " in line and
                       f"{prefix}_world_size_4_rank_{rank}.pt" in line
                       for line in text.splitlines()):
                missing.append(f"{kind}/rank{rank}")
    if "Warning: No dataloader state found" in text:
        missing.append("dataloader")
    if "Setting global step to 2" not in text:
        missing.append("global_step")
    if missing:
        raise ValueError(f"Missing actual restore evidence: {missing}")
    return {"model_optimizer_rng_scheduler_ranks": 4, "restored_step": 2,
            "dataloader_missing_warning": False,
            "bitwise_equivalence_to_uninterrupted_training": False}


def audit_resume_schedule(root, result):
    """Check the saved data pointer and the tasks actually consumed next."""
    from collections import Counter

    import pyarrow.parquet as pq
    import torch
    import yaml

    before = torch.load(root / "prior-data.pt", weights_only=False)
    after = torch.load(result / "global_step_3/data.pt", weights_only=False)
    for state, step in ((before, 2), (after, 3)):
        if (state["_num_yielded"] != step or state["_sampler_iter_yielded"] != step
                or state["_sampler_iter_state"]["samples_yielded"] != step * 8):
            raise ValueError("Restored dataloader did not advance exactly one eight-task batch")
    config = yaml.safe_load((result / "resolved-hydra.yaml").read_text())
    if config["data"]["shuffle"] is not False:
        raise ValueError("Schedule audit requires the recorded non-shuffled protocol")
    schedule = pq.read_table(result / "train_schedule.parquet").to_pylist()[16:24]
    expected = Counter(row["extra_info"]["task_id"] for row in schedule)
    expected = Counter({task: count * 8 for task, count in expected.items()})
    rows = [json.loads(line) for line in (result / "rollouts/3.jsonl").read_text().splitlines()]
    actual = Counter(json.loads(row["trajectory_facts_json"])["identity"]["task_id"] for row in rows)
    if actual != expected or len(rows) != 64:
        raise ValueError("Resumed trajectories differ from the next eight scheduled tasks")
    return {"before_batches": 2, "after_batches": 3, "before_tasks": 16, "after_tasks": 24,
            "actual_task_trials": dict(actual), "next_schedule_batch_matches": True,
            "bitwise_equivalence_to_uninterrupted_training": False}


def execute(root, source, label, resume, wall_seconds):
    if wall_seconds not in (4500, 5400):
        raise ValueError("Engineering stage must use its declared 75/90 minute cap")
    if root.exists():
        raise ValueError("Existing attempt: do not reset its budget or repeat updates")
    plan = json.loads((source / "phase-b-plan.json").read_text())[label]
    result = source / label
    checkpoint, tracking = validate_before_resume(result) if resume else (None, None)
    if not resume and ((result / "metrics.jsonl").exists() or list(result.glob("global_step_*"))):
        raise ValueError("Update stage already contains training results")
    if plan["estimator"] not in ("grpo", "tau_gigpo", "mt_gtpo"):
        raise ValueError("Unknown estimator")
    command = resume_command(plan, checkpoint, 2) if resume else plan["command"]
    import shutil
    if shutil.disk_usage(source).free < (60 if resume else 110) * 2**30:
        raise RuntimeError("Insufficient space to preserve checkpoint through rotation")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU is occupied")
    ensure_port_available(8100)
    load_tracking_env()
    env = dict(os.environ, **plan["environment_changes"], RAY_DEDUP_LOGS="0")
    for key in REMOVED_ENV:
        env.pop(key, None)
    env.pop("TAU3_DRY_RUN", None)
    root.mkdir(parents=True)
    atomic_json(root / "authorization.json", {
        "user_authorization": "2026-09-19: recovery and remaining-work GPU experiments approved",
        "phase": "C" if resume else "B", "estimator": plan["estimator"],
        "dynamic_filter": plan["filtering"], "source": str(source),
        "wall_seconds": wall_seconds, "gpus": 5,
        "max_candidates": 64 if resume else 128, "automatic_retry": False})
    atomic_json(root / "launch-plan.json", {"command": command, "result": str(result)})
    if resume:
        atomic_json(root / "prior-swanlab.json", tracking)
        atomic_json(root / "prior-checkpoint-complete.json",
                    json.loads((checkpoint / "checkpoint-complete.json").read_text()))
        for name in ("data.pt", "checkpoint-complete.json"):
            shutil.copy2(checkpoint / name, root / ("prior-" + name))
    from tau3_grpo.training.rl.runner import snapshot_source

    snapshot_source(root / "source-execute")
    script = Path(__file__)
    shutil.copy2(script, root / script.name)
    atomic_json(root / "controller-source.json", {
        "sha256": hashlib.sha256(script.read_bytes()).hexdigest()})
    started = time.time()
    atomic_json(root / "budget.json", {"started_unix": started,
                                       "deadline_unix": started + wall_seconds})
    supervisor = Supervisor(root)
    status = {"status": "running", "phase": "C" if resume else "B",
              "result": str(result), "started_unix": started}
    atomic_json(root / "status.json", status)

    def interrupt(signum, frame):
        raise TimeoutError(f"Stage interrupted by signal {signum}; preserve evidence")

    signal.signal(signal.SIGALRM, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    signal.setitimer(signal.ITIMER_REAL, wall_seconds - 60)
    try:
        simenv = dict(os.environ, TAU3_USER_CUDA_DEVICES="4", TAU3_USER_PORT="8100",
                      TAU3_USER_MAX_NUM_SEQS="16", TAU3_USER_MAX_MODEL_LEN="16384",
                      TAU3_USER_GPU_MEMORY_UTILIZATION=".65", TAU3_USER_ENFORCE_EAGER="1")
        simulator = supervisor.launch(["bash", str(CODE_ROOT / "scripts/serve/simulator_qwen38.sh")],
                                      simenv, "simulator")
        while True:
            supervisor.observe()
            if simulator.poll() is not None:
                raise RuntimeError("Simulator exited during startup")
            try:
                if requests.get("http://127.0.0.1:8100/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() - started > 1200:
                raise TimeoutError("Simulator startup exceeded 20 minutes")
            time.sleep(1)
        process = supervisor.launch(command, env, "trainer")
        supervisor.wait(process)
        status["status"] = "training_completed_audit_pending"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        supervisor.close()
        status["gpu_finished_unix"] = time.time()
        status["reserved_gpu_hours"] = (time.time() - started) * 5 / 3600
        status["gpu_processes_after_cleanup"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True).splitlines()
        atomic_json(root / "status.json", status)
    if status["status"] == "training_completed_audit_pending":
        try:
            status["audit"] = audit_update(result, plan["estimator"], 3 if resume else 2)
            if resume:
                status["restore"] = restore_log_evidence((root / "trainer.log").read_text())
                status["data_schedule"] = audit_resume_schedule(root, result)
                current = json.loads((result / "swanlab-run.json").read_text())
                assert current["run_id"] == tracking["run_id"]
                assert current["sessions"][-1]["restored_step"] == 2
                assert current["sessions"][-1]["first_update"] == 3
            if not resume:
                records = [json.loads(line) for step in (1, 2)
                           for line in (result / f"rollouts/{step}.jsonl").read_text().splitlines()]
                status["interfaces"] = audit_interfaces(records, result, trials=8)
                if plan["estimator"] == "mt_gtpo":
                    from tau3_grpo.analysis.calibrate_process_rewards import analyze_update

                    status["process_advantage_filter_replay"] = [
                        {"step": step, "replayed_trajectories": len(analyze_update([
                            json.loads(line) for line in (result / f"rollouts/{step}.jsonl").read_text().splitlines()
                        ]))} for step in (1, 2)]
                status["status"] = "passed"
            else:
                status["status"] = "passed_with_mapping_scope_limit"
        except BaseException as exc:
            status.update(status="audit_failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] in ("passed", "passed_with_mapping_scope_limit") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.root.resolve(), args.source.resolve(), args.label,
                             args.resume, args.wall_seconds))
