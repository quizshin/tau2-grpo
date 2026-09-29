"""Export one engineering checkpoint and run the public independent evaluator."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from architecture_acceptance import Supervisor, ensure_port_available

from tau3_grpo.evaluation.controller import check_summary, file_hashes, sha, validate_hf
from tau3_grpo.evaluation.service_attestation import write_service_attestation
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.tracking.rl_continuity import atomic_json
from tau3_grpo.training.rl.checkpoints import validate_checkpoint
from tau3_grpo.training.rl.runner import snapshot_source


def prepare(root, run, step):
    if root.exists():
        raise ValueError("Existing evaluation attempt requires inspection")
    cp = validate_checkpoint(run, step)
    root.mkdir(parents=True)
    original = CODE_ROOT / "data/manifests/areal_airline_selection_seed42.jsonl"
    entries = [json.loads(line) for line in original.read_text().splitlines() if line.strip()]
    if len(entries) != 60 or any(row["split"] != "selection" for row in entries):
        raise ValueError("Expected registered selection60")
    entries = entries[:2]
    (root / "manifests").mkdir()
    (root / "manifests" / original.name).write_text("".join(json.dumps(row) + "\n" for row in entries))
    atomic_json(root / "plan.json", {"source": str(cp), "checkpoint_step": step,
                                    "task_ids": [row["task_id"] for row in entries],
                                    "trials": 4, "max_trajectories": 8,
                                    "source_manifest_sha256": sha(original),
                                    "scope": "engineering; no ranking or effect claim"})
    snapshot_source(root / "source")
    staging = root / "model.merging"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with (root / "merge.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", str(cp / "actor"), "--target_dir", str(staging)],
                       cwd=CODE_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
    validate_hf(staging)
    verification = json.loads((staging / "export-verification.json").read_text())
    if not verification["native_format"] or verification["verified_tensors"] <= 0:
        raise ValueError("Export lacks exact source-tensor verification")
    atomic_json(root / "export.json", {
        "source": str(cp), "source_model_sha256": {
            p.name: sha(p) for p in (cp / "actor").glob("model_world_size_4_rank_*.pt")},
        "verification": verification, "files_sha256": file_hashes(staging)})
    staging.rename(root / "model")
    print(json.dumps({"prepared": str(root), "gpu_started": False}), flush=True)


def execute(root):
    if (root / "budget.json").exists():
        raise ValueError("Existing evaluation attempt: no automatic retry")
    plan = json.loads((root / "plan.json").read_text())
    if len(plan["task_ids"]) != 2 or plan["trials"] != 4:
        raise ValueError("Engineering evaluation requires exactly 2 x 4 trials")
    checkpoint = root / "model"
    validate_hf(checkpoint)
    if file_hashes(checkpoint) != json.loads((root / "export.json").read_text())["files_sha256"]:
        raise ValueError("Export changed since verification")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               text=True).strip():
        raise RuntimeError("GPU occupied before independent evaluation")
    for port in (8100, 8200):
        ensure_port_available(port)
    atomic_json(root / "authorization.json", {
        "user_authorization": "2026-09-19: all engineering and interface GPU acceptance approved",
        "gpus": [0, 4], "wall_seconds": 5400, "max_trajectories": 8,
        "automatic_retry": False, "official_final": False})
    started = time.time()
    atomic_json(root / "budget.json", {"started_unix": started, "deadline_unix": started + 5400})
    supervisor = Supervisor(root)
    status = {"status": "running", "phase": "D", "started_unix": started}
    atomic_json(root / "status.json", status)

    def interrupted(signum, frame):
        raise TimeoutError(f"Evaluation interrupted by signal {signum}")

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 5340)

    def healthy(process, port):
        deadline = min(time.time() + 1200, supervisor.deadline - 60)
        while time.time() < deadline:
            supervisor.observe()
            if process.poll() is not None:
                raise RuntimeError(f"Service at {port} exited")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(f"Service at {port} startup deadline")

    try:
        simenv = dict(os.environ, TAU3_USER_CUDA_DEVICES="4", TAU3_USER_PORT="8100",
                      TAU3_USER_MAX_NUM_SEQS="16", TAU3_USER_MAX_MODEL_LEN="16384",
                      TAU3_USER_GPU_MEMORY_UTILIZATION=".65", TAU3_USER_ENFORCE_EAGER="1")
        simulator = supervisor.launch(["bash", str(CODE_ROOT / "scripts/serve/simulator_qwen38.sh")],
                                      simenv, "simulator")
        healthy(simulator, 8100)
        name = "tau3-engineering-restored"
        args = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(checkpoint),
                "--served-model-name", name, "--host", "127.0.0.1", "--port", "8200",
                "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "24576",
                "--gpu-memory-utilization", ".80", "--max-num-seqs", "8", "--enforce-eager",
                "--generation-config", "vllm", "--seed", "42", "--language-model-only",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                "--reasoning-parser", "qwen3", "--default-chat-template-kwargs", '{"enable_thinking":false}',
                "--enable-prefix-caching"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", VLLM_WORKER_MULTIPROC_METHOD="spawn",
                   VLLM_CACHE_ROOT=str(root / "cache"), PYTHONUNBUFFERED="1")
        atomic_json(root / "policy-command.json", args)
        policy = supervisor.launch(args, env, "policy")
        healthy(policy, 8200)
        write_service_attestation(checkpoint_path=str(checkpoint), served_model_name=name,
                                  base_url="http://127.0.0.1:8200/v1", pid=policy.pid,
                                  output=root / "attestation.json")
        command = [sys.executable, "-m", "tau3_grpo.evaluation.run", "--target", "selection",
                   "--checkpoint", str(checkpoint), "--manifest-dir", str(root / "manifests"),
                   "--results-dir", str(root), "--policy-model", name,
                   "--policy-base-url", "http://127.0.0.1:8200/v1",
                   "--policy-attestation", str(root / "attestation.json"),
                   "--user-model", os.environ.get("TAU3_USER_SERVED_MODEL_NAME", "Qwen/Qwen3.8-27B-AWQ-INT4"),
                   "--user-base-url", "http://127.0.0.1:8100/v1", "--seed", "42", "--data-seed", "42",
                   "--trials", "4", "--ks", "1", "2", "4", "--include-pass-hat",
                   "--max-concurrency", "4", "--max-steps", "30", "--policy-temperature", ".4",
                   "--user-temperature", "1.0", "--output-dir", str(root / "evaluation")]
        atomic_json(root / "evaluation-command.json", command)
        evaluation = supervisor.launch(command, dict(os.environ, CUDA_VISIBLE_DEVICES=""), "evaluation")
        while evaluation.poll() is None:
            supervisor.observe()
            if simulator.poll() is not None or policy.poll() is not None:
                raise RuntimeError("Evaluation dependency exited")
            time.sleep(1)
        if evaluation.returncode:
            raise RuntimeError(f"Independent evaluation exited {evaluation.returncode}")
        status["summary"] = check_summary(root / "evaluation", plan["task_ids"], trials=4)
        status["status"] = "passed"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        supervisor.close()
        status["finished_unix"] = time.time()
        status["two_card_reservation_gpu_hours"] = (time.time() - started) * 2 / 3600
        status["gpu_processes_after_cleanup"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).splitlines()
        atomic_json(root / "status.json", status)
    print(json.dumps(status), flush=True)
    return int(status["status"] != "passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--step", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.run is None:
            parser.error("prepare requires --run")
        prepare(args.root.resolve(), args.run.resolve(), args.step)
    else:
        raise SystemExit(execute(args.root.resolve()))
