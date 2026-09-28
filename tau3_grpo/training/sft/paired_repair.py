"""Bounded, same-data SFT from two explicitly authorized initialization models.

Uses the existing trainer/exporter and owned process lifecycle. Refuses to reuse
run outputs; does not run evaluation services, retry training or alter data.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import time
from pathlib import Path

from tau3_grpo.training.services import launch_process, stop_process
from tau3_grpo.training.sft.continuation import preserve_base_tokenizer, validate_training


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run(planpath):
    plan = read(planpath)
    root = Path(plan["root"])
    code = Path(plan["code_root"])
    assert socket.gethostname() == plan["hostname"]
    assert plan["authorized_epochs"] == 1 and len(plan["arms"]) == 2
    for file, sha in plan["frozen_files"].items():
        if digest(file) != sha:
            raise ValueError("Frozen file changed: " + file)
    for arm in plan["arms"]:
        path = root / arm["name"]
        if any((path / x).exists() for x in ["train.log", "adapter", "train.exit"]):
            raise ValueError("Prior attempt exists")
        path.mkdir(parents=True, exist_ok=True)
    children = []
    state = dict(status="starting", pid=os.getpid(), plan_sha256=digest(planpath), arms={})

    def status(value):
        state.update(status=value, updated_at=time.time())
        dump(root / "state.json", state)

    def signal_handler(sig, frame):
        raise InterruptedError(f"Controller received signal {sig}")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    with (root / ".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        dump(
            root / "started.json",
            dict(
                started_at=started,
                pid=os.getpid(),
                deadline=started + plan["train_timeout_seconds"],
            ),
        )
        try:
            env = dict(os.environ, **plan["environment"])
            running = {}
            for arm in plan["arms"]:
                out = root / arm["name"]
                aenv = dict(
                    env,
                    SFT_GPU=str(arm["gpu"]),
                    SFT_CONFIG=arm["config"],
                    SFT_MODEL_NAME_OR_PATH=arm["model"],
                )
                proc = launch_process(
                    arm["train_argv"],
                    cwd=code,
                    env=aenv,
                    log_path=out / "train.log",
                    pid_path=out / "train.pid",
                )
                children.append(proc)
                running[arm["name"]] = proc
                state["arms"][arm["name"]] = dict(
                    status="training", pid=proc.pid, gpu=arm["gpu"], initialization=arm["model"]
                )
            status("training")
            deadline = time.monotonic() + plan["train_timeout_seconds"] - 45
            while running:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Paired training deadline; no automatic retry")
                for name, proc in list(running.items()):
                    rc = proc.poll()
                    if rc is None:
                        continue
                    (root / name / "train.exit").write_text(str(rc) + "\n")
                    if rc:
                        raise RuntimeError(f"{name} training exited {rc}")
                    summary = validate_training(
                        root / name,
                        read(plan["data_audit"]),
                        expected_steps=plan["expected_steps"],
                        expected_epochs=1,
                    )
                    if summary.get("observed_training_dialogues") != plan["train_size"]:
                        raise ValueError("Wrong number of consumed dialogues")
                    state["arms"][name].update(
                        status="trained",
                        steps=summary["actual_optimizer_steps"],
                        train_loss=summary["train_loss"],
                        validation=summary["validation_metrics"],
                        baseline_validation=summary["baseline_validation"],
                    )
                    del running[name]
                    status("training" if running else "trained")
                if running:
                    time.sleep(3)
            for proc in children:
                stop_process(proc)
            state["gpu_phase_seconds"] = time.time() - started
            state["allocated_gpu_hours_upper"] = 2 * state["gpu_phase_seconds"] / 3600
            status("exporting_on_cpu")
            for arm in plan["arms"]:
                out = root / arm["name"]
                menv = dict(env, CUDA_VISIBLE_DEVICES="")
                proc = launch_process(
                    arm["merge_argv"],
                    cwd=code,
                    env=menv,
                    log_path=out / "merge.log",
                    pid_path=out / "merge.pid",
                )
                children.append(proc)
                deadline = time.monotonic() + plan["export_timeout_seconds"]
                while proc.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("CPU export deadline")
                    time.sleep(3)
                (out / "merge.exit").write_text(str(proc.returncode) + "\n")
                if proc.returncode:
                    raise RuntimeError("CPU export failed: " + arm["name"])
                preserve_base_tokenizer(
                    Path(arm["model"]), out / "merged", out / "provenance/export_tokenizer"
                )
                files = list((out / "merged").glob("*.safetensors"))
                if not files or not (out / "merged/config.json").exists():
                    raise ValueError("Missing merged weights")
                dump(
                    out / "export_receipt.json",
                    {
                        "initialization": arm["model"],
                        "files": {
                            p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files
                        },
                        "tokenizer_sha256": digest(out / "merged/tokenizer.json"),
                        "gpu_inference_verified": False,
                    },
                )
                state["arms"][arm["name"]]["status"] = "trained_exported"
                status("exporting_on_cpu")
            status("trained_exported_evaluation_pending")
            (root / "controller.exit").write_text("0\n")
        except BaseException as exc:
            state["error"] = str(exc)
            status("failed")
            (root / "controller.exit").write_text("1\n")
            raise
        finally:
            for proc in children:
                stop_process(proc)
            state["owned_processes_stopped"] = True
            status(state["status"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    run(p.parse_args().plan)
