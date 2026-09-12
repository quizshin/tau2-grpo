"""Wait for SFT and export to shared storage; RL requires explicit --run-rl.

Run remotely under run_guarded.py. Each child owns a process group; cleanup
stops only these jobs and preserves all checkpoints and logs on failure.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


ROOT = Path(os.environ.get("TAU3_ROOT", "/root/shared-nvme/tau3"))
CODE = ROOT / "code"
BOOT = ROOT / "bootstrap"
ADAPTER = ROOT / "artifacts/sft4b_lora_5090/adapter"
MERGED = ROOT / "artifacts/sft4b_lora_5090/sft_merged_seed42"
PYTHON = sys.executable
STATUS_NAME = "sft-export-status.json"


def status(stage, **details):
    value = {"stage": stage, "time": time.time(), **details}
    target = BOOT / STATUS_NAME
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(value), flush=True)


def stop(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(stage, command, seconds):
    status(stage)
    with (BOOT / f"{stage}.log").open("w") as output:
        process = subprocess.Popen(command, cwd=CODE, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=seconds)
        finally:
            stop(process)
    (BOOT / f"{stage}.exit").write_text(str(result) + "\n")
    if result:
        raise RuntimeError(f"{stage} exited {result}; see its log")


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-rl", action="store_true",
                        help="Explicitly opt into simulator and RL after export")
    args = parser.parse_args()
    global STATUS_NAME
    if args.run_rl:
        STATUS_NAME = "sft-rl-status.json"
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    simulator = None
    simulator_output = None
    try:
        status("waiting_for_sft")
        deadline = time.monotonic() + 10860
        sft_exit = BOOT / "sft-train.exit"
        while not sft_exit.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("SFT completion record did not arrive")
            time.sleep(5)
        if int(sft_exit.read_text().strip()) != 0:
            raise RuntimeError("SFT failed; export and RL were not started")
        summary = json.loads((ADAPTER / "train_summary.json").read_text())
        if summary["actual_optimizer_steps"] != 30 or summary["effective_batch_size"] != 8:
            raise RuntimeError("SFT did not preserve the frozen optimizer schedule")
        if not math.isfinite(summary["validation_metrics"]["eval_loss"]):
            raise RuntimeError("SFT validation loss is not finite")
        if MERGED.exists():
            raise FileExistsError(f"Refusing to overwrite an existing export: {MERGED}")

        if args.run_rl:
            simulator_output = (BOOT / "simulator-sft-validation.log").open("w")
            simulator = subprocess.Popen(
                [PYTHON, "-m", "tau3_grpo.launch", "simulator", "--config",
                 "configs/simulator/qwen38_27b_paratera_5090.yaml"], cwd=CODE,
                stdout=simulator_output, stderr=subprocess.STDOUT, start_new_session=True)
            (BOOT / "simulator-sft-validation.pid").write_text(str(simulator.pid) + "\n")
        run("sft-merge", [PYTHON, "-m", "tau3_grpo.training.sft.merge",
            "--base", str(ROOT / "models/Qwen3.5-4B"), "--adapter", str(ADAPTER),
            "--output", str(MERGED)], 1200)
        (MERGED / "sft_provenance.json").write_text(json.dumps({
            "adapter": str(ADAPTER), "summary": summary,
            "rl_requested": args.run_rl}, indent=2) + "\n")

        if not args.run_rl:
            status("complete", sft_summary=str(ADAPTER / "train_summary.json"),
                   merged_model=str(MERGED), rl_started=False)
            return 0

        status("waiting_for_simulator", simulator_pid=simulator.pid)
        deadline = time.monotonic() + 600
        while True:
            if simulator.poll() is not None:
                raise RuntimeError("Simulator exited during initialization")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=3) as response:
                    models = json.load(response)
                if any(item["id"] == "Qwen/Qwen3.8-27B-AWQ-INT4" for item in models["data"]):
                    break
            except (OSError, ValueError, KeyError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Simulator readiness check timed out")
            time.sleep(3)
        run("rl-sft-validation", [PYTHON, "-m", "tau3_grpo.launch", "rl", "--config",
            "configs/train/rl/qwen35_4b_lora_paratera_sft_validation.yaml",
            "--experiment", "e0"], 10800)
        status("complete", sft_summary=str(ADAPTER / "train_summary.json"),
               rl_log=str(BOOT / "rl-sft-validation.log"))
        return 0
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop(simulator)
        if simulator_output is not None:
            simulator_output.close()


if __name__ == "__main__":
    raise SystemExit(main())
