"""Run an explicitly authorized standalone selection queue under one deadline.

Unlike the post-RL controller, this never waits for or exports training arms.
Uses the common owned-process lifecycle and original evaluation CLI.
Version3 supports separately retained, evidence-approved simulator-drift retries.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from tau3_grpo.training.services import launch_process, stop_process


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def validate_plan(plan):
    if plan.get("version") not in (
        "bounded_dual_selection_v1",
        "bounded_repair_selection_v2",
        "bounded_repair_selection_v3",
    ):
        raise ValueError("Unknown plan")
    if plan.get("trials") != 1 or plan.get("seed") != 42 or plan.get("task_count") != 60:
        raise ValueError("Only full selection60 one-trial smoke is authorized")
    if not 0 < plan.get("wall_seconds", 0) <= 7200 or plan.get("gpus") != [0, 1]:
        raise ValueError("Authorization is limited to two GPUs and two hours")
    if (
        plan.get("prior_wall_seconds", 0) < 0
        or plan.get("prior_wall_seconds", 0) + plan["wall_seconds"] > 7200
    ):
        raise ValueError("Recovery must stay within the original combined GPU budget")
    expected = (
        ["base", "sft1", "sft3"]
        if plan["version"] == "bounded_dual_selection_v1"
        else ["base", "sft1", "repair_base", "repair_sft1"]
    )
    if [a["name"] for a in plan["arms"]] != expected:
        raise ValueError("Unexpected model queue")
    if plan["version"] in (
        "bounded_repair_selection_v2",
        "bounded_repair_selection_v3",
    ) and not plan.get("simulator_probe"):
        raise ValueError("Repair evaluation requires an independent simulator calibration gate")
    if plan["version"] == "bounded_repair_selection_v3":
        from tau3_grpo.evaluation.simulator_retry import validate_retry_plan

        validate_retry_plan(plan)
    elif plan.get("probe_failure_policy", "stop") != "stop":
        raise ValueError("Diagnostic probe policy requires explicit v3 authorization")
    if plan.get("training") is not False or plan.get("formal_evaluation") is not False:
        raise ValueError("This controller does not train or run formal evaluation")
    if not plan.get("source_sha256") or not plan.get("authorization_sha256"):
        raise ValueError("Missing frozen authorization or sources")


class BoundedEvaluation:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.plan = json.loads(self.path.read_text())
        validate_plan(self.plan)
        self.root = self.path.parent
        self.code = Path(self.plan["code_root"])
        self.children = []
        self.state = {
            "status": "preflight",
            "arms": {},
            "controller_pid": os.getpid(),
            "plan_sha256": digest(self.path),
        }
        self.env = dict(os.environ, **self.plan["environment"])
        self.deadline = None

    def status(self, phase, **fields):
        self.state.update(status=phase, updated_at=time.time(), **fields)
        save(self.root / "state.json", self.state)
        print(json.dumps({"status": phase, **fields}), flush=True)

    def launch(self, name, command, environment):
        p = launch_process(
            command,
            cwd=self.code,
            env=dict(self.env, **environment),
            log_path=self.root / (name + ".log"),
            pid_path=self.root / (name + ".pid"),
        )
        self.children.append(p)
        return p

    def guard(self, dependencies=()):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Approved combined two-hour GPU workload budget reached")
        if (self.root / "STOP").exists():
            raise InterruptedError("Explicit STOP requested")
        if any(p.poll() is not None for p in dependencies):
            raise RuntimeError("Owned service exited")

    def wait_ready(self, services):
        pending = dict(services)
        while pending:
            self.guard([v[0] for v in services.values()])
            for name, (p, port, model) in list(pending.items()):
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/v1/models", timeout=2
                    ) as r:
                        names = [m["id"] for m in json.load(r)["data"]]
                    if names != [model]:
                        raise ValueError("Wrong service model identity")
                    pending.pop(name)
                except (OSError, TimeoutError):
                    pass
            time.sleep(2)

    def preflight(self):
        if self.plan.get("hard_deadline_unix", float("inf")) - time.time() < 120:
            raise ValueError("Original queue deadline leaves insufficient time to start")
        if socket.gethostname() != self.plan["hostname"]:
            raise ValueError("Wrong host")
        if digest(self.root / "authorization.json") != self.plan["authorization_sha256"]:
            raise ValueError("Authorization changed")
        for name, expected in self.plan["source_sha256"].items():
            if digest(self.code / name) != expected:
                raise ValueError("Frozen source changed: " + name)
        for name in ("bundle", "manifest"):
            if digest(self.plan[name]) != self.plan[name + "_sha256"]:
                raise ValueError("Frozen " + name + " changed")
        for name, expected in self.plan.get("recovery_inputs_sha256", {}).items():
            if digest(name) != expected:
                raise ValueError("Frozen recovery receipt changed")
        from tau3_grpo.data.manifest import read_manifest
        from tau3_grpo.evaluation.outcome_contract import DUAL_VERSION, load_bundle

        bundle = load_bundle(self.plan["bundle"], read_manifest(Path(self.plan["manifest"])))
        if bundle["version"] != DUAL_VERSION or bundle.get("approval_scope") != "smoke_only":
            raise ValueError("Requires frozen dual smoke bundle")
        from tau3_grpo.tracking.judge_budget import Budget

        budget = Budget(self.code / bundle["budget_directory"] / "budget.json", 100, max_calls=5000)
        if budget.accounted >= 100:
            raise ValueError("Existing API budget exhausted")
        for port in (8000, 8100):
            with socket.socket() as s:
                s.bind(("127.0.0.1", port))
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(gpu) != 2 or not all("A800" in x for x in gpu):
            raise ValueError("Expected exactly two A800 GPUs")
        busy = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        )
        if any(x.strip().isdigit() for x in busy.splitlines()):
            raise ValueError("GPU already occupied; no process killed")
        for arm in self.plan["arms"]:
            p = Path(arm["checkpoint"])
            if not (p / "config.json").is_file() or not list(p.glob("*.safetensors")):
                raise ValueError("Missing checkpoint")
        self.status(
            "cpu_preflight_passed", gpu_inventory=gpu, budget_before=budget.billing_summary()
        )

    def reviewable_errors(self, out):
        if not self.plan.get("defer_communication_review", False):
            return False
        path = Path(out) / "errors.jsonl"
        rows = (
            [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            if path.exists()
            else []
        )
        return bool(rows) and all(
            (r.get("execution_evidence") or {}).get("stage") == "communication_scoring"
            and (r.get("execution_evidence") or {}).get("simulation")
            for r in rows
        )

    def evaluate(self, arm, user):
        self.guard([user])
        name = arm["name"]
        for source, expected in self.plan["source_sha256"].items():
            if digest(self.code / source) != expected:
                raise ValueError("Frozen source changed between arms: " + source)
        for path, expected in self.plan.get("recovery_inputs_sha256", {}).items():
            if digest(path) != expected:
                raise ValueError("Frozen recovery receipt changed between arms")
        self.state["arms"][name] = {"status": "starting", "started_at": time.time()}
        self.status("starting_policy", current_arm=name)
        policy = self.launch(name + "_policy", arm["policy"]["argv"], arm["policy"]["environment"])
        try:
            self.wait_ready(
                {
                    "policy": (policy, 8000, "Qwen/Qwen3.5-4B"),
                    "user": (user, 8100, "Qwen/Qwen3.8-27B-AWQ-INT4"),
                }
            )
            self.state["arms"][name]["status"] = "evaluating"
            self.status("evaluating", current_arm=name)
            proc = self.launch(
                name + "_evaluation", arm["evaluation"]["argv"], {"CUDA_VISIBLE_DEVICES": ""}
            )
            out = Path(arm["output"])
            last_seen = (0, 0)
            last_progress = time.monotonic()
            while proc.poll() is None:
                self.guard([policy, user])
                counts = tuple(
                    sum(1 for line in (out / f).open() if line.strip()) if (out / f).exists() else 0
                    for f in ("trajectories.jsonl", "errors.jsonl")
                )
                if counts != last_seen:
                    last_seen = counts
                    last_progress = time.monotonic()
                    self.state["arms"][name].update(scored=counts[0], unresolved=counts[1])
                    self.status("evaluating", current_arm=name)
                if counts[1] >= 3 and not self.reviewable_errors(out):
                    raise RuntimeError(
                        "Three unresolved trials: stop expansion and preserve evidence"
                    )
                if time.monotonic() - last_progress > 1200:
                    raise TimeoutError("No completed attempts for twenty minutes")
                time.sleep(5)
            if (out / "run.json").exists():
                from tau3_grpo.evaluation.dual_metrics import write_report

                dual = write_report(out, self.root / name / "analysis")
                self.state["arms"][name].update(
                    status="complete" if dual["metrics_valid"] else "unresolved",
                    ended_at=time.time(),
                    dual_summary=str(self.root / name / "analysis/summary.json"),
                )
            else:
                raise RuntimeError("Evaluation exited without run record")
            if proc.returncode or not dual["metrics_valid"]:
                if self.reviewable_errors(out) and (out / "summary.json").exists():
                    self.state["arms"][name]["status"] = "awaiting_communication_review"
                    self.state["scoring_review_pending"] = True
                else:
                    raise RuntimeError(
                        "Evaluation has unresolved trials; no expansion to later arms"
                    )
            self.status("arm_complete", current_arm=name)
        finally:
            # Stop evaluation before services if an error interrupted the wait.
            if "proc" in locals():
                stop_process(proc, timeout=5)
            stop_process(policy, timeout=5)

    def run(self):
        with (self.root / ".controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.root / "started.json").exists():
                raise ValueError("Prior GPU attempt; no automatic replay")
            self.preflight()

            def interrupted(signum, frame):
                raise InterruptedError("Controller interrupted")

            signal.signal(signal.SIGTERM, interrupted)
            signal.signal(signal.SIGINT, interrupted)
            start = time.time()
            hard_deadline = min(
                start + self.plan["wall_seconds"], self.plan.get("hard_deadline_unix", float("inf"))
            )
            self.deadline = time.monotonic() + hard_deadline - start - 60
            save(self.root / "started.json", {"started_at": start, "hard_deadline": hard_deadline})
            code = 1
            try:
                self.status("starting_user", deadline_unix=hard_deadline)
                u = self.plan["user"]
                user = self.launch("user", u["argv"], u["environment"])
                if self.plan.get("simulator_probe"):
                    self.wait_ready({"user": (user, 8100, "Qwen/Qwen3.8-27B-AWQ-INT4")})
                    probe = self.plan["simulator_probe"]
                    self.status("calibrating_user")
                    proc = self.launch(
                        "simulator_probe", probe["argv"], {"CUDA_VISIBLE_DEVICES": ""}
                    )
                    while proc.poll() is None:
                        self.guard([user])
                        time.sleep(2)
                    diagnostic = (
                        self.plan.get("probe_failure_policy") == "record_and_review_each_trajectory"
                    )
                    if proc.returncode and not diagnostic:
                        raise RuntimeError("Simulator calibration failed; no selection sampling")
                    receipt = json.loads(Path(probe["receipt"]).read_text())
                    if diagnostic:
                        if (
                            proc.returncode not in (0, 1)
                            or receipt.get("cases") != 12
                            or receipt.get("replies") != 24
                        ):
                            raise RuntimeError(
                                "Simulator probe infrastructure/incomplete output failure"
                            )
                        self.state["probe_diagnostic_only"] = True
                        self.state["probe_passed_replies"] = receipt.get("passed_replies")
                    elif receipt.get("passed") is not True:
                        raise RuntimeError("Simulator calibration gate not passed")
                    self.state["simulator_probe_sha256"] = digest(probe["receipt"])
                    if probe.get("semantic_review"):
                        self.status("awaiting_probe_semantic_review")
                        review_path = Path(probe["semantic_review"])
                        review_deadline = time.monotonic() + 600
                        while not review_path.exists():
                            self.guard([user])
                            if time.monotonic() >= review_deadline:
                                raise TimeoutError("Probe semantic review not received")
                            time.sleep(2)
                        review = json.loads(review_path.read_text())
                        if review.get("passed") is not True or review.get(
                            "probe_summary_sha256"
                        ) != digest(probe["receipt"]):
                            raise ValueError(
                                "Probe semantic review did not pass or has stale identity"
                            )
                        self.state["probe_semantic_review_sha256"] = digest(review_path)
                for arm in self.plan["arms"]:
                    self.evaluate(arm, user)
                if self.plan["version"] == "bounded_repair_selection_v3":
                    from tau3_grpo.evaluation.simulator_retry import process_retry_queue

                    process_retry_queue(self, user)
                self.status(
                    "awaiting_communication_review"
                    if self.state.get("scoring_review_pending")
                    else "complete"
                )
                code = 0
            except BaseException as exc:
                self.status("stopped_for_review", error_type=type(exc).__name__, error=str(exc))
            finally:
                for p in reversed(self.children):
                    stop_process(p, timeout=5)
                for arm in self.plan["arms"]:
                    out = Path(arm["output"])
                    analysis = self.root / arm["name"] / "analysis"
                    if (out / "run.json").exists() and not analysis.exists():
                        try:
                            from tau3_grpo.evaluation.dual_metrics import write_report

                            write_report(out, analysis)
                        except Exception as exc:
                            self.state.setdefault("report_errors", []).append(str(exc))
                self.state.update(
                    gpu_workloads_stopped=True,
                    wall_seconds=time.time() - start,
                    allocated_gpu_hours_upper=2 * (time.time() - start) / 3600,
                )
                self.status(self.state["status"])
                (self.root / "controller.exit").write_text(str(code) + "\n")
            return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    c = BoundedEvaluation(a.plan)
    if a.preflight:
        c.preflight()
        return 0
    return c.run()


if __name__ == "__main__":
    raise SystemExit(main())
