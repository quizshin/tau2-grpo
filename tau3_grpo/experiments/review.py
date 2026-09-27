"""Generate review drafts from explicit receipts without editing human indexes.

Receipts are reported evidence, not a substitute for rechecking checkpoints,
cloud tracking or raw evaluation trials. Unknown formats fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

KINDS = ("gpu_acceptance", "formal_controller", "evaluation")


def read_receipt(kind: str, path: Path) -> dict:
    content = path.read_bytes()
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("receipt must be a JSON object")
    status = data.get("status")
    if not isinstance(status, str) or not re.fullmatch(r"[a-z_]+", status):
        raise ValueError("receipt needs a machine-readable status")
    facts = {}
    if kind == "gpu_acceptance":
        if data.get("schema") != "tau3_architecture_acceptance_summary_v1":
            raise ValueError("unknown GPU acceptance schema")
        keys = ("training_candidates", "parameter_updates", "phase_a_trajectories",
                "independent_evaluation_trajectories", "elapsed_seconds",
                "five_card_reservation_gpu_hours_upper_bound")
        facts = {key: data[key] for key in keys if key in data}
        # Keep raw failure and boundary-stop interpretation visible simultaneously.
        for key in ("raw_controller_status", "phase_c", "phase_d", "limitations",
                    "phase_b_remaining"):
            if key in data:
                facts[key] = data[key]
        attention = status != "passed"
    elif kind == "formal_controller":
        if not {"pid", "target", "session", "updated_at"} <= data.keys():
            raise ValueError("not a formal runner controller-state receipt")
        facts = {key: data[key] for key in ("target", "completed_step", "cloud_steps_verified",
                                           "paused") if key in data}
        attention = status != "completed"
    elif kind == "evaluation":
        keys = ("planned_trajectories", "completed_trajectories", "failed_trajectories",
                "missing_trajectories")
        if type(data.get("metrics_valid")) is not bool or any(
            type(data.get(key)) is not int or data[key] < 0 for key in keys
        ):
            raise ValueError("evaluation receipt needs validity and nonnegative trial counts")
        planned, completed, failed, missing = (data[key] for key in keys)
        if planned == 0 or completed + failed + missing != planned:
            raise ValueError("inconsistent evaluation denominator")
        valid = data["metrics_valid"]
        if valid != (completed == planned) or status != ("complete" if valid else "incomplete"):
            raise ValueError("inconsistent evaluation completion status")
        if not valid and data.get("metrics") is not None:
            raise ValueError("incomplete evaluation cannot publish aggregate metrics")
        metrics = data.get("metrics")
        if valid and (not isinstance(metrics, dict) or not metrics or any(
            not re.fullmatch(r"pass[@^][1-9][0-9]*", key)
            or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for key, value in metrics.items()
        )):
            raise ValueError("invalid evaluation metrics")
        facts = {key: data[key] for key in (*keys, "metrics_valid", "metrics")}
        attention = not valid
    else:
        raise ValueError(f"unsupported receipt kind: {kind}")
    # No arbitrary environment/config/log dump: credentials and freeform errors are omitted.
    return {"kind": kind, "path": str(path.resolve()),
            "sha256": hashlib.sha256(content).hexdigest(), "reported_status": status,
            "attention_required": attention, "reported_facts": facts}


def render(experiment_id: str, run_id: str, receipts: list[dict]) -> tuple[str, str]:
    heading = f"# 待审阅草稿：{experiment_id} / {run_id}\n\n"
    caveat = ("本草稿只转录指定机器回执；未重新验证 GPU、checkpoint、云端记录或原始 trial。"
              "人工确认后将结论归入 EXPERIMENTS.md / ERRORS.md，勿把本文件作为第二份总索引。\n\n")
    body = []
    issues = []
    for receipt in receipts:
        facts = json.dumps(receipt["reported_facts"], ensure_ascii=False, indent=2, allow_nan=False)
        # Make freeform limitations inert inside the Markdown code block.
        fence = "`" * max(3, 1 + max((len(x) for x in re.findall(r"`+", facts)), default=0))
        section = (f"## {receipt['kind']}：{receipt['reported_status']}\n\n"
                   f"来源：{receipt['path']}\n\nSHA256：{receipt['sha256']}\n\n"
                   f"{fence}json\n{facts}\n{fence}\n\n")
        body.append(section)
        if receipt["attention_required"]:
            issues.append(section)
    tail = ("## 人工结论（待填写）\n\n"
            "- 实际完成范围、未完成项及原始证据：\n"
            "- 原因与影响：区分预算、基础设施、算法、数值和记录问题。\n"
            "- 后续验证与预算：\n"
            "- 增幅结论：需要匹配协议的独立比较；本草稿不推断算法提升。\n")
    errors = "".join(issues) or "指定回执未标出需关注状态；这不证明不存在其他错误。\n\n"
    return heading + caveat + "".join(body) + tail, heading + caveat + errors + tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", required=True, metavar="KIND=PATH")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for value in (args.experiment_id, args.run_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            parser.error("experiment/run IDs must use letters, digits, dots, underscores or hyphens")
    receipts = []
    for spec in args.receipt:
        kind, separator, path = spec.partition("=")
        if not separator or kind not in KINDS:
            parser.error(f"receipt must be KIND=PATH; kinds: {KINDS}")
        receipts.append(read_receipt(kind, Path(path)))
    experiment, errors = render(args.experiment_id, args.run_id, receipts)
    record = json.dumps({"schema": "tau3_review_draft_v1", "experiment_id": args.experiment_id,
                         "run_id": args.run_id, "receipts": receipts},
                        ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Re-runs require a new directory so human annotations and older drafts survive.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in (("experiment.draft.md", experiment), ("errors.draft.md", errors),
                          ("sources.json", record)):
        (args.output_dir / name).write_text(content, encoding="utf-8")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
