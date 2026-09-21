"""Strict, CPU-only comparison of saved independent evaluations.

No model calls, no intersection-only scoring and no merging of retry artifacts.
Protocol equality is limited to recorded evidence; legacy missing source/data
hashes are disclosed, never synthesized from today's checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tau3_grpo.evaluation.artifacts import EvaluationArtifacts, read_evaluation
from tau3_grpo.evaluation.diagnostics import trajectory_diagnostics
from tau3_grpo.evaluation.metrics import paired_bootstrap
from tau3_grpo.evaluation.scoring import summarize_trials

PROTOCOL_FIELDS = (
    "benchmark_revision", "spec.target", "spec.trials", "spec.seed", "spec.max_steps",
    "spec.max_errors", "spec.max_concurrency", "endpoints.policy.temperature",
    "endpoints.user.model", "endpoints.user.temperature", "provenance.tool_protocol",
    "provenance.agent_system_prompt_sha256", "provenance.benchmark_policy_modified",
)
EXTENDED_FIELDS = (
    "provenance.task_manifest_sha256", "provenance.task_db_sha256",
    "provenance.evaluator_source_sha256", "provenance.harness_source_sha256",
)
IDENTITY_FIELDS = ("provenance.checkpoint_hash", "provenance.policy_service_attestation_hash")


def _field(metadata, name):
    value = metadata
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _plan(artifact):
    return sorted((j["task_id"], j["trial"], j["seed"]) for j in artifact.metadata["planned"])


def _score(artifact: EvaluationArtifacts, ks):
    return summarize_trials(
        planned=artifact.metadata["planned"], results=artifact.trajectories,
        errors=artifact.errors, trials=artifact.metadata["spec"]["trials"],
        ks=ks, include_pass_hat=True,
    )


def compare_evaluations(baseline: Path, treatment: Path, *, ks=None, resamples=10000,
                        confidence=.95, seed=42) -> dict:
    if not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    base, treat = read_evaluation(baseline), read_evaluation(treatment)
    if ks is None:
        trials = min(base.metadata["spec"]["trials"], treat.metadata["spec"]["trials"])
        ks = [k for k in (1, 2, 4) if k <= trials]
    scores = [_score(a, ks) for a in (base, treat)]
    mismatches = []
    missing = []
    for name in (*PROTOCOL_FIELDS, *EXTENDED_FIELDS):
        left, right = _field(base.metadata, name), _field(treat.metadata, name)
        if left is None or right is None:
            if name in PROTOCOL_FIELDS:
                mismatches.append({"field": name, "reason": "required protocol evidence missing"})
            missing.append(name)
        elif left != right:
            mismatches.append({"field": name, "baseline": left, "treatment": right})
    # Missing versions in old runs remain unknown. Two historical runs retain
    # the old comparison rules, but cannot silently compare to a versioned run.
    name = "provenance.harness_protocol"
    left, right = _field(base.metadata, name), _field(treat.metadata, name)
    if left is None and right is None:
        missing.append(name)
    elif left != right:
        mismatches.append({"field": name, "baseline": left, "treatment": right})
    from tau3_grpo.evaluation.harness import TOKENS_V4

    if any(_field(a.metadata, "provenance.harness_protocol.version") == TOKENS_V4 for a in (base, treat)):
        name = "provenance.tokenizer_files_sha256"
        left, right = _field(base.metadata, name), _field(treat.metadata, name)
        if not left or not right or left != right:
            mismatches.append({"field": name, "reason": "token protocol requires identical tokenizer evidence"})
    for name in IDENTITY_FIELDS:
        if any(_field(a.metadata, name) is None for a in (base, treat)):
            missing.append(name)
    if _plan(base) != _plan(treat):
        mismatches.append({"field": "planned", "reason": "task/trial/seed sets differ"})
    complete = all(score["metrics_valid"] for score in scores)
    comparable = not mismatches and complete
    report = {
        "schema_version": 1, "comparable": comparable,
        "status": "incompatible_protocol" if mismatches else "complete" if complete else "incomplete_evaluation",
        "protocol": {
            "mismatches": mismatches, "missing_extended_or_identity_evidence": missing,
            "assurance": "recorded_protocol_only" if missing else "recorded_protocol_and_hashes",
            "note": "hash presence is recorded provenance, not independent re-attestation of historical services",
        },
        "runs": {}, "differences": None,
        "inference_scope": "paired task uncertainty within these fixed runs, not variation across training seeds; CIs are unadjusted for multiple comparisons",
    }
    for label, artifact, score in zip(("baseline", "treatment"), (base, treat), scores):
        report["runs"][label] = {
            "directory": str(artifact.directory), "artifact_sha256": artifact.files,
            "policy_model": artifact.metadata["endpoints"]["policy"]["model"],
            "provenance": artifact.metadata.get("provenance", {}),
            "score": score, "diagnostics": trajectory_diagnostics(artifact.trajectories),
        }
    if comparable:
        tasks = sorted(scores[0]["per_task"])
        differences = {}
        for metric in scores[0]["metrics"]:
            paired = paired_bootstrap(
                [scores[0]["per_task"][t]["metrics"][metric] for t in tasks],
                [scores[1]["per_task"][t]["metrics"][metric] for t in tasks],
                resamples=resamples, confidence=confidence, seed=seed,
            )
            result = paired.to_dict()
            result.update(
                difference_pp=100 * paired.difference,
                ci_pp=[100 * paired.ci_low, 100 * paired.ci_high],
                relative_change_percent=(100 * paired.difference / paired.baseline_mean
                                         if paired.baseline_mean else None),
            )
            differences[metric] = result
        report["differences"] = differences
    return report


def _fmt(value, digits=2):
    return "NA" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict) -> str:
    lines = ["# 独立评测离线比较", "", f"状态：`{report['status']}`；可比较：{report['comparable']}。",
             f"证据范围：`{report['protocol']['assurance']}`。", ""]
    for label, run in report["runs"].items():
        score = run["score"]
        lines.append(f"- {label}：`{run['directory']}`；{score['completed_trajectories']}/"
                     f"{score['planned_trajectories']} 已评分，{score['failed_trajectories']} 异常，"
                     f"{score['missing_trajectories']} 缺失。")
    lines += ["", "增幅是 treatment − baseline；CI 按任务成对 bootstrap。训练 seed 不确定性未覆盖，多个指标的区间未做多重比较校正。", ""]
    if report["differences"] is not None:
        lines += ["|指标|基线 %|实验 %|增幅 pp|相对变化 %|CI pp*|", "|---|---:|---:|---:|---:|---|"]
        for name, value in report["differences"].items():
            lines.append(f"|{name}|{_fmt(value['baseline_mean'] * 100)}|"
                         f"{_fmt(value['treatment_mean'] * 100)}|{_fmt(value['difference_pp'])}|"
                         f"{_fmt(value['relative_change_percent'])}|"
                         f"[{_fmt(value['ci_pp'][0])}, {_fmt(value['ci_pp'][1])}]|")
        confidence = next(iter(report["differences"].values()))["confidence"]
        lines += ["", f"*实际置信水平：{confidence:.1%}；基线为 0 时相对变化记 NA。"]
    else:
        lines.append("不输出增幅或排名；不能删除失败任务、仅取交集或拼入补跑结果后冒充原评测。")
    if report["protocol"]["mismatches"]:
        lines += ["", "协议差异：", ""]
        lines += [f"- `{m['field']}`：{json.dumps(m, ensure_ascii=False)}"
                  for m in report["protocol"]["mismatches"]]
    if report["protocol"]["missing_extended_or_identity_evidence"]:
        lines += ["", "缺少的历史证据（未从当前代码补造）：", ""]
        lines += [f"- `{name}`" for name in report["protocol"]["missing_extended_or_identity_evidence"]]
    lines += ["", "## 执行诊断", "", "|指标|baseline|treatment|", "|---|---:|---:|"]
    diagnostics = [report["runs"][key]["diagnostics"] for key in ("baseline", "treatment")]
    for label, key in (("平均 assistant 消息数", "assistant_messages_per_trajectory"),
                       ("平均 assistant 工具调用数", "assistant_tool_calls_per_trajectory"),
                       ("平均轨迹耗时 s", "simulation_duration_seconds")):
        values = [f"{_fmt(d[key]['mean_observed'])} ({d[key]['observed']}/{d[key]['expected']})"
                  for d in diagnostics]
        lines.append(f"|{label}（字段覆盖）|{'|'.join(values)}|")
    values = [f"{_fmt(None if d['assistant_tools']['error_rate_observed'] is None else 100 * d['assistant_tools']['error_rate_observed'])}%"
              f" ({d['assistant_tools']['responses_with_error_flag']}/{d['assistant_tools']['requested']})"
              for d in diagnostics]
    lines.append(f"|工具错误率（已知 error flag / 请求数）|{'|'.join(values)}|")
    for role in ("assistant", "user"):
        for kind in ("prompt_tokens", "completion_tokens"):
            values = []
            for d in diagnostics:
                value = d["usage_by_message"][role][kind]
                values.append(f"{_fmt(value['sum_observed'], 0)} ({value['observed']}/{value['expected']})")
            lines.append(f"|{role} {kind} 已记录总量（消息覆盖）|{'|'.join(values)}|")
    lines += ["", "消息数包含开场模板；token 仅汇总 usage，缺失不当作 0。轨迹耗时总和不是运行墙钟时间或 GPU 小时。", "",
              "终止原因：", ""]
    lines += [f"- {label}：`{json.dumps(d['termination_reasons'], ensure_ascii=False)}`"
              for label, d in zip(("baseline", "treatment"), diagnostics)]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", type=int, nargs="+")
    args = parser.parse_args(argv)
    try:
        report = compare_evaluations(args.baseline, args.treatment, ks=args.ks,
                                     resamples=args.resamples, confidence=args.confidence, seed=args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (args.output_dir / "comparison.md").write_text(render_markdown(report))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "comparable": report["comparable"],
                      "output_dir": str(args.output_dir)}, indent=2))
    return 0 if report["comparable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
