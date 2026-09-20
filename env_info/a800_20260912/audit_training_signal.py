"""Audit frozen rollout evidence without reconstructing missing anchor payloads."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tau3_grpo.algorithms.dynamic_filtering import apply_dynamic_filter
from tau3_grpo.algorithms.tau_gigpo import (
    StepRecord, combine_advantages, episode_advantages, step_advantages,
)


def mechanism_checks():
    """Adversarial examples, not measurements of historical signal coverage."""
    steps = [StepRecord(i, 0, "shared", (0, 1)) for i in range(8)]
    adv, stats = step_advantages(steps, [0.] * 8)
    assert stats.usable_step_coverage == 1 and stats.step_advantage_active
    assert not np.any(adv)
    self_steps = [StepRecord(0, j, "self", (j, j + 1)) for j in range(2)]
    self_adv, self_stats = step_advantages(self_steps, [1.])
    assert self_stats.usable_step_coverage == 1
    np.testing.assert_allclose(self_adv, [-.025, .025])
    lengths = list(range(2, 10))
    all_steps = [StepRecord(i, j, "initial" if j == 0 else f"unique-{i}-{j}", (j, j + 1))
                 for i, length in enumerate(lengths) for j in range(length)]
    rewards = [1.] * 8
    uids = ["task"] * 8
    episode = episode_advantages(rewards, uids)
    raw_step, _ = step_advantages(all_steps, rewards)
    mask = np.array([[int(j < n) for j in range(9)] for n in lengths], dtype=float)
    raw = combine_advantages(episode, raw_step, all_steps, num_trajectories=8,
                             response_length=9, response_mask=mask)
    filtered, _ = apply_dynamic_filter(mask, rewards, uids)
    actual = combine_advantages(episode, raw_step, all_steps, num_trajectories=8,
                                response_length=9, response_mask=filtered)
    assert np.count_nonzero(raw) == 8 and not np.any(actual)
    undiscounted, _ = step_advantages(all_steps, rewards, gamma=1.)
    assert not np.any(undiscounted)
    return {"checks_passed": 4,
            "all_zero_coverage": stats.usable_step_coverage,
            "all_zero_nonzero_step_count": int(np.count_nonzero(adv)),
            "single_trajectory_repeated_state_advantages": self_adv.tolist(),
            "synthetic_all_success_nonzero_before_filter": int(np.count_nonzero(raw)),
            "synthetic_all_success_nonzero_after_filter": int(np.count_nonzero(actual)),
            "gamma1_all_success_nonzero": int(np.count_nonzero(undiscounted)),
            "scope": "synthetic inputs to repository functions; not historical coverage"}


def analyze(snapshot):
    result = {"captured_at": snapshot["captured_at"], "arms": {}, "checks": mechanism_checks()}
    for arm, data in snapshot["arms"].items():
        metric_rows = {row["step"]: row["metrics"] for row in data["metrics"]}
        schedule = {row["update_index"] + 1: sorted(row["task_ids"])
                    for row in data["experiment_manifest"]["schedule"]}
        assert len(metric_rows) == len(data["metrics"]) == 20
        assert sorted(u["step"] for u in data["updates"]) == list(range(1, 21))
        total = Counter()
        rows_out, all_success_groups, discrepancies = [], [], []
        failures, terminations = Counter(), Counter()
        unique_sessions = set()
        for update in data["updates"]:
            step = update["step"]
            rows = update["rows"]
            metrics = metric_rows[step]
            groups = defaultdict(list)
            assert len(rows) == 64
            for row in rows:
                assert row["score"] in (0., 1.)
                groups[row["task_id"]].append(row)
                session_id = row["trajectory"]["session_id"]
                assert session_id not in unique_sessions
                unique_sessions.add(session_id)
                assert not (row["has_anchor_ids"] or row["has_anchor_spans"] or row["has_response_mask"])
                failures[row["failure_category"]] += 1
                terminations[row["termination_reason"]] += 1
            assert len(groups) == 8 and all(len(group) == 8 for group in groups.values())
            assert sorted(groups) == schedule[step]
            assert all(len({r["gts"] for r in rs}) == 1 for rs in groups.values())
            counts = Counter()
            episode_values = []
            for task, group in groups.items():
                rewards = np.array([row["score"] for row in group])
                kind = "all_zero" if rewards.max() == 0 else "all_one" if rewards.min() == 1 else "mixed"
                counts[kind] += 1
                if arm != "e3" or kind == "mixed":
                    episode_values.extend((rewards - rewards.mean()).tolist())
                lengths = [row["trajectory"]["assistant_turns"] for row in group]
                if kind == "all_one":
                    all_success_groups.append({"step": step, "task_id": task, "assistant_turns": lengths,
                                               "variable_lengths": len(set(lengths)) > 1})
                if kind != "mixed":
                    counts["degenerate_assistant_turns"] += sum(lengths)
            scores = sum(row["score"] for row in rows)
            assert abs(scores / 64 - metrics["critic/score/mean"]) < 1e-8
            if arm in ("e1", "e3"):
                for key, field in [("all_zero", "all_zero_groups"), ("all_one", "all_one_groups"),
                                   ("mixed", "effective_groups")]:
                    assert counts[key] == metrics["dynamic_filter/" + field], (arm, step, key)
                assert metrics["dynamic_filter/effective_rollouts"] == counts["mixed"] * 8
            turns = sum(row["trajectory"]["assistant_turns"] for row in rows)
            one = {"step": step, "groups": dict(counts), "successes": scores, "session_assistant_turns": turns}
            if arm in ("e2", "e3"):
                assert metrics["gigpo/fnorm"] == 1 and metrics["gigpo/omega"] == 1
                lo, hi = min(episode_values), max(episode_values)
                actual_lo, actual_hi = metrics["critic/advantages/min"], metrics["critic/advantages/max"]
                # Logged extrema are over policy tokens AFTER the DF response mask.
                # An extremum beyond the episode-only envelope certifies some
                # nonzero step contribution; it cannot measure its count or cause.
                certified = actual_lo < lo - 1e-5 or actual_hi > hi + 1e-5
                one.update({"episode_only_range": [lo, hi], "observed_advantage_range": [actual_lo, actual_hi],
                            "nonzero_step_contribution_certified": certified})
                total["updates_with_certified_step_contribution"] += int(certified)
                total["logged_anchor_steps"] += metrics["gigpo/steps"]
                total["logged_repeated_anchor_steps"] += metrics["gigpo/steps_in_usable_groups"]
                total["logged_anchor_groups"] += metrics["gigpo/anchor_groups"]
                total["logged_repeated_anchor_groups"] += metrics["gigpo/usable_anchor_groups"]
                if turns != metrics["gigpo/steps"]:
                    discrepancies.append({"step": step, "session_assistant_turns": turns,
                                          "logged_anchor_steps": metrics["gigpo/steps"]})
            total.update(counts)
            total["successes"] += scores
            total["trajectories"] += 64
            total["actor_seconds"] += metrics["timing_s/update_actor"]
            total["rollout_seconds"] += metrics["timing_s/gen"]
            total["recorded_assistant_turns"] += turns
            rows_out.append(one)
        assert len(unique_sessions) == 1280
        assert all(group["variable_lengths"] for group in all_success_groups)
        total["groups"] = total["all_zero"] + total["all_one"] + total["mixed"]
        total["degenerate_trajectories"] = (total["all_zero"] + total["all_one"]) * 8
        total["mixed_trajectories"] = total["mixed"] * 8
        out = {"totals": dict(total), "updates": rows_out,
               "failure_categories": dict(failures), "termination_reasons": dict(terminations),
               "all_success_groups": all_success_groups, "step_count_discrepancies": discrepancies,
               "raw_signal_files_found": data["raw_signal_files"],
               "nonzero_step_coverage": None, "cross_trajectory_anchor_coverage": None,
               "noninitial_anchor_coverage": None, "filtered_nonzero_step_signal_fraction": None}
        if arm in ("e2", "e3"):
            out["repeated_anchor_step_coverage"] = total["logged_repeated_anchor_steps"] / total["logged_anchor_steps"]
        result["arms"][arm] = out
    # Schedule/task identity matches, but trajectories are independently generated.
    schedules = [[sorted(row["task_id"] for row in u["rows"]) for u in d["updates"]]
                 for d in snapshot["arms"].values()]
    assert all(schedule == schedules[0] for schedule in schedules)
    result["validation"] = {"rollouts": 5120, "updates": 80, "groups": 640,
                            "reward_means_match_metrics": True, "df_counts_match_all_40_updates": True,
                            "same_task_schedule": True, "historical_exact_anchor_replay_possible": False}
    return result


def report(result):
    lines = ["# Agentic RL 信号审计（2026-09-14）", "",
             "只读审计四组已完成的 20-step 训练，未重训、未调用 GPU、未更改正在运行的独立评测。",
             "训练采样成绩不等于 selection 或官方 final 成绩。", "", "## 实际奖励与过滤", "",
             "|组|轨迹|成功采样|全失败组|全成功组|混合组|等回报轨迹|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, data in result["arms"].items():
        t = data["totals"]
        lines.append(f"|{arm.upper()}|{t['trajectories']}|{int(t['successes'])} ({t['successes']/t['trajectories']:.2%})|"
                     f"{t['all_zero']}|{t['all_one']}|{t['mixed']}|{t['degenerate_trajectories']} ({t['degenerate_trajectories']/1280:.2%})|")
    lines += ["", "每组 160 个 task/update 组、每组 8 次采样。E1/E3 为实际过滤；E0/E2 的等回报组只是离线分类。",
              "E1/E3 全部 40 个 update 的分类与线上 DF 日志逐项一致；80 个 update 奖励均值与线上记录一致。",
              "训练仍然存在大量全失败组；固定 rollout 过滤不会新增成功样本或奖励。", "",
              "## 步级信号是否实际进入更新", "",
              "|组|有非零步级贡献证据的 update|日志锚点步骤|重复锚点步骤|重复覆盖率|",
              "|---|---:|---:|---:|---:|"]
    for arm in ["e2", "e3"]:
        d = result["arms"][arm];t = d["totals"]
        lines.append(f"|{arm.upper()}|{t['updates_with_certified_step_contribution']}/20|{t['logged_anchor_steps']}|"
                     f"{t['logged_repeated_anchor_steps']}|{d['repeated_anchor_step_coverage']:.2%}|")
    lines += ["", "证据方法：从每题实际奖励重建 episode-only 优势范围（Fnorm=1，omega=1，无 reward KL）。",
              "线上 critic/advantages 极值来自过滤后的有效策略 token；超出 episode-only 范围，足以证明该 update 有非零步级贡献。",
              "它不能证明每个锚点都有效，不能测量非零 token 比例，也不能证明步级项带来因果收益。",
              "日志 groups_with_signal、step_advantage_active 与 usable_step_coverage 只检查组大小至少为 2，包含同一轨迹内重复状态。",
              "因此表中重复覆盖率不能写成非零优势、跨轨迹或非初始状态覆盖率。", "",
              "## E3 过滤是否可能损失信号", "",
              "E3 按终局等回报分类，在 compute_advantage 前清零整个 response_mask。奖励与 anchor payload 没有删掉，最终 token 优势仍被该 mask 归零。",
              "实际 15 个全成功组共 120 条轨迹，全部存在 assistant_turns 长度差异；这些组确实被过滤。",
              "gamma=0.95 时，全成功轨迹仍可能因剩余步数不同产生不同折扣回报。合成反例通过实际算法函数验证：",
              "8 条全成功、不同长度且共享初始锚点的轨迹，过滤前有 8 个非零步级位置，过滤后全部为 0；gamma=1 时该反例消失。",
              "这是机制层面的反例与历史候选组定位，不能把 120 条等同于实测被误删的非零信号条数。缺失逐步锚点，无法给出历史实际损失比例。",
              "而且长度产生的优势可能只是偏好较短轨迹，不自动等于更好的动作信用。不能直接据此解释 E3 与 E2 的成绩差。", "",
              "## 可恢复性与缺口", "",
              "- 现有 JSONL 保存了奖励、解码后的对话、session 元信息；未保存 anchor_ids、anchor_spans、response_mask、逐 token 优势或 optimizer-update batch。",
              "- 日志聚合值无法反推出逐轨迹分组；解码时跳过特殊 token，不能把文本重分词当作在线 span 的精确恢复。",
              "- 非零步级覆盖率、跨轨迹覆盖率、非初始覆盖率、过滤掉的非零信号比例：均明确记为 null，而非零。"]
    for arm in ["e2", "e3"]:
        for d in result["arms"][arm]["step_count_discrepancies"]:
            lines.append(f"- {arm.upper()} step{d['step']}：session 记录 {d['session_assistant_turns']} 个 assistant turn，"
                         f"线上 anchor 统计 {d['logged_anchor_steps']} 个；两个计数来自不同位置，不能未经核验用 session 长度替代在线步骤。")
    lines += ["", "## 后续最小补充", "",
              "1. 下次在线更新前保存紧凑审计 payload：update、uid、task、session、原始奖励、anchor ID/span、过滤前/后策略 mask 的区间、初始步骤标记及代码/配置 hash。",
              "2. 从同一 payload 同时计算实际优势与关闭 DF 的反事实优势，统计跨轨迹/非初始非零步骤、被屏蔽的非零步骤及 token 加权绝对优势质量。反事实幅值不是反事实模型效果。",
              "3. 先用 CPU/固定数据验证，再在用户授权的短程训练中验证 1–2 step；不为了补历史计数自动重训四组。",
              "4. 是否修改 E3 过滤策略，等真实信号审计后决定；DB-only 仍需匹配预算的独立训练，不能由本次审计替代。", "",
              "## 复现与验证", "",
              "snapshot.json 保存远程文件 SHA256 和精简记录；audit.json 保存逐 update 分类、优势范围和全部全成功候选组。",
              "审计验证 5,120 条记录、80 个 update、640 个采样组，四组 task 调度一致。另有 4 个算法行为反例检查通过。",
              "源码入口：env_info/a800_20260912/collect_signal_audit.py 与 audit_training_signal.py。",
              "运行：PYTHONPATH=. <cpu-python> env_info/a800_20260912/audit_training_signal.py --snapshot <snapshot.json> --output <目录>", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    root = Path(__file__).resolve().parents[2]
    for name, expected in snapshot["source_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    result = analyze(snapshot)
    result["snapshot_sha256"] = hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.output / "report.md").write_text(report(result))
    print(json.dumps({arm: data["totals"] for arm, data in result["arms"].items()}, ensure_ascii=False))
