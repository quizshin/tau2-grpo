# SwanLab RL 实验清理 · 2026-09-09

用户要求精简 RL 实验列表。删除前为 7 个 RL；已删除 2 个确认首次更新前 OOM 的运行，删除后重新查询确认剩余 5 个 RL。所有 SFT 实验保持原样。

## 已删除的云实验

| 编号 / run ID | 实验名称 | 原因与保留证据 |
|---|---|---|
| RL-004 / `i7utseim` | `rl-validation-E0-Qwen3.5-0.8B-full-obs64k-g2x4-u3-s42` | 参考模型 log-prob OOM，训练退出 1，未完成更新；持久盘原始日志与归档回执保留 |
| RL-006 / `nqslriqa` | `rl-validation-E0-Qwen3.5-0.8B-full-offload-obs64k-g2x4-u3-s42` | 启用 CPU offload 后仍 OOM，训练退出 1，未完成更新；持久盘原始日志与归档回执保留 |

删除通过 SwanLab SDK 的单实验 delete 接口执行，返回成功，随后列表查询确认两 ID 均消失。没有删除本地或持久盘失败记录，也没有把失败运行的奖励补成 0/24。

## 保留的五个 RL

| 实验 | run ID | 保留原因 |
|---|---|---|
| RL-003：旧观察上限 LoRA | `3stth2pz` | 完整三步，作为观察修复前的工程对照 |
| RL-005：完整观察 LoRA | `2vnhl1qv` | 完整三步及精确 LoRA 同步审计 |
| RL-007：full 第一步 | `llml3h9i` | 虽后续磁盘满，但首步已完成、是最终 full 链路的一部分 |
| RL-008：full 续跑第二、三步 | `lhlz7jms` | 完成逻辑三步链路，保存最终完整检查点 |
| 旧 TRIO 4B RL | `0uk586zt` | 已存在训练指标和对话记录，不能按过时描述当作空实验删除 |

旧 TRIO 4B 实验的 description 仍写“尚未训练”，但实际有 53 个自定义 scalar key、6 个 media key，包含 actor loss、训练 token、奖励与对话记录。初次仅依据 description 判断占位记录不准确，进一步核验后已纠正并保留。此历史运行未在本次重新评估，也不混入本轮 0.8B veRL 的训练结果。

SwanLab 的 FINISHED 不必然代表训练成功：已删除的两次 OOM 曾显示 FINISHED，判断以本地退出码和完整更新产物为准。RL-007 当前云状态为 CRASHED，仍不影响它已保存的第一步证据。

## 归档与本地文档

删除前的列表、所选实验配置/指标快照、已有持久归档状态核对记录先写入持久盘，18 文件逐项 SHA256 核验后才删除。旧 4B 的只读核验快照也包括在这批文件中；媒体快照只含云端记录信息，旧 4B 云端正文继续保留。`archive-verification.json` 覆盖删除前快照；删除结果和删除后列表分别为 `deletion-receipt.json`、`after.json`。

本地证据：`results/validation/swanlab_cleanup_20260909/`。持久证据：`/root/autodl-fs/tau3_grpo_fix/results/swanlab_cleanup_20260909/`。RL-004/006 原始日志仍在各自同名持久结果目录。

实验结果与测试结果已写入 [EXPERIMENTS.md](../EXPERIMENTS.md)、[RL 对照报告](rl_validation_comparison_20260909.md)、[接管文档](training_handover_20260909.md)。相关显存修复回归是本地 103 passed、远程 100 passed / 3 skipped；本次仅整理云端实验及文档，没有重新运行或改写这些历史测试结果。
