# 本地历史结果归档（2026-09-09）

外层 `results`、`validation` 已按实际来源迁入本项目，原始指标、JSON、轨迹、源码压缩包与脚本字节均保留。Markdown 引用和 SwanLab 本地软链接按新位置调整。没有删除云实验、重写历史运行身份、启动训练或改动远程存储。

## 归属核对

外层结果的实际训练配置指向 `runtime/code` 和 veRL，源码压缩包包含 `code/verl`。本地 27B AWQ 模拟器是为这条链路部署的服务；不能因型号与 PyTRIO 用户模型同为 27B 就认作 PyTRIO 推理结果。

SwanLab 项目清单包含两套后端的实验身份，保留为当时完整的项目级管理快照。它不是清单里所有实验的训练数据。Agent/PyTRIO 的原始运行仍保留在各自 `runs` 中，没有将其他后端的训练结果改归 veRL。

| 原目录/内容 | 当前位置 | 依据与说明 |
|---|---|---|
| 外层 results/rl_qwen35_lora_20260908 | [RL 原配置与补传材料](../../results/rl_qwen35_lora_20260908/) | 实际配置 backend=veRL；对应 bd743csc |
| qwen38_simulator 中服务、模型、推理测试 | [模拟器部署](../../results/simulator/qwen38_simulator_20260908/) | 本地 AWQ 模型、vLLM 服务与下载 revision |
| qwen38_simulator 中 RL 轨迹、云端快照、源码包与测试日志 | [RL 补充材料](../../results/rl_qwen35_lora_20260908/supplemental/) | 与模拟器文件混放，现按实际用途拆开 |
| 外层 validation 的 7 个目录 | [验证证据](../../results/validation/) | RL、梯度诊断、边界修复、磁盘清理、code 重构和训练接管 |

## 验证记录入口

- [第一轮 RL](../../results/validation/rl_qwen35_20260908/README.md)
- [补全用户场景后的 RL](../../results/validation/rl_qwen35_scenariofix_20260908/README.md)
- [代码审计](../../results/validation/code_audit_20260909/README.md)
- [边界修复](../../results/validation/rollout_fixes_20260909/README.md)
- [磁盘清理](../../results/validation/disk_cleanup_20260909/README.md)
- [code 重构](../../results/validation/layout_refactor_20260909/README.md)
- [训练接管](../../results/validation/takeover_20260909/README.md)

详细实验结论以 [EXPERIMENTS.md](../../EXPERIMENTS.md) 为准；归档不会把失败运行变成成功实验。原始回执中的绝对路径、hash 和云状态仍代表当时快照，不能当作当前库存。

## 保存与复现

`results/` 为本项目忽略的运行资产目录，复制源码或 Git 克隆不会自动携带这些文件；需要显式同步结果归档。小型导航和迁移清单在 `docs` 中。历史收集/部署脚本原样保存，不能按旧绝对路径直接重跑。

[文件迁移与逐文件 SHA256 清单](../architecture/results-migration-20260909.json)。
