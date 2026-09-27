# MT-GTPO 保守奖励配方离线校准（2026-09-18）

## 结果与边界

远程 CPU 完成 20 个历史训练 buffer、1,280 条轨迹的原始奖励／优势／mask 回放及重新计分。
独立保守配置通过其预先声明的数值检查，状态为 `development_checks_passed`。
没有生成 `frozen-recipe.json`，没有新增采样、训练或调用 LLM judge。
这是项目的环境适配与校准策略变体，不能称为原论文算法的原样复现，也不能证明模型优于 GiGPO。

使用此前已检查过的任务划分和旧策略轨迹，因而下文的“原留出划分”仅用于开发诊断，已经不是未查看的验证集。
本轮没有搜索多套权重、降低阈值或改变划分种子。报告的通过只适用于这套新增策略，原始 IRC 配置的失败结论保留。

## 独立配置

配置：`configs/analysis/mt_gtpo_irc_paper_env_conservative_20260918.yaml`。
奖励匹配继续使用 `paper_env_v2`，Hybrid 保持 `gamma=0.9`、`lambda_outcome=0.3`，动态过滤仍由独立开关控制，本轮不修改过滤实现。

| 类别 | 候选权重 | 依据 |
| --- | ---: | --- |
| gold_exact | +1.000000 | 预先指定的参考锚点，不是数据拟合结果 |
| soft_match | 0 | 未验证的部分匹配不给额外 shaping |
| duplicate | 0 | 重复且状态未变化不自动等于无用，暂不规定惩罚方向 |
| state_change | -0.20961006222627246 | 仅在拟合集按出现性与终局结果的相关系数拟合 |
| error | -0.35298915110170204 | 同上 |
| read_only / message / unknown | 0 | 固定中性；出现 unknown 仍拒绝验收 |

`fixed_anchor_policy: gold_reference_v1` 必须显式声明，且只允许 `gold_exact=1` 这一个非零固定权重。
旧配置没有此字段时仍只允许中性固定权重，不能通过把任意类别放入 fixed 绕过拟合检查。
gold 在两组仍必须满足最低出现数量、即时 proxy 与 Hybrid 的平均优势均为正。
gold 的出现性支持数仍如实报告为 5 / 2、不足以拟合；锚点策略没有把这一估计标成可靠。

保留 `alpha=1`、`delta=0.05`、`eta=0.1`、`min_support=20`、`holdout_fraction=0.2`、`split_seed=42`。
拟合类别仍要求支持数、预期权重方向、两种平均优势方向；总体平均过程奖励与终局结果相关性仍须超过 eta。
中性指即时 shaping 为零，不要求 Hybrid 优势为零：终局项、未来回报、同轮其他调用及组内归一化仍会分配信用。

## 实测检查

| 指标 | 拟合集 | 原留出划分 |
| --- | ---: | ---: |
| 任务 / 轨迹 | 40 / 960 | 10 / 320 |
| 历史成功轨迹数 | 426 | 101 |
| 平均过程奖励与终局相关性 | 0.1216437862 | 0.2387247551 |
| gold 平均 proxy / Hybrid 优势 | 0.875119 / 0.530540 | 0.810170 / 0.545917 |
| state_change 平均 proxy / Hybrid 优势 | -0.496349 / -0.702139 | -1.002305 / -0.314610 |
| error 平均 proxy / Hybrid 优势 | -0.796525 / -0.442709 | -0.731923 / -0.373631 |
| state_change 拟合支持数 | 94 | 31 |
| error 拟合支持数 | 278 | 101 |
| unknown 出现轨迹数 | 0 | 0 |

这些成功数来自旧训练数据，重算奖励不会改变它们。拟合集相关性只是略高于 0.1 的门槛，没有给出显著性或泛化保证。

仍有三个实际限制：

1. `state_change` 出现性与成功率的相关系数在拟合集是 -0.209610，在原留出划分是 +0.118562。当前既定验收检查的是拟合权重及平均优势方向，不要求留出出现性相关系数同号，因此数值通过仍不代表该惩罚跨任务稳定。需要新轨迹和该类别的工具／任务分层审计，不能事后改阈值来美化结论。
2. gold 覆盖 955/960 与 318/320 条轨迹，包含大量参考查询。固定 +1 是设计假设，并未证明这些查询都带来任务进展，也没有解决潜在的查询刷分问题。
3. 负奖励不保证每次错误调用的最终优势为负。本轮拟合集／原留出中仍有 240 / 148 次 error 调用得到正的 Hybrid 优势；duplicate 虽然 shaping 为零，在原留出中的平均 Hybrid 优势仍是 +0.309289。这与共享回合信用和相对归一化一致，不能声称所有错误激励已经消除。

## 代码与验证

`tau3_grpo/analysis/calibrate_paper_rewards.py` 新增显式锚点策略、锚点验收和 `--development-only`。
诊断模式即使数值通过也不导出正式冻结配方；直接调用 `freeze_recipe` 同样拒绝带 development 标记的报告。
报告保存实现、配置、训练 manifest 和每个输入 buffer 的 SHA-256。
正式训练控制器的冻结配方门禁保留，本报告不能作为正式配方传入。

远程对应环境设置 `CUDA_VISIBLE_DEVICES=""`、`NVIDIA_VISIBLE_DEVICES=void`，离线运行：

```bash
python -m pytest -q tests/test_paper_irc.py tests/test_environment_reward.py tests/test_paper_credit_audit.py
```

结果：45 passed，14.38 秒；仅有依赖 `audioop` 的弃用提示。
覆盖旧 IRC 配方／冻结加载、拟合与留出隔离、无支持拒绝、锚点方向／固定值约束、诊断禁止冻结，以及 DF 开关下的正式配置渲染。
这是 CPU 和配置验证，不是 GPU 更新或新策略效果验证。

## 证据与重现

报告：`results/analysis/paper_env_conservative_20260918/development/report.json`。
命令、测试日志、回执、同步清单及源码备份：`results/maintenance/paper-env-conservative-20260918/`。
训练 manifest SHA-256：`641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae`。
输入：`results/runs/mt_gtpo_reference_write_v3/20260917_s42_df0/rollouts/1.jsonl` 至 `20.jsonl`，全部为训练任务。
精确命令保存在 `calibration-command.json`；复跑必须换用新的输出目录。

## 后续门槛

这轮开发诊断已完成，停止在同一批数据上追求更高相关系数。
下一阶段从共同 `new-off` SFT 起点采集完整训练任务组的新 buffer，单独登记校准预算、模型身份、采样参数和 seed；本轮候选权重与规则先固定，不根据新检验结果偷偷修改。
先检验 state_change 的方向稳定性、gold 读取／写入分层以及错误调用的正优势来源，再决定是否接受此变体或建立新的独立修订。
同一 50 任务池的新轨迹只增加轨迹证据，不构成新任务泛化验证。
冻结后才能按既有 20-step、每步 SwanLab、整十完整保存与 selection60×4 协议做正式算法对照；final50 不用于这轮调参。
