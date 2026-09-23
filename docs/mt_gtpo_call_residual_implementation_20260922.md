# call_residual_v1：实现、固定轨迹回放及数值边界

2026-09-22。用户授权完成本地实现与 CPU 检查；本次未同步远端、未启动新 GPU 服务。状态为 CPU verified，不代表训练效果或通用数值稳定性已验证。

## 固定公式与边界

```text
A_call(t,j) = A_turn(t) + (q(t,j) - mean_j(q(t,j))) / (std_original_G(t) + eps)
```

beta 固定为 1；mode 名称决定公式，不新增可调 beta。奖励分类、实际支付分数、原始 returns、gamma/lambda 及 DF 默认值不变。文本 token 使用原轮级优势，精确归属的调用块才可能修改。单调用及同轮奖励完全相同的轮保留原值，不经浮点加减再写回。

沿用旧候选的精确 token、UID、trajectory、call ID、mask 校验和回退条件：归属歧义、组支持不足、调用同伴支持不足、原 return 零方差均回退。调用同伴支持门槛对轮内中心化并非数学必需，本版为匹配前次固定诊断，保守保留。损坏或缺失元数据仍报错，不伪造调用区间。

保留 `turn_v1` 默认路径和 `call_local_v1` 旧公式。新候选复用 `algorithms/mt_gtpo_call_credit.py` 的输入验证、统计和写回流程，使用显式分支；veRL/runner/CLI/YAML 贯通 `call_residual_v1`。复用 `mt_gtpo_call_replay_v1` 记录结构，由 `credit_mode` 区分公式；逐调用记录标明 `baseline_scope=within_turn`、实际调用均值、`residual_beta=1`，保存真实逐 token 优势及原 returns。

沿用原调用信用遥测，另增加 correction_abs_max、token_weighted_correction_sum、call_mean_drift_abs_max、applied_return_std_min、near_zero_scale_turns。SwanLab 中使用现有 `mt_gtpo/call_credit/` 前缀。调用等权平均优势保持原值，但不同长度调用的 token 加权和、PPO 梯度不守恒。

## 验证结果

分析前保存 `results/analysis/call_residual_v1_20260922/spec.json`，不改奖励或搜索参数。

### 5090 两步真实 token 回放

原始输入 SHA256 与先前 GPU 审计一致。128 轨迹、1,246 轮、864 调用；重新计算过程奖励一致。旧 call_local_v1 实际 token 优势仍逐值完全复现。新模式生产 core、veRL adapter 及 JSON 重放记录逐值一致，与独立 `statistics.mean/pstdev` 调用公式比较最大误差 1.11e-16。非调用 token 和 returns 不变。

| 指标 | 原 turn_v1 | 旧 call_local_v1 | call_residual_v1 |
| --- | ---: | ---: | ---: |
| 正奖励调用正优势（123次） | 114 | 81 | 117 |
| 成功轨迹中的正奖励调用正优势 | 71 | 39 | 71 |
| 错误调用正优势（41次） | 17 | 15 | 17 |

先前 35 次翻转全部恢复原值；这是保留同奖励轮信号的设计性质，不是独立收益证明。新增 3 次正优势来自官方失败 2 次、未评分 1 次。共 6 个轮、17 次调用、1,373 个 token 被调整。调用等权均值漂移不超过 2.23e-16。

最大绝对修正 0.917349，最大绝对优势 3.404961。实际调整轮的原 return 标准差范围 0.675426–0.859995；没有 `std<=eps` 的调整轮。两步 token 加权修正和分别 +6.031104、-0.814299，合计 +5.216805；这不是模型梯度或概率变化。

### 更大历史 buffer：仅调用级公式分析

使用 `mt_gtpo_reference_write_v3/20260917_s42_df0` 原 20 步、160 组、1,280 轨迹、9,680 调用，原 v3 支付奖励不变。重建原 turn_v1 的回报/优势与原记录逐值一致。历史记录没有精确调用 token 身份，因此只对 parsed 且 nontruncated 的记录做假定归属可用的调用级公式分析；不伪造 spans，也不声称新 estimator 在这些历史 token 上已获验证。

| 指标 | 原 turn_v1 | 轮内残差公式 |
| --- | ---: | ---: |
| 正奖励调用正优势（1056次） | 962 | 990 |
| 成功轨迹中的正奖励调用正优势 | 638 | 638 |
| 错误调用正优势（1396次） | 387 | 381 |

370 次调用被调整；没有新增正奖励信号损失，新增 28 次正优势均来自官方失败轨迹，不能记作收益。错误正优势减少 6 次，仍剩 381 次。最大绝对修正 2.262229，最大绝对优势 4.157584；调整调用所属轮标准差最小 0.0330719，没有接近 epsilon 的样本。两批数据都是已查看过的开发数据，不是盲测。

### 小方差与长度压力检查

构造同轮奖励 `[+1,-1]`、同伴总奖励 `1e-8`，原 return 标准差为 `5e-9`。按声明公式和 `eps=1e-6`，修正可超过 900,000，仍有限但不可视为合理训练尺度。测试及遥测明确捕捉该风险；本次未悄悄引入截断、上限或新归一化公式。

不等长调用测试同时确认：调用等权均值保留，token 加权和不保留。真实 buffer 上没有上述极端小方差，仅能说明这两批数据未触发，不能证明未来采样安全。若后续推进 GPU，应先明确小方差回退或限幅方案及其新版本证据，再决定实验范围。

### CPU 测试

相关集合 **169 passed、0 failed、0 skipped**。覆盖独立公式、单调用/全同奖励精确不变、归属和重叠错误、空 mask/padding、缺轮与支持不足、真实原生执行到 veRL 的新旧两分支、DF、旧 GRPO/GiGPO/MT 配置、runner/Hydra、探索门禁与 credit mode 续训身份。确定性模型替身用于原生工具链路测试，不是模型推理。lint 无新增，`git diff --check` 通过。

日志 `tests.log`、`replay.log`；报告 `real-token-report.json`、`historical-report.json`；逐调用分解 `real-calls.json`、`historical-calls.json`；输入身份 `source-hashes.json`，均在上述分析目录。`replay.py` 使用排他输出，拒绝覆盖旧结果。

## 入口与交接

模式选择：`--credit-mode call_residual_v1`，组合 profile 为 `configs/train/rl/formal50_5090_a45_mt_gtpo_call_residual_v1.yaml`。runner 自动启用驱动及 Ray worker 的调用归属记录，关闭不需要的 critic。已有模式不改默认值。

新模式只开放已有的全新 1–2 步 `--engineering-smoke` 未校准短测路径；原版 30 步 `--uncalibrated-exploration` 仍限 `turn_v1`，不会因接入本候选而扩大。正式路径仍要求通过的 frozen recipe。续训拒绝在三个 credit mode 之间切换。

可在部署并解决上述数值边界后使用下列命令进行**仅预检**（不启动服务）；本次没有在 5090 上执行：

```bash
python -m tau3_grpo.training.rl.runner \
  --estimator mt_gtpo \
  --profile configs/train/rl/formal50_5090_a45_mt_gtpo_call_residual_v1.yaml \
  --reward-version paper_env_split_v4 \
  --credit-mode call_residual_v1 \
  --token-protocol tau3_token_budget_v1 \
  --engineering-smoke --updates 2 \
  --result-dir "$TAU3_RUN_ROOT/mt_call_residual_smoke/test01" \
  --dry-run
```

结论：本地训练接入及可回放性已完成，保留同奖励轮信号的目标得到验证；当前仅修正轮内信用，时间信用问题仍存在，小方差风险未解决，不能宣称候选优于原版或直接扩训至 30 步。
