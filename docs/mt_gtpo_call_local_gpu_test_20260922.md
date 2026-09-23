# call_local_v1 GPU 工程测试入口

**2026-09-22 10:18:39更新：用户授权的test01两步GPU运行已完成，以下未执行说明保留为初始历史。** 两步128条、非零梯度、实际调用级优势修正、完整step2保存及SwanLab1/2步回读均通过；CPU重算误差0。错误调用正优势17→15，但正奖励调用正优势114→81（35个正转非正、2个非正转正）。尚无模型效果评测，不以工程通过替代效果结论。详见[完整结果](../results/runs/mt_call_local_smoke/test01/report.md)。

2026-09-22。本地实现已接通。默认信用方式仍为 `turn_v1`，候选仅在显式选择 `call_local_v1` 时生效。没有自动部署服务器或启动 GPU；以下命令供用户在已同步代码的原八卡 5090 环境执行。

## 两步测试

沿用 A45 SFT、八卡 FSDP2、GPU4–7 共享模拟器、train50、每步8组×8条，DF off、token budget v1。测试总共2个外层更新、128条候选轨迹；第2步保存完整检查点，不跑selection或final评测。SwanLab单独记录，结束后控制器清理它启动的训练/模拟器服务。

`--engineering-smoke` 显式允许 split-v4 **未校准的默认初始权重**，不是此前拟合失败的候选权重。该测试只验证真实更新链路，不验证算法收益。正式入口仍要求通过的冻结配方。该开关限定1–2步、新结果目录，不支持续训，也不能将烟雾测试run作为正式run续训。

先同步完整相关源码，不能只复制新YAML：新算法/数据模块、MT适配器、runner、配置，以及此前的调用归属模块与三个 agent_loop/parser vendor文件都必须在服务器上。启动前可执行 `python -m tau3_grpo.integrations.vendor_inventory` 检查固定vendor清单；这不代替全部部署身份核对。

```bash
source /root/shared-nvme/tau3/activate.sh
cd "$CODE_ROOT"
export TAU3_ENV_FILE=/root/shared-nvme/tau3/config/swanlab.env

python -m tau3_grpo.training.rl.runner \
  --estimator mt_gtpo \
  --profile configs/train/rl/formal50_5090_a45_mt_gtpo_call_local_v1.yaml \
  --reward-version paper_env_split_v4 \
  --credit-mode call_local_v1 \
  --token-protocol tau3_token_budget_v1 \
  --engineering-smoke --updates 2 \
  --result-dir "$TAU3_RUN_ROOT/mt_call_local_smoke/test01"
```

第一次仅想看实际Hydra、模型和数据预检时，在同一命令末尾加 `--dry-run`。检查后可以在这个仅预检的目录启动；若已开始过服务或出现训练状态，则换新的结果目录。GPU被占用时控制器会拒绝启动，不会抢占其他任务。

正常收尾核验完整第2步检查点及SwanLab步数1、2；GPU加载恢复尚需另做实验。两步运行不启用正式十步停止标记控制器；终端中断仍进入服务清理，但不承诺中断时保存一个额外检查点。

## 确认实际测到了候选

查看该run的 `launch.json`、`resolved-hydra.yaml`、`metrics.jsonl` 和 `rollouts/*.jsonl`：

- `credit_mode=call_local_v1`，驱动进程及Ray worker都有 `TAU3_RECORD_CALL_ATTRIBUTION=1`。
- replay schema为 `mt_gtpo_call_replay_v1`，含实际 `token_advantages`、原 `baseline_turn_advantages`、调用ID/span、组统计、fallback原因和原始事实。
- 指标前缀 `mt_gtpo/call_credit/` 下有 `applied_calls`、`changed_tokens`、`token_coverage`、`rms`、`baseline_rms`、`correction_rms`、`abs_max`、`abs_p95` 和fallback计数。
- 同时记录 `error_calls`、`error_positive_before/after`、`positive_reward_calls`、`positive_reward_signal_lost`。这些是本批优势诊断，不能冒充概率变化或任务成功率。
- 若 `changed_tokens` 始终为0，或者只有回退，不能仅凭训练退出码0说调用级变化已获GPU验证。还需看非零梯度、损失、KL、actor→rollout同步和实际checkpoint。

候选仍保留长程信用，不保证每个错误调用总优势为负。调用奖励相同、归属不足或零方差可能使本批没有局部修正；这些情况均应保留并报告，不能重采样到结果好看才记账。

需要同条件旧版基线时，使用同一profile、相同2步预算，将 `--credit-mode` 改为 `turn_v1`，并换新结果目录。这样仍记录调用归属供审计，但实际优势走旧函数；不从另一算法checkpoint续训。

## 本地检查与实现

- 纯数组核心：`tau3_grpo/algorithms/mt_gtpo_call_credit.py`。
- 实际IDs、mask、UID、trajectory和call ID校验：`tau3_grpo/data/call_credit.py`。
- 适配、遥测和版本化重放：`tau3_grpo/integrations/verl/mt_gtpo.py`。
- 模式切换、续训身份、两步测试生命周期：`tau3_grpo/training/rl/runner.py`。
- [公式与研究限制](mt_gtpo_call_local_credit_design_20260922.md)。

本地完整相关回归首轮165项通过，覆盖旧MT、GRPO/GiGPO共存、调用边界、原生工具/verifier到trainer的数据链路以及真实shell→Hydra解析。最后增加的坐标检查及诊断指标定向复核35项通过；两次结果按用例去重后166项通过、0失败、0跳过。确定性策略/用户测试不是模型采样或GPU效果评测。日志与源码身份在 `results/analysis/call_local_v1_20260922/`。

## 2026-09-22 用户授权执行

用户明确要求执行上面的两步 call_local_v1 命令。目标 `ackcs-00gjhp53` 已运行、八卡空闲；仅同步21个相关文件，远端覆盖前源码备份于 `/root/shared-nvme/tau3/bootstrap/call-local-smoke-20260922/before.tgz`，逐文件SHA256核对通过，vendor inventory无漂移。真实shell/Hydra及数据预检通过，确认2步128条、DF off、v4未校准默认初始权重、完整schema、调用归属记录、save2与不评测。计划墙钟上限60分钟。对比是同一批次的旧turn优势与call-local优势，不把两步训练称作成功率对照实验。初次远端CPU测试存在tokenizer路径缺失，已补共享模型逻辑链接并用A45 tokenizer复核；最终结果后补。

09:49：远端相关CPU检查去重共99项通过，最终token harness 47项无跳过；vendor inventory通过。控制器已按用户命令启动（launcher PID3398），墙钟上限60分钟，目录 `runs/mt_call_local_smoke/test01`。目前仅启动，不宣称已完成GPU更新。
