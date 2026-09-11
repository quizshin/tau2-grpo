# 独立评测：pass@k 为主，pass^k 为辅

固定使用 τ³-bench v1.0.1（`fc0055dc4e0a316c3f83133267fbd6faaa770992`），
保留本项目 Airline 数据库隔离补丁。包名仍为 `tau2`。
评测调用官方 Orchestrator 和 `EvaluationType.ALL`，使用冻结模型实际生成对话。
训练 reward、E0–E3 的 24 条训练轨迹不能代替最终模型的独立评测。

## 指标与分母

每个任务采样 n 次，其中 c 次成功：

- 主指标 `pass@k = 1 - C(n-c,k) / C(n,k)`：k 次中至少一次成功。
- 可选 `pass^k = C(c,k) / C(n,k)`：k 次全部成功。不是把样本成功率直接取 k 次方。
- 成功采用官方规则：`1 - 1e-6 <= reward <= 1 + 1e-6`，部分正奖励不算成功。
- 先逐任务计算，再对全部计划任务等权平均。默认 n=4，报告 @1/@2/@4；
  `--include-pass-hat` 同时报告 ^1/^2/^4。`--ks` 可指定其他 k，但 k 不能超过 n。
- 默认 k 自动限制在采样次数内，例如 `--trials 2` 只报告 @1/@2。

任何任务缺少试验、端点异常、官方 `infrastructure_error` 或奖励不可用时，
整次评测标记 `incomplete`、`metrics_valid=false`，总指标、mean_reward、solve_rate 为 null，
命令返回 1。完整任务的诊断分数仍保存在 `per_task`；不得据此给模型排名。
正常完成且得到官方奖励的失败轨迹保留在分母中。
所有计划试验评分完成后，才输出可比较的总指标；基础设施失败不会被当作模型失败。

## SFT / E0–E3 的运行方式

先使用 `selection` 比较冻结的 SFT、E0、E1、E2、E3；使用同一份 selection manifest、
每任务 4 次、相同 seed、模拟器、温度、步数上限和生成配置。
标准 selection 为 60 个任务，即每模型 240 条，5 个模型共 1200 条轨迹。
实际任务数量以 dry-run 输出为准。τ³ 官方 Airline base 的 50 个任务留给冻结赢家后的
`tau3-final`，该入口继续要求 winner lock，不能用它选择 E0–E3。

评测入口要求已服务的**确切 merged checkpoint**及服务 attestation。
RL LoRA 需要先与其精确的 SFT 合并基座合并，再启动服务；不要把 adapter 目录当作完整模型，
也不要误用原始 Qwen 基座合并 RL adapter。当前评测入口未添加在线 LoRA 加载支持。
此改动本身不启动模型服务、不执行 GPU 实验。

以下命令在 `code` 根目录、已配置的 GPU Python 环境执行。
先把 `CHECKPOINT` 和 `TAU3_USER_SERVED_MODEL_NAME` 设为实际模型路径与模拟器服务的模型名；
27B INT4 模拟器的 API 模型名必须与 `/v1/models` 一致。

```bash
export QWEN35_SIZE=4B
export CHECKPOINT=/absolute/path/to/exact_merged_checkpoint
export TAU3_USER_SERVED_MODEL_NAME=actual-simulator-served-name
export TAU3_USER_BASE_URL=http://127.0.0.1:8100/v1

# 只检查评测目标、试验数量和指标；不发起生成。
bash scripts/eval/run_qwen35.sh selection "$CHECKPOINT" \
  --seed 42 --trials 4 --ks 1 2 4 --include-pass-hat --dry-run

# 在独立终端启动冻结 policy 服务，自动写入 checkpoint/PID attestation。
bash scripts/serve/policy_qwen35.sh "$CHECKPOINT"

# policy 和模拟器服务就绪后运行；每个模型使用新的独立输出目录。
bash scripts/eval/run_qwen35.sh selection "$CHECKPOINT" \
  --seed 42 --data-seed 42 --trials 4 --ks 1 2 4 --include-pass-hat \
  --max-concurrency 4 --max-steps 30 \
  --policy-temperature 0.4 --user-temperature 1.0 \
  --output-dir results/evaluation/sft_selection_seed42
```

运行会保存：

- `run.json`：完整 task/trial/seed 计划、评测参数、benchmark revision、模型名与温度、
  checkpoint 路径/内容哈希、服务 attestation 哈希，以及 final 的 winner lock 信息。
  不保存 API key。
- `trajectories.jsonl`：每条已评分轨迹和官方完整 simulation，完成一条写入并 flush 一条。
- `errors.jsonl`：每条异常及其 task/trial/seed；官方基础设施异常保留 simulation。
- `summary.json`：完成状态、分母、总指标与逐任务指标；与 CLI 输出一致。

已有评测文件的输出目录会被拒绝，防止覆盖证据。当前不支持自动续跑；失败后应使用新目录
重跑同一冻结计划，不能反复重试模型失败直到成功，再挑选成功轨迹计算指标。

## 离线重算

不需要 GPU、vLLM、在线服务或 tau2 的运行依赖；安装项目 CPU 依赖即可。
同一批轨迹可以追加 pass^k，不需要重新采样：

```bash
python -m tau3_grpo.evaluation.rescore \
  --run-dir results/evaluation/sft_selection_seed42 \
  --ks 1 2 4 --include-pass-hat \
  --output results/evaluation/sft_selection_seed42/rescored.json
```

未指定 `--output` 时只打印 JSON；指定时要求目标文件不存在。
重算校验计划、重复/未知任务试验、seed 和有限奖励；不会静默丢掉缺失任务或重复计数。
返回码：0 表示完整，1 表示试验未完成，2 表示参数/文件/证据错误。
该入口要求本次新增的 `run.json`；旧训练 telemetry 不能直接当作独立评测输入。
