# GRPO 最终结果与 MT-GTPO 最小接入

2026-09-22 后续：已新增显式 `--uncalibrated-exploration` 供 `turn_v1 + split-v4` 运行未校准探索；正式校准门禁保持。部署、30 步命令及续训说明见 [探索入口](mt_gtpo_v4_exploration_20260922.md)。下文的预检记录保留原始时点状态。

2026-09-22。本轮仅读取已归档结果并增加组合配置；没有开机、采样或启动训练。

GRPO v4 完成 30 步，每个节点 selection60×4。A45 SFT 成功 110/240；GRPO step10/20/30 分别 110/108/130，即 45.83%/45.00%/54.17%。step30 较 SFT 增加 20 次成功、8.33 个百分点；经验 pass@4 同为 46/60。相同覆盖数不代表成功任务集合完全相同，也尚无多种子显著性结论。训练 reward 每十步均值为 46.25%、49.375%、54.375%，任务组成不同，不能单独用这条趋势判定收益。日志累计训练 step 时间约 4.63 小时，三次评测约 1.15 小时；不含全部初始化开销。

## 最小代码路径

- 调度、服务、保存、评测、SwanLab：`tau3_grpo/training/rl/runner.py`。
- Hybrid 优势核心：`tau3_grpo/algorithms/mt_gtpo.py:compute_mt_gtpo`。
- veRL batch/token span 适配：`tau3_grpo/integrations/verl/mt_gtpo.py`。
- 过程奖励：`tau3_grpo/evaluation/process_reward.py`。
- 配置：`configs/train/rl/formal50_5090_a45_mt_gtpo_v4.yaml`，仅组合本次 GRPO 硬件配置、MT-GTPO、split-v4 奖励。

这些模块仍依赖现有项目、veRL、tau2-bench 和部署环境；不是复制四个 Python 文件即可独立运行。历史 `scripts/train/rl/run_mt_gtpo_formal.py` 仅转发至同一 runner，无需复制训练实现。

## 入口

在原 5090 环境开机并同步新增 YAML 后，可先执行无 GPU 服务的预检：

```bash
source /root/shared-nvme/tau3/activate.sh
cd "$CODE_ROOT"
export TAU3_ENV_FILE=/root/shared-nvme/tau3/config/swanlab.env
python -m tau3_grpo.training.rl.runner \
  --estimator mt_gtpo \
  --profile configs/train/rl/formal50_5090_a45_mt_gtpo_v4.yaml \
  --reward-version paper_env_split_v4 \
  --token-protocol tau3_token_budget_v1 \
  --updates 30 \
  --result-dir "$TAU3_RUN_ROOT/mt_gtpo_5090_a45/v4-preflight" \
  --dry-run
```

正式训练需要去掉 `--dry-run`，增加 `--reward-recipe` 指向实际通过校准的 `frozen-recipe.json`，并改用独立的新结果目录。当前已查阅的最新校准报告未通过，未发现可用冻结配方；不能将上面的未校准预检当作可立即正式训练的证明。这个限制来自现有 `runner.resolve` 的代码校验。探索性未校准训练需明确标注和另定预算，不能冒称校准通过。

建议先保持与 GRPO 相同的 A45 起点、train50、30 步、8×8 轨迹/步、温度 0.7、lr1e-6、KL0.01、完整 schema、DF off。不要使用 GRPO step30 的 `--resume-from` 作为算法对照；runner 也拒绝续训时改变 estimator。

本地 `.venv-cpu` 已验证组合加载、8 rank、FSDP2、A45 路径、模拟器共享、MT-GTPO 身份、v4 奖励和 DF off。尚未在目标机执行该配置的完整 Hydra 预检或 MT-GTPO GPU 更新。GRPO 的 30 步成功只验证了共享硬件路径，不能代替 MT-GTPO 专项运行证据。
