# 代码组织与调用关系

`code` 是唯一业务项目根目录。原 `tau3-grpo-longhorizon/src/tau3_grpo` 已提升为 `tau3_grpo`；包安装名暂时保持 `tau3-grpo-longhorizon`，以便升级已有环境，文件目录中不再保留旧的嵌套。

## 日常操作与实现

- `configs` 存参数；`scripts` 存启动入口；`tau3_grpo` 存可导入、可测试的实现。
- 统一入口是 `scripts/train/{sft,rl}/run.sh` 和 `scripts/serve/simulator.sh`。
- `run_base.sh`、`run_qwen35.sh` 是统一入口调用的底层实现，保留 veRL 参数兼容与 Qwen3.5 特殊设置。直接调用它们时沿用原先的位置参数；日常使用统一入口即可。
- `configuration.py` 合并 YAML 的 `includes`（相对当前 YAML）并记录有序来源与 SHA256，`launch.py` 调用真实入口。外部环境变量和 `.env` 覆盖配置中的 environment 默认值；显式原生参数位于命令末尾。
- SFT YAML 的 `model/data/train/output` 仍由 SFT Trainer 读取；`launch` 只负责服务/环境选择，不复制一套训练设置。
- 不同模型家族需要选择相应的已安装环境，启动器不会自动切换或安装 Torch/vLLM。
- 兼容默认由 `training/rl/runtime_defaults.py` 读取 YAML；Qwen3.5 默认和接线参数集中在 `configs/runtime/rl_qwen35.yaml`。一般入口与正式 runner 的优先级保持，原生 Hydra 覆盖最后追加。收尾验证与历史文件保留依据见 [配置与记录收尾](cleanup_20260919.md)。

## 训练链路

```text
scripts/train/rl/run.sh
  → tau3_grpo.launch（配置、模型、硬件、算法选择）
  → scripts/train/rl/run_qwen35.sh 或 run_base.sh
  → tau3_grpo.training.rl.train（当前进程注册算法）
  → verl.trainer.main_ppo
  → veRL agent loop ↔ envs.interaction / tools / session
  → evaluation.verifier（官方 reward）
  → algorithms（优势、动态过滤）→ veRL 参数更新
  → tracking（SwanLab 指标、轨迹与源码附件）
```

SFT 则由 `training.sft.train` 调用 `training.sft.trainer` 和 `training.sft.dataset`，随后使用 `training.sft.merge` 导出。独立评估在 `evaluation.run/runtime`；它与训练阶段 reward 记录是不同流程。

## 从旧位置找新位置

| 原模块 | 当前模块 |
|---|---|
| `cli.sft_train`、`sft` | `training.sft.train`、`training.sft` |
| `cli.train` | `training.rl.train` |
| `algo`、`anchors` | `algorithms`、`algorithms.anchors` |
| `env`、`integration.tools/interaction/registry` | `envs` |
| `evaluation.verifier`、`experiment.eval_runtime/metrics` | `evaluation` |
| `experiment.swanlab_tracking/telemetry`、`integration.trainer_telemetry` | `tracking` |
| `integration.anchor_hook/patch_contract` | `integrations` |
| `experiment.manifest/winner_lock` | `experiments` |

工具/交互 YAML、Ray/veRL 回调、测试、安装入口和源码附件路径都随包名修改。旧 CLI 模块路径不再保留影子副本；已有 `tau3-*` console 命令由重装后的 pyproject 指向新模块。

## 存储与部署

源码唯一主路径为 `/root/autodl-fs/tau3-core/code`，不另建长期运行副本。发布前后按变更文件 SHA256 与权限清单核对。虚拟环境的 editable 安装和 `.pth` 必须指向实际运行副本；不能只更新文件而继续导入旧 FS 源码。

原始模型、数据集和结果通过 `TAU3_MODEL_ROOT`、`TAU3_DATA_ROOT`、`TAU3_RUN_ROOT` 指定，源码替换不删除这些资产。旧结果中的配置快照记录的是当时路径，不批量修改历史证据；再次执行旧命令时按本表迁移。


## 第一批后的职责边界（2026-09-18）

```text
configuration → launch / training.rl.runner → shell → training.rl.train → veRL
                                                                  ↓
               policy generation ↔ harness（agent loop / interaction / tools / session）
                                                                  ↓
              官方 verifier + 独立版本 shaping → integrations.verl → algorithms
                                                                  ↓
                       veRL actor 更新 → 权重同步 → 下一批 rollout
                       checkpoint / selection / telemetry / SwanLab
```

Harness 是组织一次 agent 与环境交互的执行流程：建立任务和独立状态，把模型输出解析成动作，执行工具并返还 observation，推进模拟用户，判断终止，收集轨迹，调用验证器。它由现有 `envs`、veRL agent loop 补丁和独立评测 runtime 协作实现，不是另一个算法或一个独立目录。

训练从 train50 调度任务，按 8×8 产生完整轨迹；GRPO 用终局 outcome 构造组内优势，GiGPO 在 episode 信号外使用可见状态 anchor 的 step 信号，MT-GTPO 使用显式版本的过程 reward 与终局信号进行混合信用分配。框架 adapter 负责精确 token/span/mask 与 tensor 结构接线，veRL 做优化更新与权重同步。benchmark 环境提供任务、工具、状态和官方验证，benchmark 本身不承担梯度更新。

独立评测使用 official Orchestrator 与同样的工具/提示词协议，执行流程和训练 rollout 并不是同一个实现。因此共用 verifier 不足以证明两套 harness 完全等价；原始终止、动作顺序、可见信息和预算仍需差分验收。官方 reward 是可重放的成功信号，但不是全部业务文本正确性的保证；训练 shaping 也不是官方成功率。

|入口 / 接口|当前实现与范围|
|---|---|
|正式单实验|`training/rl/runner.py`，三种 estimator + DF，默认 20 step；旧 MT 脚本转发|
|子进程所有权|`training/services.py`，正式 runner、旧 matched 队列、独立评测 controller 复用|
|veRL adapter|`integrations/verl/{gigpo,mt_gtpo}.py`；旧路径是同一模块对象，避免双份全局统计|
|可见消息|`data/messages.py`，语义模型和离线分析共用，去掉隐藏 reward/gold 字段|
|模板 token|`models/tokenization.py`，prepare_sft 与 SFT dataset 共用|
|独立评测证据|`evaluation/artifacts.py`，rescore/compare 共用读取；不信任 cached summary|
|严格比较|`evaluation/compare.py` + `diagnostics.py`，完整计划/协议校验、增幅/CI/成本覆盖率|
|记录草稿|`experiments/review.py`，从显式回执生成含 SHA256 的实验/错误草稿，不覆盖唯一索引或人工结论|

不要把 `configuration_sources` 当作逐字段 Hydra 完全来源追踪；它现在记录 YAML 包含链与启动环境来源，最终 Hydra 仍由正式 runner 保存。当前硬件组件和独立评测 controller 已整理如下；历史队列及环境诊断脚本保留原复现职责。

## 后续批次后的接口（2026-09-18）

当前 formal profile 已解除对旧 pilot 的继承：`configs/train/rl/formal50_a800.yaml` 组合 `protocols/models/hardware/data/simulator`；MT-GTPO profile 再增加 `algorithms/mt_gtpo.yaml` 和显式 reward 组件。旧日期 profile 仍供历史复现与兼容测试引用，不通过移动文件制造旧记录断链。

|接口|实现与边界|
|---|---|
|完整 checkpoint|`training/rl/checkpoints.py` 统一验证；`integrations/boundary_checkpoint.py` 发布与轮换；文件结构证据不等于 GPU 可恢复性|
|独立评测 controller|`evaluation/controller.py`，保留历史 E0–E3 队列管理；任意单模型用 `evaluation/run.py`|
|轨迹事实|`data/trajectory.py`；新正式 runner 显式启用 `TAU3_RECORD_TRAJECTORY_FACTS=1`，记录真实 token/mask、发出及保留区间、工具与原始结束原因|
|奖励 recipe|`evaluation/rewards/recipe.py` 为公开读取/校验，`analysis/calibrate_paper_rewards.py` 拟合与导出；v2 记录新身份，已知 v1 显式兼容|
|本批诊断|estimator 通过 `diagnostics_out` 写 `batch.meta_info`；trainer 与 tracking 不依赖另一次调用留下的全局统计|
|评测来源|`evaluation/provenance.py` 在新运行前记录实际任务/DB及导入的 benchmark 源码身份；旧历史 hash 不补造|
|补丁与测试|`env_info/vendor_patches.json` 固定 revision 对照；`cpu_test_suites.json` 分类执行；`lint_debt.json` 显式旧债务与新增拒绝|

harness 差分测试已经覆盖固定回复下的数据库写入、写后读取、错误响应、顺序、终止和官方成功/失败分数。它不证明真实模型下两套 executor 的全部分布相同：训练采用 assistant/user 各 15 轮与 token 上限，独立评测使用 Orchestrator step/error 上限；真实服务解析、生成/恢复和成本开销仍需 GPU 证据。
