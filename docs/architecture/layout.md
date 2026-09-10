# 代码组织与调用关系

`code` 是唯一业务项目根目录。原 `tau3-grpo-longhorizon/src/tau3_grpo` 已提升为 `tau3_grpo`；包安装名暂时保持 `tau3-grpo-longhorizon`，以便升级已有环境，文件目录中不再保留旧的嵌套。

## 日常操作与实现

- `configs` 存参数；`scripts` 存启动入口；`tau3_grpo` 存可导入、可测试的实现。
- 统一入口是 `scripts/train/{sft,rl}/run.sh` 和 `scripts/serve/simulator.sh`。
- `run_base.sh`、`run_qwen35.sh` 是统一入口调用的底层实现，保留 veRL 参数兼容与 Qwen3.5 特殊设置。直接调用它们时沿用原先的位置参数；日常使用统一入口即可。
- `launch.py` 合并 YAML 的 `includes`（相对当前 YAML），再调用真实入口。外部环境变量和 `.env` 覆盖配置中的 environment 默认值；显式原生参数位于命令末尾。
- SFT YAML 的 `model/data/train/output` 仍由 SFT Trainer 读取；`launch` 只负责服务/环境选择，不复制一套训练设置。
- 不同模型家族需要选择相应的已安装环境，启动器不会自动切换或安装 Torch/vLLM。

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

源码应持久保存在 `autodl-fs`；需要时可在数据盘保留同版本运行副本。两者部署前后按源码 SHA256 清单核对。虚拟环境的 editable 安装和 `.pth` 必须指向实际运行副本；不能只更新文件而继续导入旧 FS 源码。

原始模型、数据集和结果通过 `TAU3_MODEL_ROOT`、`TAU3_DATA_ROOT`、`TAU3_RUN_ROOT` 指定，源码替换不删除这些资产。旧结果中的配置快照记录的是当时路径，不批量修改历史证据；再次执行旧命令时按本表迁移。
