# MT-GTPO split-v4 未校准探索入口

## 最终结果（2026-09-22）

后续[本地审计](../results/analysis/mt_gtpo_v4_collapse_20260922/report.md)定位到第 25–27 步已出现大量生成长度终止，以及全失败批次仍由过程优势更新。需更正根因表述：`context_window_exceeded` 混合了单轮 1024 token 截断与真正的上下文不足，不能直接把 217 条都解释为用满总上下文；47.84 是裁剪前梯度范数。原始轨迹/批次与权重同步审计尚未完成，用户已要求先审计本地，具体诱因未定。

30 步及三次 selection60×4 评测已全部结束，控制器状态 completed。成功数依次为 119/240（49.58%）、101/240（42.08%）、6/240（2.50%）；第 30 步比 A45 SFT 110/240 低 43.33 个百分点，比 GRPO 第 30 步 130/240 低 51.67 个百分点。2026-09-22 17:57 左右独立回读 SwanLab，云端 1–30 步及三点评测与本地结果一致。

第 30 步有 217/240 条因 context_window_exceeded 提前终止，其余 23 条为 user_stop；提前终止按全量评测口径计零，不从分母中剔除。第 20 步仅有 17 条超上下文、7 条轮数上限。末段训练也出现异常：第 27/28 步训练终局奖励均值为 0.03125/0.0625，第 29/30 步为 0；第 30 步梯度范数 47.84、actor/ppo_kl 0.359。不同步的训练任务不同，这些信号用于定位异常，不能单独确定因果。当前证据表明本次未校准探索末段性能严重退化，不能宣称增益；具体根因尚待轨迹与更新审计，不建议直接续训该配置。

最终小型回执位于 `results/runs/mt_gtpo_5090_a45/v4-explore-u30-01/`：completion.json、metrics.jsonl、evaluation-summary.json、cloud-final-readback.json。以下为启动过程的历史记录。

> 11:18 启动复核：模拟器健康检查通过，8 卡模型与 rollout 服务已加载，控制器处于 `training`，完成更新仍为 0/30。新 [SwanLab 实验](https://swanlab.cn/@quizshin/tau3-grpo-pytrio/runs/172a7b7987984f52af8cd) 已独立通过 API 回读，云端配置确认为 `uncalibrated_exploration`、`irc_calibrated=false`、30 步、save/test=10；此回读验证的是配置，不是尚未产生的训练指标。回执为 `cloud-startup-readback.json` 和 `startup-verification.json`。

> 执行状态更新（2026-09-22 11:09，北京时间）：用户后续明确授权后，已部署到 grpo-long / ackcs-00gjhp53，远端 45 项 CPU 测试和真实 dry-run 均通过，并已启动 30 步训练。下面“未部署/未启动”的文字描述初版交付时状态。当前启动验证仍在进行，不能据此声称已完成 GPU 更新或续训验证。

运行目录为 `/root/shared-nvme/tau3/runs/mt_gtpo_5090_a45/v4-explore-u30-01`，控制器 PID 18853，后台独立会话运行，SSH 断开不影响训练。部署前核对 1,156 文件基线，只同步 runner/profile；原件备份位于共享盘 `bootstrap/uncalibrated-exploration-20260922/before.tgz`。

预检确认 A45 新起、30 步、8×8 候选/步、8 ranks、FSDP2 offload、save/test=10、最新完整检查点保留 1 份、DF off、turn_v1 及未校准标记。小型回执保存在 `results/runs/mt_gtpo_5090_a45/v4-explore-u30-01/`。没有配置墙钟超时，6–7 小时仅为估计。旧 GRPO 关机自动检查保持暂停，不用于此新实验；本次未配置平台自动关机。

2026-09-22。用户要求增加支持，由用户在 GPU 上运行探索实验。本次修改本地代码及 CPU 检查，不启动 GPU 服务。新开关 `--uncalibrated-exploration` 显式允许原 `turn_v1 + paper_env_split_v4` 使用当前配置中的未校准奖励，支持 10 的正整数倍更新及同一实验的整十步续训。未提供这个开关时，原正式训练仍要求通过校准的冻结配方。

## 本次实验

- A45 SFT 重新初始化；不能接着 call_local_v1 的 step 2 或 GRPO 检查点换算法续训。
- 八张 5090，原 FSDP2/shared8 路径；train50，每步 8 组×8 条，30 步共 1,920 条候选轨迹。
- 保持原 turn_v1 优势，gamma=0.9、lambda_outcome=0.3、DF off；不加入本地研究的轮内残差候选。
- 使用 split-v4 默认初始权重，gold_write=1、error/state_change=-0.1，其他已配置类别为 0；不是之前拟合未通过的候选权重。
- 使用 token budget v1、温度 0.7、学习率 1e-6、KL 系数 0.01；每 10 步保存完整检查点并执行 selection60×4。只保留最新完整检查点，历史评测与轨迹保留。final50 不用于此次选择。
- 探索结果只能作为未校准经验对照。按已完成 GRPO 的硬件用时，预留约 6–7 小时供参考，MT 实际耗时另行记录；30 步是本命令的硬更新预算，不是额外墙钟自动中断保证。

## 部署与启动

此文档生成时未同步到服务器。对于已经完成 `mt_call_local_smoke/test01`、具有全部现有依赖的 5090 代码，只需同步本轮运行所需两文件：

```text
tau3_grpo/training/rl/runner.py
configs/train/rl/formal50_5090_a45_mt_gtpo_v4.yaml
```

`results/analysis/uncalibrated_exploration_20260922/deploy/` 保存上述文件、SHA256 清单及压缩包；不包含模型、环境、轨迹或凭据。覆盖前检查远端未提交修改并备份两文件，覆盖后核对清单。如果远端不是已完成两步 smoke 的那份代码，应先核对依赖，不能把两文件当独立项目运行。

在 5090 已激活环境的 bash 中，先组成唯一一套参数并做无服务预检：

```bash
source /root/shared-nvme/tau3/activate.sh
cd "$CODE_ROOT"
export TAU3_ENV_FILE=/root/shared-nvme/tau3/config/swanlab.env

exploration_args=(
  --estimator mt_gtpo
  --profile configs/train/rl/formal50_5090_a45_mt_gtpo_v4.yaml
  --reward-version paper_env_split_v4
  --credit-mode turn_v1
  --token-protocol tau3_token_budget_v1
  --uncalibrated-exploration
  --updates 30
  --result-dir "$TAU3_RUN_ROOT/mt_gtpo_5090_a45/v4-explore-u30-01"
)
python -m tau3_grpo.training.rl.runner "${exploration_args[@]}" --dry-run
```

预检核对 A45 模型、train50、30 updates、64 candidates/update、8 ranks、save/test=10、DF off，以及未校准标记。预检成功后，在同一个 shell 手动启动：

```bash
python -m tau3_grpo.training.rl.runner "${exploration_args[@]}"
```

可在已有 tmux 会话里执行以避免 SSH 断开影响前台进程。本命令启动真实 GPU 训练；本次代码修改没有执行该命令。不要添加 `--engineering-smoke` 或 `--reward-recipe`。使用新的 run 目录；只做过 dry-run 的目录可直接用于首次启动。控制器仍检查 GPU 空闲、端口、磁盘余量和既有训练状态。

## 结果标识及续训

SwanLab/trace 实验名带 `uncalibrated-exploration`；实际 Hydra 保存 `tau3_experiment_kind=uncalibrated_exploration`、`tau3_irc_calibrated=false`。`launch.json` 记录 `uncalibrated_exploration=true`、`uncalibrated_initializer=true` 和实际奖励/算法设置，`preflight.json`、控制器状态和完成回执带探索/校准状态。不会生成冒充校准通过的 frozen recipe。

需要正常停止时，在本 run 目录创建 `STOP_AFTER_BOUNDARY`；现有控制器会在整十步保存和评测后退出。恢复前先检查并归档停止标记，核对最新完整检查点与 SwanLab 身份。使用原参数、原结果目录和原开关，加 `--resume-from "$TAU3_RUN_ROOT/mt_gtpo_5090_a45/v4-explore-u30-01/global_step_10"`（步数按实际已完成节点填写），`--updates 30` 仍指累计目标。

续训只接受同一探索 run。不能切换 credit mode、奖励版本、DF、token 协议、实际奖励权重/选项、gamma/lambda 等优势设置，也不能把探索 run 转成校准 run。不同实验应从共同起点新建目录。恢复时奖励/算法检查通过之后才替换已保存的 Hydra 配置。

## 本地验证范围

真实 shell→Hydra 组合对照验证：除探索标记、实验名称及名称插值产生的 trace 标签外，全部训练配置与旧版未校准 dry-run 一致。测试覆盖 30 步预算、每 10 步保存/评测、A45 路径、8 卡、DF off、拒绝混用开关/算法、续训身份及有效奖励/优势设置变化、CLI dry-run 元数据与不启动服务。

最终相关 CPU 集合 **64 passed、0 failed、0 skipped**，覆盖新入口、旧 smoke、GRPO/GiGPO/MT-GTPO 历史配置对照、SwanLab 连续性和整十步预算控制；lint 无新增，`git diff --check` 通过。

CLI 编排测试中的模型/数据预检与远端 Hydra 子进程使用替身，Hydra 组合单独真实执行；本地不声称完成远程模型资产预检、GPU 更新或 GPU 恢复验证。检查日志保存在 `results/analysis/uncalibrated_exploration_20260922/`。首次测试未将虚拟环境加入 shell PATH，导致 Python 子进程找不到；补 PATH 后发现测试需同时排除名称插值生成的 trace 标签，修正测试后再执行相关集合。完整结果见该目录 `pytest-final.log`。
