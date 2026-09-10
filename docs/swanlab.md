# SwanLab：SFT 与 RL 指标和样例

> 历史 smoke runs `bd743csc`、`8qja7qdf`、`vwg965wg` 已按要求删除；本文下方的相关运行链接与命令仅作历史记录。新训练使用新的 run，当前状态见 [接管记录](training_handover_20260909.md)。

固定 SDK `swanlab==0.10.0`。新环境通过项目的 `tracking` extra 安装。
SFT 入口默认 `--report-to swanlab`，RL 入口默认 `trainer.logger=[console,swanlab]`。
入口自动读取 `code/.env`（python-dotenv，不执行 shell），显式环境变量优先。
凭据沿用现有 pytrio 的 SwanLab 账户，项目为 `quizshin/tau3-grpo-pytrio`。
`code/.env` 权限为 0600，Git 和源码上传均排除它；不复制无关服务的密钥。

## 环境与项目

```bash
source /root/autodl-fs/tau3_grpo_fix/activate.sh
export TAU3_SWANLAB_PROJECT=tau3-grpo-pytrio
# 团队空间按需设置；不设置则使用个人空间。
# export SWANLAB_WORKSPACE=your-workspace
export SWANLAB_LOG_DIR=/root/autodl-fs/tau3_grpo_fix/results/swanlog
export SWANLAB_MODE=online
```

实验名沿用 `code_pytrio` 的阶段、模型、方法、学习率、预算、种子格式，例如
`sft-airline-Qwen3.5-0.8B-lora16-lr1e-4-ep5-s42`。全量标记为 `full`；
RL 根据最终解析的配置命名，合并 LoRA 后进行全量 RL 时会标记 `full`。
SFT 曲线使用 `train/loss`、`train/lr`、`train/grad_norm`、`val/loss`，横轴
`train/global_step`；RL 使用 veRL 原生指标，横轴 `trainer/global_step`。
适用于 Qwen2.5 和 Qwen3.5 系列，具体模型兼容范围见 Qwen 文档。

项目的两个原生对比视图按配置 `trainer.phase=sft` / `trainer.phase=rl` 筛选，
不是按名称或标签。SFT 和 RL 入口现在均写入该字段；RL 同时记录 `RL-E0` 等
group 和 RL/veRL/方法标签。打开单实验链接不自动切换视图，比较时请从项目
`/runs` 页面选择“RL · 实验对比”。Ray 内训练异常会显式标记 SwanLab 失败，
避免 logger 析构函数把未完成的验证误标为完成。

本地离线检查可用 `SWANLAB_MODE=offline`。离线写入不会上传；必须登录并使用
`swanlab sync <run-directory>` 或在线执行上传命令，才能在网页看到。

## 记录内容

| 阶段 | 曲线/指标 | 样例与附件 |
|---|---|---|
| SFT | 按真实 optimizer step 记录 loss、验证 loss、学习率、梯度等；结束后记录最佳验证、参数量、训练时间、峰值显存 | 原始训练/验证参考对话各 3 条，明确标记为 reference；配置、训练状态、源码快照 |
| RL | veRL 已生成的 reward、actor loss、KL、梯度、吞吐、耗时、动态过滤、GiGPO/anchor 指标等 | 每次更新抽取真实 rollout，保留角色、工具参数、工具返回、reward、终止原因；完整所选 JSONL 附件 |
| SFT 后评测 | 最佳导出模型的完整预算成功率、正常结束数、上下文失败数、严格回放错误数 | 12 条按成功/失败抽样的生成对话；全部评测轨迹压缩附件；两方法对比报告 |

RL 默认每 1 次更新上传最多 4 条，排除 padding；使用独立随机数生成器确定性抽样，
不改变训练随机数状态。可通过 `SWANLAB_ROLLOUT_INTERVAL` 和
`SWANLAB_ROLLOUT_SAMPLES` 调节。表格每条预览最多 24,000 字符并标明截断；
附件保留选中轨迹全文。原生验证样例开关仍是 `trainer.log_val_generations`；
项目的训练轨迹上传不依赖开启定期验证。

同一更新还写入兼容 pytrio 的 `cases/batch`（真实非 padding 批次摘要）与
`cases/preview_1..4`（抽样中不同任务的文本），保留原 `train/rollout_examples`
表格。样例在参数更新后上传；若首次更新失败，该实验可能只有配置而没有曲线
和样例。`trainer.phase=rl` 负责加入对比视图，媒体字段负责显示具体内容。
`trajectory_json` 若仅包含 DB hash/session 等元数据，会保留为 metadata，并
从实际 token IDs 解码对话；按 attention mask 去掉 padding，保留不参与 loss
的用户和工具轮次。抽样优先覆盖不同任务，同时尽量保留成功/失败案例。

本次已启动的早期 RL 进程只上传了 metadata 预览，可在实验完成后从原生
`rollouts/1.jsonl` 等文件补传正文：

```bash
python -m tau3_grpo.tracking.upload_rl \
  --run quizshin/tau3-grpo-pytrio/bd743csc \
  --rollout-dir /root/autodl-tmp/tau3/runs/rl_qwen35_lora_20260908/smoke_e0_seed42/rollouts \
  --output /root/autodl-tmp/tau3/runs/rl_qwen35_lora_20260908/smoke_e0_seed42/recovered-examples
```

补传使用同一实验的新媒体字段 `cases/recovered_dialogues` 与
`cases/recovered_preview_*`，保留原更新步号和全部原始 JSONL，避免 SDK 拒绝
覆盖已经写入的媒体步号。只恢复案例，不补造 scalar 或额外训练步；完成回执
要求云端媒体步号已到达且原训练指标未改变。`--dry-run` 只读本地数据。

源码附件 `source-sft-rl.zip` 包含项目 SFT/RL 源码、配置、脚本和使用到的
veRL/tau2 源码。只归档指定源码目录，不包含模型、数据盘、运行日志、环境目录、
隐藏凭据文件或软链接。历史补传的源码是**补传时的代码**，包含训练后修复；
训练配置和原始 step 曲线来自保存的训练结果，二者不会混称为同一时间的快照。

## 补传本轮两组 SFT

```bash
cd /root/autodl-fs/tau3_grpo_fix/code
python -m tau3_grpo.tracking.upload_sft \
  --results-root /root/autodl-fs/tau3_grpo_fix/results/sft_compare_20260908 \
  --require-evaluation --dry-run

python -m tau3_grpo.tracking.upload_sft \
  --results-root /root/autodl-fs/tau3_grpo_fix/results/sft_compare_20260908 \
  --require-evaluation
```

分别创建全量与 LoRA 实验。保留第 0 步基座验证和 30 步训练曲线；同一 step 的
训练/验证指标合并后上传。成功的在线补传会在各方法的 `swanlab-upload/`
写入 `upload-complete.json`，包含网页地址，重复执行会读取回执而不重复上传。
离线回执单独命名，不会误当成已上传。

本轮只接受隔离数据库后严格重评分的 `selection-isolated/`，拒绝原始虚高分数。
上下文超限按失败纳入 240 次尝试的分母；SFT 参考答案与模型生成样例分别展示。

需要关闭跟踪时，SFT 传 `--report-to none`；RL 设置 `TRAINER_LOGGERS='[console]'`。
本轮 RL 云端实测见下文；源码接入与执行链路验证不等于已经产生有效学习。

## 本轮上传与验证

- 全量：[sft-airline-Qwen3.5-0.8B-full-lr2e-5-ep5-s42](https://swanlab.cn/@quizshin/tau3-grpo-pytrio/runs/jt7m41hy)
- LoRA：[sft-airline-Qwen3.5-0.8B-lora16-lr1e-4-ep5-s42](https://swanlab.cn/@quizshin/tau3-grpo-pytrio/runs/x8l5l11u)
- RL：[0.8B LoRA 三步验证](https://swanlab.cn/@quizshin/tau3-grpo-pytrio/runs/bd743csc)

RL 已完成三步，云端回读核对 12 个关键 scalar 的 step 1/2/3。24 条轨迹
reward、advantage 均为 0，梯度为 0，adapter 未改变；不能据此声称学到有效
策略。初次失败的 `vwg965wg` 已标记 CRASHED；新旧运行均归入 RL 对比视图。
真实正文已从三批原始 rollout 补传到 `cases/recovered_dialogues` 和
`cases/recovered_preview_*`，按原步号展示；原训练 scalar 点保持不变。
补传还上传完整 24 条 JSONL 和修正日志逻辑后的源码，后者在 `logging-code/`
下，与原训练时的源码快照区分。

两组 SFT 均已在线上传，云端回读验证了 30 个训练 loss 点、6 个验证 loss 点、
第 0–30 步横轴和完整预算成功率，并查询到生成对话表格。SDK 确认每组
同步 11 个附件，含源码、完整评测轨迹和对比报告。源码 ZIP 共 817 个文件。
本地完整测试：404 passed；包含原生 veRL/SwanLab 离线指标、表格和附件集成。

`.env.example` 提供无凭据模板。项目字段采用 `TAU3_SWANLAB_PROJECT`，
避免 SwanLab 0.10 将 `.env` 中的 `SWANLAB_PROJECT` 当作嵌套 JSON 解析。
