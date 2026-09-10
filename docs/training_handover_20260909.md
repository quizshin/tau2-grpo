# 训练接管记录 · 2026-09-09

**最新4B主线：** SFT-003与RL-009已完成，SFT最佳验证loss 0.354217（第24步）；RL三步8/24官方成功，全部为LoRA更新，actor/vLLM同步通过。策略为4B，模拟器为27B INT4。最终4B RL检查点位于 `/root/autodl-fs/tau3_grpo_fix/results/rl009_4b_lora_qwen38_20260909/checkpoints/global_step_3`，SFT合并起点位于同级 `sft4b_lora_20260909/sft_merged_seed42`。训练服务已停止，实例未关闭；后续扩大预算前需处理GPU0仅约2.11GiB的实测余量。详见[4B实验记录](4b_lora_sft_rl_20260909.md)。下文0.8B恢复入口属于历史链路。

**历史云端实验变更：** RL-004 `i7utseim`、RL-006 `nqslriqa` 的 OOM 云记录已删除，原始日志与配置归档保留；清理时 RL 列表5个，随后新增RL-009，详见[SwanLab 清理记录](swanlab_cleanup_20260909.md)。

**后续存储变更：** 本轮结束后已完成[服务器整盘清理](server_cleanup_20260909.md)。RL-007 不完整 step2 已删除；数据盘六处旧模型/检查点改为持久副本链接，完整 step1、最终 step3 和实验记录保留。下文迁移/归档数量是历史快照。

来源：Codex 任务「检查 Qwen35 运行支持」`01a07f49-4f7a-7153-96b0-c7d4df29ec6c`、本地实验与迁移记录，以及本次 SSH 实机检查。新接管任务为 `01a08443-34bc-70f2-9cd2-5a63cc5c8c54`。

## 当前状态与目标

当前主线只使用 `tau3_grpo_fix/code` 中的标准 veRL。`code_agent`、`code_pytrio` 是其他实现，不作为本轮训练入口。最终希望以小模型验通 SFT → RL → 独立评估，再比较 full / LoRA RL，之后扩展到 Qwen3.5-9B；E0～E3 算法对比尚未完成。

| 阶段 | 已观察到的结果 | 接下来怎么用 |
|---|---|---|
| 0.8B 全量语言参数 SFT | 最佳 val loss 0.565518；严格内部评估 38/240（15.83%） | 保留为对照；视觉模块冻结 |
| 0.8B LoRA SFT | 最佳 val loss 0.529768；严格内部评估 54/240（22.50%） | 本轮 RL 起点；不将单轮差异推广为普遍优势 |
| RL-001、RL-002 | 各 3 步、24 条轨迹；reward、advantage、grad norm 全零，adapter 未变 | 仅用于诊断，不算有效学习 |
| 代码修复 | 用户场景 known_info、末轮执行/记录、工具伴随正文已修复；另修复 Qwen3.5 FSDP、模板及加载问题 | 已在 RL-003 观察到真实非零优势、梯度和同步 |
| 目录迁移 | 本地历史完整回归 440 passed；远程历史 437 passed / 3 skipped，收尾超时；独立入口 17 passed | 不能把历史超时命令写成完整正常退出 |
| RL-003 | 双 A800 三步完成，官方成功 3/24；各步 LoRA actor / vLLM 全部映射匹配 | 发现旧工具观察上限 256 字符，保留作诊断 |
| RL-005 | 工具观察 65,536 字符；三步完成、3/24 官方成功，2 条上下文超限；LoRA 同步全部通过 | 完整观察修复后仍需独立评估 |
| RL-004 / RL-006 | full 首批 ref log-prob OOM；仅启用状态 offload 仍不足 | 失败记录保留，不计入完成指标 |
| RL-007 / RL-008 | 分块修复后 full 首步非零更新与视觉冻结通过；第二步保存时磁盘满，已从完整 step1 恢复并完成 step2/3、退出 0 | 检查点改为直接写持久盘；最新状态见对照记录 |

完整证据与限制见 [EXPERIMENTS.md](../EXPERIMENTS.md)。SFT 两组学习率、参数精度不同，是两套配置比较。官方 Tau3 最终 50 题只在模型锁定后评估，不用于训练或选型。

本次开卡后的记录见 [RL 验证与对照](rl_validation_comparison_20260909.md) 和 [工具观察修复](observation_budget_20260909.md)。下文无卡检查与首次启动说明保留其历史时点；不能据此判断实例当前仍无 GPU。

当前续跑配置 `configs/train/rl/qwen35_full_rl008.yaml` 记录的是一次具体恢复：从持久盘 RL-007 的完整 `global_step_1` 恢复模型、优化器、调度器、随机和数据进度；后续检查点直接写 `/root/autodl-fs/tau3_grpo_fix/results/rl008_full_resume_20260909/checkpoints`，日志和轨迹留在数据盘同名 run。未来恢复应使用新的运行目录与选定完整检查点，不能盲目重复此已登记运行。

保留最近一个检查点也需要容纳新旧两份的峰值空间；该配置会在新检查点成功写完后才删除旧的。RL-007 的原始检查点及部分写入的 step2 已逐文件 SHA256 核验转存，后者仅供诊断，不用于恢复。

## 本轮最终检查点与后续恢复

截至 15:29，RL-008 退出 0，控制器已停止其拥有的训练/模拟器进程；未关闭实例。最新完整检查点为：

```text
/root/autodl-fs/tau3_grpo_fix/results/rl008_full_resume_20260909/checkpoints/global_step_3
```

持久盘直接写入的检查点和样例共 17 文件、10,471,292,718 字节，已逐文件计算 SHA256；这不是两份副本对拷核验。回载确认 320 份有效优化器状态均为 step3、矩有限，scheduler last_epoch=3、数据 `_num_yielded=3`，保留 RNG。模型 SHA256 为 `82ffd79cb718d894e7806ef32cc601e199554057fac17dd248b5ee4f5444704d`，与参数数值审计一致。原 RL-007 完整 step1 仍保留；其不完整 step2 不可用。

后续续训应创建新的实验配置和输出目录，固定新的训练任务计划和预算，再设置以下 veRL 参数；总更新目标必须大于 3，不能把它误当“额外更新数”。同时保留 SFT 起点作为固定 KL reference，并检查新计划与恢复的数据进度一致。

```text
trainer.resume_mode=resume_path
trainer.resume_from_path=/root/autodl-fs/tau3_grpo_fix/results/rl008_full_resume_20260909/checkpoints/global_step_3
trainer.del_local_ckpt_after_load=false
trainer.default_local_dir=<新的持久盘实验目录>/checkpoints
```

先用 `bash scripts/train/rl/run.sh --config <新配置> --experiment e0 --dry-run` 核对解析结果，再依照下文激活环境、启动和检查模拟器、运行新配置。不要直接重跑 RL-008：它固定从 step1 恢复且使用已完成的运行身份。持久盘目录下 `swanlab-artifacts/` 同样由 trainer 写入；数据盘 SwanLab 文件链接在归档时已实体化。

## 架构与执行路径

```text
configs/                模型、SFT/RL 方法、E0～E3、GPU、模拟器、SwanLab 参数
scripts/                data / train / eval / serve / maintenance 启动入口
tau3_grpo/
  training/sft/         数据标签、训练、LoRA 合并和模型导出
  training/rl/          算法注册，启动 veRL PPO trainer
  algorithms/           GRPO 适配、动态过滤、Tau-GiGPO 与状态锚点
  envs/                 Airline 任务、独立会话、用户模拟器与工具调用
  evaluation/           官方 reward、独立评估、报告
  data/                 AReaL 数据准备、固定划分、防泄漏
  models/               Qwen 兼容、下载和运行检查
  tracking/             SwanLab 指标、真实对话、源码快照
  integrations/         veRL 补丁契约和回调
  experiments/          运行清单和最终模型锁定
verl/                   训练、FSDP、vLLM rollout；含本项目适配
tau2-bench/             官方 Tau3 环境依赖，Python 包名仍为 tau2
```

RL 调用：`scripts/train/rl/run.sh` → `tau3_grpo.launch` 合并 YAML → `run_qwen35.sh` / `run_base.sh` → `training.rl.train` → `verl.trainer.main_ppo` → agent loop ↔ 工具与用户模拟器 → 官方评分 → 优势计算 → 参数更新 → actor 到 vLLM 权重同步 → SwanLab。

E0=GRPO，E1=GRPO+动态过滤，E2=Tau-GiGPO，E3=Tau-GiGPO+动态过滤。SFT 使用 LoRA 后导出合并模型，后续 RL 可以选择 full 或新建 LoRA；这两种选择互相独立。

Qwen3.5 使用原生 padded forward，关闭不安全的 packing/fused 路径，冻结视觉模块；工具使用 Qwen XML 格式，策略与 4B 模拟器关闭 thinking。9B 的 untied embedding、传输缓冲区和显存需要单独验收。

## 远程连接：本次已验证

实例：AutoDL A800 专区 051 机，容器 `autodl-container-857546be50-4fdafb0e`。SSH 私钥只引用路径，不复制私钥内容。

```bash
ssh -i /Users/apple/.ssh/id_ed25519_a800 \
  -o IdentitiesOnly=yes -o ConnectTimeout=12 \
  -p 28713 root@connect.nma1.seetacloud.com
```

自动化命令再加 `-o BatchMode=yes`，避免密码提示挂起。`scp` 使用大写 `-P 28713`；`rsync` 通过 `-e 'ssh -i ... -p 28713 -o IdentitiesOnly=yes'` 指定连接。当前密钥认证已成功，无需重新登录或重新提供密码。平台重建/迁移实例后应从控制台核对新的主机和端口。

进入当前训练运行环境：

```bash
source /root/autodl-fs/tau3_grpo_fix/activate.sh
source /root/autodl-tmp/tau3/runtime/qwen35/bin/activate
export TAU3_DATA_ROOT=/root/autodl-tmp/tau3/data
export TAU3_MODEL_ROOT=/root/autodl-tmp/tau3/models
export TAU3_RUN_ROOT=/root/autodl-tmp/tau3/runs
export TAU3_CACHE_ROOT=/root/autodl-tmp/tau3/cache
cd /root/autodl-tmp/tau3/runtime/code
```

第一条设置持久环境及数据盘缓存，第二条切到读取较快的数据盘 venv。非交互 SSH 不依赖 `.bashrc` 自动激活。当前 Qwen3.5 环境为 Torch 2.11.0、vLLM 0.20.0、Transformers 5.5.1、PEFT 0.18.1、SwanLab 0.10.0；旧 `a800` 的 Torch 2.8 环境是 Qwen2.5 配置，不能混用。

模拟器仅监听服务器本机的 `8100`，策略通过 `http://127.0.0.1:8100/v1` 使用。如需本地调试，可另开 SSH 隧道 `-L 18100:127.0.0.1:8100 -N`，本地访问 `http://127.0.0.1:18100/v1/models`；接管初检时无 GPU；随后本轮双卡实验已实际使用该服务。SSH 本地端口转发命令未另行验收。

## 存储与换机

| 内容 | 持久副本 / 工作位置 |
|---|---|
| 源码 | FS：`/root/autodl-fs/tau3_grpo_fix/code`；运行：`/root/autodl-tmp/tau3/runtime/code` |
| Linux Python、venv | FS：`runtime/python`、`runtime/venvs/qwen35`；运行 venv：`/root/autodl-tmp/tau3/runtime/qwen35` |
| 基础模型 | FS：`/root/autodl-fs/tau3_grpo_fix/model_store`；运行：`/root/autodl-tmp/tau3/models` |
| 数据 | FS：`/root/autodl-fs/tau3_grpo_fix/data`；运行：`/root/autodl-tmp/tau3/data` |
| SFT 起点 | `/root/autodl-tmp/tau3/runs/sft_compare_20260908/lora/sft_merged_seed42` |
| 新结果、缓存、临时目录 | `/root/autodl-tmp/tau3/{runs,cache,tmp}` |

本次 11:47（北京时间）实查：系统盘 84%（可用 5.0G），数据盘 53%（可用 24G），FS 14%（可用 173G）。原部署清单的 1,425 个文件在 FS 和运行源码中全部匹配。两组 SFT 导出和 0.8B/4B 权重文件、tokenizer 均存在；本次检查记录文件大小和配置 hash，没有重新计算全部大型权重 hash。

换机时挂载同一 FS、保持相同绝对路径，使用兼容 Linux x86_64 和 CUDA 13 / R580+ 驱动。先激活 FS 环境，执行 `bash /root/autodl-fs/tau3_grpo_fix/restore_models.sh` 恢复基础模型，再恢复数据、源码运行副本及选定的 SFT 成品。若使用缓存 venv，重新检查 editable 安装和 `.pth` 指向新源码，实际打印 `tau3_grpo.__file__`、`verl.__file__`、`tau2.__file__`。

数据盘结果不会自动持久化；在释放或更换实例前，把选定模型、适配器、日志、轨迹和配置同步到 FS 并校验。旧全量 SFT 优化器检查点已清理，不能精确断点恢复；保留的合并模型可开始新的 RL。恢复基础模型的脚本不等于恢复 SFT 成品。

Qwen3.8-27B-AWQ-INT4 历史上通过单 A800 推理验证，8 并发约 171 输出 token/s、整卡约 51.6GiB；这是旧实测。当前 FS 存在其模型目录，独立服务脚本/venv 在 `/root/tau3-staging/qwen38`，服务未运行，本次未重新验收该模型完整性。RL-003 仍使用 4B，避免同时改变模拟器。

## RL-003 的固定配置与开卡步骤

配置：[RL-003](../configs/train/rl/qwen35_lora_rl003.yaml)；[4B 模拟器](../configs/simulator/qwen35_4b_rl003.yaml)。沿用 RL-002 的 SFT 起点，seed 42、E0、新建 LoRA r16/alpha32、lr=1e-6、3 次更新、每步 2 组×4 条，GPU0 策略训练和 rollout，GPU1 模拟器。模拟器显存比例显式保持 0.60，策略 rollout 0.35；参数与优化器 offload 关闭。新增独立输出目录和权重/训练批次诊断记录，核心学习预算不变。

接管初检时（历史记录）是无卡启动：`/dev/nvidia[0-9]*` 为空，没有训练或模拟器进程。因此本次没有创建 RL 云实验、没有开始优化器更新，也没有排队或设置自动开卡。

开卡后先检查实际 GPU 与 CUDA：

```bash
nvidia-smi
python -c 'import torch; import vllm._C; assert torch.cuda.device_count() == 2; print(torch.__version__, torch.version.cuda); print(torch.ones(1, device="cuda"))'
python -m tau3_grpo.models.check_qwen35 \
  --model /root/autodl-tmp/tau3/runs/sft_compare_20260908/lora/sft_merged_seed42 --gpu

bash scripts/serve/simulator.sh --config configs/simulator/qwen35_4b_rl003.yaml --dry-run
bash scripts/train/rl/run.sh --config configs/train/rl/qwen35_lora_rl003.yaml --experiment e0 --dry-run
```

检查 GPU 无其他任务，启动模拟器并确认 `/v1/models` 返回 `Qwen/Qwen3.5-4B`，再做一次实际生成请求。确认后才启动 RL：

```bash
# 终端 A：先激活上述运行环境
bash scripts/serve/simulator.sh --config configs/simulator/qwen35_4b_rl003.yaml

# 终端 B：先激活同一运行环境；首次执行前确认输出目录没有旧实验
bash scripts/train/rl/run.sh \
  --config configs/train/rl/qwen35_lora_rl003.yaml --experiment e0
```

正式接管执行时使用独立后台进程或 tmux 保存 PID、stdout 和退出码，使 SSH 断线不影响任务；接管初检时以上配置仅完成预检；后续已执行的实验以顶部状态和对照记录为准。外部环境变量及 `.env` 会覆盖 YAML 默认值，因此实际启动前保存解析配置。重复实验使用新的输出目录和 run 身份，不覆盖历史运行。

验收顺序：真实工具和用户交互 → 每组 reward 分布 → 非零 advantage / grad norm → adapter 实际变化 → vLLM 收到更新后权重 → 下一批 rollout 使用新策略。只看到进程退出 0 或 checkpoint 文件存在不代表通过。现有权重审计可记录 rollout LoRA 缓冲区 hash，但仍需和实际 actor 更新对应核对。

若 24 条仍全零，逐条分类轮数、工具、任务结果错误并检查分组；全零或全一组都会导致 GRPO 无组内学习信号。先定位再决定任务预算或模型升级，不改官方 reward 以制造非零曲线。通过后比较相同 SFT 起点、模拟器、数据和评估预算下的 full / LoRA RL，再扩展到 9B。

## SwanLab 与维护入口

本次接管验证：本地相关回归 **94 passed，24.49 秒，退出 0**；远程 RL-003 和模拟器 dry-run 均正常退出，已核对 SFT 路径、三步预算、LoRA、offload 与模拟器 0.60 分配；远程相关回归 **240 秒超时、退出 124**，只输出部分进度，没有完整结果，不记为通过。新增配置和更新文档均同步到两处远程源码并通过 SHA256 校验。随后已完成实际 CUDA/服务/LoRA 训练验收，full 三步恢复链路也已完成，状态见顶部。

固定项目：[quizshin/tau3-grpo-pytrio](https://swanlab.cn/@quizshin/tau3-grpo-pytrio/runs)。RL 记录 `trainer.phase=rl` 并加入 `RL-E0` 分组，从“RL · 实验对比”查看。每次更新上传 scalar、真实 rollout 表格、`cases/batch`、`cases/preview_*` 和源码附件；抽样默认最多 4 条，完整轨迹另存本地。

`code/.env` 已在本地和远程配置，入口自动读取；不展示、不上传凭据。旧 smoke 实验 `bd743csc`、`8qja7qdf`、`vwg965wg` 已删除；旧文档里的相关链接和上传命令是历史记录，不要复用旧 run ID。

每次新训练都更新 [EXPERIMENTS.md](../EXPERIMENTS.md)，分别记录“已启动/已完成”和“是否有效学习”。本次原始清单、dry-run 和测试日志保存于本地 `tau3_grpo_fix/code/results/validation/takeover_20260909/`，远程预检日志为 `/root/autodl-tmp/tau3/runs/takeover_20260909/`。
