# 4B LoRA SFT → LoRA E0 / 27B INT4 模拟器（2026-09-09）

SFT-003 与 RL-009 均已完成、退出 0，两个阶段均使用 LoRA。RL 是 3 次更新、24 条训练轨迹的 E0/GRPO 工程验证，尚未进行独立任务评估。

## 固定设置与运行身份

| 阶段 | 设置 | 当前状态 |
|---|---|---|
| SFT-003 | Qwen3.5-4B，BF16 冻结基座，LoRA r16/alpha32/dropout0.05；45/5 完整对话，5 epoch、30 次更新，lr1e-4、seed42 | 已完成，恢复 checkpoint-24 并导出 |
| RL-009 | 合并 SFT-003 后新建 r16/alpha32 RL adapter，E0/GRPO、lr1e-6、seed42；3 步×2任务×4轨迹 | 三步完成，训练轨迹官方成功 8/24 |
| 用户模拟器 | Qwen3.8-27B-AWQ-INT4，独占 GPU1，16k、thinking off、0.65 显存预算、最多8并发 | 已完成本轮服务并由控制器停止 |

策略独占 GPU0，TP1 rollout 最大并发4，micro-batch1、梯度检查点、actor参数与优化器阶段间 CPU offload。策略上下文24576，prompt8192、交互16384、单轮1024、最多15轮，工具观察65536字符。没有改变官方 reward，也未使用官方最终50任务。

SFT 使用固定参考对话，不调用用户模拟器生成训练数据。4B 原生分词核验：训练45条、488834 tokens（监督97413），最长16473；验证5条、62000 tokens（监督15492），最长16702，全部保留。RL 起点由 SFT 验证 loss 选择；这不代替独立任务效果评估。

## 显存探针与熵统计修复

为了先检查 24k 路径，使用原始4B、新建LoRA和一条本项目先前保存的训练 token 序列做内存探针，形状1×24576、有效9772 token、助手监督3078 token；不是新的 RL 成绩，没有保存探针模型。未包含 FSDP/vLLM 常驻，所以不能当完整 RL 显存保证。

原始路径实测 allocated 峰值：旧策略概率+熵统计70,310,041,088字节（65.48GiB），参考概率21,804,700,672字节，反向+一次AdamW更新62,928,126,464字节（58.61GiB），梯度有限。反向探针约68.26秒；reserved峰值77,579,943,936字节。

新增熵统计修复仅在无梯度、三维 logits、序列超过128时按token切块，保留原softmax/logsumexp运算、dtype与autocast。反向路径保持原样。相同4B输入的旧策略阶段 allocated 峰值降至21,869,699,584字节（20.37GiB）。新探针没有重复反向阶段；随后 RL-009 已通过三步完整更新。

本地20项回归通过；远程20项通过。CUDA FP32/BF16/FP16、有/无autocast六组输出逐元素完全一致；1×1024×16384的BF16 autocast受控探针，临时峰值由约192MiB降至24MiB。生产改动在 `verl/verl/utils/torch_functional.py`，独立回归为 `tests/test_entropy_memory.py`。

4B 的GDN Q/K/V投影宽度不等（2048/2048/4096），RL-009 的适配器核对脚本使用累计分块偏移，不能沿用0.8B等宽切分假设。三个步骤的496个actor适配器张量均覆盖640个vLLM缓冲区，按BF16、alpha/r与分块映射后hash全部一致。

## 实际结果

SFT 初始验证 loss **0.5577036142**，最佳/恢复后 **0.3542169929**；训练 loss 0.3316994404。30次更新耗时4281.02秒（71.35分钟），最佳为第24步。最终导出适配器与 checkpoint-24 的496个张量逐元素完全一致且有限；合并模型文件哈希清单已保存。SFT Torch allocated 峰值24.03GiB。

| RL 步骤 | 任务 | 官方成功 | 有组内奖励差异的任务组 | 梯度范数 | 整步耗时 |
|---|---|---|---|---|---|
| 1 | 709 / 1072 | 6/8 | 2/2 | 0.311708 | 17.85分钟 |
| 2 | 760 / 871 | 1/8 | 1/2 | 0.214437 | 22.90分钟 |
| 3 | 542 / 382 | 1/8 | 1/2 | 0.140381 | 20.30分钟 |

三步总计3662.70秒（61.05分钟），每次actor更新约552秒，平均每8条轨迹整步20.35分钟；不含服务/模型启动与事后归档。失败分布为任务结果错误8、轮数上限7、工具错误1。第一步优势范围[-1.499997,0.499999]，第二、三步[-0.499999,1.499997]；各步old/ref log-prob均有限。六个任务组中四个有奖励差异，两个全零。

整卡5秒采样峰值 GPU0 **79,755MiB（77.89GiB）**、GPU1 **51,119MiB（49.92GiB）**。GPU0仅约2.11GiB剩余，尚未达到8GiB部署余量目标；本轮通过不保证更长有效轨迹或更高并发仍有足够余量。策略24576 token预算没有缩短。24k是每条序列长度上限，与本轮每步8条、总计24条轨迹分别计量。

这是已验证非零奖励优势、LoRA更新及实际推理权重同步的短程实验。**8/24不是独立评估成绩，也不能证明RL提升**：首批采样在第一次更新前，三批任务不同，同时相较旧实验更换了策略规模和模拟器。官方reward也不保证覆盖所有对话事实错误。正式40/60步、128条/步和最终50题均未执行；不能把当前8条/步耗时直接当128条/步耗时。

## 保存与控制

- SFT运行日志：`/root/autodl-tmp/tau3/runs/sft4b_lora_20260909`。
- SFT适配器、检查点和最佳模型导出：`/root/autodl-fs/tau3_grpo_fix/results/sft4b_lora_20260909/{adapter,sft_merged_seed42}`。
- RL运行日志：`/root/autodl-tmp/tau3/runs/rl009_4b_lora_qwen38_20260909`。
- RL检查点：`/root/autodl-fs/tau3_grpo_fix/results/rl009_4b_lora_qwen38_20260909/checkpoints`，新旧两份空间已预留；每步单独保存适配器快照。
- SFT控制器PID65160，RL控制器PID66057，仅描述此次启动。控制器只停止自己拥有的进程，不关闭实例。SFT失败时RL不会启动。
- SFT云实验：[vr3e6ezj](https://swanlab.cn/@quizshin/tau3-grpo-pytrio/runs/vr3e6ezj)。RL云实验：[6rwf81a4](https://swanlab.cn/@quizshin/tau3-grpo-pytrio/runs/6rwf81a4)，FINISHED，三步关键指标与6条实际预览正文回读核对通过。

收尾时发现训练退出0后云端最后一次上传未完整结束。先把原始SwanLab日志同步回同一run ID，恢复第三步指标，再从原始抽样JSONL恢复第三步媒体并关闭云记录。SDK resume曾清空配置展示，已按训练时 `files/config.yaml` 恢复全部原值并逐项回读核对；未重写scalar数值。相关操作和原始失败证据保留在运行目录。归档脚本另增加了对两份确切SFT输入文件链接的识别，复制时仍核对SHA256。

最终验收于18:54完成：最新检查点标记为3；496份活动优化器状态均为第3步、动量有限，调度器与数据进度均为3，随机状态存在。持久检查点目录21个文件共10,893,565,422字节已计算SHA256。运行资产分别归档SFT 37个文件（5,428,434字节）、RL 118个文件（431,255,689字节），复制前后哈希全匹配；直接写持久盘的模型资产与复制回执分别计量。仅最近step3保留完整RL恢复状态，每步LoRA快照均归档。

本地小型证据：[SFT完成审计](../results/validation/sft4b_lora_20260909/sft-completion-audit.json)、[RL报告](../results/validation/rl009_4b_lora_qwen38_20260909/validation-report.json)、[LoRA同步](../results/validation/rl009_4b_lora_qwen38_20260909/adapter-sync-verification.json)、[恢复状态与文件清单](../results/validation/rl009_4b_lora_qwen38_20260909/persistent-checkpoints-verification.json)、[云端回读](../results/validation/rl009_4b_lora_qwen38_20260909/cloud-content-verification.json)、[最终状态](../results/validation/rl009_4b_lora_qwen38_20260909/finalization-status.json)。完整权重、debug批次与适配器快照保留远程持久盘，不下载到本机。

入口配置为 `configs/train/sft/qwen35_4b_lora_20260909.yaml`、`configs/train/rl/qwen35_4b_lora_rl009.yaml` 与 `configs/simulator/qwen38_27b_rl009.yaml`。当前阶段只验证三步RL，不自动扩展为128条×40/60步的数天实验；正式规模需先参考本轮实际显存、奖励与耗时。
