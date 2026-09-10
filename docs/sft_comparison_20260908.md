# Qwen3.5-0.8B：全量 SFT 与 LoRA 对照

状态：两组训练、生成评测和独立数据库严格重评分均已完成；本轮建议选 LoRA 继续。

## 固定设置

- 同一 Qwen3.5-0.8B 固定权重版本；冻结原始 45 条训练、5 条验证完整对话。
- seed 42；micro-batch 1，梯度累计 8；5 epoch，共 30 次 optimizer update。
- 最大长度 24,576；完整保留上下文和工具 schema，仅监督 assistant 内容、工具调用和 EOS。
- 训练前验证、每个 epoch 验证；按验证 loss 保存并导出最佳 checkpoint。
- GPU 0 / GPU 1 均为 A800 80GB；两组同时训练，模拟器在 SFT 完成后启动。

| 项目 | 全量 SFT | LoRA SFT |
|---|---|---|
| 可训练参数 | 752,393,024 | 10,822,656 |
| 范围 | 全部语言参数，含 embedding / LM head | 语言层线性模块，含 Gated DeltaNet |
| 视觉模块 | 冻结 | 冻结 |
| 学习率 | 2e-5 | 1e-4 |
| 数值精度 | FP32 参数/Adam 状态，BF16 autocast | BF16 冻结基座，PEFT 可训练参数 |
| LoRA 配置 | 不适用 | r=16、alpha=32、dropout=0.05 |

这是两套对应训练配置的比较，学习率不同。验证集仅有 5 条，因此还要用同一
Qwen3.5-4B 模拟器、同一内部 selection 集和评测随机种子比较任务效果。
不使用官方最终 50-task 测试集选择 SFT 方法。

## GPU 启动时修复的问题

1. 原始全量前向为所有 token 建立大词表 logits，在第二步反向传播时因大块内存分配失败而 OOM。
   当前仍让完整对话进入 transformer，但只将有监督的 next-token 位置送入 LM head；
   原生 loss 与所有参数梯度的等价性已覆盖全量/LoRA、共享/独立 embedding 两种布局。
   两组均使用 `PYTORCH_ALLOC_CONF=expandable_segments:True`。
2. Transformers 5.5.1 与 Accelerate 1.12.0 在该路径重复除以梯度累计次数。
   微型模型实测：累计两次时，参数更新只有直接 batch 的一半。
   现在仅由 Trainer 按实际累计窗口归一化；直接 batch 与累计更新（含不足一整窗的尾批）通过回归检查。
3. 上述修复后的本地完整测试：389 passed。前两次启动为排查尝试，不纳入效果比较；
   当前两组均于 2026-09-08 18:30:56 重新从相同基础权重开始。
4. 收尾发现 Transformers 5.5.1 的默认反向权重名称转换与 Trainer 的直接
   `load_state_dict` 不兼容，全量组未能自动恢复最佳模型。现在训练 checkpoint
   和 BF16 导出均保存原生名称，并检查回载后的验证 loss 必须匹配最佳记录。
   已从保留的 `checkpoint-12` 恢复全量模型，实测 loss 为 0.5655176043510437，
   与原记录完全一致，无需重训。新增全量/LoRA、共享/独立 embedding 的真实回载测试；
   最新完整测试为 393 passed。
5. 任务轨迹抽查发现注入的 Airline `FlightDB` 被模型回放与标准答案回放共享，
   会使不同操作的最终 DB hash 错误相等。已在 Airline 环境构造时深拷贝注入的 DB，
   不改变官方评分器、工具或 reward basis。真实“漏做取消 / 正确取消”回放测试通过，
   最新完整测试为 395 passed。两组保存的轨迹统一进行严格回放重评分，
   使用 `evaluation/selection-isolated/` 中的结果；原始 `selection/` 分数不用于比较。
   抽查样例 `airline_740` 原 reward=1，隔离后为 0，证明原始虚高分数不可用。

## 已完成的训练结果

| 指标 | 全量 SFT | LoRA SFT |
|---|---:|---:|
| 初始验证 loss | 0.753222 | 0.751903 |
| 最佳验证 loss | 0.565518 | 0.529768 |
| 最佳 epoch / step | 2 / 12 | 5 / 30 |
| 第 5 轮验证 loss | 0.619913 | 0.529768 |
| 训练时间（分钟） | 38.68 | 37.55 |
| 峰值已分配显存（GiB） | 26.13 | 14.99 |
| 峰值预留显存（GiB） | 30.36 | 18.08 |

两组初始 loss 的小差异来自全量 FP32 权重配 BF16 autocast、LoRA BF16 基座的精度差异。
全量第 2 轮以后验证 loss 回升；LoRA 在这 5 条验证对话上更好。
以下内部 selection 评测同时用于判断任务效果。

## 内部 selection 评测（已完成）

固定 60 个任务，每任务 4 次，seed 42–45，最大 30 步，并发 4；策略上下文
上限 24,576 token，模拟器上限 16,384 token。上下文超限明确计失败，保留在
240 次尝试的总分分母中；其他运行异常会阻止有效结论。全部有效轨迹使用
隔离 DB 的官方评分器严格重放，不调整工具或 reward basis。

- 全量：38/240 成功（15.83%）；228 条有效轨迹，12 次上下文超限，0 个重放错误。
- LoRA：54/240 成功（22.50%）；233 条有效轨迹，7 次上下文超限，0 个重放错误。

LoRA 比全量高 6.67 个百分点。按任务进行配对 bootstrap（每任务 4 次保持同组，
10,000 次重采样），全量减 LoRA 的差值为 -6.67 个百分点，95% 区间
[-13.75, 0.00] 个百分点。区间触及零，因此不能声称任务优势已被统计上明确证实。
结合较低验证 loss、较少显存和本次成功率，建议先选 LoRA；结论仅适用于本轮
小数据和这两套不同学习率的配置，不推断所有模型大小或最优调参后的比较。

全量原始评分受 DB 共享缺陷影响，141 条有效轨迹的分数在修复后改变。
比较仅使用 `selection-isolated/` 的完整预算分数。`selection/summary.json` 的
原始 `mean_reward` 还会排除运行失败，不能直接用作本实验最终指标。

## 结果位置

以 `/root/autodl-fs/tau3_grpo_fix` 为文件存储根：

- 日志、PID 和阶段退出码：`runtime/validation/sft-compare-20260908/`。
- 最佳 checkpoint、BF16 导出及训练统计：`results/sft_compare_20260908/{full,lora}/`。
- 当前数据盘运行目录：`/root/autodl-tmp/tau3/runs/sft_compare_20260908/{full,lora}/`。
- 中间 checkpoint 和优化器状态暂存在数据盘；最佳模型、`train_summary.json`、
  `trainer_state.json` 和导出模型会自动同步到文件存储。

比较指标包括初始/最佳验证 loss、任务平均 reward / solve rate、运行失败数、
训练时间和峰值显存。完整机器可读报告：`results/sft_compare_20260908/comparison-report.json`。

训练曲线、任务样例和源码附件已补传至同一 SwanLab 项目，链接与验证说明见
[SwanLab 文档](swanlab.md)。最终本地完整回归测试 404 passed。
