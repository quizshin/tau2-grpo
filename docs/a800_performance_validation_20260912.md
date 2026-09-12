# A800 方案 2：保留 5,120 条轨迹的性能验证

本次只做短测试，没有恢复或启动正式 E0，也没有生成新的完整模型检查点。
结论按数值验证结果选择，不按最快的一项选择。通过分量测试不等于完整在线 RL 验收通过。

## 环境与复现范围

- 基线代码 `30a68b0`；独立分支 `perf/a800-e0-20260912`，未覆盖主工作目录的其他修改。
- 远程独立代码 `/root/autodl-tmp/tau3-perf-20260912/code`。
- 原始结果 `/root/autodl-fs/tau3-core-20260912/runs/performance-20260912`。
- 本地结果副本 `results/a800-performance-20260912`，不提交原始轨迹或张量。
- 4 × A800 80GB PCIe 做 FSDP 回放，第 5 张卡单独做采样测试；未启动模拟器。
- Python 3.12 / Torch 2.11+cu130 / Transformers 5.5.1 / vLLM 0.20。
- 模型：四模型合并结果中的 `new-off`，Qwen3.5-4B，新版非 thinking SFT。
- 语言参数全量训练、视觉冻结；FP32 主参数、BF16 计算，LM head 可训练。
- FLA 0.5.2 只安装在独立 overlay，未改主虚拟环境。没有安装 causal-conv1d。

正式预算保持 40 × 16 × 8 = 5,120 条轨迹。测试使用此前验收保存的真实
`update_000001.pkl`：8 条轨迹，4 个 rank 各处理 2 个 micro-batch，输入均为
`[1,24576]`，沿用真实 token、response mask 和 advantage。

`performance_probe.py` 回放 old log_prob、同初始权重的 reference scoring、
PPO 类 clipped loss 加 KL 的前反向、AdamW 更新。它不包含在线采样、用户模拟器、
独立 reference 副本、veRL 完整 loss scaling/offload 调度或同卡 vLLM。
因此下表显存是这一分量的 PyTorch allocated 峰值，不是完整 RL 的 nvidia-smi 显存。
真实模型梯度只采样每个参数分片最多 256 个值；小模型测试比较完整梯度。

## 计算与显存

时间为四 rank 同步后的最大值；显存取四 rank 的最大 allocated 峰值。

| 实现 | old log_prob | reference scoring | 前向与反向 | 前反向显存 | 数值结论 |
|---|---:|---:|---:|---:|---|
| 原实现 | 14.67 s | 14.18 s | 144.13 s | 59.81 GiB | 基线 |
| 只分块输出层 | 14.13 s | 13.67 s | 139.48 s | 31.66 GiB | 通过 |
| 分块输出层＋FLA 默认归一化 | 43.46 s（首次编译） | 5.82 s | 85.46 s | 22.40 GiB | 未通过 |
| 分块输出层＋FLA、原生归一化 | 7.66 s | 6.00 s | 31.09 s | 22.40 GiB | 未通过 |
| 分块输出层＋裁左、右 padding | 4.62 s | 4.09 s | 30.84 s | 17.27 GiB | 未通过，改变左 padding 状态 |
| 分块输出层＋只裁右 padding | 6.92 s | 6.34 s | 53.32 s | 20.69 GiB | 梯度通过，log_prob 未通过 |

只分块输出层减少约 47% 的 allocated 训练显存，前反向仅快约 3.2%。它保留原生
backbone 和完整输入，只对 response_mask 指定的策略 token 做词表投影；用
非重入 checkpoint 分块计算统计值，保留可训练 LM head 和空 mask 的 FSDP 参与。
不需要 bypass，也不冻结 LM head。

统一预设的真实模型门槛：log_prob 最大绝对差 < 0.04、梯度采样 relative L2 < 0.03、
cosine > 0.999。没有根据失败结果放宽门槛。

- 只分块：全部 rank 的 log_prob 相同，梯度采样 relative L2 约 0.0068–0.0179，
  cosine ≥ 0.999846；四 rank 均发生实际参数更新。
- FLA 首组：最大 log_prob 差 0.561，梯度采样 relative L2 0.527–1.382。
  注意 overlay 还会自动替换 RMSNorm，不能将这一组差异全部归因于 GDN kernel。
- 裁左右：最大 log_prob 差 4.278，未通过。
- 只裁右：梯度采样 relative L2 0.0119–0.0207、cosine ≥ 0.999809；
  最大 log_prob 差仍为 0.1888，不能作为已通过的等价优化。
- 初版 fused head 没有通过完整梯度小模型门槛，候选配置采用 checkpoint backend。

保留原生 RMSNorm、只切换 FLA GDN 的补测仍未通过：最大 log_prob 差 0.3042，
梯度采样 relative L2 0.264–0.952、cosine 0.616–0.974。四 rank 均有参数更新，
但“能更新”不等于数值一致。该组复用了前一组的 Triton 缓存，不能将 85→31 秒
全部归因于归一化层变化，也没有据此认定正式训练能够稳定达到这一速度。

## 采样并发与 bypass

同一个 vLLM TP=1 引擎、显存比例 0.30、eager、无 prefix cache。
从两个真实 batch 抽出 32 个 assistant-turn 前缀，长度 4,134–14,338；
每请求 seed=42+i、temperature=1、top_p=1、top_k=-1、最多 1,024 token。
输入 hash 为 `d5b4c90befd8af40e01e7a501b47e7f1ed2e6acc718c4107e44ed5b65ff6d343`。

| max_num_seqs | 不返回 log_prob：秒 / token/s | 返回 log_prob：秒 / token/s |
|---:|---:|---:|
| 4 | 70.43 / 86.55 | 106.50 / 57.23 |
| 8 | 56.16 / 113.68 | 59.60 / 107.11 |
| 16 | 35.33 / 168.03 | 39.22 / 152.20 |

每轮 32 个请求均完成；并发 8 有 1 个请求达到长度上限，并发 4/16 没有。
调度改变了部分生成 token 数，因此这不是固定输出长度的严格 kernel benchmark。
各设置只测一批，含统计噪声。并发 16 值得用于下一次在线短验收，尚不能认定
工具调用、用户模拟器参与之后也能提速 1.94 倍。

返回 log_prob 有实际开销，不能把 bypass 省掉的 actor forward 当成净收益。
本次没有启用 bypass 训练，也没有声称训练 ratio 或 off-policy correction 通过。

现有 request_id → TrajectorySession 结构已经隔离各轨迹的 DB、用户和环境。
4/16/32 并发的交错多工具调用、取消和清理测试共 3 项通过；环境与用户使用 stub，
这不是“真实 32 路模型＋环境 100% 安全”的证明。无需仅为提并发重写 ContextVar。

## 更优先的正确性问题：singleton GDN padding

当前 Transformers 5.5.1 的 `modeling_qwen3_5.py` 中：

```python
if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
    hidden_states = hidden_states * attention_mask[:, :, None]
```

当 micro-batch=1 时，GDN 跳过 padding mask。当前模型的左 padding 因而能够影响递归状态。
这解释了为什么直接裁左 padding 不能保持现有行为，也说明不能把“保持原实现数值”
和“训练、推理语义已经正确”混为一谈。

额外用 vLLM 已生成的 8 条回复、共 619 token 对照 HF，输入/输出 token 与温度完全相同：

| HF 输入方式 | token 加权平均绝对 log_prob 差 | 最大差 | 差值 > 0.1 的 token 比例 |
|---|---:|---:|---:|
| 无 padding | 0.00818 | 0.21072 | 1.45% |
| 按生产 prompt 长度 8,192 补左 padding | 0.09036 | 1.58181 | 20.03% |

这不是所有任务的总体统计，也不证明全部差异都来自这一条件；但已足以要求先做
padding 一致性修复与回归，再做正式 RL。不能根据这 8 条短前缀认定 bypass 可用。
本次没有修改主环境的 Transformers，也没有重训/否定此前 SFT 模型。

## 后续顺序与时间目标

1. 在独立分支修正 singleton padding 行为；以相同真实 token 的 unpadded scoring 为参考，
   验证 batch=1/2、有/无左右 padding 的 log_prob、完整梯度和 HF/vLLM 差异。
   这是正确性修复，需要记录新的实验基线，不能悄悄混入纯性能对照。
2. 保留已通过的分块输出层。核验新的数值基线后再评估去 padding 与 FLA，避免带着
   原有 padding 差异给更快的实现错误背书。
3. 用完整模拟器和多工具执行器做 1–2 个在线短 step，验证 Ray 参数传递、old/reference
   scoring、更新后权重同步及稳定性；之后再用正式每步 128 条轨迹的实测时间估算 E0。
4. 确认端到端预算后再开始 40-step E0。本次没有自动开始。

目前不能承诺保留 5,120 条轨迹能在 6–10 小时内结束。按已通过的分块版、这批序列长度
简单线性外推，前反向为 `139.48 × (5120/8) / 3600 ≈ 24.8 小时`，尚不包括
采样、reference、old log_prob、offload、评估和保存。此式只是分量外推，首次调用开销、
长度分布和完整 trainer 调度都会改变它；它不是完整 E0 的实测 ETA。

候选配置为 `configs/train/rl/qwen35_4b_full_a800_perf_candidate.yaml`：
预算不变、并发 16、分块 head、bypass 关闭、padding trim 关闭、每 10 step 保存模型。
模型分片此前实测约 19.3 GiB/次；完整模型＋optimizer 约 50.6 GiB/次。
只保存模型可用于评估或重新初始化训练，不支持原 optimizer 状态的精确续训。
保存频率从每 5 步改为每 10 步只减少写盘次数，不会直接加快 rollout 或反向计算。

## 复现入口

激活远程原环境后，所有测试应使用独立代码路径：

```bash
source /root/autodl-fs/tau3-core-20260912/activate.sh
C=/root/autodl-tmp/tau3-perf-20260912/code
export PYTHONPATH="$C:$C/verl:$C/tau2-bench/src"
cd "$C"
python env_info/a800_20260912/check_perf_candidate.py
```

该命令只做配置解析和 Hydra `--cfg job`，不会启动 RL。
本次配置检查已通过；最终 GPU 输出层测试 7 项通过，并发隔离测试 3 项通过。
输出层 GPU 测试见 `tests/test_qwen35_compact_head.py`；并发隔离见
`tests/test_rollout_concurrency_stress.py`。性能回放与比较入口分别是
`performance_probe.py`、`compare_probes.py`；采样与同 token score 审计分别是
`rollout_throughput.py`、`rollout_logprob_probe.py`。
