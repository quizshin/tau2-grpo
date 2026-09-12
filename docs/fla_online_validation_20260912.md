# FLA IEEE 在线 RL 单步实验（2026-09-12）

已完成一轮真实 E0 在线 RL：8 个任务 × 每题 8 次采样，共 64 条新轨迹，包含采样、old/ref 打分、actor 更新和推理权重同步。**不开外部 FA2，完整 step 实测 18.83 分钟；此前为 32.15 分钟，耗时减少 41.43%。** 实验成功退出，所属服务已停止，5 张 GPU 均已释放。

这是单步工程验证，没有运行正式多步 pilot、保存训练 checkpoint 或评测更新后模型；更新后的参数未作为后续训练检查点保留。

## 实测时间

|阶段|此前在线基线|本次 FLA 在线|变化|
|---|---:|---:|---:|
|Rollout（含工具和模拟用户等待）|476.86 s|533.27 s / 8.89 min|增加 11.83%|
|old log-prob|123.42 s|87.79 s / 1.46 min|减少 28.87%|
|reference log-prob|123.95 s|97.76 s / 1.63 min|减少 21.13%|
|actor 更新|1,198.16 s|404.68 s / 6.74 min|减少 66.22%|
|更新后权重同步|5.94 s|5.85 s|基本不变|
|完整 step|1,928.79 s / 32.15 min|1,129.76 s / 18.83 min|减少 41.43%，约 1.71 倍速度|

此前条件估计为 18.5 分钟，本次实测接近该估计。训练进程总墙钟为 2,159.44 s / 35.99 分钟，其中启动及其他开销为 17.16 分钟；模拟器在训练进程之前启动，另花约 5 分钟。不能把 18.83 分钟当成从零启动整个实验的时间。

新旧批次均从同一 SFT 初始化、采样同一组任务，但属于不同随机在线轨迹。本次 policy token 为 122,604，之前 116,370；本次有效上下文总长 560,739 token，单条 5,295–18,367 token。该比较证明这套在线配置可用且本轮明显变快，不是控制相同轨迹的单因素 A/B。

本次同时改变了 FLA、padding 裁剪、计算精度和 actor offload。FLA 的独立收益以此前相同 64 条轨迹回放为准：native cropped FP32 actor 519.78 s → FLA IEEE cropped FP32 377.65 s，减少 27.34%。不能把本次 actor 的 66.22% 降幅全部归因于 FLA。

## 配置与正确性

- 独立代码目录 `/root/autodl-tmp/tau3-gdn-diagnosis-20260912/code`；本地分支 `diagnose/gdn-20260912`。远程为无 `.git` 的代码副本，以源文件 SHA256 对应实际实现。其他工作目录未修改。
- 5 × A800 80GB PCIe；GPU 0–3：FSDP actor/reference 与 vLLM，GPU 4：Qwen3.8-27B AWQ INT4 模拟用户。策略从 `new-off` Qwen3.5-4B SFT 初始化，全语言参数训练，视觉分支冻结，关闭 thinking。
- FLA 0.5.2；Triton FP32 为 IEEE，含 triangular solve 精度设置；原生 GDN normalization；FP32 attention 显式重复 KV，避免原生 GQA 进入高显存 math 路径。
- actor/reference 均使用 FP32 参数及计算；actor 参数和优化器不 offload，reference 参数 offload。此前在线基线为 FP32 actor 主参数、BF16 计算、actor 参数和优化器 offload。
- 保留 compact checkpoint head（chunk 256）、singleton padding guard；新增裁掉轨迹外部 padding。每个 microbatch 单条独立轨迹，不复用递推状态，不做 sequence packing。
- 8 任务 × 8 次采样，两个 PPO minibatch 更新，学习率 1e-6、KL 系数 0.01、temperature 1.0，最大训练形状 24,576 token，既有轮数限制和 sequential multi-call 协议不变；vLLM `max_num_seqs=16`。
- 外部 FA2 overlay 未进入在线 PYTHONPATH。未编译或继续试验外部 FA2。正式 E0/E1/E2/E3 配置未改，只新增 opt-in 单步 profile。

接入前，实际完整模型的短轨迹和最长旧轨迹（17,581 token）log-prob 与已验证 IEEE FSDP 回放结果最大差均为 **0.0**。远程 Python 3.12 编译检查通过，既有补丁契约、课程配置与 launcher 测试 **57 项通过**，Hydra 配置展开核对通过。

在线日志证实 24 个 GDN 层安装 FLA，实际前向与反向调用均为 FP32、IEEE。64 条记录对应预定 8 个任务，每题恰好 8 条；64 个唯一 session ID，无 dummy 行。old/ref log-prob、advantages、returns 在 policy mask 内均有限；112,404 个 token 有非零 advantage，`actor/grad_norm=1.61886`。

四个 vLLM worker 的初始和更新后权重审计均与传入 actor 权重精确一致，每个 worker 检查的 24 个卷积张量在更新后全部改变。该证据确认发生实际参数更新与同步；未进行下一轮 rollout 或独立评测。

## 轨迹结果

|任务|成功 / 8|组类型|
|---|---:|---|
|airline_61|2|混合|
|airline_574|2|混合|
|airline_892|7|混合|
|airline_367|6|混合|
|airline_43|2|混合|
|airline_209|0|全零|
|airline_614|7|混合|
|airline_739|3|混合|

评分器成功率 **29/64 = 45.3125%**；这是更新前 SFT 的训练采样得分，不能与此前 34/64 的另一批采样直接比较并宣称 RL 提升或退化。

7 个混合组、1 个全零组。`airline_209` 的组内 advantage 全零，符合 E0；这一组不提供组内相对奖励的策略梯度信号，KL 项仍按配置计算。失败为错误结果 24 条、工具错误 6 条、轮数上限 5 条；59 条完成 verifier，5 条因轮数上限记零，无上下文超限。日志可见工具参数缺少 `payment_id`、日期等字段；本轮未观察到此前的 Python set JSON 序列化异常，不能据此宣称该问题已修复。

## 显存与下一步

训练器汇总 actor allocated 峰值 32.65 GiB、reserved 74.65 GiB。它不是整卡占用，也不是各 rank 峰值保证。运行中抽查整卡占用最高观察到 80,493 MiB（约 78.61 GiB），随后回落，未发生 OOM。采样时策略卡约 30 GiB、模拟器约 52 GiB。长轨迹下显存余量有限，不能从本轮通过推断所有更长批次都能通过。

当前主要耗时已变为 rollout（约 47.2%），actor 更新约 35.8%。建议下一轮先验证少量连续 step 的耗时和最长轨迹显存，再扩大至 10 步，并安排固定独立评测与完整 checkpoint。若新批次出现显存压力，先单独验证 allocator reserved 释放、训练/推理共驻内存和 offload 的取舍，再考虑增大 microbatch；本次没有修改这些机制。

按本轮外推，10/15/20 步纯 step 时间约 3.14/4.71/6.28 小时；若后续每步慢 20%，约 3.77/5.65/7.53 小时。均不含启动、checkpoint 和评测，不构成多步运行保证。没有自动启动这些后续实验。

## 产物

- 运行目录：`/root/autodl-fs/tau3-core-20260912/runs/fla-online-20260912`
- `online.log`、`simulator.log`、`controller.log`：完整日志。
- `online-smoke/update-batches/update_000001.pkl`：85 MiB 训练 batch；`online-smoke/rollouts/1.jsonl`：64 条轨迹；`online-smoke/weight-audits/`：初始与更新后权重审计。
- 本地证据：`env_info/a800_20260912/gdn_evidence/online/`，包括 `online-analysis.json`、配置展开、bridge 验证、CPU 测试日志、源文件哈希与 manifest。
- 新配置：`configs/train/rl/qwen35_4b_full_a800_fla_online_smoke_20260912.yaml`；入口：`env_info/a800_20260912/launch_fla_online_smoke.sh`。
- 先前完整数值验证见 `docs/rl64_gdn_validation_20260912.md`；旧在线基线见 `docs/curriculum_speed_validation_20260912.md`。
