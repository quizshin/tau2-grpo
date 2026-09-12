# 40 任务课程的 A800 加速与在线验证（2026-09-12）

已完成一次真正的 E0 在线更新：8 个任务 × 每任务 8 次采样，共 64 条轨迹，训练一步 **32.15 分钟**。建议先以 **10 批 / 640 条**作为 6–7 小时目标的诊断实验；若前段速度与评测支持，再累计至 **15 批 / 960 条**，尝试控制在 10 小时内。这里只测了一批，时间估计不构成保证。

本次测试已经结束，所属训练与模拟器服务已停止，5 张 GPU 已释放。没有启动正式 10 批、20 批或原文的 50 批实验，没有保存完整检查点。单步工程测试独立记录，不计入以后正式实验的采样预算。

## 实测环境与改动

- 独立分支：`perf/a800-e0-20260912`。本地目录 `/Users/apple/Projects/program-llm/tau3_grpo_fix/perf-20260912`；远程目录 `/root/autodl-tmp/tau3-perf-20260912/code`。主工作目录里其他任务的未提交修改没有被覆盖。
- 5 × A800 80 GB：4 张 policy/FSDP 卡，1 张 Qwen3.8-27B AWQ INT4 用户模拟器卡。模型从 `new-off` 合并 SFT 初始化，Qwen3.5-4B 全语言参数训练、冻结视觉分支，关闭 thinking；不是 LoRA 实验。
- Torch 2.11、Transformers 5.5.1、vLLM 0.20；FP32 主参数、BF16 计算。沿用 24,576 token 训练形状、15 次 assistant / 15 次 user 轮数上限、学习率 1e-6、KL 0.01、温度 1.0。多工具调用继续按列表顺序执行。
- 沿用上一轮通过验证的 compact checkpoint head；生成端 `max_num_seqs` 从原正式配置的 4 提高到 16。新增 singleton GDN padding 修复；未打开 FLA、padding 裁剪或 bypass。
- 补齐独立 manifest 路径、10/15/20 批配置及展开检查。解析脚本覆盖激活环境中旧预算值，并清除可能绕过课程的 `TRAIN_PARQUET`。没有修改通用 launcher 的环境优先级规则。

## 哪些优化有效，哪些没有通过

|措施|实测证据|本次在线使用|
|---|---|---|
|compact checkpoint head|上一轮同一批 8 条真实轨迹回放：反向峰值 allocated 59.81→31.66 GiB，约降 47%；反向 144.13→139.48 秒，仅快约 3%|是，主要解决显存峰值|
|生成并发上限 4→16|上一轮同一组 32 个真实前缀，生成吞吐 86.55→168.03 token/s；未包含完整工具交互|是；本次不是完整在线 A/B，不能声称整轮翻倍|
|GDN singleton padding guard|修复 batch=1 时 padding hidden states 未清零的问题；同一批 619 个生成 token，带生产左 padding 的 HF/vLLM log_prob 加权 MAE 0.09036→0.00821；未 padding 为 0.00818|是，属于正确性修复|
|同时裁掉左右 padding|修复后真实 4B 回放反向 137.32→31.11 秒，但各 rank 的梯度相对 L2 0.194–0.740，log_prob 最大差 0.302|否，数值不通过|
|FLA + 原生 q/k 归一化 + FP32 递推输入|第一版混合 dtype 编译失败；修正后反向 103.17 秒、allocated 24.00 GiB，但梯度相对 L2 0.144–0.617，log_prob 最大差 0.530|否，数值不通过|
|bypass old_log_prob|本次没有进行 bypass 在线训练；old_log_prob 只占整轮约 6.4%，即使完全消除此项也不足以大幅缩短 E0|否|

真实回放的数值门槛沿用原设定：log_prob 最大差 <0.04、抽样梯度相对 L2 <0.03、cosine >0.999。没有为了接受加速结果放宽门槛。真实 4B 梯度是每个参数 shard 最多抽 256 个值；tiny 模型检查的是全部梯度。

padding guard 是一个新的正确性基线：它有意改变了旧版本 batch=1 跳过 mask 的行为，不是与旧错误实现逐位等价的加速。4 个 GPU 小模型测试覆盖 batch=1/2、tied/untied embedding、非零 padding embedding 和 activation checkpointing，输出与全梯度检查通过。短序列验证不代表所有长序列数值问题已解决，因此裁剪与 FLA 仍关闭。后续四组算法实验应统一使用这个修复，避免新旧行为混比。

## 一次完整在线更新的结果

|阶段|耗时|占训练一步|
|---|---:|---:|
|在线 rollout（含工具与用户模拟器等待）|476.86 秒 / 7.95 分钟|24.7%|
|old_log_prob|123.42 秒 / 2.06 分钟|6.4%|
|reference log_prob|123.95 秒 / 2.07 分钟|6.4%|
|actor 参数更新|1,198.16 秒 / 19.97 分钟|62.1%|
|同步更新后权重|5.94 秒|0.3%|
|整步|1,928.79 秒 / 32.15 分钟|100%|

训练进程总墙钟 2,810.41 秒 / 46.84 分钟，其中启动及其他开销 14.69 分钟；模拟器启动发生在此之前。没有在这一步保存 checkpoint 或执行 selection 评测。

actor 日志汇总的 allocator 峰值为 allocated **39.55 GiB**、reserved **68.20 GiB**。它不是整卡 `nvidia-smi` 占用，也不是每个 rank 最大值的保证；共驻 vLLM 和 CUDA/NCCL 还占显存。模拟器运行时另占一张卡约 50.8 GiB。CPU memory 指标 183.11 GiB 来自机器级 `psutil.virtual_memory().used`，不能当作本训练进程独占内存。

64 条轨迹属于预定的 8 个 task，每个恰好 8 条、64 个唯一 session ID，没有 dummy padding 行。共有 116,370 个 policy token，其中 109,275 个具有非零 advantage；old/ref log_prob、advantage、return 在 policy mask 内均有限。`actor/grad_norm=1.478`。4 个推理 worker 的初始和更新后权重审计均与 incoming actor weights 一致；每个 worker 检查的 24 个卷积权重哈希全部在更新后改变。本次没有更新后的下一次 rollout，因此只证明更新与同步，没有独立验证第二轮行为。

|任务|成功数 / 8|组类型|
|---|---:|---|
|airline_61|1|混合|
|airline_574|1|混合|
|airline_892|8|全 1|
|airline_367|6|混合|
|airline_43|6|混合|
|airline_209|1|混合|
|airline_614|7|混合|
|airline_739|4|混合|

共 7 个混合组、1 个全 1 组、0 个全 0 组。全 1 组的 E0 advantage 确实为零。34/64=53.125% 是**初始 SFT 模型在这一批训练采样上的评分器成绩**，不是训练后的独立评测，也不能据此证明能力提升。

失败类别：错误结果 18、工具错误 5、轮数上限 5、agent error 2。57 条完成 verifier 评分；5 条轮数上限和 2 条上下文超限未进入 verifier，按失败记 0。日志还有一次 XML 工具参数被解析成 Python set，随后 JSON 序列化失败；任务进程继续运行。该解析边界尚未修复，不能把工具错误都解释成模型不懂业务。

`airline_892` 属于应拒绝取消并转人工的任务。读取的一条成功样本完成了这段流程；但当前评分为 DB + COMMUNICATE，且没有 communicate_info，不能从 8/8 推断已严格验证所有解释和转人工行为。名单保持冻结，不因本批全 1 或失败率更换任务；正式效果比较前应继续检查这类奖励覆盖弱项。

## 三份课程材料的核对

读取了用户给出的课程计划、40 个候选与 SFT 对照表、10 批 YAML。40 个候选逐条等于冻结 train200 中的原条目，与 selection60 的 ID 交集为 0，4 份 DB 文件哈希一致。能力映射与 ID 检查不等于全部任务的语义认证，也没有另行完成与官方 final 的语义去重。

实际 `new-off` 输入与原筛查快照的身份差异已经查清：

- 筛查快照 SHA256：`a930dbffab46ebb5ce2e9df453ec7c6ecc69752dbd8de9a1f3f1969f974cfe55`。
- 实际 SFT 输入 SHA256：`e0aec9fb93c25f4f63e6580f6cdfcc0b95c1dfabe47bdf284745fa1759d1384b`，与远程历史 `experiments/multicall-aa267bb/preflight.json` 一致。
- 两份均为 45 条，44 个共有 ID 的 messages 完全相同；快照独有 `airline_dialog_360`，实际输入独有 `airline_dialog_804`，共有条目的 metadata 也发生变化。
- **40 个候选的 best_clean_match 全部存在于实际 new-off 的训练输入中**。该替换没有使这 40 条能力映射失效；仍不能把能力关联说成完全相同的 SFT/RL 任务。

40 候选 manifest SHA256：`79e317a824e8b55f7ce338c9f30a5811d83f2dce17dd896397eaffb003f5efbb`。1→10→15→20 以及 10→50 的 schedule 前缀检查通过。10/15/20/50 批分别使每个任务出现 2/3/4/10 次；每次采样 8 条。

## 怎样控制在 6–10 小时

任务池从 200 缩到 40，本身不决定运行时间。决定总量的是 `updates × groups_per_update × group_size`。原 E0 是 40×16×8=5,120 条；本配置每步 8×8=64 条。

|累计批数|总轨迹|每任务轨迹|按本批外推纯训练|每批慢 20% 时纯训练|建议|
|---|---:|---:|---:|---:|---|
|10|640|16|5.36 h|6.43 h|首段诊断；最有希望满足 6–7 h|
|15|960|24|8.04 h|9.64 h|速度稳定后才考虑，争取总计 10 h|
|20|1,280|32|10.72 h|12.86 h|当前速度不符合总计 10 h|
|50|3,200|80|26.79 h|32.15 h|不符合本次时间目标|

以上没有包含启动、模拟器启动、保存与评测。若暂预留 1–1.5 h，10 批约 6.4–6.9 h，15 批约 9.0–9.5 h；该预留是规划值，selection60×4 的实际评测时间尚未在新配置测量。如果中途停机、再启动并重复评测，要额外计时。后续任务更长或保存变慢都可能超出窗口，尤其 15 批不能保证 10 h 内完成。

建议保持 40 个任务、每组 8 次采样，先运行 10 批，检查奖励有效性、工具/截断失败、前 5–10 批的实际耗时和固定 selection60。若速度与评测收益支持，再从 step10 完整恢复并累计至 15 批。15 批每任务 24 条，接近原 E0 平均每任务 25.6 条；它仍是缩小任务分布和优化步数的新实验，不能声称等同原 200 任务预算。E0/E1/E2/E3 后续必须使用同一个最终预算、任务顺序和评测设置。

最大瓶颈是 actor 更新，占整步 62%。继续提高生成并发或仅 bypass old_log_prob 的收益有限。下一项值得独立测试的是利用 compact head 节省的显存，把训练 microbatch 从 1 调到 2；需要重新检查真实峰值、梯度与每步时间，目前没有验证，未写入推荐配置。更激进的 FLA/裁剪等数值问题解决后才能重新考虑。

## 保存、配置与验证产物

快配置 `qwen35_4b_full_a800_curriculum40_fast_pilot_20260912.yaml` 使用 `save_freq=10`、仅保留 1 份完整状态，保存内容为 `[model,optimizer,extra]`。对应 15/20 批 extension 复用相同 E0 输出目录与 10 批 schedule 前缀。trainer 会在最后一批额外保存，所以累计 15 批会保存 step15，即使 15 不是 10 的倍数。

上一轮实测一份完整检查点约 **50.6 GiB**（模型状态 19.3 + optimizer 31.3），一次保存约 **6 分钟**；本次没有重复写入验证。保留一份主要减少持久存储，不能把保存瞬间的需求理解成绝不超过一份。跨进程从 10 批继续必须保留 optimizer/RNG 等状态，合并模型或 model-only 导出不能替代完整 resume。实际恢复 step10 尚待正式阶段从日志验证。

改成每 10 批保存主要减少 I/O；在当前预算中只节省若干分钟，无法把几十小时训练压到 6 小时。

新增的审计/控制/分析脚本位于 `env_info/a800_20260912/`：`prepare_curriculum_speed.py`、`run_curriculum_smoke.py`、`analyze_curriculum_smoke.py`。控制器只运行 1 步，并使用独占启动标记与进程组清理；不会继续正式实验。分析脚本只读现有日志和 batch，不启动 GPU。

本轮验证：原相关 CPU 测试 35 项通过；补充快配置断言后 `test_curriculum_profile.py` 共 5 项通过，其中 3 项为新增；padding GPU 全梯度测试 4 项通过；Hydra 单步配置展开通过；真实在线 64 条轨迹完整更新通过。失败的 FLA/裁剪比较如实保留。

完整小型证据在本地 `results/curriculum-speed-20260912/`，远程 `/root/autodl-fs/tau3-core-20260912/runs/curriculum-speed-20260912/`。关键文件：`online-analysis.json`、`data-audit.json`、`sft-identity-comparison.json`、`online.log`、`online-smoke/rollouts/1.jsonl`、`corrected-logprob-audit.json`、`trimmed-comparison.json`、`fla-fp32-comparison.json`。88.6 MB 的训练 batch 保留远程；其 SHA256 与日志/轨迹 SHA256 记录在 `online-analysis.json`。Git 中另保留摘要与证据哈希，模型权重和大规模结果没有提交。
