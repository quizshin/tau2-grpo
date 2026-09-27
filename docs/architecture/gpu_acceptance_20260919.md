# GPU 工程验收执行记录（2026-09-19）

用户已授权执行 [预算方案](gpu_acceptance_20260918.md)。SSH 使用端口 53458；主环境 qwen35，模拟器 qwen38-sim，复用 `/root/autodl-fs/tau3-core/activate.sh`。

**最终状态：部分验收完成，已停止所有本次 GPU 服务。** A 的三算法轨迹检查与 B 的 GRPO 两步检查通过；其余 B、C、D 未完成。2026-09-19 北京时间 18:46:27 清理完成。机器汇总为 `final-summary.json`，最终资源检查为 `final-resource-check.json`。

## 预算与隔离

- 4 张 A800 80GB 策略卡 + 1 张模拟器卡，启动前均空闲。
- 总墙钟上限 3 小时、最多 15 GPU-hours，从首次服务启动开始计时；不是运行时间预测，不保证在预算内完成全部阶段。
- A 最多 12 条训练 harness 验证轨迹，B 最多 512 条训练候选，C 最多 64 条续训候选，D 最多 8 条独立评测。
- 失败即停止后续阶段，不自动重试、增加采样、扩充时间或运行 final50。
- 独立输出 `results/runs/architecture_gpu_acceptance/20260919_s42/`；本文件是执行索引，实际进程与阶段状态以该目录的回执为准。

## 实现与当前状态

阶段 A 于北京时间 16:20:03 开始，17:28:41 通过并完成服务清理；全程预算截止仍为 19:20:03。控制器 `env_info/a800_20260919/architecture_acceptance.py` 复用正式 profile、runner 的配置解析及数据校验；明确记录工程覆盖，正式入口的整十步限制不变。

A 使用真实 veRL `val_only` 分支，固定 train50 的前两个任务、每任务两条轨迹，分别执行 GRPO、GiGPO、MT-GTPO。保持正式上下文、工具与模型设置；logger 为 console，保存和参数更新关闭，避免把 step 0 验证写成训练更新。初始 actor→rollout 同步仍会执行。所有阶段采样统计按实际发生记录，不能以计划数冒充完成数。

源码、启动参数、完整 Hydra、任务 ID、数据预检和授权预算在启动前落盘。工程异常记录保留；短程 reward 不能作为算法排名。

阶段 A 的最终证据：

|算法|轨迹|生成回合|工具调用|多工具回合|官方成功|参数更新|
|---|---:|---:|---:|---:|---:|---:|
|GRPO|4|38|31|5|0/4|0|
|GiGPO|4|41|32|4|1/4|0|
|MT-GTPO|4|43|33|5|0/4|0|

每组 4 个 worker 各 24 个 GDN conv1d 张量均与传入 actor 权重逐值相同；补查回执为 `phase-a-independent-audit.json`。A 结束后 5 卡占用均为 0 MiB，compute process 列表为空。官方成功仅描述工程小样本，不能比较算法收益，尤其这些运行均未训练。

所有 122 个原始生成结束原因均为 `stop`，没有真实 `length` 覆盖；精确 token/mask 已有，行为 logprob 不完整，事实层的 `sample_group_uid/trial/seed` 仍为空，不能据此宣称完整采样身份或行为概率已经接通。DB 哈希与轨迹身份记录已验证；广泛的会话隔离依赖此前 CPU 用例，12 条轨迹不构成穷尽证明。

B 的四组配置完成 CPU 解析后，于 17:29 左右进入执行，使用原预算的剩余秒数设置外部 timeout。GRPO 两步完成后的具体结果见下表。续训和独立评测控制器已准备，但因 B 不完整未执行。现有真实同步审计检查全部 GDN conv1d 张量，不覆盖全模型所有参数，不能将其描述为全参数映射验收通过。

运行可观测性限制：当前 MFU 估算器不支持 Qwen3.5，日志中的 0 不能解释为真实利用率；LiteLLM 未登记本地 27B AWQ 模拟器单价，美元成本不可用。文本推理启动出现 multimodal warmup 与 NIXL 不可用警告，A 的实际推理随后正常完成，不能把这些警告直接当作推理失败。

## B 的完成范围

|GRPO step|候选轨迹|非零优势 token|梯度范数|跳过更新|SwanLab 云端真实 step|
|---|---:|---:|---:|---|---:|
|1|64|40,579|1.2104773521|否|1|
|2|64|64,427|1.4493535757|否|2|

- 128 条 rollout 均包含共享轨迹事实；实际更新输入保存为 `update-batches/update_000001.pkl` 与 `update_000002.pkl`，优势及 actor 数值指标有限。
- 12 份权重审计全部匹配，观察到初始 / step 1 / step 2 三个不同的卷积参数状态，支持本组发生了有效更新及所覆盖卷积权重正确同步的结论。
- 最新完整 `b-grpo-df0/global_step_2/` 通过统一结构校验；新完整检查点发布后旧 step 1 才轮换。轮换过程中曾单独核验 step 1 完整且 step 2 尚在写入。没有据此宣称 GPU 恢复通过。
- 同一 SwanLab run `0aed4bc93a9e4c79b2504` 的 step 1/2 已云端回读，与本地一致。没有执行断点恢复，因此这不是“续训云端连续性”的验证。
- 补充审计：`b-grpo-df0/step1-input-cpu-audit.json`、`step1-completion-audit.json`、`cloud-step1-audit.json`；最终两步审计在 `phase-bc-status.json` 的 `arms.b-grpo-df0` 中。

## 预算收尾与未完成项

本次从 16:20:03 到 18:46:27 共 8,783.27 秒，即 2 小时 26 分 23 秒。若保守地按 5 卡全时预留计，资源上界为 12.20 GPU-hours；这不是实际利用率积分，不包含美元价格推算。

两步通过后仍剩 2,018.96 秒（33.65 分钟）。以本组策略冷启动 908.36 秒、最快完整 step 1,202.10 秒计算，下一组“启动 + 两步 + 60 秒清理”的估计为 3,372.57 秒（56.21 分钟）。这是按已观测速度的预算估计，并非严格运行时间下界。为避免新组在硬截止前只能留下不完整结果，独立预算检查在完整组边界发出 SIGTERM，未延长预算。

控制器当时已切换 `active_arm` 并刚发起 GiGPO launcher；其日志为 0 字节，无 rollout、更新输入、指标或检查点，随后清理完成。应记录为“启动被预算检查提前停止、未采样”，不能写成 GiGPO 更新失败，也不能写成 GiGPO 通过。

原 `phase-bc-status.json` 保留 `failed / InterruptedError: ... signal 15`，不改写历史。`budget-boundary-stop.json` 记录收尾依据，`final-summary.json` 使用 `partial_budget_boundary_stop` 明确区分资源决策与数值故障。实际未完成项：

|项|状态|
|---|---|
|B GiGPO 两步|启动后、采样前停止，未验收|
|B MT-GTPO DF off/on|未启动|
|C step 2 → 3 GPU 恢复|未启动，B 前置未完成|
|D 导出与独立 2×4 评测|未启动，B/C 前置未完成|
|完整参数映射 / live length / 完整行为 logprob|不在已有通过结论中|

最后检查：5 张 A800 均为 0 MiB、无 compute process，无验收控制器进程，持久盘剩余约 140.65 GiB。没有 formal20、额外补采样、GPU 重试或 final50。

## 证据与维护

远程运行目录：`/root/autodl-fs/tau3-core/code/results/runs/architecture_gpu_acceptance/20260919_s42/`。本地小型证据副本在 `results/maintenance/architecture-20260919-gpu/`，包含按哈希校验的日志、配置、轨迹与回执；模型、检查点和更新 tensor 包保留远程。采集清单明确记录收集范围，不能将未下载的大型资产视为本地备份。

运行结束后仅删除 `architecture_acceptance.py` 未使用的 `sys` import；数值、控制器行为和采样参数不变。实际执行过的源码仍保存在运行目录的 `source/`、`source-bc/` 归档及哈希清单，最终维护版本与执行快照不混淆。文档通过有前像校验和备份的逐文件同步发布；不批量提交或覆盖既有工作区修改。
