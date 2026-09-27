# GiGPO 两步 GPU 工程验收（2026-09-19）

结论：`passed_with_mapping_scope_limit`。本次批准的 GiGPO、DF off、2 step / 128 条候选已完成，两步都有非零学习信号、有限梯度和真实参数变化；完整 step 2、SwanLab 1/2 云端回读及 GDN 卷积同步审计通过。5 张卡已释放。不是正式算法效果实验，未运行 MT-GTPO、续训、独立评测或 final50。

## 授权、时间和身份

- 用户在上一条具体预算请求后回复“批准”：只补 GiGPO 两步，4 张策略 A800 + 1 张模拟器 A800，90 分钟硬上限 / 7.5 GPU-hours，不自动重试、扩时或继续下一阶段。
- 计时从北京时间 **20:06:41.981** 开始，原硬截止 **21:36:41.981**；自有 GPU 服务清理完成于 **21:15:00.840**。耗时 **1 小时 8 分 19 秒**，按 5 卡全时预留计算约 **5.69 GPU-hours**；不是实际 GPU 利用率积分，也不是美元费用。
- 远程 run：`results/runs/architecture_gpu_acceptance/20260919_gigpo_s42/`；训练子目录 `b-tau_gigpo-df0/`。旧 `20260919_s42` 的结果和失败回执保留原样。
- Qwen3.5-4B `new-off`、全语言参数、非 thinking、FP32/IEEE FLA、独立 Qwen3.8-27B-AWQ-INT4 模拟器、seed42、LR1e-6、KL0.01、GiGPO structured v1，沿用既有 `legacy_mean` episode normalization 和 omega=1。
- 使用 formal20 的前两批 8×8 调度。输入预检回执仍描述原 20-step / 1,280 候选协议，实际 engineering overrides 与完整 Hydra 明确为 **2 step / 128 候选**，不能把预检预算误认为本次实际运行量。
- 工程例外：每步保存完整状态、关闭训练内评测、仅保留最新完整检查点。本次不覆盖正式每 10 step 保存及 selection60×4 的规则。
- 复用前一轮已通过的 A 轨迹证据；847 个算法、环境、数据、验证器及 vendor 运行时源码文件哈希相同，配置整理有 71 组旧/新命令等价证据。启动快照保存 1,902 个源码文件及 Git dirty patch。

## 两步更新

|指标|step 1|step 2|
|---|---:|---:|
|候选轨迹|64|64|
|策略 token|198,206|110,623|
|联合优势非零 token|40,177|101,005|
|GiGPO 步级非零 token|34,178|69,337|
|非零 step 数|220|397|
|跨轨迹、非初始轮非零 step|184|329|
|梯度范数|0.80470836|1.27853233|
|跳过 actor 更新|否|否|
|整步秒数|1,586.11|1,232.72|

DF off 时前后 policy mask 一致，被过滤的非零 step/token 均为 0。共享 facts 不注入 MT-GTPO shaping，GiGPO 用官方终局信号与 anchor 分组进行 episode/step 信用分配。

12 份 worker 同步回执均 `all_conv_match=true`，覆盖初始、step1、step2 三种不同卷积参数状态。两步参数变化与非零梯度相符。范围是所有 GDN conv1d 张量、TP1，**不是全模型参数映射验收**。

最新完整检查点：`b-tau_gigpo-df0/global_step_2/`，约 50.63 GiB。观察到先同时保留 step1/step2，step2 完整校验通过后才移除 step1。完整文件集合与 latest marker 验证通过；本次没有加载恢复，不能宣称 GPU 续训通过。

SwanLab run ID：`56e3545143db4a0da46b5`；云端真实 step/value 为 `(1,1)、(2,2)`，没有重置 step、重启训练或创建第二条训练曲线。

## 轨迹、终止与 reward 的边界

两批共 **1,302 个生成回合、960 次工具调用、146 个同轮多工具回合**。128 条轨迹的 token ID、mask、发出/保留 span 可按公共 facts 契约精确重建，均有初始/最终 DB 身份，未发生 token 丢弃。工具错误响应保留，共 228 次；工具缺参错误是环境执行事实，不等同于推理服务退出。

|训练终止事实|step 1|step 2|
|---|---:|---:|
|完成官方评分（user_stop）|42|56|
|其中官方成功|24|30|
|max_steps，未完成官方评分|21|8|
|context_window_exceeded，未完成官方评分|1|0|
|原始生成 stop|700|599|
|原始生成 length|3|0|

训练 fallback 的 0 分与已完成官方评分的 0 分保持区分。上述 30 条早停不是“官方验证已完成且失败”；本次不计算独立 pass@k，也不能把 step2 高于 step1 的成功数量解释为提升，两批任务本来就不同。

真实 `length` 补到了前次 A 只有 stop 的缺口：3 个生成回合都达到 1,024-token 单轮上限，token/span/mask 一致，分布在两条轨迹。一个回合所在轨迹最后 max_steps，另外两个所在轨迹后来 user_stop 并完成官方 0 分评分。**逐轮 length 不是轨迹终止原因的同义词。** 这仍不证明全部上下文/长度/重试边界覆盖。

facts 的 sample_group_uid/trial/seed 和完整生成时行为 logprob 仍缺失；训练更新包里的 uid、重算 old_log_probs/ref_log_prob 不自动填补这些 facts 字段。保留缺失能力，不从当前数据猜造历史身份。

释放 GPU 后对两份实际 update packet 做 CPU 重放：逐轨迹 reward/mask、step advantage、全部信号统计精确相等，联合优势字节 SHA256 也相同；两个包都包含 old_log_probs/ref_log_prob，均没有 rollout_log_probs。首次审计辅助脚本直接比较 Python tuple span 与 JSON list，出现容器类型断言失败；v2 仅将容器按 JSON 归一化后比较，未使用数值容差或舍入，原失败脚本及日志保留。这是审计辅助代码问题，未改训练结果。

## 控制器缺陷与本次处置

扩展既有 `architecture_updates.py`，新增仅 GiGPO 的 prepare/execute 模式，复用原有 Supervisor、配置解析、检查点及同步审计。拒绝多算法计划、超过两步配置、已有 attempt 重启；保留原 execute-bc 入口。初版单组入口 7 项 CPU 检查通过。

运行中检查发现：初版每次轮询只要最新 metrics 仍是 step1，就可能再次检查“剩余是否足够开始下一步”；第二步已开始后也会触发，可能在不足 1,263 秒时错误中断正在进行的 step2。这是本批控制器缺陷，不是 GiGPO 算法失败。

修正为每个已完成边界只判断一次，增加两项回归，共 **9 passed**；新增 lint 0，历史债务仍 382。对已在运行的旧控制器，没有替换进程内源码或重启训练：独立 CPU 守护于 **20:58:18** 暂停原控制器的监督循环，持续记录原训练进程及后代的 PID/创建时间；训练、模拟器、SwanLab 和截止时间不变。训练正常结束后 **21:14:56** 恢复原控制器完成服务清理与最终审计；未使用截止时间兜底终止。接管脚本、前后时间和原始回执均保存在 run 中。

实际执行控制器的原始 SHA256 为 `0ef43b41338894c3e1cc57961a86f1c078bde669bbfeed71eca1f2ad6fea7432`，在 `source-bc` 归档中保留；当前源码包含上述修正，不能用当前哈希冒充启动时版本。GPU 训练没有重试，这次控制流程修正也不改变两步数值算法。

## 耗时与存储

|阶段（秒）|step 1|step 2|
|---|---:|---:|
|生成|505.03|287.94|
|重算 old logprob|106.00|81.53|
|reference logprob|114.95|87.30|
|actor 更新|491.69|381.13|
|完整保存|361.34|388.24|
|权重同步|6.37|6.01|

模拟器启动约 5 分钟，策略侧冷启动约 16 分钟。没有把总墙钟全部归入算法计算，也不把 MFU 的不支持值当作有效效率测量。

最后核对：5 张 A800 均 0 MiB，无 compute process，无控制器或守护；持久盘空闲约 **89.82 GiB**。低于本控制器新算法组开始所需 110 GiB 轮换门槛。后续 MT-GTPO 两组应先解决空间或明确资产保留方案；本次未删除原 GRPO 或其他实验的完整检查点。

## 证据与剩余验收

远程 run 保存 `authorization.json`、`budget.json`、完整配置/命令、源码归档、两步 rollout、update packets、signal-audit、weight-audits、checkpoint-complete、SwanLab 身份、控制接管和 `final-summary.json`。本地维护证据在 `results/maintenance/gigpo-acceptance-1789819378/`；选定小型证据逐文件哈希核验，模型与完整训练包留在远程。

本次只关闭 B-GiGPO 的两步更新缺口。MT-GTPO DF off/on、C GPU 恢复、D 模型导出及独立评测、全参数同步、完整行为概率与采样身份、正式消融与增幅置信区间仍未验收。不会因剩余时间自动执行下一组。
