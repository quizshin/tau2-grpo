# Base → SFT → GRPO → MT-GTPO：现有证据与下一轮比较

**最终状态（2026-09-20 23:00完成，现已核验）：四模型960条全部完成，0执行异常/0缺失；Base/SFT/GRPO/MT成功率50.00%/45.00%/41.25%/39.58%。本次GPU服务已清理，051实例未关机。下文较早进度、失败与退役路径只作为历史记录。**

**20:57 核对：16/16 真实预检通过，全部 user_stop、零执行异常，108 对工具调用/返回 ID 相符；五服务模型身份及四模型协议/任务/DB/源码一致。正式 960 条已于 20:50:43 开始；核对时 55 条完成、零执行异常。截止仍为 2026-09-21 00:43:09，不重启、不补样。**

**当前更新（2026-09-20 20:43）：按用户明确要求，中断上一轮、修复 STOP 优先级并从头重评。新控制器 PID 9658，截止 2026-09-21 00:43:09；当前五服务加载中。之前三次中断尝试的运行产物及本地对应副本已清理，保留原始检查点、独立目录中的两份导出及简短删除回执。下文旧尝试路径已退役，旧运行状态为历史记录。**

**最新执行：用户明确要求启动，新的 legacy 四模型评测已于北京时间 2026-09-20 20:22:51 启动。硬截止 2026-09-21 00:22:51（4 小时/最多 20 预留 GPUh）；当前为五服务加载阶段，16 预检后自动 960 正式。单条执行异常保留现场并继续其他正式 trial，不补样。**

**当前决定：保留 legacy 主要行为，统一 0.7/0.7；取消新增 1024 请求限制，异常留完整已观测记录并继续其余 trial。CPU 修改已验证，未启动新 GPU；见第 9 节。**

日期：2026-09-20。**最新状态：正式评测19:41因HTTP上下文边界处理缺口停止，186条完成、1条异常，结果不完整，不能排名。GPU进程已退出，未重试。CPU五场景复现及内存候选通过，生产修补尚未部署。**下文早期进度保留历史语义。

本轮按用户最新要求，以 Base、SFT、GRPO、MT-GTPO 四组为主。这里的 GTPO 明确指已有 MT-GTPO `reference_write/v3`、DF off；GiGPO 与 DF 组保留历史诊断，不进入这次 960 条主比较。研究目标是判断 SFT 是否改善起点、GRPO 是否改善 SFT、MT-GTPO 是否进一步改善 GRPO，并用任务与轨迹解释差异。

## 1. 模型身份已经核对

远程代码根 `/root/autodl-fs/tau3-core/code`，本次只读访问该仓库与既有模型。下表路径相对此根。

|组|精确资产|本次核对|
|---|---|---|
|Base|`models/Qwen3.5-4B`|两份权重完整 SHA256 与历史基座清单一致；来源 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`|
|SFT|`checkpoints/sft-merged/new-off`|合并权重完整 SHA256 与 9 月 14 日 SFT 导出回执一致|
|GRPO|`results/runs/rl-c50-matched6h-a800-20260912/e0_seed42/global_step_20`|四 rank 模型完整 SHA256 与历史 E0 导出源清单一致；检查点完成标记所列文件存在且尺寸一致|
|MT-GTPO|`results/runs/mt_gtpo_reference_write_v3/20260917_s42_df0/global_step_20`|四 rank 模型重新计算完整 SHA256；完成标记所列文件存在且尺寸一致；记录为本次新固定身份，不虚构旧模型哈希比对|

共读取并哈希 11 个权重文件、59,801,495,056 字节。没有加载模型执行推理，也没有重新做恢复实验；优化器只核对存在性与登记尺寸，没有重新哈希。GRPO/MT 的 `actor/huggingface` 不含推理权重，GPU 前需要从保留的 step20 重新 CPU 导出并逐张量核验。此次没有生成新的大型导出副本。

GRPO 与 MT 均登记从 `new-off` 起步，前 20 step 任务调度相同、每步 8×8、学习率 1e-6、KL 0.01。历史实际策略训练温度为 1.0、训练内评测 0.4、用户模拟器 0.0；它们不会因当前默认温度改为 0.7 而改变。模型来源、算法实现和日期差异仍需保留，不能把两组之差完全归因于一个过程奖励开关。

机器证据：[模型身份](../results/analysis/grpo_improvement_evidence_20260920/model_identities.json)、[权重完整哈希](../results/analysis/grpo_improvement_evidence_20260920/sources/model-weight-hashes.json)。

## 2. 历史成绩：独立评测与训练内验证分开

### 历史独立 selection

|模型|有效轨迹|pass@1|pass@4|四次全部成功|
|---|---:|---:|---:|---:|
|Base|缺完整对照证据|NA|NA|NA|
|SFT new-off|240/240|43.75%|68.33%|18.33%|
|GRPO step20|240/240|47.50%|73.33%|25.00%|
|MT-GTPO step20|缺同协议独立评测|NA|NA|NA|

SFT/GRPO 原始 run、轨迹与错误文件重新从远程取回，字节哈希与此前审计完全一致，使用现有严格比较器重新汇总。GRPO−SFT 为 **+3.75 pp，任务配对 95% CI [-2.50, 10.00] pp**。60 题中 19 题提高、12 题下降、29 题持平；这是这两次固定跑次的描述，区间包含 0，未证明跨训练 seed 的稳定改善。

已保存历史协议相符，但缺少当时完整 task/DB/harness 源哈希，仍标记 `recorded_protocol_only`。旧评测为策略温度 0.4、用户温度 1.0，不能与拟运行的 0.7 新协议拼表计算增幅。

辅助现象：SFT→GRPO，出现工具错误的轨迹 26→40，错误调用 36→50，有重复相同调用的轨迹 23→37，预算终止 21→27。重复不自动等于无效，错误也不自动是失败原因。这说明总体成功率点估计提高，不能推断所有行为都改善。

[60 题可读表](../results/analysis/grpo_improvement_evidence_20260920/task_comparison.md)；[CSV](../results/analysis/grpo_improvement_evidence_20260920/task_comparison_historical_selection.csv)；[严格比较与区间](../results/analysis/grpo_improvement_evidence_20260920/historical_sft_grpo_comparison.md)。

### 历史训练内 selection 验证

|模型节点|成功/240|成功率|完成评分但失败|预算未评分|
|---|---:|---:|---:|---:|
|GRPO step10|118|49.17%|99|23|
|GRPO step20|114|47.50%|107|19|
|MT step10|118|49.17%|108|14|
|MT step20|104|43.33%|127|9|

MT20−GRPO20 = −4.17 pp，描述性任务配对 CI [-12.92, 4.17] pp；MT20−MT10 = −5.83 pp，CI [-12.92, 1.25] pp。两区间也包含 0。这是已记录相同主要控制条件下的训练内诊断，不是新独立评测，也没有完整证明历史执行代码等价。

值得优先检查的现象：MT 的预算未评分从 14 降到 9，但已评分失败从 108 增到 127；有正过程奖却最终未成功的轨迹从 34 增到 45。因此不能把本轮退化简单归因于收尾超时，也不能把更密集的过程信号直接视为提点。

[训练内任务表](../results/analysis/grpo_improvement_evidence_20260920/task_comparison_training_validation.csv)、[学习曲线数值](../results/analysis/grpo_improvement_evidence_20260920/learning_curves.csv)。旧 step10 仅保留成绩与轨迹，不能假定还存在可加载权重；本轮预定使用两组 step20，不据旧分数临时换赢家。

## 3. Badcase 索引与已读案例

索引包含 2,284 条失败/未评分记录：SFT/GRPO 独立评测 135/126 条；GRPO/MT 训练 rollout 764/753 条；两者 step10/20 训练内验证 248/258 条。按数据来源、task、step、trial（已有才填写）、session、源文件行号、终止原因、错误与过程奖检索。自动字段只是观测标签，不冒充人工原因标注。

另外复用既有 6,400 条训练审计，生成 250 行「run×task」聚合表；本次重新取回的 40 个 GRPO/MT rollout 源文件哈希与旧审计全部一致。没有重新生成轨迹、改写旧 reward、补造旧 token 优势或 anchor span。

抽查 11 份可读案例，形成 4 组解释：

- **airline_657：GRPO 退化切片。**SFT 3/4，GRPO 0/4。所读 GRPO 轨迹反复讨论删除乘客，在实际工具返回人数不匹配后原样重试，随后开始处理行李请求时耗尽预算。政策理解、错误后修正与轮数消耗同时存在，不能只贴一个「超时」标签。
- **airline_219：评分形式敏感切片。**SFT/MT 的所读失败轨迹与 GRPO 成功轨迹更新的是同三位乘客，差异是名单顺序；参考顺序为 Nora/Ethan/Liam。保留官方结果，并标注顺序敏感，不能写成换错了乘客，也不擅自放宽评分器。
- **airline_23：支付参数与对话约束切片。**MT step20 给航班变更使用原支付卡 `1955700`，参考与 step10 使用 `4421486`；行李更新匹配并得到 0.5 过程奖。对话中「退款原卡」与「行李指定卡」的解释不同，属于参数/任务解释待复核，不能仅因工具返回成功就认为任务完成。
- **airline_934：局部奖励与额外操作切片。**MT step20 取消参考目标 B9E6D7 得到 +1，另外又取消两张订单，额外写操作为中性，最终官方失败。3D0729 在场景当前时间下已过 24 小时 15 分钟；CB3689 则为 18 小时 50 分钟，不能将两项都仅凭不匹配参考判为业务违规。该例同时提示奖励覆盖与参考规范边界，尚不能证明过程奖励导致了额外操作。

以上按提升/退化切片选取，不是随机抽样，不能由四例估计总体原因占比。详见 [案例索引与逐条说明](../results/analysis/grpo_improvement_evidence_20260920/badcase_review.md)；[全部失败索引 CSV](../results/analysis/grpo_improvement_evidence_20260920/badcase_index.csv)。

## 4. 现在能决定什么

当前证据不支持直接宣布 MT 比 GRPO 更好，也不支持先增加训练步数或扩大轮数。优先诊断主题是**写操作是否满足完整任务约束，以及过程奖是否与最终结果一致**。同时把业务错误、参考顺序差异、模拟用户变化分开，避免针对评分假象改算法。

下一步先完成四模型统一 selection，主对比预先指定 MT−GRPO，另报 SFT−Base、GRPO−SFT。用同任务的成功次数变化和案例核验定位最主要问题。只有新结果支持某一机制后，才选择一个训练改动；不同时改工具 schema、奖励、信用公式和预算。

「MT 比 GRPO 高」即使成立，首先也是方案效果；现有 MT 同时改变过程信号和信用计算，不能归因于过程奖励单项。需要机制解释时，后续单独设 GRPO 保持不变的单因素对照。该后续研究不属于本次 GPU 评测预算。

## 5. 四模型统一 selection 的具体预算

- 模型：Base、SFT new-off、GRPO step20、MT reference_write/v3 step20；后两组 DF off。
- selection60×4×4 模型 = **960 条正式轨迹**；另有固定 4 题×1 次×4 模型 = **16 条预检**，不混入成绩，也不按预检成功率选择模型。
- 协议：`tau3_eval_train_inputs_v3`；策略/用户温度均 0.7；非 thinking；策略每次最多 1,024 token；策略上下文 24,576，模拟器 16,384；trial seed 42/43/44/45。四个策略均显式 `--generation-config vllm`，避免各 checkpoint 默认生成配置不一致。
- 资源假设：**5 张 A800 80GB**，四张各运行一个 TP1 策略，一张运行 27B INT4 模拟器；每模型并发 4，总并发 16。
- 预计墙钟 **2–3 小时**，**4 小时硬上限**：若必须分配 GPU 才能获得导出所需内存，从 GPU 分配开始计时；否则从首个 GPU 进程启动计时。包含占卡期间的导出、加载、预检和正式评测；按五卡预留上限 **20 GPU-hours**。这是基于旧轨迹耗时的规划估计，不保证新协议一定在此时间内完成。
- CPU 导出、权重和源码身份准备在分配 GPU 前进行。已有历史导出回执不替代新服务 PID/内容证明。
- 触发身份/数据/协议不匹配、OOM、服务退出或未评分执行异常即停止并保留证据；预算到达停止。只管理本次进程，不全局清理其他作业；不重置截止时间，不自动重试补成完整分数。不完整评测不得删除困难任务后排名。
- 正常的协议预算终止按实际运行记录处理，不与服务异常混为一类。不得临时扩上下文、加轮数或改变采样条件。
- 不启动 RL 训练，不访问 final50，不自动测试其他 reward/anchor 候选。

本地与远程四个 CLI 计划均成功解析，每组 240 条；dry-run 不验证尚未导出的 HF 权重，也不证明模型服务运行成功。本次远程 `nvidia-smi -L` 返回 **No devices found**，需要先分配 GPU，并按用户要求在启动前确认上述预算。

CPU 导出曾做一次准备尝试：GRPO merger 被 SIGKILL 终止。现场 cgroup 内存上限仅 **2,147,483,648 字节（2 GiB）**，不足以合并 4B FSDP 分片；没有生成推理权重，原 checkpoint 未改动。失败记录与现场内存证据保留，不在原资源配置下重试。准备脚本现先检查至少 64 GiB 容器内存（建议分配 128 GiB 或以上），资源就绪后再新建一次 CPU 导出尝试；CPU 导出仍须先于任何模型服务启动。CPU证据整理已完成，推理导出与真实GPU评测尚未完成，二者明确区分。见 [资源失败证据](../results/analysis/grpo_improvement_evidence_20260920/cpu-export-failure-diagnostics.json)。

[机器可读评测计划与 argv](../results/analysis/grpo_improvement_evidence_20260920/evaluation_plan.json)、[远程 CPU dry-run](../results/analysis/grpo_improvement_evidence_20260920/remote_cli_dry_runs.json)。

## 6. 复现与验证范围

分析代码在 `results/analysis/grpo_improvement_evidence_20260920/`，不新增生产框架。`collect_remote.py` 只读远程文件，`hash_models.py` 只读取权重字节，`analyze.py` 复用项目比较器，`inspect_cases.py` 按固定切片提取案例，`finalize.py` 生成身份与评测计划。

本次没有修改生产训练、奖励或评测实现。验证采用实际产物完整性、旧哈希一致、每模型/节点任务与条数、指标复算和双端 CPU dry-run；没有把历史测试数量重复计成本轮新验收。首次本地解包遇到 Python 版本不支持 `tarfile.extractall(filter=...)`，已改为白名单普通文件提取；原始归档保留，未影响远程记录或评分。


## 7. 19:22 预检与正式运行回执

Base/SFT/GRPO/MT各4条预检，均正常user_stop，零执行异常；工具调用/对应返回分别22/23/29/20，共94对。多工具消息分别3/4/5/5条，模型生成usage均有记录。四组模型内容回执与各自run记录相符，任务/DB/评测器/harness哈希跨组一致；正式四份run各240条，同一协议与计划。这里只证明该预检范围，没有把4题成功率作为模型比较结论。30份下载证据均核验hash。详见[预检核对](../results/analysis/grpo_improvement_evidence_20260920/execution-1852/smoke-evidence-1922/smoke-audit.json)。


## 8. 本轮失败收尾与CPU边界诊断

MT-GTPO airline_698/trial0/seed42的HTTP请求超过上下文至少1 token，被通用异常分支归类为基础设施错误，触发19:41:20整组停止。具体23553输入+1024申请输出=24577，上限24576；不代表模型实际生成了1024个新token。Base40、SFT49、GRPO48、MT49共186条完成，另1异常和773条无最终回执。严格比较器核对协议/身份记录相符，但三组均因不完整返回differences=null。不得据这批任务子集排名或决定主要算法改动。

85份失败证据已下载并逐文件hash通过；控制器清理了19个本轮进程/后代，存活0，GPU核对无进程。实例051未关机，平台计费仍继续。原完整checkpoint与两份精确导出均保留，不重跑已通过导出。

CPU原生循环5场景验证了修补方向：只将策略/模拟器生成端typed ContextWindowExceededError收束为预算终止，保留之前已执行工具、消息及错误来源；不补造生成，不缩短输出、不扩上下文、不自动重试。一般BadRequest仍是异常，正常STOP不变。候选仅在内存中测试，生产源码未修改或部署，真实服务验证待后续有界尝试。原错误轨迹缺少simulation，不能靠事后改标签恢复成完整数据。

[失败回执](../results/analysis/grpo_improvement_evidence_20260920/execution-1852/failed-attempt-1952/failure-receipt.json)、[不完整比较审计](../results/analysis/grpo_improvement_evidence_20260920/execution-1852/failed-attempt-1952/incomplete-audit.json)、[CPU候选验证](../results/analysis/grpo_improvement_evidence_20260920/execution-1852/context-boundary-cpu/verification.json)。

## 9. 用户确认：保留旧评测行为，仅统一条件并补全异常记录

2026-09-20，用户确认旧评测主要行为可接受，优先比较 Base / SFT / GRPO / MT-GTPO。使用现有 `tau3_eval_legacy_v1`，不新增 v5、不改训练算法，也不把训练和独立评测完全一致作为必要条件。

| 条件 | 下一次四模型统一取值 |
| --- | --- |
| 开场 / 工具返回 | 原生问候后模拟用户开场；原始工具返回，不启用 v3 固定开场或字符投影 |
| 交互预算 | 原生 30 Orchestrator step；累计工具错误上限 10；不是 30 次策略生成 |
| 策略 / 用户温度 | 0.7 / 0.7 |
| 单次输出请求 | 不显式传 `max_tokens=1024`，统一服务端 `generation-config=vllm`；不等于无限输出 |
| 上下文 / thinking | 策略 24,576；模拟器 16,384；两侧 non-thinking |
| 任务 / seeds | 同一 selection60，各 4 次，42/43/44/45；四模型 960 条，固定 16 条预检另计 |
| 模型 | 已核验的 Base、SFT new-off、GRPO E0 step20、MT reference_write/v3 DF off step20；复用导出 |
| 错误处理 | 单条执行异常完整保留已有消息/工具结果/待处理消息/双方状态/task/当前DB/角色/step/堆栈，其他正式 trial 继续；不重试不补样 |
| 全局停止 | 硬时间上限、服务退出/OOM、身份不一致、产物缺失或损坏仍停止并保存已产生记录 |

旧原生 STOP 与 step 边界优先级仍保留；这是用户选择复用旧主要行为的明确边界，不宣称已修完旧协议所有问题。上下文异常仅对真实 `ContextWindowExceededError` 作预算终止，保留轨迹和角色/堆栈，记 0 结果但不标记官方已评分。普通 BadRequest 等异常写入 `errors.jsonl`，附 `execution_evidence` 与 traceback；若某个状态组件不可捕获，记录 `capture_errors`，不以保存错误覆盖原始异常。只能保留实际发生且程序可获取的记录，不补造未生成内容。

运行器原本已逐条收集错误；上一轮外层有界控制器会在发现非空 errors 文件时停止整个任务。本次下一轮控制器草稿已取消该行为，允许 formal 阶段按“成功记录 + 错误记录”核对全部 960 个身份。预检仍须通过服务/工具完整性验证，正式阶段的单条错误不触发全局停止。最终 `completed_with_errors` 与 `passed` 分开，异常不冒充模型失败，不把不完整四模型结果用于完整排名；已有每任务诊断保留。

生产修补位于 `envs/orchestrator.py` 与 `evaluation/runtime.py`，协议元数据记录异常处理语义；原 v2/v3 也复用该防护，v4 路径未变。本地/远程针对性 CPU 回归各 85 passed，四模型 CLI dry-run 两端均通过；控制器合成 960 条（959 完成 + 1 异常）验证全部收集、无补样、禁止未授权启动。真实 GPU 效果未验。历史 0.4/1.0 成绩、本轮失败 v3 的 186 条与下一次 legacy 0.7 结果分别保存，不能拼接成 960 条或直接相减。

下一次命令/五服务配置及控制器保存在 [CPU 准备目录](../results/analysis/grpo_improvement_evidence_20260920/legacy-t07-revision/next-attempt/execution_plan.json)。沿用通用 CLI，四份 argv 明确 legacy、0.7/0.7、30/10。草稿 `gpu_authorized=false`、`deadline_unix=null`，不能启动；上一轮预算和失败目录原封保留。确认下一次 GPU 预算、重新核对源码快照和服务身份后才可发布到新运行目录。


## 10. 新授权 legacy 正式评测启动（2026-09-20 20:22）

用户在第 9 节条件与完整异常记录确认后明确要求“你赶紧启动评测吧”，据此发布新尝试，没有重启旧失败尝试或修改其预算。新根 `results/runs/grpo_improvement_selection/20260920_legacy_s42_t07_a800`，控制器 PID 8131 / create_time 1789906979.09。准备开始 1789906971.4004269，包含本次准备/校验/加载的绝对硬截止 1789921371.4004269（北京时间 2026-09-21 00:22:51）；五卡预留最多 20 GPUh，完成后只停止本次服务。

启动前核对 051 身份 `857546be50-ec463c07`、5×A800、GPU 空闲且无其他评测进程。1,021 源码/配置/脚本文件与本地逐文件 hash 匹配，归档 source snapshot、Git HEAD 与 dirty patch。GRPO/MT 复用此前精确校验的 724 张量导出；四策略及模拟器重新计算模型身份并与上次固定哈希相符后才加载服务。正式条件见第 9 节，16 预检与 960 正式分别记录。

控制器已确认存活并进入 service_loading；尚无正式分数。后续按真实回执更新，单条异常继续收集，completed_with_errors 不冒充 passed。自动跟进已恢复，只在实质进展/结束/失败时通知，不自动增加试验。未获 051 平台关机授权，作业结束不等于平台计费结束。

本地回执：[controller-launch.json](../results/analysis/grpo_improvement_evidence_20260920/execution-legacy-20260920/controller-launch.json)；已下载 9 份启动小文件并逐文件 hash 核对。第 9 节“GPU 草稿禁止启动”描述为此次授权前的准备状态，原草稿继续保留；实际发布的 execution_plan 使用新路径和本次授权/截止时间。


## 11. STOP 边界修复、旧产物删除与从头重评

用户明确要求立即修复并重跑。`EvaluationOrchestrator` 仅在 `done` 且终止原因为 USER_STOP/AGENT_STOP 时跳过后续预算覆盖；STOP 不直接赋成功分，仍由原官方评分器判断。legacy 元数据记录 `stop_priority=completed_stop_before_budget_v1`，未改变开场、30 step、10 工具错误、raw 返回或 0.7/0.7 条件。typed context/普通异常完整记录策略保留。

两端各 90 项 CPU 回归通过；新增真实原生循环对照：第 5 step 收到 STOP 时，模型调用与消息完全相同，旧原生结果为 max_steps，新适配器为 user_stop。其他预算边界保持一致，只有 done 的 STOP 才优先；未完成状态仍执行步数/错误检查，lint 与 diff 检查通过。

旧 legacy 控制器 8131 在 smoke 阶段（4/4/4/3 条已保存）按用户请求 SIGTERM，16 个已捕获进程清理后存活 0，GPU 无进程。远程三个目录 `20260920_s42_t07`、`20260920_s42_t07_a800`、`20260920_legacy_s42_t07_a800` 及旧证据副本已删除；本地 execution-1852、execution-legacy-20260920 与失效启动草稿也已删除。未删除历史完整实验、原始检查点或代码。两份推理导出原子迁移至 `results/exports/grpo_improvement_selection_step20_20260920/{grpo,mt_gtpo}`，文件集合/inode/大小保持一致，新服务重新核对原固定内容哈希再启动。保留[清理回执](../results/maintenance/selection-restart-20260920-stop-priority/cleanup-manifest.json)、[本地清理](../results/maintenance/selection-restart-20260920-stop-priority/local-cleanup.json)及[迁移回执](../results/maintenance/selection-restart-20260920-stop-priority/model-relocation.json)，未另行保留完整中断轨迹档案。

新运行根 `results/runs/grpo_improvement_selection/20260920_legacy_stop_s42_t07_a800`。2026-09-20 20:43:09 开始，固定截止 2026-09-21 00:43:09（1789922589.115497），4 小时/20 预留 GPUh；PID9658/create_time1789908192.26。1,021 源码文件两端一致并归档源码/dirty patch。五服务进程存活且处于加载阶段；16 新预检后从头跑 960 新正式轨迹，不混入旧结果。异常继续留档、不补样、全局故障和预算保护保留。自动跟进已切换本轮；未授权 051 平台关机。

[本轮启动](../results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/controller-launch.json)、[实际计划](../results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/execution_plan.json)。


### 11.1 修复后真实预检完成与正式评测开始（20:57 核对）

四模型各 4 条，16/16 均为正常 user_stop，零执行异常；工具调用/返回对数 Base26、SFT25、GRPO26、MT31，共108对，ID逐一对应；多工具消息分别3/3/3/7。五服务实际加载身份与固定模型哈希一致；预检与正式四组的任务/DB/harness/evaluator身份分别对齐，seed与0.7/0.7温度符合计划。STOP元数据为completed_stop_before_budget_v1。此预检不用于模型排名，也不能单凭正常结束声称覆盖了真实STOP恰好撞预算的情况。

正式评测自北京时间20:50:43开始。20:57核对共55/960条，Base12、SFT16、GRPO13、MT14，执行异常0，进程存活；结果尚不完整，不排名。33份稳定小文件已下载逐文件SHA256核验。控制器原硬截止2026-09-21 00:43:09保持不变。

[预检审计](../results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/smoke-evidence-2057/smoke-audit.json)、[进度快照](../results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/smoke-evidence-2057/progress.json)。


### 11.2 真实上下文预算终止与留档验证（2026-09-20 21:34）

SFT 的 airline_698 trial1/3（seed43/45）在策略生成端实际触发 typed ContextWindowExceededError：输入至少24,577 token，超过24,576上下文。两条均按budget_limit结束、结果记0且不标为官方已评分，没有重试或补样，也未中止其他trial。各自保留20条消息、7条工具返回和异常堆栈；下载记录SHA256核验通过。这覆盖了ERR-029策略端真实HTTP边界，模拟用户端目前只有CPU回归覆盖。

21:34快照为392/960（Base89/SFT100/GRPO103/MT100），四组执行异常均0，控制器继续selection；这不是最终成绩或模型排名，00:43:09硬截止不变。

[真实边界审计](../results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/context-evidence-2132/audit.json)。


## 12. 四模型统一 selection 最终完成

STOP修复版于北京时间2026-09-20 23:00:08完成并清理；含准备2.283小时，按五卡预留折合11.416 GPUh，未触及4小时/20 GPUh上限。全部四组各240条，0执行异常、0缺失；16条真实预检单列。960个task/trial/seed身份与计划逐一一致，四模型内容/服务回执及共同任务、DB、harness、评分器身份通过核对，分数从轨迹复算与summary完全一致。44份最终文件逐文件hash通过；旧预检33份已核验，不重复下载。原始轨迹、工具调用/返回、奖励细节、预算原因与上下文堆栈远程/本地均保留，errors文件为空是因为本轮无执行异常。

|模型|成功条数|pass@1|pass@4|pass^4|预算终止|
|---|---:|---:|---:|---:|---:|
|Base|120/240|50.00%|70.00%|30.00%|25|
|SFT new-off|108/240|45.00%|71.67%|16.67%|25|
|GRPO E0 step20|99/240|41.25%|68.33%|21.67%|26|
|MT reference_write/v3 DF off step20|95/240|39.58%|65.00%|16.67%|22|

预算终止按预定规则记0留在240条分母中，不能标为官方已评分。SFT−Base为−5.00pp（95%任务配对CI −13.33至+2.92）；GRPO−SFT为−3.75pp（−11.25至+3.33）；主对比MT−GRPO为−1.67pp（−8.33至+5.00）。区间全部跨0，本轮未显示提点，不能据此断言跨训练seed稳定变差。历史0.4/1.0与当前legacy STOP修复/0.7条件不混合。另一个获用户授权的校准客户端复用了SFT/模拟器服务，本轮不是独占资源吞吐实验，不据完成速度作算法效率比较。

60任务四模型表、538条失败/预算索引、10条定向人工复核已产出。业务参数错误、参考形式差异、模拟用户偏离固定参考、预算终止分别标注；定向案例不估计原因占比。只建议一个下一步主要因素：**写操作业务参数与目标状态的一致性**，先用历史训练buffer离线核对信号，不修改当前模型/算法、不启动训练、不将selection参考直接喂回训练。历史v4过程奖励/信用审计不当成本次MT-v3效果的因果证明。

清理核验：本次进程及后代存活0，nvidia-smi计算进程为空；停止服务不等于平台关机，051仍未执行关机。原训练检查点和导出均保留。正式多seed/单因素训练及final50仍是后续另行决定的研究。

[完整结果与案例](../results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/final-analysis/report.md)、[停止/下载回执](../results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/completed-evidence/final-download-and-stop-audit.json)。


## 2026-09-21 四模型 badcase 全量 CPU 重放

960条/60题、6,430次工具返回一致，862条官方结果全部复现；98条预算只诊断DB，不改分。GRPO相对SFT补回27条、丢36条，MT相对GRPO补回29条、丢33条；净失分主要是官方DB失败增加，不是预算增加。纯乘客数组顺序失败6条单列，模拟用户偏离固定参考与真实参数错误分开标注。历史GRPO63/160组终局奖励全同；MT reference_write/v3有25条官方失败拿满正工具奖励，其中12条有中性非参考写操作。以上是单因素反馈研究假设，不是因果证明；未启动采样/训练，未修改奖励或分数。

[完整诊断与复现证据](selection_badcase_analysis_20260921.md)。


## 2026-09-21 按 MT→SFT→GRPO 完成训练信号链审计

实际MT reference_write/v3全部1280条、12591轮的奖励/优势/mask回放一致；独立优势分解误差4.663e-15。25条拿满正工具奖励的官方失败中，12条中性非参考写入所在轮仅3条正优势、9条负优势，三正轮都混有参考正奖励调用。59个官方失败工具错误轮为正优势，其中52轮无后续正奖励写入；同轮混合信用及相对未来贡献是已定位风险，非算错公式证明。非参考取消存在用户同意且24小时内的合法替代，不能一律负奖。

实际new-off输入/远程effective messages及原生token渲染通过核对，最佳为checkpoint30、val loss0.422255，不能混用旧SFT-003的checkpoint24。E0原始20份rollout hash与前审计一致；正/负/零终局优势396/380/504，20步极值与在线日志一致，无失败轨迹正终局优势。E0缺逐token历史payload，未冒充重现优化器更新。下一步优先检查MT局部与长程信用组合，随后处理已证实SFT数据问题；不立即另起奖励/训练。仅CPU，未改任何官方分数。

[完整报告与来源](training_credit_chain_20260921.md)。
