# 错误与风险回顾索引

**最终状态（2026-09-20 23:00完成，现已核验）：四模型960条全部完成，0执行异常/0缺失；Base/SFT/GRPO/MT成功率50.00%/45.00%/41.25%/39.58%。本次GPU服务已清理，051实例未关机。下文较早进度、失败与退役路径只作为历史记录。**

**当前更新（2026-09-20 20:43）：按用户明确要求，中断上一轮、修复 STOP 优先级并从头重评。新控制器 PID 9658，截止 2026-09-21 00:43:09；当前五服务加载中。之前三次中断尝试的运行产物及本地对应副本已清理，保留原始检查点、独立目录中的两份导出及简短删除回执。下文旧尝试路径已退役，旧运行状态为历史记录。**

## 2026-09-20 harness固定轨迹差分

第二步核对在真实训练/独立评测循环中复现STOP边界覆盖、预算/错误上限差异和可见observation裁剪差异。固定生成不调用模型；记录新的问题与协议边界，本轮不修改生产行为或旧结果。未知工具失败仍缺DB hash回执，亦在报告中保留。[第二步报告](docs/architecture/harness_step2_audit_20260920.md)。

## 2026-09-20 训练分析新增风险

第一步CPU审计发现两项需要在后续效果分析中披露的边界：官方DB评分不能覆盖所有对话要求；当前任务留出不是数据库/实体留出。它们是评测覆盖与结论范围风险，不等于新发现了训练实现故障。历史分数、数据划分与奖励实现保持原状；详见[第一步审计](docs/architecture/training_step1_audit_20260920.md)。

## 2026-09-20 本轮收尾状态

本轮所有有界GPU工程验收已通过；原失败尝试仍保留，未覆盖为成功。ERR-017的CPU/GPU参数化混跑已通过隔离复验，ERR-018端口预检修复有15项回归；三算法完整采样身份/生成logprob和活动文本权重、MT两组重放及GiGPO真实恢复均有回执。DF on真实两批均无零信号组，因此只证明开关/重放及正常更新路径，未证明真实剔除收益。最终范围与剩余独立研究见[接口报告](docs/architecture/interface_acceptance_20260919.md)。

下表历史状态按对应实验时间解读，不把后续通过回写成原失败不存在。

更新：2026-09-19。这里集中导航与结论，原始失败日志、修复细节仍保留在各报告；不复制一份失去同步的故障全文。修复实现、回归通过与新实验效果分别记录。

|编号|问题及影响|状态 / 回顾依据|
|---|---|---|
|ERR-001|实际执行与 gold replay 共用可变 DB，可能制造虚高成功率|已隔离并重评分；历史旧分数不能继续作为有效基线。[记录](docs/sft_comparison_20260908.md)|
|ERR-002|SFT 梯度累积重复除法、最佳 checkpoint 标记问题|已有修复及对应 SFT 对照；不能沿用文件名推断真实最佳模型。[记录](docs/sft_comparison_20260908.md)|
|ERR-003|工具 observation 截断到 256 字符，模型无法看到完整结果|提高上限并审计参考返回；reward=1 仍不保证回复业务文本完全正确。[记录](docs/observation_budget_20260909.md)|
|ERR-004|全词表 logprob OOM / 磁盘满中断|未完成更新不能记作“模型成功率 0”；分块与续训分别验收，tracking FINISHED 不是成功依据。[记录](docs/rl_validation_comparison_20260909.md)|
|ERR-005|Qwen3.5 导出重复 language_model 前缀，推理加载失败|修复格式导出并精确验证 724 个张量；训练、导出和评测状态分开。[记录](docs/post_rl_selection_20260914.md)|
|ERR-006|E2 独立评测一条上下文异常|原 239/240 仍无有效总排名；补测是单独新尝试，不覆盖原失败。[记录](docs/post_rl_selection_20260914.md)|
|ERR-007|过程奖励参数匹配未对齐实际转换；排序列表改变真实 DB 语义|使用新版本环境语义匹配，保留旧版本回放，不能用排序掩盖状态差异。[记录](docs/mt_gtpo_environment_fix_20260918.md)|
|ERR-008|提前终止的记录 0 分与已完成官方评分混淆，查询奖励跨任务相关性误导|读取器与分层诊断已修；新 split 奖励只有开发验证，未证明新模型收益。[读取修复](docs/mt_gtpo_reward_transfer_fix_20260918.md)、[split](docs/mt_gtpo_split_reward_revision_20260918.md)|
|ERR-009|重复 anchor 不等于跨轨迹有区分性的 GiGPO 信号；DF 前后位置会改变可用信号|历史审计事实已记录；因果收益与新行为修复待独立实验。[信号审计](docs/training_signal_audit_20260914.md)|
|ERR-010|语义状态与直接判等误合并/弃权|候选仍用于研究，不默认上线；双向一致不能保证语义正确。[记录](docs/semantic_pair_direct_v1_20260916.md)|
|ERR-011|rollout 原始 stop/length 被归并 completed|已增加原始原因及全算法事实；GiGPO 两步又覆盖 3 个真实 length，精确 token/span/mask 对齐。单轮 length 与轨迹终止分开；不宣称所有长度边界已覆盖。[新增证据](docs/architecture/gigpo_acceptance_20260919.md)|
|ERR-012|完整 process-turn 记录只走 MT-GTPO 分支；旧评测缺 task DB / harness source hash|新正式 runner 统一事实层，新评测自动记录任务/DB/源码哈希；旧记录保持原状，缺失身份仍披露。9月20日增强A三算法实际采样身份、完整生成logprob和活动文本权重检查全部通过；MT更新后的接口仍待B。[接口证据](docs/architecture/interface_acceptance_20260919.md)|
|ERR-013|续训曾只校验收据列出的文件，可能漏掉优化器/rank 状态；本批统计曾依赖全局 last_stats|统一完整恢复文件集合、路径/大小/marker 校验；trainer 使用本批显式 diagnostics。CPU 回归及GiGPO真实GPU step2→3恢复已通过，四rank状态/同云端run 1/2/3/数据16→24核对一致；不宣称与不中断训练逐位等价。[恢复证据](docs/architecture/interface_acceptance_20260919.md)|
|ERR-014|完整 GPU 工程验收受冷启动、更新与全状态保存耗时限制|9 月 19 日完成 A 和 B-GRPO 后，剩余约 34 分钟低于下一组约 56 分钟的实测推算，主动收尾；原控制器 SIGTERM/failed 回执保留，汇总明确为预算收尾，并非 GRPO 数值失败。其余 B/C/D 未完成，5 卡释放。[记录](docs/architecture/gpu_acceptance_20260919.md)|
|ERR-015|CI push 触发仅列 master/dev，遗漏实际默认分支 main|已补 main 与手动触发，远程 CPU 检查通过；未发布本批代码，尚无托管运行回执，不能视为 CI 已验收。[收尾记录](docs/architecture/cleanup_20260919.md)|
|ERR-016|单组验收控制器重复检查 step1 边界，可能在 step2 已开始后按“下一步预算不足”误停|改为每个完成边界只判断一次，9 项控制器 CPU 回归通过；运行中通过有 PID/创建时间记录的 CPU 守护接管监督，原训练未重启、未延长截止时间。两步完成，原执行快照和接管过程保留。[记录](docs/architecture/gigpo_acceptance_20260919.md)|
|ERR-017|GPU数值补验混入CPU参数化用例，GPU可见时CPU tensor误进Triton|原尝试保留failed（17通过/4 CPU失败）；隐藏CUDA的CPU完整回归已通过，GPU-only另目录17项全部通过。未修改生产计算逻辑。[记录](docs/architecture/interface_acceptance_20260919.md)|
|ERR-018|独立评测结束后增强A的端口bind预检瞬态失败，队列又因无阶段回执掩盖原错误|未开始新采样；保留原日志。预检使用SO_REUSEADDR但仍拒绝监听占用，15项控制器回归通过；新v3队列保留原退出码/日志入口。[记录](docs/architecture/interface_acceptance_20260919.md)|
|ERR-019|paper_env_split_v3的state_change仍混入GENERIC转人工；自由文本summary未精确匹配时扣作状态修改|2026-09-20远程CPU诊断确认36次state_change中21次是转人工，8次发生于官方成功轨迹；基准函数不改DB。尚未修订，需独立版本分开GENERIC/DB_WRITE并重做校准；不改旧分数、不推断所有转人工都正确。[记录](docs/mt_gtpo_engineering_reward_diagnosis_20260920.md)|
|ERR-020|train50有2题、selection60有4题参考执行不改变DB，正常结束的无工具合成探针均得官方1分|已复现，评测覆盖风险待专项设计：DB+空COMMUNICATE不能认证查询、回复、授权或转人工；不删除原任务、不覆盖官方分数，后续辅助指标独立版本化。[CPU证据](docs/architecture/training_step1_audit_20260920.md)|
|ERR-021|train50/selection60全部共享四套DB模板，391/3,000任务对共享参考实体|精确任务检查无重叠发现，但不能声称world/entity-disjoint泛化；语义去重仅复核前三对，final50未打开。后续如需新环境结论须另建隔离评测协议。[数据审计](docs/architecture/training_step1_audit_20260920.md)|
|ERR-022|独立Orchestrator在已收到用户STOP后仍以step_count>=max_steps覆盖为MAX_STEPS，导致原本正常结束的轨迹记为未评分0|新 `tau3_eval_train_control_v2` 项目适配器已修复终止优先级；legacy 原样保留。两端原生 CPU 对照通过，未宣称真实模型效果提升。[修复](docs/architecture/harness_fix_20260920.md)、[原反例](docs/architecture/harness_step2_audit_20260920.md)|
|ERR-023|训练15轮与独立30step计数语义不同；独立累计10个工具错误会终止，训练仍可能继续|v2 显式采用当前正式训练的 15 轮/观察预算、无单独累计工具错误上限；两个相同目标 DB 的反例已在两端恢复为训练/评测均官方1。token/context 边界仍待对齐，旧协议保留。[修复与范围](docs/architecture/harness_fix_20260920.md)|
|ERR-024|训练长工具返回按65536字符middle裁剪，独立LLMAgent收到完整返回，模型可见输入不同|v2 与训练共用可见工具投影；76736→65553字符及 raw/visible hash 两端一致，官方轨迹保留原文。已补派发失败的现场 DB hash，含两次真实写入之间的回执测试。实际裁剪对模型效果的因果影响仍未验证。[修复](docs/architecture/harness_fix_20260920.md)|

## 新错误模板

- ID / 首次发现日期 / 实验 ID / 当前状态 / 负责人或后续任务。
- 触发条件与症状：task/trial/seed、step、算法/reward/harness 版本、错误日志和最小复现。
- 原因：已证实事实与假设分开；说明是算法、环境、数据、服务、数值、资源还是记录问题。
- 影响：哪些分数/检查点/实验结论失效，哪些可以继续使用；是否涉及数据曝光。
- 修复：变更源码、配置版本、兼容与旧结果保留方式。
- 验证：CPU / GPU 更新 / 续训 / 独立评测各自证据，未验证项明确列出。
- 关闭条件：对应风险的验收证据；不要用“单测通过”替代真实训练/推理结果。

### ERR-019 后续：2026-09-20 split v4 分类修订

新增独立v4将成功转人工归为generic=0，错误执行保持惩罚。1,536条固定权重回放确认仅108次转人工即时奖励改变，旧split v3完整payload保持一致。该分类问题在v4中已修复；支持不足和错误轮Hybrid正方向仍未解决，不能认定正式配方通过。旧版本保留历史语义。见[修订与对照](docs/mt_gtpo_split_v4_20260920.md)。


## 2026-09-20 Harness 修复与新旧协议核验

新增显式 `tau3_eval_train_control_v2`：修正 STOP 优先级，按当前训练轮数/错误策略控制独立评测，共享模型可见工具裁剪；保留 legacy 及原评分。派发失败现补录真实 DB hash。已完成本地/远程各 129+1=130 项 CPU 用例，新旧协议各 22 例、跨主机共 44 组诊断见证匹配；两端 lint 零新增、vendor 通过，21 文件最终哈希一致。追加远程单项首次 300 秒超时保留，独立离线诊断尝试 66.09 秒通过。尚未证明真实 token/context 等价或模型效果提升，ERR-020/021 保持开放。详见[修复与验证记录](docs/architecture/harness_fix_20260920.md)。

## 2026-09-20 输入协议与工具 schema 修复

- **ERR-025**：训练固定 reason_for_call 开场，独立评测先问候再生成开场；新 v3 共用开场、显式每轮1024 token及non-thinking。策略评测温度按Mercor参考设为1，用户模拟器评测保留1；不能把训练温度0与评测1的区别本身当故障。旧协议保留。
- **ERR-026**：veRL schema注册丢失 `$defs`/数组items/嵌套字段。新增 opt-in `tau3_full_schema_v2` 保留完整序列化；旧默认保持复现。与v3配对后selection60首轮token匹配，未做GPU或证明收益。
- **ERR-027**：真实Qwen模板多轮重渲染与训练原始token累加不等价，已用空thinking历史和合法紧凑XML复现。首轮对齐不能关闭此问题，累计context/HTTP/parser仍开放。

实现、失败尝试与验证范围见 [本轮报告](docs/architecture/harness_inputs_20260920.md)。

本轮验证收尾：远程主回归148项、两端收尾配置回归各38项均通过（有重叠）；19文件hash一致。ERR-025/026在显式新版本下完成CPU修复，ERR-027仍开放；旧默认没有自动启用新训练schema。


## 2026-09-20 多轮 token 与预算协议 v4

新增 opt-in `tau3_token_budget_v1`（训练）与 `tau3_eval_token_v4`（同条件诊断 selection），共用原生执行循环和完整 schema、保留真实 token、统一累计预算。本地/远程主要 CPU 回归各256项、收尾协议专项各42项通过（两组有重叠，不相加），两端lint零新增/vendor通过，tokenizer三文件hash及27份本轮变更文件一致；未启动 GPU，无新模型效果结论。ERR-027 的多轮重渲染/预算差异已有共享实现，真实服务 token/logprob 验收仍开放；旧版本与旧结果保留。详见 [实现与验证](docs/architecture/token_harness_20260920.md)。

本次定位澄清：它不是所有评测必须与训练一致的要求；Mercor正文明确160k训练/256k评测，并进行跨harness评测。公平比较要求被比较模型使用同一评测条件，TITO主要保护训练采样与学习的token身份；独立正式评测协议仍另行确定。

ERR-027 定位补充：已确认的是两种执行路径存在差异，不是“训练与独立评测不同即为错误”。只有实验声称同条件、却遗漏这些差异时，才形成可比性问题。v4提供诊断选择；不以消除全部差异作为正式训练的强制完成条件。


## ERR-028：2026-09-20 CPU 推理导出受容器内存上限阻断

CPU 推理导出准备受资源阻断：当前远程容器内存上限仅 2 GiB，GRPO 合并进程被 SIGKILL，未生成推理权重；MT 导出未启动，原检查点未改。失败证据已保留（ERR-028），不在同资源下重试。重新导出前至少分配 64 GiB 容器内存（建议 128 GiB），完成导出核验后再启动获批的 GPU 评测。

现场内存上限为 2,147,483,648 字节，峰值约到上限、failcnt=36，merger 返回 SIGKILL。现场累计 oom_kill=19 没有运行前计数，不能当作本次导致 19 次 OOM；日志中的 DeviceMesh 警告也不足以判定代码错误。本次为资源失败，不是算法或模型成绩失败。已增加导出脚本内存预检；当前没有成功导出回执，不能将 dry-run 写成真实推理通过。关闭条件：资源调整后使用独立尝试目录，完成 GRPO/MT 两份导出、逐张量验证和来源校验。

证据与计划：[四模型 CPU 报告](docs/grpo_improvement_evidence_20260920.md)；现场记录：`results/analysis/grpo_improvement_evidence_20260920/cpu-export-failure-diagnostics.json`。


### ERR-028 后续：新资源导出通过

用户提供43294新连接，实测cgroup内存600 GiB、5×A800。GRPO与MT两份CPU导出均完成724张量精确核验，源权重与已固定哈希一致，资源阻断已解除。原2 GiB失败记录保留。新运行目录为 `results/runs/grpo_improvement_selection/20260920_s42_t07_a800`。

### 2026-09-20 本轮运行产物路径纠正

一次性CPU导出模板的相对输出字面量仍指向旧准备目录，覆盖了旧300字节失败日志；该原日志在本地既有失败证据中完整留存。完成本次两份导出后，将新模型与日志移入新尝试目录，导出文件逐个hash相符，保留迁移前收据并恢复远程旧日志。未修改原模型/续训检查点，未重跑导出或重置预算。证据：`results/analysis/grpo_improvement_evidence_20260920/execution-1852/path-correction.json`、`relocation.json`。


## ERR-029：2026-09-20 独立评测的HTTP上下文超限被归类为基础设施异常

首次真实触发：051四模型selection，MT-GTPO airline_698/trial0/seed42。vLLM拒绝至少23553输入+1024输出（至少24577）超过24576上下文的请求。协议声明context_window_exceeded为预算结果，但项目v3原生循环没有接住typed ContextWindowExceededError，runtime通用except将其归为基础设施异常。控制器按既定停止条件正常停掉全组；不是GPU OOM或模型权重加载失败。

影响：186条已完成、1条异常、773条无最终回执；无法四模型完整排名，不能说明MT算法更差。该异常行没有simulation，无法凭错误字符串恢复完整轨迹或追溯改成可信已完成0分。原日志与分数保留。

CPU使用真实环境/工具与mock生成，在原生循环复现通用异常归类与轨迹缺失；内存候选在生成角色边界仅接住typed context异常，设置context_window_exceeded并保留原工具/消息/错误来源后走正常finalize。策略和用户两角色覆盖；普通BadRequest即使文本提到context仍报异常，正常STOP保持原样，5场景通过。无模型API/GPU，未修改生产文件或部署。建议修补中保持24576/1024和官方评分不变，不自动缩短输出、不扩上下文、不重试；保留明确处理修订及源码hash，真实服务边界仍需验证。

关闭条件：生产补丁与针对性回归、真实HTTP边界回执通过；重新评测需独立预算决定。证据：`results/analysis/grpo_improvement_evidence_20260920/execution-1852/context-boundary-cpu/verification.json`；失败包SHA256 `f7bc4b16649139ba8f29116b18e3d8a87989d65660a707de4c40dac3d6b02387`。


### ERR-029 后续：legacy 条件与完整异常记录（2026-09-20）

本次生产适配器仅将 typed 生成上下文超限转为预算终止；普通执行异常保留原类型、堆栈、此前消息/工具结果、待处理消息、双方状态、任务与当前数据库。正式单条错误不再要求外层草稿控制器取消其他模型/trial；无补采样，服务崩溃及硬预算仍停止。上下文适配器与原生 legacy 正常路径 CPU 对照一致，本地/远程相关回归各 85 passed；控制器合成 959 完成/1 异常保留通过。真实服务仍待后续有界尝试，旧失败不能倒填为成功。未设置 1024 token 本身不是旧协议错误；训练/评测输入不同也不自动构成错误。[报告第 9 节](docs/grpo_improvement_evidence_20260920.md)。


### ERR-022 / ERR-029 最新状态：2026-09-20 20:43

用户明确选择修复 legacy 的 STOP/预算覆盖边界并中断重评。项目适配器已保护完成的 USER_STOP/AGENT_STOP，官方评分器不变；两端各90项CPU通过，真实服务重新验正在加载。用户授权删除中断产物已执行，详细失败轨迹不再保留，简短停止/删除/模型迁移记录见 `results/maintenance/selection-restart-20260920-stop-priority/`。新异常记录仍要求保留完整可获取现场。


### ERR-029 真实服务验证：2026-09-20 21:34

STOP修复版正式selection中，SFT airline_698 trial1/3（seed43/45）实际策略输入至少24,577 token超出24,576上限，typed ContextWindowExceededError被正确保留为budget_limit终止，结果0且不冒充官方评分。每条20条消息、7条工具返回、原异常堆栈均已下载并核对SHA256；无自动重试，其他trial继续，21:34全组执行异常0。生产补丁、双端90项CPU回归及真实策略端HTTP边界已验证；模拟用户端真实触发与普通执行异常的真实触发尚未发生，不能扩大验收范围。

[真实边界审计](results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/context-evidence-2132/audit.json)。


### ERR-022 / ERR-029 最终真实服务回执（2026-09-20）

四模型STOP修复版960正式全部完成，0执行异常/0缺失。六条真实agent端ContextWindowExceededError（Base2/SFT3/GRPO1）均按budget_limit终止，已有消息/工具返回/堆栈保留，零重试补样；这些0结果不冒充官方评分。模拟用户端本轮无真实超限，普通执行异常本轮也未触发，覆盖仍以已有CPU回归为界。完成STOP恰好撞预算的优先级有CPU原生循环见证，不能把本次普通STOP结束当作新的撞边界实证。

完整hash、身份与停止核验通过，全部GPU计算进程清空。业务评分失败与模拟用户偏离参考、格式差异、预算结果已分开索引；不改写官方分数。538条失败/预算记录、10条定向复核见[最终报告](results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/final-analysis/report.md)。旧中断详细轨迹按用户授权已删除，不再恢复；本次960条与16预检完整保留。


### 2026-09-21 四模型失分诊断补记（非新增执行故障）

960条CPU重放的6430次工具返回、862条官方评分全部一致；本轮执行异常仍为0。模型业务错误与严格DB排序、模拟用户/参考偏离分开记录，保持原分数与完整轨迹。MT参考写入过程奖励与终局失败的并存属于待验证研究问题，不记为已证实框架故障。[报告](docs/selection_badcase_analysis_20260921.md)。


### 2026-09-21 训练信号链核对（机制风险，未认定框架故障）

MT-v3原奖励/优势与1280条保存payload一致；E0二值终局优势范围与20步在线指标一致。定位同轮正确/错误调用共享优势、无后续进展却获正相对未来贡献的风险，未证明实现算错或某动作概率实际上升。实际new-off输入/有效消息/渲染一致。远程曾连接拒绝，恢复后确认051并补取31份来源hash，不涉及GPU启动或平台操作。[报告](docs/training_credit_chain_20260921.md)。


### 2026-09-21 MT信用对照补记（非新增执行故障）

原始20份MT buffer hash、奖励/优势/mask回放一致，基线误差5.774e-15；独立重算时间候选误差1.388e-11。移除后续过程信用减少错误正信号，但仍有混合调用和前置无奖行为信用损失，不认定已修复训练效果。调用事件诊断不具备历史token边界，未上线。[结果与限制](docs/mt_credit_comparison_20260921.md)。


### 2026-09-26 双起点 SFT72 训练后验收接口不一致

两组9次更新和最终adapter已保存，原控制器因远端冻结trainer未输出`observed_training_dialogues`而报错并停止导出。实际源码哈希无漂移，但本地计数扩展与远端版本不同；启动前缺少验收字段兼容性检查。Base退出码0；SFT1退出码未记录，不补造。独立CPU恢复核对epoch1、optimizer step9、finite指标、effective数据哈希、最终adapter与checkpoint-9哈希一致后导出，未重训。72条消费量仅按调度推算，缺少运行时逐样本证明。原失败记录保留；详见[执行记录](docs/sft_decision_repair_pair_20260926.md)。
