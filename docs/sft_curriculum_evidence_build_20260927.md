## 2026-09-27 18:04 — 用户新指令：Codex审核到累计200，随后交Kimi K3

当前主账本180=历史70+teacher110，距200还20。owner审06后10（125不通过/126hold，预计8通过到188）；curriculum_audit仅07前10（131–140），最多10通过；owner精确补最后缺口，达到200后interrupt审核子agent且不再唤醒。停止的是Codex逐条语义审核，DeepSeek继续生成，后续全部pending Kimi K3 review，不擅自调用Kimi或猜端点。准备200冻结索引和待审原始对话/DB/API回执/hash/rubric交接资料。100元共享历史预算、每goal最多3总尝试、新unknown即停不变。自动化保持现有PAUSED状态。

## 2026-09-27 17:51 — 173合格，100轮继续batch06

batch05逐条完整审完16/20通过：096/100缺少旅行凭证更新禁用说明，097误称凭证可支付更新且未真正纠正，106多个信用卡中未事先披露/确认实际选择。失败保持同task/hash待freshDB重生成，未改教师gold可见性。完整native40calls（7writes）和全DBdiff已核验；账本133候选，103teacher+70历史=173，仍非最终500。控制器94943继续batch06子PID99669，本轮已生成40/100；09:50:38UTC预算快照38.497049元/3655calls，只2259历史reserve，无新增unknown。该金额仅时间戳快照，实时须读唯一budget。

验证38源审计报告已收：284只读calls/572独立核验、全DB不变，无源hold；9目标措辞矛盾需独立新草稿修正，21/28近重复只保留28，推荐37独立目标，尚无验收验证对话。curriculum_audit已接06前10独立审核。

## 2026-09-27 17:43 — batch04全20审核完成，157合格

batch04后10完整逐句判定并native回放15只读调用、全DB不变，20/20已合并。总157=历史70+teacher87，113候选审核、26drop/hold；不是最终500。新增fixed157真实tokenizer/mask审计：max11520，total1055417，assistant51213，0 exact duplicates；精确消息和mask未改，frozen render projection显式。round100_01继续生成05–08，不新增per20启动门禁。

## 2026-09-27 17:37 — 用户更新：100条一轮，147合格（未最终打包）

用户明确要求 DeepSeek 每100条生成后审核，不合格按原目标重新生成，替代旧每20条审完再启动的门禁。round100_01=remaining20_04–08，100个不同目标；技术分批最多20，可并发审核。控制器PID94943，远端 `round100_01_runtime.json` 是当前子PID与预算的唯一实时来源；最新子PID94987/batch05。不得另起重复控制器。每目标最多3总尝试，新unknown账单即停，全局历史100元预算及2259保留金额不变。

batch04前10独立逐条审核已按候选/消息mask/全DB证据哈希合并，累计147=历史70+teacher77；主账本103候选、26drop/hold。剩余353，147并非最终500，完整token审计仍只覆盖此前固定137。frozen prompt最小兼容补丁已部署远端，25项相关CPU测试通过，保留本地/远端已有训练代码差异；未进行SFT GPU训练。

## 2026-09-27 17:20 进展（部分137，非最终500）

第三批20/20通过：13行李额度与7订单摘要，33native读调用精确回放、完整DB不变。累计137=历史70+teacher67，主账本93候选，仍需363。partial137实际tokenizer/mask检查：max11520,total932316,assistant46777,0精确重复。

批次04已启动PID90872，20目标065–086（跳过071此前pilot和079源hold），起点37.562673元/3405calls，批次cap47.562673元/5405calls，全局100元。独立agent前10、owner后10审查。独立agent还交付reserve021–040草稿（17strategy+3constraints、114只读native与59条件核数），只在results/analysis/sft_reserve21_40_20260927/，尚未生成或算合格。最新唯一索引及automation已更新。

## 2026-09-27 17:10 进展（部分117，非最终500）

批次02的20条完成：19通过，039因没有profile观察却断言regular会员/0免费basic行李而drop，原候选保留。主账本73候选、teacher47通过，加历史70共117，仍需383。真实Qwen3.5 tokenizer/mask审计partial117：max11520,total809590,assistant43418,0精确重复。

批次03已启动PID86997：existing042–064中的20目标，跳过源hold049/063，056为此前pilot而不重复；起点37.289126元/3321calls，批次绝对47.289126元/5321calls、全局100元。代码/任务/独立rubric与上批review_gate绑定，不修改历史2259 reserve。批次04准备中；独立agent准备reserve021–040真实条件分支草稿。

验证旧60来源中22条与历史72来源用户重叠而剔除，仅剩38可作为新目标来源的条目/7用户。validation_source_goal_inventory仅来源盘点，未算合格验证样本，旧assistant不会进入验证集。

## 2026-09-27 17:00 后进展（部分98，非最终500）

remaining20_01 全部20候选终止并完成独立/逐条件原生复核：18通过、006因用户模拟器quote→book→否认自身请求而hold、018因错误宣称其余航班均超出窗口而drop（KJQ88IV/HURYM0C实为13:30有效项）；不改原文、不切成功前缀。011允许澄清between的端点歧义：原闭区间319与明确strict-before18后的407均有完整回执比较。

累计98=历史70+teacher28；主评审53候选。partial98真实tokenizer/mask审计：最大11520、总678412、assistant35525、0精确重复。该审计固定98，不是最终数据包。

新进程PID81616生成remaining20_02（existing022–041），起点36.885717元/3213calls，批次绝对46.885717元/5213calls，总上限100元。旧2259reserve0.139384保持。前10独立agent复核，后10由owner复核。下一批尚未启动。源hold排除与corrected prompt身份继续生效。

最新执行更新（2026-09-27 16:40 北京）：首20目标33候选已全审，10新teacher+70历史=80合格，全部80已做实际Qwen3.5token/mask审计：max11520，总547589、assistant28860、精确重复0；不是500完成。新20目标existing002–021已启动PID76288，run=sft_deepseek500_remaining20_01_20260927；启动总账36.533477/3121calls，增量≤10/绝对46.533477/请求cap5121，全局100不变。先前PID69397已结束。teacher datetime freeze原hash声明因混入旧envelope口径错误保留，独立correctedidentity及retryhashcorrection附件已核对全部10实际record/code；新批已用校正freeze。当前主状态见sft500_status.json，首批gate见first20_pipeline_gate.json，详细审核见teacher_quality_reviews.jsonl。下文旧计数均为历史记录。

最新执行更新（2026-09-27 16:18 北京）：语义合格76=历史70+DeepSeek teacher6，仍未包装；76不等于500完成。PID61358已结束，13候选8完整/5已结算格式失败，08–14审核新增11/12合格，09/13日期推理错误拒收，14确认细节hold。15–20由独立审核员复核。现PID69397运行10个明确重试目标，task06为第三且最后次，其余第二次；task08已耗尽3次禁止再试。启动占用35.903626元，首20所有续跑绝对cap45.222625，增量仅9.318999元，请求cap4853。格式echo精确兼容和通用完整日历时间提醒已冻结，local20及remote20测试全过，无GPU/无共享RL改动。详情以计划目录sft500_status.json、teacher_quality_reviews.jsonl与pilot_retry10_datetime_launcher.json为准。旧下文计数为历史记录，不表示当前接受集合。

# 2026-09-27 SFT 课程生成与证据账本

**当前主责状态（以sft500_status.json/launcher为准）**：语义审核暂合格74（历史70+teacher4），尚未打包；原72另2条断裂源场景unknown保留但不入训练。PID56228已因模型坏JSON停止并保留，已把teacher可见历史统一为flat action、保持事实与原回执不变；坏模型输出不修补。最新PID61358运行`pilot13_flat_v2`，task08第三且最后attempt、09–20首次，已结算格式失败保留后只前进至下一不同任务，不自动重试该任务；未知usage/预算cap停止。本地19专项、远端对应19检查通过。原100元总账本及首20绝对45.222625元上限不变，最新启动35.44539元。后续新28的取消退款不可假设即时补足礼卡，独立rubric已修订。

**接管/协议v2更新（15:46 北京）**：用户授权SFT子agent全程推进500，root专注RL。续批15也已结束：task06/07完整，task08因text+tools混合被旧互斥协议拒绝。两批9条均已独立真实runtime重放并另存`*_runtime_reconstructed`及manifest（3条原快照错误），原始保持。已冻结teacher局部`airline_teacher_visible_text_multicall_v2`，合法混合动作按assistant→全部native工具→独立用户次序处理，用户只看文字；写入仍须事先详细明确同意，不能提前宣称成功。共享RL prompt未改；本地17项与远端runtime16+mixed1通过。最新剩13任务（task08第二次，09–20首次）已启动PID56228，`results/analysis/sft_deepseek500_pilot13_protocol_v2_20260927/`，实际费用上限仍45.222625元，启动35.432207元，增量9.790418元。历史72已交curriculum_audit独立fresh审查；余380+new28必须待首20质量流程完成后才生成。最新rubric见pilot20_protocol_v2_rubrics.jsonl，协议和预算见teacher_protocol_v2_freeze.json及pilot13_protocol_v2_launcher.json。

**状态审计缺陷与续批（15:35 北京）**：首20批原run已失败保留，完成5条对话，第6条因纯文本JSON省略字段失败；窄修复远端15项CPU检查通过。原索引5:20续批已启动，PID54763，输出 `results/analysis/sft_deepseek500_pilot15_continuation_20260927/`，launcher位于 `results/analysis/sft_deepseek_500_20260927/pilot15_continuation_launcher.json`。首20含续批的费用绝对上限45.222625元（续批开始35.319414元，增量9.903211元），累计请求上限4853；历史unknown reserve 0.139384元保留全额占用、不重试或释放。

逐条审核发现：native `build_environment` 会深复制DB，旧generator/reviewer却从传入原DB取`final_db`，导致写操作的完整终态快照错误，环境哈希和真实工具回执仍可独立复现。**旧native_replay passed不能证明完整state；所有已生成候选尚未正式接受。** 新修复从`environment.tools.db`读取真实状态，拒绝错误旧快照。保留首批及正在运行续批的原记录和hash，运行结束后独立重放全部工具、核对原回执和全部hash，另存可追溯派生state，再逐条rubric。不得覆盖原候选，不得声称磁盘修复影响已加载旧代码的进程。首5运行时重建证据：`results/analysis/sft_deepseek500_pilot20_20260927_review_envelope_fix/first5_runtime_state_diff.json`。其中第5真实有新订单、席位减少及用户订单索引三处变化；第1存在最低价选择错误（报价145美元，但有合格140美元选项），应拒收。

运行更新（15:15 北京）：首20个任务的 DeepSeek teacher/独立user真实交互候选批已启动，远端PID53584，输出 `results/analysis/sft_deepseek500_pilot20_20260927/`，20×1、40assistant turns、4096 max output tokens/request、本批增量≤10元/累计≤100元；不启动GPU。新入口`data/teacher_rollout.py`及无卡独立回放`analysis/teacher_review.py`已部署。本地相关24项、服务器5专项检查通过，source/protocol/DB/API回执保留。每条候选仍需Codex按冻结五维rubric逐项评审，生成数不等于合格数；当前不能称500完成。新SFT heartbeat每15分钟继续生成/审阅，独立于RL监控。

**最新用户修正（2026-09-27 15:00）：最终训练目标500条。原72条保留，现有400条降为待DeepSeek agent改进/重执行的候选，再新增至少28条（淘汰继续补足），验证另计。以下v4的“完成”只指确定性模板数据和工具/DB/长度检查完成，不是DeepSeek生成或完整rubric评估完成，不是当前500条交付。当前不执行纯模板筛选合并。**

现400没有DeepSeek评分或agent逐轮生成；复查发现单向目标“in each direction”歧义、部分替换航段无法衔接、价格说明与回执冲突等。排错记录在 `results/analysis/sft_quality_merge_20260927/`；未逐条完整审查的行不能称合格。原72的两个机场查询样本source metadata用户/订单与v4 dev重叠，虽visible正文无该身份，新的来源隔离也要排除这些dev用户。原v4仅visible曝光隔离的结论不等于合并72后的来源隔离。

新方法：DeepSeek仅看到当前政策、工具schema、可见历史和真实工具结果，独立用户会话只看到角色所需信息；不向teacher提供旧assistant答案、gold动作或独立grader。CPU原生环境执行工具；完整轨迹逐条经Codex按rubric审核及独立回放，失败/未知不入训练。原100元累计API账本不重置，15:04核对已占用35.222625元、剩余64.777375元（含已确认实际13.93元与之后估算/预留，并非最终账单）。先20任务生成试批，确认费用/成功率再扩量。尚无本轮DeepSeek生成候选或500条成品。

最终新增生成产物为 `data/sft/generated_curriculum_20260927_v4/`：400 条训练、60 条开发完整对话；累计课程为 A100→B250→C400。全程本地 CPU，没有启动 SFT、GPU、远端任务或付费 API，没有改动 RL 代码。它们是新编写的训练子任务，不代表原始 23 个参考任务已被解决。

## 新生成 v4：验收范围

| 桶 | 训练 | 开发 | 覆盖 |
|---|---:|---:|---|
| foundation | 100 | 16 | 40 条真实直飞搜索/时段与价格筛选；45 条读取/额度计算；15 条政策拒绝 |
| constraints | 150 | 14 | 40 条真实新预订（20 信用卡、20 礼卡+卡拆付）；110 条变更、取消、退款及保护字段约束 |
| strategy | 150 | 30 | 61 条搜索新预订（21 中转、35 往返、5 真实无座日期替代）；89 条有状态依赖的变更/退款/行李链 |

每条保留原生工具完整回执、源 task/DB 身份、明确写入确认、独立构造的完整预期数据库检查和新鲜环境独立回放证据。新预订包含完整乘客字段、保险选择、用户要求的行李、准确 `payment_id+amount` 和座位/预算检查。明知余额或座位不足时先调整方案，不故意发起失败写操作。直飞最低价仅在用户明确限定的直飞范围内选择；中转实际调用 `search_onestop_flight`，检查同日出发、机场衔接、至少一小时等待和两段库存。

自然用户目标由助手拆成搜索、筛选、计算、支付和确认步骤。依赖链目标中有直接指定目标航班或舱位的情况；按保守口径，61/150属于搜索选择，89/150属于指定目标的依赖执行，不能把后者夸大为通用自主规划。原生取消工具只登记退款历史，不立即增加礼卡余额；取消后重读账户并按实际余额决定新预订支付，不消费尚未到账的退款。

## 隔离与版本纠正

最终开发集只有 **9 个用户**，明确为 **entity_heldout_template_shared**：用户和源订单与最终 train 隔离，模板共享，DB 文件也有共享，不能宣称独立任务家族或 DB 泛化。冻结的 117 个保护用户继续排除。正式 RL50 的直接 `user_id` 与订单所有者关联并集为 47 个，均不进入开发集。

v1/v2 是保留的原型候选，**已被 v4 取代，不能使用它们的 dev 隔离结论**。此前历史曝光提取只读工具参数，漏掉“只查询订单、user_id 仅在工具回执出现”的用户。补读 old100/repair72/repaired149 原生回执发现额外 18 人，其中 6 人与 v2 dev 相撞。v3 已重新冻结实体划分，v4 保持这9个用户，并进一步扫描历史完整可见消息，9 个 dev 用户均无出现。repaired149 作为更保守的既有候选曝光也被排除，不把其文件名当作已完成梯度训练的证据。

原生新订单分配器在独立 DB 克隆里重复产生字面 ID `HATHAT`。最终验证的是源订单实体隔离；新订单按独立克隆和用户命名空间标识，不能声称新生成订单的字面字符串全局唯一，也没有篡改原生 ID。

v4 按严格相同 messages+assistant mask 复用了本轮 v3 合格候选，v3 的部分记录精确复用自 v2。记录保留源文件 SHA、原生成代码/计划 SHA、内容身份和原生回放证据；v4 全部重新使用本地 Qwen3.5-4B tokenizer 渲染，并逐项核对复用行的 input_ids/labels SHA。复用证据与 v4 新原生执行数量分别列出，不把复用包装为又执行了一次。

v3 全460条后续日期审计发现1条 `book_date_replan` 使用了内部键 `2024-05-25_late`。v3 同样作为 superseded 候选保留，v4 对工具参数、原生回执和自然目标同时执行严格 `YYYY-MM-DD` 与真实日历校验，源头过滤非 ISO 日期，按真实剩余容量重新选取分支；没有伪造失败日期或更改原生回执。

## 统计、复现与限制

| 集合/桶 | p50 | p90 | p95 | p99 | max | assistant tokens |
|---|---:|---:|---:|---:|---:|---:|
| train/foundation | 5,973 | 7,270 | 8,161 | 10,080 | 10,104 | 15,851 |
| train/constraints | 7,254 | 8,676 | 9,434 | 11,391 | 11,801 | 74,261 |
| train/strategy | 9,046 | 10,958 | 11,696 | 13,450 | 13,468 | 140,682 |
| validation/foundation | 5,997 | 7,130 | 7,632 | 7,632 | 7,632 | 2,598 |
| validation/constraints | 7,414 | 10,019 | 10,096 | 10,096 | 10,096 | 6,713 |
| validation/strategy | 8,727 | 9,855 | 11,041 | 11,341 | 11,341 | 27,983 |

训练400条合计 **3,110,763 total / 230,794 assistant tokens**。A100/B250/C400 的 assistant tokens 为15,851/90,112/230,794。最大完整上下文13,468，超长拒绝0；原始完整回执没有截断。

三个训练桶分别有11/16/9个编写分支标签，工具调用序列结构数为 **3/14/10**，参数签名数为100/150/150。strategy按保守口径为 **61条自然目标搜索选择（40.7%）/89条指定目标依赖执行（59.3%）**，后者虽由助手决定步骤和检查状态，不能算作开放式自主规划。开发桶工具序列结构数3/9/7。完整family计数、工具序列及参数指标保存在 `summary.json` 的 `stats`。

460条均有本轮原生执行与独立回放证据；v4精确复用453条、重新原生生成7条，460条全部重新tokenize。历史精确去重挡住6条已选旧对话和2条候补（8次拒绝），同分支补入6条合格新对话，最终与old100/repair72的messages+mask精确交集为0。旧稿保留在`superseded_pre_exact_dedup/`及拒绝账本，未靠修改姓名或措辞规避去重。

主要交付：`train.jsonl`、`validation.jsonl`、六个分桶JSONL、`A100_train.jsonl`、`B250_train.jsonl`、`C400_train.jsonl`、`entity_split.json`、`group_split_manifest.json`、`summary.json`；最终审计在 `results/analysis/sft_generated_curriculum_20260927_v4/final_audit.json`。组划分复用既有`curriculum_split.build_split`锁定用户；它的组难度统计采用该用户最高难度，不替代实际逐行100/150/150分桶统计。

所有行都通过严格 assistant-only mask、工具 schema/回执顺序、写入前明确 yes、完整对话长度不超过 24,576，以及 train/dev 内容去重检查；没有裁剪工具回执或确认。报告独立统计分支标签、参数签名和工具序列结构，不能将400条参数化实例说成400种策略。每条的代码/输入/模板/token身份可追溯；并非460条都经独立人工逐字语义审核，审核范围是编写分支合同、代表性原生集成测试和逐条程序检查。

复现入口复用 `tau3_grpo/data/grounded_repair.py`，未另建平行生成体系。本地相关测试 **26 passed**（21.43秒，1条warning），日志为 `results/analysis/sft_generated_curriculum_20260927_v4/tests_final.log`。以下生成命令为本轮执行记录；重新生成时应改用新输出目录：

```bash
.venv-cpu/bin/python -m tau3_grpo.data.grounded_repair --output data/sft/generated_curriculum_20260927_v4 --curriculum-phase foundation --reuse-curriculum data/sft/generated_curriculum_20260927_v3
# constraints / strategy 同入口，输出各自文件；冻结计划不允许漂移。
.venv-cpu/bin/python -m tau3_grpo.data.grounded_repair --output data/sft/generated_curriculum_20260927_v4 --curriculum-finalize
.venv-cpu/bin/python -m pytest -q tests/test_composed_sft_probe.py tests/test_source_reuse_ledger.py
```

最终数据目录已禁止继续追加；只读重验可重复执行 finalize，第二次一致性检查已通过。重新生成应使用新输出版本。源码执行快照保存在 `results/analysis/sft_generated_curriculum_20260927_v4/`（精确复用行的旧代码快照在对应 v2/v3 results 目录），不会在 data 目录维护源码副本。

## 先前 v6 审核账本（独立保留，不与新增400混算）

该历史阶段可审查产物为 `data/sft/curriculum_evidence_20260927_v6/`。本轮仅推进数据：继承历史合格记录、逐条语义审核、精确去重、保留训练曝光，补两条真实依赖的完整对话。没有启动 SFT、GPU、付费 API 或改变远端 RL；主线仍是 Base→repair72。

## 结果与统计口径

| 维度 | 当前结果 | 含义 |
|---|---:|---|
| 数据集成员身份 | 1,245 | 9个输入cohort；不能当作独立对话数 |
| 精确 messages + assistant mask 去重 | 932 | 原消息与监督位置保留，metadata不参与身份 |
| 实验训练候选 | 168 | 94原始来源 + 72既有 authored + 2新组合样例 |
| 本轮完整可见对话语义阅读 | 52 | 25原始 + 25既有 authored + 2新组合；37通过、15 hold |
| 仅继承资格、未在本轮重新完整审核 | 131 | 84原始 + 47既有 authored；不能标作本轮审核通过 |
| 候选中有历史梯度曝光 | 132 | 其余36无确认曝光，其中2是本轮新建 |
| 原有课程标签保留 | 152 | 均为历史信息，未自动升级为新语义标签 |
| 先前adjudication联接 | 149 | 与visible evidence及实际critic文件SHA绑定 |
| 原23多步参考任务 | 23 | 原参考动作已执行≠原任务完整对话已接受；后者仍0 |
| 新组合完整对话 | 2 | 新 authored 子任务，另行计数，不能称解决原23任务 |
| 独立family-dev可用 | 否 | 训练数据资格与独立评测资格分开 |

精确重叠：repaired149中的40条authored完全在repair72中；old train100与repaired149交集71；capability dev14完全在repair dev22中。相同source的内容变体0，精确train/dev内容冲突0，train/dev可见用户实体冲突0。22个通用模板跨train/dev只作诊断，不自动判为语义泄漏或剔除训练数据。

old100不能整体报废，也不能整体继承：71条与repaired149精确重合，剩余29保留原始身份与曝光，未因旧训练身份自动取得本轮训练资格。训练候选总数168已排除两批发现的15个语义hold。

## 本轮语义审核

证据位于 `results/analysis/sft_curriculum_evidence_20260927_v6/semantic_reviews.jsonl`，每条包含消息SHA、零基消息索引、用户目标、具体判断、难度、依赖标志、限制。Reviewer为Codex读取，不冒称人审。

第一批原始通过：384、999、468、390、762、924、422、609、969。原始hold：763、997、24、617、577、637、702、575、239、816。主要问题包括无依据升级人工/例外承诺、往返与返程陈述不一致、其他订单错误摘要、未证明晚间筛选；816为礼品卡350余额再退款350但最终声称余额恢复350的明确财务陈述问题。577/637保留“后续用户指定ID可能缩小目标”的不确定性，未虚构原始条件证据。

第一批既有authored通过11条：airport_lookup479、baggage_paid1114、profile_identity1103、flight_refusal656、cause_unavailable444、cancel_after24_668、cancel_within24_761、card_tail_binding615、historical_upgrade171、downgrade_refund564、multileg_cabin566。564/566存在源支付历史与存储票价不一致，资格限于基准原生工具按存储票价计算的合同，不能宣称历史财务账已完整核平。

422协调局部航段变更、加袋与后续查询，保留hard；但行李额度并未依赖新舱位，`needs_dependent_subgoals=false`，不把多个写操作自动当作真实政策依赖。924和已有614也不能仅凭多工具/多写操作升为策略类。

原始来源多数没有可验证的原任务DB映射；本轮原始审核是完整可见证据阅读，不是原始环境fresh replay。账本保留unknown，不用成功receipt或既有judge标签填补缺失事实。


第二批从此前151条未复核候选中再完整读取20条：6原始、14既有authored，15通过、5 hold；独立逐条证据另存 `results/analysis/sft_curriculum_evidence_20260927_v6/semantic_reviews_batch2.jsonl`。

- 原始854通过：后移出发航班后，对三个不同舱位/人数订单加袋，增量50/0/50和gift余额350→250正确；这些操作相互独立，标medium而不增加策略覆盖。
- 原始121 hold：AXRDYS2实际11:30，首次展示为10:15；后续虽纠正并确认，但原监督消息仍错误；另有无依据人工政策弹性暗示。
- 原始96 hold：正确返程改签及94退款之后，提出无依据“取消但不退款”和人工例外选项。
- 原始625 hold：改签与免费加袋正确，但最后成人核对没有证明所选返程属于用户要求的下午航班；不能把未知当错误成年人身份。
- 原始945 hold：取消重复订单/加袋/拒绝减袋主体正确，但无依据断言人工也不可能帮助；晚间条件未证明，而用户后来明确选择D01B39需保留这一缩小目标的证据。
- 原始614 hold：七日早班比价及55+50+30=135订票正确，但最后虚称确认邮件已发出；这是dialogue614，不是另一个映射中的task airline_614。
- 既有authored14条通过：transfer_flown618；compensation603/739/865；multileg_cabin1142/1079/550；downgrade_refund208/40/530；cancel_within24_845；cancel_after24_1081；card_tail_binding551；historical_upgrade105。逐条检查状态优先级、证书资格、每航段座位与存储票价、乘客倍数、分拆退款、24小时边界、尾号到实际ID映射和终止方式，均标medium且无重规划/真实依赖声明。

1079源三段行程的时序、550的one_way回原点标签、551的断开航线都存在原DB结构限制；审核只接受明确限定的舱位变更与源结构保护，不宣称原行程规划有效。1079支付620与存储票价×2=1240不一致，沿用明确的原生存储票价合同限制。105的336支付=306票价+30保险则可解释。第二批未重新执行原生DB回放，继承已有receipt/历史执行证据，与本轮语义阅读分别记录。

## 两条组合流水线样例

文件：`data/sft/curriculum_composed_probes_20260927_v1/train.jsonl`。

复用 `tau3_grpo/data/grounded_repair.py::build_cabin_then_baggage` 与现有 `grounded_gap_pilot.Builder`，没有新建平行生成体系。helper不加入旧单动作KINDS，不改变既有生成配方。

| 来源训练实体 | 第一写操作 | 第二写操作 |
|---|---|---|
| airline_497 / DX8C9F | HAT257 basic→economy；187−96=$91 | silver新economy额度2；总3袋，1付费袋，$50 |
| airline_254 / Z65U3F | HAT082/HAT068均basic→economy；133+169−98−76=$128 | silver新economy额度2；总3袋，1付费袋，$50 |

用户先明确顺序与保护字段；助手观察reservation/profile/实际航班价格及座位，报价后获得第一次明确yes，执行升舱。第二次行李额度来自实际升舱receipt的cabin，而非旧舱位；如果误用旧额度会收2袋费用。第二次费用单独报价与yes后写入。两条均经新鲜环境独立原生回放、outcome hash比对、独立构造的完整预期DB相等检查；包含未请求订单、用户、航班、乘客、保险等全局状态保护。两位用户都排除在冻结的selection/validation保护实体之外，仅复用既有repair72训练实体。

这两条共享模板、silver会员、单乘客，只有单段与中转差异。它们是流水线样例，不证明strategy覆盖充分，也不是原始source task已被解决；不含模型在线rollout、用户模拟或新API。新记录原metadata难度留待审核，只有明确的新语义审核把它们纳入策略课程。未使用新付费critic；语义阅读与生成属于同一Codex任务，不是独立外部评审。

## 可累积课程与token统计

`reviewed_training_manifest.json`只涵盖锁定为train、且本轮语义通过的子集。`manifest.json`为全局family独立性待审清单；不会把子集family标签伪装成独立dev保证。

| 累积课程 | 对话数 | 历史梯度曝光 | 历史/新计数记录 | assistant tokens | total tokens |
|---|---:|---:|---:|---:|---:|
| A foundation | 4 | 4 | 4 / 0 | 370 | 22,620 |
| B A+constraints | 34 | 33 | 34 / 0 | 21,492 | 261,721 |
| C B+strategy | 37 | 33 | 35 / 2 | 25,754 | 296,948 |

实际新增桶为4/30/3。A⊆B⊆C已校验，第二批没有新增策略依赖样本。168候选计数完整覆盖：166条继承相同messages+mask的历史计数，2条本地新tokenizer渲染；合计assistant253,164、total1,701,191。未宣称168条全部重新tokenize。batch8、单epoch的1/5/5更新数仍仅示意，未冻结配置或训练run。

本地已有 `models/Qwen3.5-4B` tokenizer，使用 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 与正式 `TrajectorySFTDataset`、现有tool schemas、max_length24576、approved targets要求计数，无下载/权重加载/GPU。497为total7,511/assistant600，254为total7,919/assistant697；合计15,430/1,297。tokenizer JSON/config/chat_template、tool schema、dataset代码及输入文件SHA已保存在 `results/analysis/sft_curriculum_evidence_20260927_v6/composed_token_counts.json`。账本分别统计历史与fresh来源，并拒绝token报告绑定到不同源文件。

## 复现、校验与变更文件

输入清单：`results/analysis/sft_curriculum_evidence_20260927_v6/input_manifest.json`。最终审计：同目录 `final_audit.json`。当前生成命令如下，复跑需换新output目录；已有输出禁止覆盖。

```bash
.venv-cpu/bin/python -m tau3_grpo.data.source_reuse \
  --ledger-config results/analysis/sft_curriculum_evidence_20260927_v6/input_manifest.json \
  --semantic-reviews results/analysis/sft_curriculum_evidence_20260927_v6/semantic_reviews.jsonl \
  --output data/sft/curriculum_evidence_20260927_v6

.venv-cpu/bin/python -m pytest \
  tests/test_source_reuse_ledger.py tests/test_curriculum_profile.py \
  tests/test_composed_sft_probe.py -q
```

本地CPU：19 passed；唯一warning为上游audioop弃用。新测试另覆盖fresh token身份绑定与来源保留，并覆盖新cohort必须明确语义pass、Unicode旧hash联接、mask身份、内容变体、过期证据、模板跨split不取消训练资格、dev独立性分离、第二动作采用新额度、每次写操作确认及完整DB保护失败拒绝。两条实际生成各自原生回放通过；不以CPU替代GPU训练验收。

本轮修改：`tau3_grpo/data/source_reuse.py`、`tau3_grpo/data/grounded_repair.py`；新增 `tests/test_source_reuse_ledger.py`、`tests/test_composed_sft_probe.py`、本文件及上述数据/结果产物。其余预先dirty文件和RL修改没有覆盖；没有提交或远程同步。

`source_reuse.py` SHA256：`e2972fdfff56a1a545d84c8e77e536f31d28d1cc514f6b86acbc52ce4bdca142`。
`grounded_repair.py` SHA256：`5d5914dc419b3391190b2c2ccf53cf6e9f36a1890514e1939dce23b86685a8ab`。
最终summary SHA256：`bfa13176fa9568dab7eabafe7fa6c89b566de0eb85235b52fbbe7ff053c34b46`。

本轮v1/v2是过严格family训练门槛的开发草稿，不能引用其“0训练可用”为现结论；v3是30审核/171候选阶段；v4新增两条组合但旧标签联接未完整处理Unicode序列化；v5修正该联接，保留152条历史标签。旧审计用ensure_ascii=True，新canonical hash用False，两者通过原序列化精确比对，不按ID强行联接。v6增加第二批20条语义复核和两条真实tokenizer计数；v1–v5产物均已被v6取代，未覆盖任何原历史数据。

## 后续缺口

- 剩余131条候选尚未获得本轮完整语义复核；其他727条未知/旧hold也未获得新资格。
- 仍需跨语义family的独立dev审核与分配，不能通过通用模板名直接下泄漏结论。
- 策略样本需要不同会员、人数、权限、已付款行李、重规划/失败分支等真实变化，不能重复这两条模板宣称覆盖。
- 原23参考任务需分别建立完整自然用户目标、可见证据、确认及终态合同；不把新子任务样例挪作原任务成功。
- 新样例token计数已完成；统一训练配置与任何SFT/GPU实验仍未执行。

## 18:14 update:187 qualified and Kimi configuration

Batch06 back10 full manual review plus34 native replays:7keep,2drop(125 chosen card not confirmed;130 unsupported cheaper-fare claims),1hold(126 correct booking but expanded ticketing scope/incomplete mandated handoff). Whole DB changes verified exactly. Total187 including historical70; hard Codex ceiling200. Kimi authorized by latest user: local env key configured, exact Kimi-K3 listed by provider GET /v1/models; no paid calls yet because provider rates have not been verified. Shared DeepSeek-only pricing must not be reused for Kimi. Automation remains PAUSED.

## 18:30 milestone: Codex200 frozen, subagent handoff

Exactly200 qualified:70historical+130independent DeepSeek teacher.172teacher candidates reviewed,42drop/hold. Last qualified tasks147/148/150;151 and all later candidates remain unreviewed by Codex. Audit agent interrupted (already completed) and never reawakened. All Codex semantic review stopped at200. Exact source messages/masks frozen with provenance and token audit:1,377,854total tokens,81,758assistant tokens,max11,543<24,576,zero exact duplicates. NoGPU. codex200_handoff_manifest.json is not final500 packaging. Root now directly owns Kimi K3 review to obtain remaining300, as latest user instructed; no Kimi paid calls were made by this subagent. Local-env endpoint and exactmodel verified with read-only GET models; no guessed prices. Current DeepSeek100controller continues; next100v3 ready, not launched. PAUSED automation updated and remains PAUSED.

## 20:30 resumed SFT supervision by latest explicit user instruction

The previous stop-all-subagents rule is superseded for sft_curriculum_build only: supervise DeepSeek generation and Kimi API review toward500unique qualified train plus isolated dev; do not wake curriculum_audit or substitute Codex semantic labels. FrozenCodex200 unchanged. Remote round02 controller134996 and batch11 generator145653 verified alive;09/10completed40, no duplicate launch. Kimi007 is240s timeout/no usage, retained and never retried. Authorized distinct remaining08 candidates resumed;012 known schema failure has full response/usage but only18/21conditions, not accepted or repaired. Kimi progress now mechanically reconciled in kimi_verified_progress.json and kimi_accepted_index.json. Future APIdeadline600s; exact response model Kimi-K3 and valid input/output usage required. New bounded supervisor preserves existing attempts, prevents frozen200 resubmission, uses exclusive lock, stops new dispatch on new unresolved request.6CPUtests pass. Round03next10014–18 prepared from exact frozen source/preflight subsets, localdryrun passed, not launched. NoRL changes/noGPU/selection-final reading.

### 20:45 supervisor mechanical progress

Kimi last07 task151 accepted with complete exact-model receipt; verified combined count219 (frozen200+Kimi19). All19 newly accepted trajectories passed exact source rendering and assistant-only token/mask audit:150694total/14312assistant,max9590<24576,zero exact duplicates including frozen200. Evidence:kimi19_exact_prompt_token_audit.json. Remote dev37 and round03next100 dryruns passed without API; prepared files uploaded, neither launched. Round02advanced tobatch12 after60complete. Existing timeout007 and incomplete-schema012 remain immutable/unretried. NoGPU/RL changes.

### 20:58 bounded handoff and future capacity

Finite local driver PID62889/session47447 waits round02, then downloads completed originals12/13, creates a ≤100 explicit Kimi plan, starts Kimi concurrency3 and dev37 generation once. Unknown Kimi usage ends dispatch without retries. It does not launchround03 or alter PAUSED automation. Runtime:bounded_round02_handoff_runtime.json. Round04prepared locally, not uploaded/launched:80 exact unstarted goals +20 frozen rejected goals, freshDB, one regeneration each, globalattempt guard<3; onlypilot06/08 globallyexhausted. No new Codexsemantic decisions. Accepted userisolation219train67users vs37dev7users haszerooverlap. Devreview receipts/index isolated atkimi_dev_reviews/kimi_dev_accepted_index.json; currently0.

21:05 update:round02 has80complete and finalbatch13 running. Round04plan/controller uploaded and remote100taskdryrun passed; still NOT launched. Retryadmission helper4CPUchecks reject acceptedgoals, changedcounts, and unapprovedduplicategoals. Before anyround04launch, synchronize latestmechanicalkimiacceptedindex for guard and inspect liveattemptcounts/budget.

### 21:41 recovery and bounded continuation

After UI interruption, verifiedKimi supervisor64087 alive:27complete/27accepted, combined227 at21:38:41. SSH120s launch timeout did NOT prevent dev:controller168690, secondbatch169571 running after20complete. Originalhandofferror preserved; no duplicate paidlaunch. Finite remote waiter172509(wait_dev_then_round03_runtime.json) waitsdev37complete and launches exactlythehashboundround03once, then exits; PAUSEDautomation unchanged. New09_008 Kimi response waslength-truncated at8192completiontokens (prompt16542,total24734), usageknown, preserved/notaccepted/notretried. FutureonlyKimi calls nowmax16384/timeout1200s bounded; existingrequestsretainloadedlimits.6CPUtests passed. Round02generated100candidates:91user_stopforKimi,9generationfailedpreserved.

### 2026-09-27 14:05 UTC continuation snapshot

- Existing Kimi round02 PID64087 and dev-review waiter PID64843 verified alive; neither duplicated. Remote round03 controller177378 / batch14 child177410 retained. RL untouched.
- Refreshed accepted receipts: frozen200 +46 unique Kimi accepted =246 provisional train;47 completed Kimi reviews include1 semantic reject; same4 preserved infrastructure failures, no new unresolved-use stop.
- `kimi46_exact_prompt_token_audit.json`:46 new rows,345191 total tokens,29716 assistant tokens,max10638<24576; exact message/mask duplicates including frozen200=0. Index snapshotSHA `8a6499dd767f20aab809e9b20be1f5ca39049e57ed2121f758b8903f1c5cd9b4`.
- `accepted_isolation_audit.json`:246 train,67 train users vs7 frozen dev users,zero overlap,zero teacher exact-goal duplicates.
- Remote generation inventory at14:05:32 UTC:309 train attempts started (284 user_stop,24 settled failed,1 running),37 validation attempts (32 user_stop,5 failed). Started candidates are not qualified examples.
- Preparing separate bounded fixed5 validation generation-failure retry controller; not launched. Runtime guard will require round03 completion, verified dev acceptance index, exact source equality, immutable original-failure hashes, settled API usages and current global attempts<3. Original train and validation generators unchanged.

### Fixed5 dev retry preparation (not launched)

`validation_retry5_01_plan.json` SHA `830deb8665470b74484c1ae7025a2745e643e8edb146319cbc5742020127f088` binds the5 original settled generation failures028/037/041/056/060. Separate `validation_retry_controller_v4.py` preserves generator behavior and enforces exact frozen validation tasks, failure file/run hashes, all prior request usages settled, current attempts equal1 and<3, and a verified separate dev acceptance guard. CPU admission fixture passed baseline plus rejection of accepted task, unknown usage, changed attempt count, changed source, and changed proof. Local and remote5-task no-API dryruns passed. Controller/plan/pool/preflight uploaded as new result artifacts; no execution or waiter has been scheduled for this retry. Start remains gated on round03 completion and verified dev index. Full dev originals downloaded locally; the existing dev waiter still owns review dispatch and remote/local tar integrity reconciliation.

### User steering: generation first, Kimi scheduling paused (14:20 UTC)

Parent relayed explicit user request to let DeepSeek generate to500 first because Kimi is slow. New target is500 complete unique training candidates pending review: historical70 +at least430 unique teacher goals; never claim500qualified. Known semantic rejects,failed/in-flight/duplicate candidates anddev are excluded. No GPU or RL restart.

Observed Kimi supervisor64087 already naturally fail-closed by14:18:45 after3 new RemoteProtocolError calls with unknown usage; all active calls had finished/failed,active{} and54pending. Existing dev-review waiter64843 also fail-closed14:19:07;both PIDs absent on subsequent check. No paid process was killed,no failed call retried. Kimi totals51completed,47accepted,4semanticrejects,7infra failures;247qualified remains distinct from generation target. `KIMI_SCHEDULING_PAUSED_BY_USER.json` records explicit scheduling hold. No further Kimi launch is authorized by current instruction.

`generation_first_inventory.json` remote14:23:26:296user_stop teacher attempts,37known rejected completed candidates excluded,259unique complete teacher goals;historical70+259=329,remaining171.27failed and1running excluded;32dev excluded. Hash/message-mask/source-state structure checks and user isolation passed;full native replay pending. Frozen200semantic and token audit unchanged.

Finite waiter `wait_round03_then_round04_generation_first.py` remotePID183525 now waits existing round03,then launches frozen round04 once. PlanSHA `2675c5744ab9742dd12b8b8c5823f49fc0b0a4357e145629f89ef08629228b2e`;remote100-task dryrun passed. No Kimi/dev wait dependency, no parallelDeepSeek owner. Dev retry5 is prepared but deferred;it will not preempt training-candidate generation. Native CPU audit runs independently against snapshot259,with frozen130teacher prior-audit provenance preserved and fresh candidates replayed against originalDB.
