## 最新状态：2026-09-28 发布计划与旧实验收尾

本地归档的 `results/runs/repair72_grpo_shared32_full_eval20_20260927/seed42/completion.json` 与 `controller-state.json` 确认旧 repair72 GRPO 已完成20步，云端20个更新步验收通过；step10/20 的 selection 终局奖励均值为0.483333/0.491667，起点记录为0.445833。收尾含 step20 缺失标量按本地原始指标精确补传，不改写实验身份。`shutdown-status.json` 记录051已于2026-09-28按原任务授权正常关机、未删除实例存储。这里是已归档事实，不再沿用下文旧PID作为当前运行状态。

新主线为已验收500条的 A109→B393→C500 累计SFT，以及对应同一SFT起点的GRPO/ARPO分支对照。dev37扩充至150仍在候选生成阶段，新课程训练尚未执行；ARPO仅CPU验收，未做GPU验收。当前发布把既有完整评测profile与ARPO合入本地Git，并准备同步GitHub，不启动训练服务器或更改其checkout。详细计划见 [README](../README.md)。下文历史预算、暂停/重启和未完成状态保留作为追溯记录。

## 2026-09-27 17:58 用户授权改为速度优先 fast32

用户明确允许每step轨迹减半，并明确选择保留日志/轨迹、停止旧运行、放弃未保存更新，从同repair72起点立即重开。旧throughput20 controller55708已SIGINT并由自有cleanup释放两卡，原状态failed是人工停止而非训练崩溃，user_authorized_fast32_stop.json记录。停止前step1约40分钟，进入update_actor，尚无完整step指标或checkpoint；不计为完成更新。

新profile configs/train/rl/repair72_grpo_2xa800_passenger_v1_fast32.yaml：4任务组×8，每步32，共20步640训练候选；两卡、模型、LoRA16、奖励、轮数/响应上限、学习率、FP32 IEEE及全重算不变。仍10/20完整checkpoint+selection60×4。初始评测复用旧run已完成240条，启动脚本逐项比对model/val输入/采样参数/reward协议并绑定结果SHA，不重复采样，也不伪造成新run采样结果。新run results/runs/repair72_grpo_passenger_v1_fast32_20260927/seed42，launcher.json为实际PID权威，禁止重复启动。绝对deadline仍1790753043.1027153，不重置；原72h已用预算包含旧运行。

流式保存已按备份/hash定向部署remote：runner→Ray env→agent-loop后处理→逐条原子json；远端3专项CPU通过+完整runner dryrun通过，GPU真实落盘待新worker验证。维护回执 results/maintenance/repair72_speed_storage_20260927/fast32-deployment.json。当地vendor文件存在既有两端差异，未整树覆盖。CUDA Graph与logprob microbatch2仍是待验证候选，未启用、不声称提速已达18分钟。采样时GPU0策略、GPU1用户；更新时GPU1空闲，暂不变更角色调度。

## 2026-09-27 逐轨迹流式保存补丁（尚未对运行中进程启用）

本地新增 tracking/rollout_stream.py：每条 agent loop 完成 postprocess 后，把未 padding 的 token、mask、logprob、raw_prompt、reward 与全部 extra_fields 原子保存到 completed-rollouts/{validation|training}/{step}/{uuid}.json，再返回 gather。UUID 隔离跨 worker 同名索引与重入；await to_thread 写入，序列化/磁盘失败不伪造完成。不代表整批完成，也不提供断点续采样。runner 对未来启动传入 TAU3_ROLLOUT_STREAM_DIR。当前已加载的远端 worker 未修改、未重启，因此本轮评测仍采用原有整批保存；不得声称流式已在线生效。

本地 CPU：流式并发/未完成同批保留/非法数值不发布/关闭开关与 vendor 清单共 5 passed，throughput profile 1 passed。尚未部署及做远端/GPU 集成验收。vendor revision 沿用清单固定版本，仅更新 agent_loop 文件 hash 与原因/测试。

历史实际计时：五卡全参数训练内同样 selection60×4，step10/20 分别 1349.60/1356.54 秒；近期两卡独立 merged 推理每 60 条分别 938.46/896.18/893.16/841.53 秒（base/sft1/repair_base/repair_sft1）。后者含每 arm 启动等开销，初始化/协议/推理方式有差异，不能作为严格 LoRA 对照。详见 results/maintenance/repair72_speed_storage_20260927/streaming-readiness.json。

# repair72 RL 主线与累积 SFT 数据并行计划

16:13检查：正式run正在执行起点selection真实工具/模拟器请求，日志持续推进；无Traceback/OOM，GPU约60561/52677MiB。实际日志确认策略max_num_seqs16；初始240候选尚未整体落盘，训练metrics/update-batches仍无，不将其误报为首步耗时。10步边界请求已存在，未修改运行配置。


15:58检查：throughput20 controller55708仍运行，模型加载/初始full与adapter GPU卷积同步审计均all_conv_match=true，日志中的conv skip warning未被当作验证失败或忽略不核对。已请求global_step0起点评测，AgentLoop workers仍初始化；尚无完整metrics/训练step。两卡占用约57075/52047MiB，无OOM。SwanLab初始化run f257ae7ef6f1416185cb2（尚无训练点可云端验收），STOP_AFTER_BOUNDARY已于15:51设置并核对。


最新正式启动15:41：throughput20 run `results/runs/repair72_grpo_passenger_v1_throughput20_20260927/seed42` controller PID55708，已核对完整Hydra预检并提交。profile hash `64d7bb0441e1b20945c8ac415455ac46ce37327f2a8e970817d2134050bb2a17`，max-wall258022，绝对deadline不变。启动时GPU均无进程；15:43核对controller实际命令一致、状态starting_simulator，模拟器尚在加载前期，无metrics/报错，不重复启动；待training后设第10步边界请求。当前18分钟/step仍是目标，尚无正式step实测。新run授权/启动/预算文件已落盘，heartbeat已更新，新run开始training后请求10步边界停止并验收续训。


## 用户最新速度/清理要求：2026-09-27

用户要求不加卡、优化速度争取64轨迹/step不超过18分钟并尽快正式跑；SFT由sft_curriculum_build子agent全权接管API/CPU生成/逐条Codex审核至500。

旧正式controller53977已于首个更新前有意SIGINT，最终failed是人为停止而非训练错误；原服务已清理、0更新。勿自动重启旧run。新throughput profile提高MAX_NUM_SEQS4→16、user8→16，actor参数/优化器常驻GPU，rollout显存比例0.35→0.50，其他学习/评测协议不改。CPU并发/配置4项通过，远端完整dry-run已通过；最新正式run启动记录见本文顶部。

GPU0固定真实buffer replay已完成：全激活重算10605/13223上下文前反向24.19/29.79秒、峰值25.14/27.14GiB；交替重算和完全关闭均OOM，拒绝采用，原FP32 IEEE及全重算保持。该probe不更新权重，不是正式RL吞吐；18分钟目标未验证。所有重试/正式仍使用绝对截止1790753043.1027153，不追加卡。

用户明确授权后，已删除两次smoke global_step_2和clean14/A100/A100_ep3三个merged，共约65GiB。完整文件清单/精确基座与adapter SHA/导出记录见results/maintenance/repair72_speed_storage_20260927/deletion-completed.json及远端authorized-deletion。通过smoke的step2已不可恢复；历史验收与原始buffer保留。repair72 from_base/merged未动，正式仍从它启动。


## 最新状态 2026-09-27 15:24

修复后 smoke 已通过全部12项验收，证据为 `results/maintenance/repair72_rl_20260927/passenger_v1/gpu-engineering-acceptance.json`。第二步平均reward=0.25、grad_norm=0.28913775；实际GPU rollout与保存adapter均有18,112,512个非零LoRA B元素，云端step1/2核验通过。仅证明有效学习链路，不代表能力提升。

正式20步controller PID53977已提交，独立run `results/runs/repair72_grpo_passenger_v1_formal20_20260927/seed42`，原repair_base重新起步，LoRA16、2 A800、8×8/步，selection60×4在0/10/20。CPU dry-run已通过；实际服务状态以controller-state为准。总墙钟上限72小时，绝对deadline epoch1790753043.1027153，恢复不能重置预算。启动后请求step10边界停止，核验完整checkpoint及云端后，同run恢复至20，始终使用updates20保持scheduler horizon。

复验每步8条，耗时24.4/16.5分钟；第二步包括采样9.1分钟、actor3.2分钟、old/ref1.5分钟、保存2.5分钟。正式64条尚未实测，不能简单乘8。旧4策略卡全参64条18.83分钟；现1策略卡、max_num_seqs4、FP32/CPU卸载，且12/16满轮数，硬件、并发和轨迹均不同。

SFT旧400现为候选素材；428个DeepSeek真实执行任务已规划，加历史72，逐条Codex rubric验收后目标500。首批5完整候选、第6因JSON省略字段中断；尚未完成质量验收或500交付。数据构造使用API/CPU，不占RL GPU。

状态：2026-09-27，用户已授权两步工程验证，通过后直接正式 RL；服务器为两张 A800 80GB，旧关机 automation 已暂停。

最新启动（2026-09-27 14:14 北京）：用户“那你进行rl吧”已授权追加修复复验。新run `repair72_grpo_passenger_v1_smoke_20260927/seed42` controller PID34233已启动，2×A800、2步/16候选、最多2小时/4 GPU小时，runner内部7100秒；启动前双卡0MiB、12个部署文件hash匹配。此前“预算待确认”为历史状态。heartbeat `repair72-rl` 已恢复15分钟检查，完成后验收有效更新再放行此前授权的正式20步；尚未获得本次GPU验收结果。

最新修复状态（本轮收尾）：显式新协议 `airline_passenger_multiset_v1` 已接入真实 RL 配置链并部署；本地 CPU 95 项、远端 CPU 76 项通过（19 项无关旧硬件/历史配置检查未运行）。远端对原16条buffer只读重放，原生分数仍全0，新协议识别3条乘客排列假阴性；第二批得到一组 `[0,1,1,1]`，有效token上5268个非零优势，范围约[-1.5,0.5]。第一批仍全0。50个训练参考均在本地独立DB执行成功。该结果只证明奖励/优势链路，不证明新GPU参数更新；正式RL仍未启动。详情见末尾“乘客排列奖励修复验收”。

最新RL结论（12:16）：两步程序完成、完整checkpoint结构和SwanLab云端step1/2通过，但学习验收失败：16条训练奖励及advantages全0，两步grad_norm=0，实际rollout LoRA B和保存adapter B均0。正式RL启动条件不成立，因此未启动。服务已由控制器清理，nvidia-smi两卡均0 MiB；平台实例没有关闭。一次性审计结果为远端run内`engineering-acceptance.json`，passed=false。

最新数据交付（12:09）：新增 SFT `generated_curriculum_20260927_v4` 已完成400 train+60 dev，累积A100/B250/C400。全460有原生执行、独立DB回放与完整预期DB相等证据；453条复用本次生成过程的精确证据、7条在v4新执行，460均重新tokenize。root核验内部及旧100/repair72完整对话+mask重复为0、9 dev用户在训练可见文本中出现为0，相关CPU32项通过。用户/订单隔离但共享模板；并非460条独立人工语义审查，未训练SFT。详见数据交接和 `results/analysis/sft_generated_curriculum_20260927_v4/root_review.json`。

首轮 `repair72_grpo_smoke_20260927/seed42` 在 actor 构建时因 BF16 模型配置与 FP32 FLA 断言冲突失败，尚无采样或更新。改为 actor/ref 的 model_dtype、compute dtype 均 float32，保留 LoRA rank16；失败记录不覆盖。新尝试 `repair72_grpo_smoke_20260927_retry1/seed42` 已启动，沿用首次 12:49:47 北京时间硬截止，启动时剩余6237秒，没有重置两小时预算。

本地配置、日志连续性、checkpoint和服务回归46项通过；远端新配置4项通过。远端较宽CPU集合41通过、3失败，失败来自未部署的旧5090 profile；不为本次A800修复同步无关文件。首次宽检查发生于本轮同步前；同步后另跑远端新日志连续性8项，全部通过。

启动器读取实际组数与组大小记账；历史8×8任务调度检查保留。新增起点评测step0日志路径，续训时禁用重复起点评测。正式计划20步，每10步保存/评测，从10步恢复至20步验证连续性；起点重新使用原repair_base，smoke权重不接入正式run。当前尚未证明有效学习或GPU续训。

正式首轮暂沿冻结的原生 train50/selection60 管线，以同管线起点评测和节点结果诊断工程与趋势；已知模拟器偏移及934参考矛盾并未解决，不将其分数当作可靠能力增幅，也不混用历史42/60或调用final50。

## 用户确定的两条线

1. 固定 Base→72条修复训练的模型，先跑通 RL，再研究 RL 提点。
2. 同时整理旧 Stage A 的100条数据与真正的分阶段课程数据，不等待这条线完成才推进 RL。

旧 A100 指100条训练对话，不是显卡。后续数据版本不在同一 RL run 中替换初始化模型；如果新课程产生更好 SFT，单开实验比较。

## RL 起点和已核查事实

- 远端模型：`/root/autodl-fs/tau3-core/code/results/runs/sft_decision_repair_pair_20260926/seed42/from_base/merged`。
- 既有评测服务回执中的 checkpoint_hash：`6c821a90123a10c678b28a8d712d87f71bab174dd7d681824cd9d9c5a851c73a`，来自 `selection_repair_four_user07_20260927/seed42/scope_audit_corrections/model_identity_v2.json`。服务器开启后已重新计算完整目录哈希并匹配。
- 历史首次原生分42/60，仅作为背景。RL前后比较须统一当前协议和trial数，不直接把42/60当新协议下的基线。
- 本轮 SSH 返回 hostname `autodl-container-857546be50-ec463c07`，远端 Git HEAD `1f58472`。服务器开启前无GPU设备；开启后已确认2×A800 80GB并核对空闲。
- 本地有大量既有未提交修改，不能整仓覆盖远端。
- 公共入口 `tau3_grpo.training.rl.runner` 支持 grpo、tau_gigpo、mt_gtpo。当前正式profile绑定旧SFT和4张策略卡，必须新建组合配置显式替换模型与硬件。
- 公共入口的engineering_smoke仅支持新开1–2步，不支持将smoke接续为正式run；完整恢复验收留在独立正式run的10步边界。

## 执行顺序

### 1. CPU准备

复用当前公共runner、配置组件、评测与checkpoint模块；解析最终模型路径、算法、采样参数、训练任务、奖励和工具模板。核对当前多轮模板与训练/推理的差异，不假定旧profile覆盖了近期SFT修复。新配置不得继承带失效绝对路径的历史实验。

先用标准GRPO、关闭动态过滤，建立可解释基线。训练任务沿用训练分区，先审查其参考和奖励可执行性；开发selection的答案及具体轨迹不加入梯度训练。付费沟通judge暂不接入训练奖励。

正式比较前处理已知934题的场景/参考矛盾，冻结新的开发评测版本，保留旧结果。修协议后重评起点和RL模型，不混用不同版本分数。

### 2. 有界工程试跑（已授权）

两卡方案候选：4B LoRA策略/rollout独占一张A800 80GB，27B INT4用户模拟器独占另一张；这属于新LoRA RL配方，不能冒充历史全参数实验。两卡全参数方案不能由LoRA通过推定可行。

建议首轮预算仅包含2个外层step，每步2任务组×4轨迹，最多16条训练候选轨迹；最多2小时墙钟、2卡共4 GPU小时，提前完成即停止。这是预算上限，不是速度或人民币费用估计。实际价格、实例可用卡和预算须在启动前确认。GPU全参数方案若被选择，应重新确定资源与预算。

验收：真实工具rollout、reward、组内差异、有限且有效的advantage/梯度、参数实际变化、第一步更新到第二步rollout的权重同步，以及末步checkpoint结构检查。所有组奖励相同而没有参数更新时，仅记执行通过，不记学习链路通过；不无限补采样。停止后可做CPU检查，不追加GPU调用。

OOM、非有限值、奖励或轨迹身份错误、错误模型路径、保存失败即停止并保留现场；不靠缩短协议或删除失败样本冒充通过。

### 3. 短程正式基线与恢复验收（独立预算）

工程试跑通过后，新开正式run。按既有每10步保存/评测约定，先完成10步，在完整checkpoint核验模型、优化器、scheduler、随机状态及SwanLab连续性；再按批准的目标恢复继续。短程目标建议总20步，用户已授权试跑通过即执行；运行硬上限由实际吞吐估算并记录。

在同一冻结协议下比较RL起点、step10、step20，保留全部计划task/trial。检查原生成功、有效组比例、KL、重复回复、超步数、严重业务错误；附加judge未知单列。final50不参与调参。

### 4. RL提点

先定位瓶颈再改单个因素：组内无差异时评估采样/动态过滤；有非零信号却重复或提前终止时检查更新强度、KL及模板/终止路径；终局信号稀疏时再评估过程信用分配。每次保存不同配置/奖励身份，比较同起点、同数据和预算。不能承诺必然提点。

## 数据并行线

审计结果见 `sft_curriculum_restart_inventory_20260927.md`。复用已有source_reuse、grounded repair、多步骤inventory和curriculum_split，先核对来源、历史训练曝光、真实工具证据与完整对话验收，避免重复生成和重复judge。

基础、约束、策略分桶依据能力依赖，不以工具出现或对话长度替代。各桶按家族隔离留dev；阶段B混入A合格数据，阶段C混入A+B，数据不足如实报告，不把旧混合train100改名为foundation100。补完整多步骤、恢复和简洁结束示范；新的训练尚未启动。

## 启动前协调

确认上一评测任务的关机监控状态，避免新的GPU实验被旧收尾流程关闭。已核实该automation为PAUSED。

实际运行状态与验证范围见本文顶部更新；数据产物见 `sft_curriculum_evidence_build_20260927.md`。

## 后续检查与数据交接

本任务heartbeat `repair72-rl` 在两步学习验收失败后已暂停，防止旧自动流程放行正式训练；旧关机任务均保持PAUSED，不代表获得平台关机许可。本地一次性smoke CPU审计为 `results/maintenance/repair72_rl_20260927/audit_smoke.py`；先核对实际产物字段，GPU权重指纹与checkpoint结构不能替代逐张量映射或GPU恢复。

以下为已被新增400+60交付取代的中间进度：SFT子agent为用户指定Astra medium，当时完成v6：168候选，52条本轮完整审核37pass/15hold，其余131继承资格；累计A4/B34/C37。两条新增组合对话已原生回放及真实tokenizer计数，19项本地CPU检查通过；独立family-dev未ready，未启动SFT。详见 `sft_curriculum_evidence_build_20260927.md`。

## 用户追加：扩大实际造数（2026-09-27 11:20）

用户指出当前SFT数据太少，明确要求造数据。已将Astra medium子agent从剩余旧样本审核转为生成：目标训练400条，新增桶基础100/约束150/策略多步150，累计100→250→400，另建约60独立dev。此为目标而非完成量；需覆盖矩阵、真实工具执行、新鲜DB独立回放、逐次写入确认、终态与保护状态、token/mask/hash。优先真实状态依赖、多约束决策和失败重规划。旧v6保留，生成使用新版本；当前无新增SFT GPU或付费API授权。

## 运行检查 11:51

retry1已通过FP32 actor加载与初始rollout权重同步，GPU审计640个LoRA buffers、卷积all_conv_match=true；初始B全零。step1的8候选实际采样已完成，原生scored=0：3个context_window_exceeded标签、5个max_steps，训练奖励全部0、advantages finite且全部0。标签不能独立证明具体是哪层token边界，完整batch已留存；SHA `84bee28aa8c57423e0bb9c5070757b9ac0e0359cf67a83dcb722e531c6dd1ee1`。当前首步反向中，native stack证实FLA wy_fast backward Triton编译/调优，非已确认死锁；尚无已完成optimizer更新。有效学习信号与正式放行待第二步及最终验收，不以程序没报错替代。原12:49:47截止不变。

## 运行检查 11:59

step1 已完成，metrics 中 actor/grad_norm=0、pg_loss=0、kl_loss=0；单步1936秒，包含首次FLA编译调优。step2正在采样，尚未正式放行。进一步逐轨迹CPU检查：5条assistant_turns=15达到轮数上限；1条保存的response tokens恰为16384；另2条末次search_onestop_flight观察内容分别16657/26673 tokens，仅内容已超过剩余response预算，尚未包含模板开销。后一条原始111304字符被当前工具字符上限投影成65553字符，仍放不下；这不是模型生成了长回答。证据 `results/maintenance/repair72_rl_20260927/observed/step1_termination_diagnosis.json`。远端tool_agent_loop SHA与run源码快照相同，但与本地vendor文件不同；诊断核对的是远端分支，未同步替换vendor。

造数最终目录改为 `generated_curriculum_20260927_v3`：v2复查发现工具回执中的历史用户曝光，不能作为最终dev划分；v3排除这些用户，目标仍为400+60，独立性限于用户/订单隔离，模板共享。当前生成与最终验收尚在进行，不把旧v6的37条当作本轮交付。

## 新增数据最终交付 v4

v3中的1条内部日期键`2024-05-25_late`被root抽查拒绝；v4严格ISO日历日期检查，并替换6条与旧100/repair72重复的对话，替补再拒绝2条，共8次重复候选拒绝。最终数据基础100（搜索40、读取计算45、拒绝15），约束150（含新订票40），策略150（21真实中转、35往返支付拆分、5合法日期替代；其余89为状态依赖执行）。400训练为3,110,763 total tokens、230,794 assistant tokens；全460最大13,468，均小于24,576且未截断工具回执。每桶参数签名100/150/150，工具序列结构3/14/10；数量不等于独立语义模板数。

最终文件：`data/sft/generated_curriculum_20260927_v4/train.jsonl`、`validation.jsonl`、`A100_train.jsonl`、`B250_train.jsonl`、`C400_train.jsonl`、`summary.json`。这里A100指新增课程第一阶段，不是旧100复用。此前generated v1-v3是保留的中间候选，不用于训练；旧evidence v6也是历史审计结果。

## 两步验收失败的具体证据

step2耗时1080.52秒，4条airline_61正常user_stop、4条airline_296达到max_steps；所有奖励与advantage仍0。四条已评分的airline_61中，一条没有换乘客而转人工；另三条成功写入正确的Chloe/Daniel/Ethan，原生DB奖励因参考顺序Chloe/Ethan/Daniel而为0。CPU按保存工具调用在独立新鲜DB回放，四条最终DB hash与保存值完全一致、全部工具回执一致；后三条整个DB唯一差异均为`/reservations/90F795/passengers`的list_order_only。证据`results/maintenance/repair72_rl_20260927/observed/step2_native_zero_diagnosis.json`。旧奖励与轨迹未修改，不能把事后语义分析写成实际训练获得了奖励。

在两步验收结束时，现有`evaluation/outcome_contract.py::canonical_state/outcome_hash`已具备仅忽略乘客排列、保留重复数量和其他字段的CPU实现，但本轮RL交互仍走原生严格DB桥，尚未将该修正版奖励接入新RL profile。后续应先建立显式新奖励版本、保留原生分数旁路并在本轮buffer验证；不要把所有列表一概排序，不放松航班顺序/付款/乘客身份。另需单独处理9条轮数终止与3条response/工具观察预算终止。CPU修复不等于GPU有效更新验收；原16轨迹上限已用完，不能偷偷追加采样或将未通过smoke放行为正式20步。

## 乘客排列奖励修复验收

研究假设：乘客名单顺序不应使正确的完整业务终态被判零。新协议仅以保留重复数量的乘客无序比较替换终局DB分量，姓名、生日、航班顺序、支付顺序、其他数据库及用户状态仍严格比较；原生非DB门槛继续相乘。预算结束/未评分轨迹不补分。新协议不等同于全面政策合规评分，不引入部分分。

实现位于 `evaluation/rewards/terminal.py`，经 `envs/interaction.py` 的实际评分入口调用。使用独立新鲜DB执行gold，参考非法、缺失/非有限分量、验证后DB变化均报错；原生SimulationRun不改，原生分数、分量和新协议回执存于 `verifier_info_json.terminal_reward_selection`。默认仍是 `tau2_native_v1`。新增formal/smoke profile显式绑定新协议，runner/Hydra/模拟器配置/快照贯通；禁止跨奖励协议续训和混用旧冻结IRC配方。三算法入口均有CPU回归。15轮及16384 response预算此次保持原值，轮数与长观察问题未解决。

证据目录 `results/maintenance/repair72_rl_20260927/passenger_v1/`：

- `local-tests.log`：本地95通过。
- `remote-tests.log`：服务器既有环境无卡76通过、19 deselected；排除未部署的5090和无关历史profile。
- `buffer-replay.json`：16条旧buffer SHA保持不变；4条正常结束轨迹逐条原生工具回执/初末DB hash一致，实际调用新交互评分链；12条未评分保持0。第二批新奖励 `[0,1,0,0,0,0,1,1]`，优势5268个非零token，finite。task61无文本评判条件，重放只用真实工具史，没有伪造用户文本。旧GPU权重与训练成绩不改。
- `train50_reference_execution.json`：50个训练参考本地新鲜DB执行全部成功；此项不证明参考语义或场景均正确。
- `deployment_files.json` / `before_deploy_sha256.json`：首批12文件部署与原文件备份身份。后续4文件import顺序整理及runner预检报告奖励身份修正（避免新协议误标official）共5文件，单独保留 `import_cleanup_files.json` 和服务器 `before_import_cleanup/`，不覆盖首批回执。

下一轮有界GPU复验配置已经具体化：原repair_base重新起步，不续训旧零更新checkpoint；2×A800、2步、每步2组×4，最多16候选，2小时墙钟/4 GPU小时上限。runner内部7100秒给清理保留余量。需验证finite非零优势/梯度、真实LoRA参数变化、actor到rollout同步、保存及云端记录；若再次全部无信号则停止，不能无限补采样。旧16条预算已耗尽，追加复验预算待用户决定；已有“通过即正式20步”授权保留。CPU反事实重放不能代替GPU验收。

拟执行命令（未执行GPU）：

```bash
source /root/autodl-fs/tau3-core/activate.sh
cd "$CODE_ROOT"
python -m tau3_grpo.training.rl.runner --estimator grpo --engineering-smoke --updates 2 --max-wall-seconds 7100 --profile configs/train/rl/repair72_grpo_2xa800_passenger_v1_smoke.yaml --result-dir results/runs/repair72_grpo_passenger_v1_smoke_20260927/seed42
```

服务器公共runner最终 `--dry-run` 已通过（`passenger_v1/runner-dry-run-final/`）：50 train/60 selection身份核对、2步/16候选、GRPO、新终局协议在preflight/launch/Hydra三处一致，smoke不做selection评测。该运行明确隐藏CUDA并未启动服务。预检报告也已更正，不再将新协议误标成official-only。最终定向Ruff与diff whitespace检查通过。


## 2026-09-27 19:53 shared32 GPU verification started

User explicitly approved immediate stop of old fast32, preservation of logs/trajectories, discarding unsaved updates and fresh shared-two-card verification. Verified controller 102381 received SIGINT; its state says failed because of manual interruption, not a training defect. All GPU processes were absent before new launch. The separate CUDA Graph probe finished before stop (no probe killed).

Shared profile b7d56204617aa0a903b409412ea54e35d2c0947f63980d0ef0cfc6fa03f93cbe passed remote smoke dry-run: world_size=2, groups4 x8, 2 updates, save_freq2, test_freq-1, no initial eval, 64 total candidates. New run results/runs/repair72_grpo_passenger_v1_shared32_smoke_20260927/seed42; controller141099, launcher/authorization retained in run. Bound7100s plus cleanup, original absolute overall deadline1790753043.1027153 unchanged. Same repair72 start, FP32 IEEE/full recompute and reward; simulator shares GPU1 then sleeps, two policy replicas TP1. No old world1 updates resumed. GPU acceptance is pending: actual simulator sleep/wake, finite meaningful update, rollout sync, world2 complete checkpoint and cloud steps1/2, wall timing. Formal20 must await acceptance.

Old fast32 step1=41.01min, step2=26.96min. CUDA Graph isolated model-only 16-prefix test completed: eager adapter0 8.756/8.716s versus graph4.925/4.929s; adapter1 eager8.404/8.253s versus graph4.949/4.922s. Engine startup97.74s vs265.50s. Nonzero adapter greedy output varies even in eager; complete equivalence and end-to-end speed not established. Graph not enabled in shared smoke. Microbatch2 probe rejected as documented.


## 2026-09-27 20:15 shared32 failure and deployment correction

Live inspection found controller141099 failed at20:05:38 before any rollout or update. Root cause: missing remote tau3_grpo.integrations.verl.sleeping_simulator; training/rl/simulator_sleep.py and tests were also missing, and remote simulator launcher lacked --enable-sleep-mode. Prior full Hydra dry-run validated config values but did not import the dynamic AgentLoopManager; the earlier deployment completeness claim was incorrect. Owned GPU processes cleaned automatically; zero metrics/trajectories, no new checkpoint. Old failed run retained.

Deployed only the two missing modules, their four-case CPU test and exact four-line sleep-mode launcher addition after remote hash check/backup. Receipt results/maintenance/repair72_speed_storage_20260927/shared32-deploy-fix/receipt.json. Remote tests pending at this entry; no claim of GPU verification. A retry must use a new run, retain the original shared-smoke absolute deadline1790517215.5077481 (19:53:35 launch +7200s), and reserve startup/cleanup within it. No extra samples beyond original2x32 or new two-hour allowance.

20:17 correction tests finished remotely: 4 passed in58.07s, including real dynamic module import. Fresh retry1 launched with controller145988 under timeout145987; run results/runs/repair72_grpo_passenger_v1_shared32_smoke_20260927_retry1/seed42. Remaining5783s at launch, runner5603s/outer5723s; original shared-test deadline1790517215.5077481 preserved. No GPU result yet. Monitoring updated to retry1.

21:03 live check: shared32 retry1 completed step1 in1598.9528s (26.65min), gen844.5636s, old_log_prob118.1294s, ref112.4803s, actor479.6542s, weight sync43.5914s. Finite nonzero grad_norm0.1253593; second32-candidate rollout now active, simulator sleep/wake cycle successful. 18min target not met. Compared old fast32step2(26.96min), total near unchanged; actor update lower(7.99 vs11.14min) but sampling longer(14.08 vs10.46min). These are different sampled trajectories, not controlled identical-workload speedup. Full2step/complete world2 checkpoint/cloud acceptance pending.

21:28 offline GPU-artifact acceptance PASSED14/14 for shared32retry1; full report results/maintenance/repair72_speed_storage_20260927/shared32-engineering-acceptance.json. Both32 batches, same-step finite nonzero advantages/gradients, both actual rollout replicas updated, finite changed saved adapter, completeworld2checkpoint, cloud1/2 and simulator lifecycle verified. This does not prove GPU restore or task capability gain. Step2=1108.5177s(18.475min), save162.5887s, excluding save945.929s(15.765min); firststep1598.95s. 18min-all-steps target false. New formal shared profile only changes experiment label from candidate; baseline oldthroughput240 result+metrics SHA and model/selection/val_kwargs/reward identity verified. Formal dryrun underway; not yet launched at this entry.

21:30 formal shared32 launched: controller169037, outer timeout169036, run results/runs/repair72_grpo_passenger_v1_shared32_formal20_20260927/seed42. New profile SHA f6ac562f87ae7d72f39e49ecbd9e2bfb44ba7a188bf9d3a48d2f6e34afb5f231, CPU full dryrun passed640 candidates/world2/save10/eval10; same model/data/val sampling/reward and initial result hashes verified. Fresh repair72, no smoke resume. Formal absolute deadline1790573409.6468837 (16h within original1790753043.1027153), runner57420s, outer57540s. STOP_AFTER_BOUNDARY still pending until controller training; heartbeat updated.

21:32 user explicitly requested deleting current saved checkpoint and extra persistent artifacts. Deleted only completed smoke retry1/global_step_2 and two synthetic CUDAgraph adapter weights; actual allocated bytes21,502,443,008 (20.026 GiB) freed. Disk free351,505,412,096 bytes (~327.36GiB). Retained all log/buffer/metrics/14-check acceptance/cloud/complete-checkpoint manifest/probe configs+identity+results. Receipt shared32-deletion-completed.json (local), remote shared32-authorized-deletion/deletion-completed.json. Smoke checkpoint no longer restorable; historical acceptance was before deletion, do not rerun old audit requiring it. Formal runner does not depend on deleted checkpoint.

21:35 verified formal controller169037 reached training; created STOP_AFTER_BOUNDARY exclusively after state transition. Scheduler remains20; stop request should select boundary10 after first update. No formal rollout metrics yet.

用户于2026-09-27约22:01明确要求立即停止远程正式训练。经核验向controller169037发送SIGINT，RL heartbeat已PAUSED；不自动恢复，不影响SFT。保留日志/轨迹/既有产物，停止记录在正式run/user-immediate-stop.json。

User reauthorized formal RL restart at22:36 and explicitly requires initial evaluation, no reuse. Before mutation, hashed1231 tracked relevant package/config/launcher/vendor files across local and remote: zero differences; receipt full_initial_eval_sync_check.json. Local HEAD b9c981a992da0d238448b2ec3b885cf470f9dea9. Added explicit shared32_full_eval profile overriding val_before_train=true; no old run altered. New fresh20 run pending full CPU dry-run; retain previous formal absolute deadline1790573409.6468837, no budget reset. Selection240 at0/10/20,640 train candidates; two GPUs, same repair72 terminal reward/LoRA16/IEEE/full recompute.

22:39 full-eval formal20 launch submitted: results/runs/repair72_grpo_shared32_full_eval20_20260927/seed42, timeout184500. Profile89c6cc34975fe6c1ed6e1cda8825d51d0d5e2edb312d00e53fe857ad2c3a40ff. Remote full dryrun passed and val_before_train=True confirmed. New640train+720eval candidates; maxwall53277 and oldformal absolute deadline retained. No baseline reuse. Monitor ACTIVE updated; new STOP_AFTER_BOUNDARY pending until training state.

22:54 full_eval controller184501 verified training process; model replicas still initializing, simulator initial sleep succeeded, no evaluation trajectories yet. Created STOP_AFTER_BOUNDARY only after training state. Heartbeat updated; current marker will select10 after first update.

23:39 full_eval live: initial validation177/240 stream records, controller184501 active, no training step yet. One logged traceback23:27 from tool_parser malformed model XML (ValueError substring not found) was caught at response parsing; subsequent trajectories continue. Not a controller/engine crash; retain this model-format error in evaluation accounting, do not silently repair outputs or restart.

23:54 live check: actual initial selection evaluation completed240/240, terminal mean reward0.4458333333 (107/240 if binary), scored0.85. Simulator wake1790521093.86246 to sleep1790524246.12261 ~=52.54min evaluation phase incl interactions; not full startup time. Initial metric persisted atstep0; cloud not separately read back this check. Trainingstep1 now active,4/32 streamed complete. No capability gain claim; known selection reference/simulator issues retained.


## 2026-09-28 04:24 boundary10 cloud verification failure

Current full_eval20 completed10 updates and240 step10 evaluation rows; checkpoint-complete world2 retained. Controller exited failed: Cloud step readback incomplete. All GPU processes cleaned; STOP_AFTER_BOUNDARY retained; repair72-rl automation PAUSED per failure instructions, no restart. Read-only live SwanLab API verified trainer/global_step contains exactly0..10 (each step=value), including real initial evaluation step0. runner.py verify_completion currently compares all points strictly to1..10, so the extra legitimate0 causes false rejection rather than missing uploads. Step10 rewardmean0.4833333333 vs initial0.4458333333 is selection-only, not established capability gain. Step10 training/save1612.073s, save155.823s, testing2806.376s. Fix and validated resume remain pending; absolute deadline1790573409.6468837 unchanged.


## 2026-09-28 08:48 explicit user-authorized resume10

Cloud verification now accepts optional genuine (0,0) in addition to exact1..N, still rejects missing/duplicate update points. Local commit0dead01, remote182af5e; changed-file SHA match. Remote20 related CPU tests passed. Live cloud0..10 and world2 CP structural verification passed; both rank optimizer states and scheduler/RNG/data loaded on CPU. GPU restore remains pending. Original failed state/config/budget retained under resume10-before; STOP_AFTER_BOUNDARY archived there after acceptance.

Remote full resume dryrun passed world2/updates20/resume_path=global_step_10/val_before_train=false. Same formalrun and SwanLab retained; actual controller309836 under timeout309835 launched at1790556483.5775347. Deadline1790573409.6468837 unchanged, remaining16926s, runner16776/outer16896. Receipt resume10-launcher.json; log resume10-controller.log. No new step10 stop marker; next required complete save/eval is20. repair72-rl monitor ACTIVE. SFT owner separately restored Kimi per user; sft-500 monitor ACTIVE.


## 2026-09-28 user override: no fixed deadline; automatic repair and continuation

User explicitly cancels both former wall-clock deadlines and authorizes diagnosing/fixing/verifying recoverable failures then automatically continuing from latest complete checkpoint to20 in the same run/SwanLab. Never blindly retry, alter the protocol, discard evidence or duplicate curve points. The step0 cloud validation fix is already deployed/tested (local0dead01/remote182af5e). Monitor prompt updated ACTIVE accordingly. Future launch omits timeout wrapper and --max-wall-seconds. Current live timer removal is being verified separately; old command-line flags alone do not establish timer state. No restart of active trainer is authorized merely to change a timer.


## 2026-09-28 11:30 live deadline removal verified

User cancels former fixed wall deadlines and authorizes diagnose/repair/test/automatic same-run continuation to step20 after recoverable failures. Future launches omit timeout and --max-wall-seconds; the monitoring automation is ACTIVE with this policy. Existing step0 cloud validation fix remains deployed (local0dead01/remote182af5e;20 related tests passed).

Both live timers were removed without restarting the trainer. After disposable-process testing, controller309836 alarm(0) returned7114 remaining seconds, a second call returned0; outer309835 POSIX timer0 deletion returned0 and /proc/309835/timers is empty. Both processes remain alive, GPUs both100%, completed_step16 with step17 update in progress. Old CLI flags persist but no longer represent armed timers. Full receipt: results/runs/repair72_grpo_shared32_full_eval20_20260927/seed42/resume10-deadline-removal.json (verified=true). No GPU workers were attached or restarted.


## 2026-09-28 user-authorized AutoDL 051 shutdown after evaluation

User explicitly requests shutting down AutoDL machine051 after evaluation. Complete step20 selection240, checkpoint and cloud verification first; verify instance identity matches051, preserve results and coordinate with SFT owner so active paid API calls drain and ledger/evidence are safe. Move needed subsequent CPU/API work locally if feasible, then shut down only051 via platform control, without destroying storage. This supersedes the earlier no-platform-shutdown instruction for051 only. RL automation updated to perform and confirm this before pausing.

User further confirms: after051 shutdown, finish SFT data on Mac first. Migrate originals and the sole authoritative DeepSeek budget ledger after remote paid calls drain and the writer releases its lock; archive stale Mac ledger, preserve history/reservations/caps. Continue Mac CPU/API only; do not restart051 or start SFT training. SFT owner coordinates migration and provides shutdown-readiness receipt.

2026-09-28 13:15 shutdown identity verified read-only in existing Chrome AutoDL console: A800专区/051机 instance857546be50-ec463c07, 2xA800 running, matches remote hostname autodl-container-857546be50-ec463c07. Other instance011 (47bb4a8b0c-e6603b21) is already off and must not be touched. Native cua Google Chrome AX available; browser-provider discovery errored, native app control works. Do not click shutdown before RL final evaluation/cloud verification and SFT full-source sync receipt.


## 2026-09-28 13:35 formal20 complete; shutdown pending Mac unlock

All20 optimizer steps and0/10/20 selection240 each finished. Scores107/240=44.5833%,116/240=48.3333%,118/240=49.1667%; selection only. Final upload omitted154 step20 scalar values despite SDK finish, so controller correctly rejected missing20 (not the former zero-step bug). Recovered only absent exact scalar values from immutable local metrics into same SwanLab run; independent all-key cloud readback and verify_completion passed0..20. Preserved original failed state and plan/verification under cloud20-recovery. Bothrank optimizer496 states, scheduler/RNG/data CPU load passed; complete CP20 retained persistent disk and all17 files SHA recorded. Logs/buffers/evaluations/adapter and checkpoint metadata synced Mac; model/optimizer tensors remain remote persistent disk. Local core and adapter hashes verified. SFT migration receipt confirms complete source and ledger handoff. GPU services exited. AutoDL051857546be50-ec463c07 identity verified; shutdown attempt blocked by locked Mac before any click. User asked to unlock; machine NOT yet shut down.

2026-09-28 14:14-14:15: Mac unlocked. Confirmed051857546be50-ec463c07 normal shutdown in AutoDL console, refreshed and independently observed已关机.011 untouched. No instance deletion. shutdown-status.json saved locally; RL monitor paused after complete20/eval/cloud/backup/shutdown. SFT continues Mac CPU/API.
