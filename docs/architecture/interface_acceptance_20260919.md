# 接口补验与工程收尾（2026-09-19）

## 2026-09-20：三份工程 step2 检查点已退役

按用户明确授权，已删除 GRPO 工程 step2 和 MT-GTPO DF off/on 各自 step2，共三份。释放 163,104,264,192 字节实际分配空间（151.90 GiB），持久盘清理后可用 219.31 / 500 GiB。五份正式 E0–E3/MT-v3 step20 检查点在清理前后均通过结构校验；78份轨迹、指标、权重审计和配置证据 hash 未变。三份工程节点不再支持精确续训；原 checkpoint 完成收据和数据进度已归档，失效的 latest 指针已移除，各 run 写入退役记录。历史验收通过结论仍有效，不代表文件仍保留。

本次未启动 GPU 实验。删除回执：`results/maintenance/engineering-checkpoint-retirement-20260920/receipt.json`。清理后、文档更新前实测本地与远程 1,912 份源码/配置/测试/文档文件 hash 全部一致；本次文档同步另保留前像及 hash 回执，模型、数据、环境和运行产物不属于源码一致性范围。

## 2026-09-20 03:22 工程验收完成（历史资产快照）

本轮已授权的工程验收与接口补验全部通过，已结束GPU作业。最终只读核对：无GPU计算进程、无本轮训练/队列控制器；8份保留的完整checkpoint结构和文件大小收据有效，6项模型/SFT资产链接均存在。持久盘剩余72,377,036,800字节（67.41 GiB）。这不是对所有历史checkpoint逐一重新GPU恢复；真实恢复证据由C提供。

|阶段|实测结果|
|---|---|
|CPU与数值|远程1302 passed / 15 GPU skipped；独立GPU-only 17 passed / 4 CPU deselected；控制器补丁15项通过；lint零新增、382历史债|
|增强A|GRPO/GiGPO/MT-GTPO各4条、共12条，身份/生成logprob/token-mask/工具事实通过；各4份×331个活动文本参数逐值一致|
|B-GRPO/GiGPO|各2步真实非零更新、完整保存与云端回读已通过；历史GDN同步范围，不追溯伪造全参数回执|
|C-GiGPO恢复|四rank模型/optimizer/RNG/scheduler实际恢复，step2→3，云端1/2/3同run，数据16→24；不宣称不中断逐位等价|
|D独立评测|724/724导出张量核验、真实加载、2任务×4次=8/8完成、0异常；仅工程小样本|
|B-MT DF off|2步128候选、两次非零更新、完整checkpoint2、云端1/2、128条奖励/优势/过滤重放、12份活动文本权重回执通过|
|B-MT DF on|同上；DF配置enable=true，两批各8组均有有效信号，实际剔除0组；不能宣称过滤收益|

DF on：SwanLab run `13e546fb6061454984ffa`；grad norm 2.0622367/2.4400443，非零优势token 204,118/113,008；94条官方评分、30条max_steps、4条context_window_exceeded。记录1,330轮、318,921生成token；12份331参数报告涵盖初始和两次更新后三套状态。耗时约72.39分钟，五卡预留上界6.033 GPU-hours。两组MT保持任务调度和128候选预算，但生成轨迹不同，不能用两步分数推断算法增幅。DF边界/空mask/零信号分支的证明来自对应CPU用例；此次真实两批没有发生整组剔除。

全部权重审计范围明确为dense文本、TP1、未量化、无LoRA，vision排除。B新`interfaces`是全活动文本证据，旧`audit`仍标注GDN范围。正式IS/RS未启用；data seed=42与生成请求seed=null及engine seed分开记录。基础设施资格策略为`tau3_execution_eligibility_v1`，预算截断保留候选但不冒充官方评分，无自动重试或补采样。

新增52份DF on小型证据已下载并核对每个文件hash：`results/maintenance/interface-acceptance-20260919/remote/mt-df1-complete-20260920-0320.tar.gz`，SHA256 `7cb6fdac370f5ce66f1777d043c5fcff86dbd11a4f157eaf035ce5e4fbd32a22`。最终资产/进程/checkpoint检查：同目录`final-review-20260920.json`。A、DF off、C/D及失败原记录均保留。

本轮代码架构和已约定工程集成验收可关闭；正式20步效果、多seed/机制消融、Git分批审查提交、托管CI、历史lint债另列。GiGPO临时step3在C/D通过后已退役，明确放弃该节点精确续训，推理导出保留；正式E0–E3与MT-v3完整checkpoint保留。

关机是单独操作：文档同步后经Google Chrome核对015实例，再执行普通关机。最终UI结果单独记录于本地`results/maintenance/interface-acceptance-20260919/shutdown-015.json`，只有该回执为verified_off才代表关机完成。AutoDL页面明确提示“连续关机15天会释放实例，释放前实例在数据在”；本次禁止释放/删除，系统盘及数据盘不能被描述为永久持久备份。Chrome锁屏阻碍已解除，当前可访问AutoDL。

下方为历史阶段记录，未完成表述仅对应当时状态。

## 2026-09-20 02:05 最新实测状态

MT-GTPO DF off两步完整验收通过：128候选、两次有效更新，grad norm分别1.9267303/1.8837945，非零优势token分别176,371/122,759；最新完整checkpoint2通过校验。同SwanLab run `a669d73898fe481eb87d4` 的云端步数1/2回读一致。128条包括93条官方评分、32条max_steps和3条context_window_exceeded；35条预算截断保留候选且标记未官方评分，没有补采样。两批各64条过程奖励/优势/DF重放通过；16实际组×8 trial，1,338轮、299,859个生成token的行为logprob检查通过。

12份全活动文本权重报告均精确匹配331个参数，涵盖初始、step1后、step2后三套状态；不含vision。旧嵌套`audit.full_parameter_mapping_verified=false`指旧GDN范围，新增`interfaces.all_active_text_parameters_match=true`和独立回执为全活动文本证据，不能混写成全多模态验证。GPU阶段约70.83分钟、五卡预留5.903 GPU-hours，阶段清理时GPU进程为空。

51份证据（含两批真实轨迹）已下载，全部文件SHA256核对；归档`results/maintenance/interface-acceptance-20260919/remote/mt-df0-complete-20260920-0205.tar.gz`，SHA256 `bb6a1cacc4c8886f04fb2015cbf45f6237ea6c2ad8cb0b472d53dca7c7db6606`。v3已自动进入DF on，execution根`20260919_mt_df1_execution`，开始时间1789840452.3728778，90分钟硬截止1789845852.3728778；不重跑DF off或重置预算。

剩余GPU项为DF on的两步验收，随后独立汇总、恢复资产检查和Chrome关闭015。正式20步效果/多seed/机制消融、Git提交与托管CI单列。Mac锁屏曾阻断Chrome，关机前必须重新操作并验证；当前尚未关机。

## 2026-09-20 00:45 最新实测状态

增强A三算法全部通过：GRPO/GiGPO/MT-GTPO各4条真实轨迹，合计12条均进行官方评分；分组/trial/data-seed语义、实际processed生成logprob、token/mask/原始结束原因和多工具事件核验通过。三算法分别38/38/42轮、6,650/7,311/7,786个生成token，真实多工具轮分别5/6/5。每算法4份权重回执各精确比较全部331个活动文本参数，无缺失；独立核对逐值标记及SHA256一致。每算法只有1个初始参数状态、0更新，vision不在范围内；更新后的同步由后续B验证。

A耗时3,685.38秒（约61.42分钟），五卡全时预留上界5.119 GPU-hours，不是利用率积分。完整52份小型证据已下载：`results/maintenance/interface-acceptance-20260919/remote/enhanced-A-complete-20260920-0045.tar.gz`，SHA256 `dc6b406b276689f97c3114367c82b5c343af5574230ce37cfbc8fc03b5108274`。

v3自动进入MT-GTPO DF off两步更新，随后DF on；各128候选/90分钟，不重置预算。剩余为这两组真实更新、完整保存、云端步数回读与过程奖励/优势/DF重放，以及最终资产与证据核对、Chrome关机。C恢复、D独立评测及GPU数值检查不重复。Mac锁屏曾阻断Chrome，已通知用户解锁；015仍运行，不能宣称本轮全部完成。

下文保留阶段定义与历史时间点记录；当前状态以上方最新实测为准。

## 本轮范围

C：GiGPO step2→3，最多新增64候选，75分钟上限。D：从该完整检查点导出并实际服务，selection固定前2任务×4次，90分钟上限。增强A：GRPO/GiGPO/MT-GTPO分别4条，无参数更新，合计12条，3小时上限。MT-GTPO B：原v3配方，DF off/on分别2步、128候选，每组90分钟上限。每次失败保留原目录，修复重试另立尝试和预算，不偷偷重置计时。

这些是工程验收，不是20步正式主对照、算法提升、多seed或机制消融。官方final50不参与。用户已授权这些GPU任务，并要求全部验收收尾后在Google Chrome的AutoDL关闭015实例。

## 接口语义

- group UID直接来自实际estimator batch；在分发到worker前按UID编号trial，避免同一组跨worker时重复从0计数。不以task ID猜分组，不把随机UUID描述为跨次重跑固定ID。
- `identity.seed`记录配置中的data seed，附`seed_semantics=configured_data_seed`。生成请求seed若未显式设置保持null；实际engine seed及温度/top-p/top-k/repetition penalty逐轮记录。独立评测的seed+trial协议单独保存，不和训练data seed混用。
- 正式公共runner显式开启`calculate_log_probs=true`、`processed_logprobs`。行为概率来自实际生成分布，旧策略/参考策略logprob仍由框架独立计算。缺失任何一轮或长度不齐不能声称完整；observation位置为0占位且mask=0。
- 不默认启用IS或拒绝采样。工程Hydra预检要求rollout_is、rollout_rs为空，bypass_mode=false；以实际YAML为准，不能用上游dataclass的不同默认值替代。
- `tau3_execution_eligibility_v1`区分官方已评分、预算截断、模型/工具失败、基础设施失败。正常终止沿用官方reward；预算截断沿用0 outcome且保留候选，明示未进行官方评分。基础设施/未知异常终止中止训练batch或作为独立评测未解决错误，不作0分替代、自动重试或补采样。
- 以上基础设施处理是明确的harness行为修订；原有reward/shaping/优势公式及DF顺序不改。旧buffer只读重放，不回写旧结果，也不补造缺失UID或行为概率。

## 权重同步

新增opt-in全套文本策略参数审计：在接收actor tensor时保存CPU副本，独立重建QKV/gate-up/GDN组合映射，待vLLM加载及后处理完成后逐值比较、保存SHA256、缺失参数和tied alias证明。范围为Qwen3.5 dense、未量化、TP1、无LoRA的活动文本参数。vision明确排除，不宣称验证多模态权重。旧GDN卷积审计继续保留。

审计仅工程计划开启；需真实GPU结果才能说映射通过。单测包含参数损坏、缺失packed slice、遗漏活动参数、tied embedding和vision排除，不能替代GPU证据。

## 当前证据

- 本地相关回归117 passed；另外真实DataProto跨worker分组与概率诊断不改mask/IS的2项通过。测试集合有重叠时不相加。
- vendor清单从原固定revision tar包重建并核验，未升级上游版本。
- C恢复日志已见四rank的model/optimizer/RNG/scheduler实际加载；prior-data.pt为已消费16个任务、2批。第3步完成及后续数据进度尚待检查。
- 本地全套初验1282 passed / 18 failed / 15 skipped：17项启动脚本因测试调用未将venv加入PATH失败，修正环境后的相关93项全部通过；另1项本地torch 2.8 CPU FSDP形状报错，在服务器固定环境对应3项全部通过（69.64秒）。不修改生产代码去掩盖环境差异。
- 远程新增代码部署、完整CPU回归、增强A、MT两组B和D结果均待填入真实回执。

## 交付边界

GPU通过后可关闭本轮主架构与工程集成验收。33个仍被引用的历史入口不因目录简短而删除；正式效果研究、分批Git审查提交、托管CI和已登记历史lint债务分别跟踪。关机前同步小型结果、源码身份和恢复记录，并确认目标为015的“关机”，不是释放实例。

## 23:04 实测更新

C通过：四rank真实恢复、step3非零更新、同SwanLab run的1/2/3回读、数据指针16→24及下一批8任务×8轨迹匹配。完整checkpoint3保存；此次上界4.263 GPU-hours。回执已下载 `results/maintenance/interface-acceptance-20260919/remote/`。

26文件部署有前像与hash收据，随后额外记录runtime import排序和MT重放验收追加。远程CPU **1302 passed / 15 GPU skipped**（504.41秒），vendor通过，lint旧债382、零新增。旧320条工程buffer检查：243官方已评分、77预算截断；已有reward/token/mask/span完全保留。

D导出完成、724张量验证及源hash记录已落盘，实际独立2×4评测运行中。增强A和MT两组B已无卡生成实际Hydra计划，尚未执行GPU。

GPU数值首轮17 passed / 4 failed：失败均为CPU参数化用例在GPU可见进程误走Triton（`Pointer argument ... cpu tensor?`），15项GPU用例通过；其CPU版本已在CUDA隐藏的完整回归通过。保留整轮failed回执；新目录 `20260919_gpu_numerical_gpuonly` 用 `-k not cpu` 分开补验，不修改生产模型逻辑。原finish-queue按失败门槛停止，接续入口为 `results/maintenance/interface-deploy-1789829068/finish_queue_v2.py`、状态目录`finish-queue-v2/`。

本任务heartbeat `tau3-015`每10分钟接续核对，避免等待长GPU阶段时丢失关机收尾；无变化不通知。所有状态仍以具体回执为准，015尚未关闭。

## 23:37 实测更新

- D通过：724/724导出张量与重建BF16源精确一致，实际vLLM加载后独立2任务×4次全部完成，8/8、0异常。pass@1=0.625仅用于证明工程指标链路，不是正式模型效果或增幅。两卡预留口径0.395 GPU-hours，整机五卡同期上界0.987 GPU-hours。
- GPU-only复验17 passed / 4 CPU deselected（52.99秒），包含前述15项GPU数值检查；原混跑失败记录保留。
- 临时GiGPO checkpoint3退役完成，删除54,368,082,499字节；源模型SHA再次核对，D导出文件hash、真实加载和评测通过且无打开FD后删除。保留完整回执、data指针证据及推理导出；明确放弃该临时节点的精确续训，不能把留存的data.pt当作可恢复checkpoint。退役后实测可用181,710,168,064字节。
- v2队列在增强A采样前遇到8100 bind占用；原日志保留。后查无监听进程，两种bind均可用，无法事后证明当时具体连接状态；预检改为与服务相同的SO_REUSEADDR，覆盖关闭连接的TIME_WAIT，仍拒绝真实监听。控制器相关15项本地与远程通过（远程7.69秒）。不是算法/奖励修复。
- 新A目录`20260919_interfaces_retry_s42`，新队列`finish-queue-v3/`；旧队列不重启、不回写成功。v3复用已通过D和退役回执，依次完成增强A和MT-GTPO DF off/on，不再次执行C/D或删除检查点。

23:40：新A准备完成，v3接续已启动（控制器PID189720）；通过后自动进入MT两组B。已完成70份小型证据已回收到本地`results/maintenance/interface-acceptance-20260919/remote/completed-evidence/`。最终验收尚未完成，015保持开机。

## 2026-09-20 00:10 实测更新

增强A的GRPO已通过：2任务×2次共4条轨迹、38轮、30个工具调用，其中5轮多工具；实际分组/trial/data-seed语义与6,650个生成token的processed logprob均通过。4份真实rollout权重回执各覆盖全部331个活动文本参数，逐值及SHA256一致，无缺失；本轮只有1个参数状态、无更新，vision明确排除。证据包已下载并独立核对，位置`results/maintenance/interface-acceptance-20260919/remote/grpo-A-evidence-20260920-0010.tar.gz`。

v3队列已进入GiGPO，后续增强A的MT-GTPO与两组MT更新仍待完成，不重新启动任何阶段。Chrome预检被Mac锁屏阻断，已通知用户手动解锁；远程验收继续，015尚未关机。
