# GiGPO 原论文对照与语义规范化 v3（2026-09-14）

后续已完成 [决策状态成对无卡审计](decision_state_pair_audit_20260914.md)：47 对定向样例显示 v1 误合并与 v2/v3 误拆分并存，当前不建议直接重训 v3。

当前是 **Tau-GiGPO 适配变体，不是原论文实验或官方训练器的严格复现**。核心两级优势结构一致。v3 已实现、同步服务器并通过无卡检查，但真实对话上的重复匹配只从 16 增至 20，尚未解决 v2 的过度拆组，不能据此宣称成功率提高或建议立即正式重训。

本轮未启动 RL、vLLM、用户模拟器或新的 SwanLab run。历史 20-step 实验、检查点和成绩没有改变。

## 语义规范化实际做了什么

实现位于 `tau3_grpo/algorithms/anchors/semantic.py`，通过 `decision_evidence(version="v3")` 接入真实 anchor hook。新配置为 `configs/train/rl/qwen35_4b_full_a800_gigpo_semantic_20260914.yaml`，使用独立结果目录；模块缺省仍为 v2，历史 c50 配置仍为 v1。协议 sidecar 阻止在旧目录静默切换锚点版本。

这是一套有界、整句匹配的语义规则，而非通用语言理解模型：

- 将有限句式的同意、拒绝、暂停、撤回表示为稳定事件；同意事件仍与上下文方案一起参与状态比较，不构成工具执行授权。
- 规范化明确的用户/订单引用、取消请求、查询经济舱价格许可，以及开场/索取 ID 问句。
- 完整取消方案必须包含订单、退款金额、币种及支付方式；支持的等义句式共享表示，金额与货币符号规范化。
- 规则必须覆盖整句。条件、否定、额外操作、变更对象或支付方式不能通过只抽几个字段被丢弃；未识别文本保留为 opaque。
- `Yes?`、`I approve?` 等疑问表达不会被当作明确同意；ID 大小写保留，缺失字段不猜测。

v3 仍保留有序对话证据链、读账本、工具事件、方案与回复证据。因此一条未识别的历史改写仍可能永久拆开后续状态。没有为了提高匹配率而直接删除历史条件。

### 真实对话检查

输入是已有 E2/E3 selection 的 479 条轨迹（E2 原始 239 条，未混入单条补测），共 5,138 个 assistant 决策，其中 4,659 个非初始决策。每个实验内按 task 比较四次 trial。

**使用固定 DB 代理，未重放中间数据库。这是对话分组诊断，不是线上训练覆盖率、非零优势覆盖率或成功率。**

|版本|非初始且跨轨迹重复的位置|占全部非初始位置|平均 hook 时间|
|---|---:|---:|---:|
|v1 旧特征|3,706|79.54%|0.22 ms|
|v2 精确对话证据|16|0.34%|1.11 ms|
|v3 有界语义规范化|20|0.43%|1.51 ms|

v1 的高重复率不代表其分组正确；已有查询许可、条件批准、变更支付方式等反例。v3 比 v2 多 4 个位置，绝对覆盖增加约 0.086 个百分点，改善很有限。

7,240 条非空 assistant/user 消息中，589 条进入语义规则，6,651 条仍为 opaque。589 条中包括 479 条开场问候；除去开场，只有 110/6,761（1.63%）被规则覆盖。这解释了为什么少量等义句合并尚不能解决实际拆组问题。覆盖率也不等同于语义准确率。

耗时是在固定 DB 下且与其他 CPU 检查同时运行时测得，不包含真实 DB hash 开销，不能用于推断 GPU 训练吞吐。

## 原论文和原官方代码的对照

- 论文：[Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978)，[本轮使用的 v1 全文](https://arxiv.org/html/2505.10978v1)。
- [官方仓库](https://github.com/langfengQ/verl-agent) 的 README 明确要求：复现原论文结果应使用 2025-06-03 Major Update 之前的版本。
- 本轮主要算例参照：[原论文时期 core_gigpo.py](https://github.com/langfengQ/verl-agent/blob/f974dc5977908d6007ad178c9f3c48f0c7b39331/gigpo/core_gigpo.py)。
- 同时核查：[当前官方 core_gigpo.py](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py)。没有把新版实现自动当作原论文实现。

### 核心一致之处

论文 Eq. 3–7 使用 episode 组内相对优势、同组内重复状态的 step 相对优势，再合成：

`A(i,t) = A_episode(i) + omega * A_step(i,t)`。

本项目保留这一结构，不训练 value critic。当前仅有终局奖励，因此 step return 为 `gamma^(T-1-t) * R_terminal`，是未来奖励折扣和在仅终局奖励下的特例。`gamma=0.95`、`omega=1` 与论文 ALFWorld/WebShop 设置相同。

### 两个需要纠正的判断

1. **Fnorm=1 合法。** 论文 Eq. 3 及实验表同时讨论 `Fnorm=std` 与 `Fnorm=1`。官方原/现实现也支持 `mean_norm` 和 `mean_std_norm`。旧项目 episode、step 均值中心化属于论文允许的选择，不能说它本身写错。问题在于 E0 默认 std-GRPO 与旧 E2 mean-only 同时改变归一化和步级信用，不能把成绩差异全部归因于锚点。官方 ALFWorld 脚本选择 std，是一个实验设置，不是论文唯一允许的设置。
2. **同一轨迹内重复状态可以入组。** 原官方 `build_step_group` 按 episode group 和相同 observation 分组，没有要求不同 trajectory UID。同轨迹回到同一状态也是算法可以比较的行动。跨轨迹、非初始统计是本项目更严格的诊断维度，不是原论文信号合法性的必要条件。

### 适配与差异

|项目|论文/官方参照|本项目|
|---|---|---|
|状态相等|同一个 episode 组内重复的环境状态/observation|task + DB + 读信息 + 确认/对话证据；v1/v2/v3 为项目自定义|
|历史信息|要求可比较状态，不要求整段历史文本完全相同|v2/v3 对完整对话证据链做 hash，可能过度拆组|
|奖励|支持逐步奖励；论文实验成功 10、失败 0、无效动作 -0.1|当前终局 0/1，无中间动作奖励|
|归一化|原官方一个模式同时作用 episode 与 step|旧版均 mean；新 audit/v3 候选为 std episode + mean step，属于混合变体|
|训练行|官方按环境 step 组织样本|整条多轮轨迹一个张量，再用 span 注入 step 优势；损失权重不能默认等价|
|等回报组过滤|不能仅凭官方也有 dynamic 变体就判断实现等价|E1/E3 额外使用固定 rollout 的 mask 过滤|

新 `episode_normalization=grpo` 模式的用途是让 `omega=0` 精确回到当前 E0 的 GRPO，以隔离步级项。它**不是恢复原论文唯一正确算法**。新 v3 候选配置也使用此混合模式，而非严格原版。

### 官方代码数值对照

CPU 脚本 `env_info/a800_20260912/compare_gigpo_reference.py` 仅从已检查的原官方源码提取六个纯函数定义，不执行下载文件的顶层导入代码。固定 mean_norm、gamma=0.95、omega=1，并构造可比的 step/trajectory 布局：

|算例|项目与原官方最大绝对差|解释|
|---|---:|---|
|两轨迹等长 [2,2]，reward [1,0]|0|核心优势算术一致|
|两轨迹异长 [2,4]，reward [1,0]|1/6|原官方对 step 行重复的 episode reward 求均值（1/3），项目按轨迹求均值（1/2）|
|单轨迹两步回到相同状态|0|两边均得到 [-0.025, +0.025]|

异长差异反映官方样本组织和 episode 统计口径；项目按轨迹求均值更贴近 Eq. 3 字面定义，不能因此判定项目违反论文。新版官方有 `compute_mean_std_cross_steps` 参数，默认仍按 step 行统计。上述只是算术对照，不是完整 trainer 等价性证明。

另一个待约束边界：项目 episode 按 uid 分组，但 step 函数只按含 task 的 anchor ID 分组。若同一 task 在一个 batch 对应多个独立 episode-group uid，step 可能跨组混合；官方还显式限定 episode group。不能把目前按任务不重复采样的使用约定视作通用正确性保证。本轮未改动此算法边界，也未据此归因历史成绩。

## 验证和部署证据

- 实现提交：`72fc81341debed3ac64f2aee9a646cb00c7c0743`，本地与服务器同步。
- 服务器 CPU 回归：**310 passed, 2 warnings**，pytest 108.81 秒；涵盖语义反例、v2/v3 接线、算法、过滤、信号回放、launcher、并发隔离和 SwanLab。两个 warning 为第三方弃用提示。未涉及模型前后向。
- 实际 launcher + shell dry-run 通过：anchor_version=v3、episode_normalization=grpo、预算 2 step、独立结果目录；未创建训练目录或启动服务。第一次检查因测试环境缺少 `TAU3_ROOT` 未进入 shell，补齐部署根目录后通过，无需修改生产 launcher。
- 本地证据：`results/analysis/semantic_v3_20260914/{execution.json,pytest.log,dialogue_partitions.json,launcher_dry_run.json,shell_dry_run.log}`。
- 原文、固定源码及算例：`results/analysis/gigpo_paper_semantics_20260914/`；原官方源码 SHA256 为 `172d6a08686fe23b95ea291a9d3bcda2f2f751bb725d7bc7653c9a638451518d`。

## 后续应如何推进

先继续无卡构建“应合并/必须分开”的成对标注，再评估有限动作领域的结构化决策状态：当前目标、对象、已知事实、待确认方案、批准范围、条件与撤回。对已完整解析的状态可研究更新/折叠历史证据，未解析条件继续保留；不能直接删除历史来追求覆盖率。单纯累加同义词规则收益目前很小。

随后固定一种归一化，单独比较锚点版本，避免 v2→v3 与归一化一起变化后归因不清。开卡后先 1–2 step，测真实 DB 下的非零步优势覆盖、加权幅度和 DF 屏蔽情况，再判断是否值得正式训练。

GiGPO 用已有奖励改善信用分配，并不自动制造新的成功轨迹：终局全零组的折扣回报也全零，仍没有步级相对信号；全成功轨迹可能因折扣距离不同而有步级差异，E3 的等回报过滤可能将其屏蔽，需要真实审计量化。当前证据不足以说项目已经解决奖励稀疏问题，也不足以把旧 E2/E3 未超过 E0 全部归因为某一个因素。
