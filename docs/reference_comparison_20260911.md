# 当前 Tau3-GRPO 与参考 Agentic-GRPO 的代码、数据及成效对照

日期：2026-09-11。当前项目 A：`/Users/apple/Projects/program-llm/tau3_grpo_fix/code`；参考项目 B：`/Users/apple/Projects/program-llm/agentic-grpo-longhorizon`，其业务代码位于第二层同名目录。本文核对本地源码、配置、已有实验文件；没有重跑模型、修改业务代码或更改历史成绩。

## 结论

A 已经发展为另一套研究方案：AReaL 合成 τ²-style 数据 → τ³ runtime → Qwen3.5 → 动态过滤/Tau-GiGPO → 独立 selection 与官方最终评估。B 是旧 τ-bench → 72B 教师在同环境采集成功示范 → Qwen2.5-7B → PRM-Lite/LATA → 自定义 40/10 切分评估。

两者复用了相近的技术框架和 LoRA SFT→合并→新 LoRA RL 思路，但数据形成方式、奖励、优势估计与效果验证均不同。A 不能通过“继续跑现有 E0–E3”自动复现 B 的 PRM-Lite+LATA 结论。用户想取得“成功率提高且能用消融解释”的成效，应优先迁移 B 的数据采集和周期性效果分析流程，并在 A 的固定环境内验证；不能把 B 的百分比当作 A 的保证。

## 阅读期间的工作区变化

A 的 HEAD 为 `0fbcef8`，B 为 `2004fcb`；A 有未提交改动。读取过程中 A 新增了可选 Thinking SFT 支持及 `configs/train/sft/qwen35_4b_lora_thinking.yaml`，包括 reasoning 字段映射、监督与历史保留开关。本文没有编辑这些文件，也未验证其 GPU 训练。历史 SFT-003、RL-009～012 仍是 non-thinking 实验；读取时 Qwen3.5 RL 入口仍明确关闭 thinking。必须区分新代码能力与历史训练成绩。

## 1. 完整差异表

| 维度 | 参考项目 B | 当前项目 A | 影响 |
|---|---|---|---|
| 研究问题 | 诊断 Vanilla 长程训练退化，验证过程奖励和 token 权重 | 验证动态过滤与基于状态分组的步骤优势 | 不同机制，不能把 E1/E2/E3 视为 PRM/LATA 的同义替换 |
| 环境 | 旧 `tau_bench`，Airline 50 题 | `tau2` Python 包的 τ³ v1.0.1 runtime；合成任务与官方任务分开 | 同名工具不等于同一任务、政策、数据库和判分 |
| RL 数据 | 自定义 40 seen，剩余 10 unseen；从同一官方 50 题拆分 | 1,148 个 AReaL Airline 任务：200 train / 60 selection / 888 reserve；官方 50 题留作 final | A 的评测范围更明确，但已变成跨来源/跨版本泛化问题 |
| SFT 来源 | 教师在目标环境实际交互，保留成功且未触发污染标记的轨迹 | 重建公开 AReaL 逐回合记录，按完整对话、长度与文本近重复规则选 45/5 | A 缺少 B 那样的本项目在线教师采样→成功过滤入口 |
| SFT 文件 | 本地 train.jsonl 45 段，覆盖 14 个 task ID；报告称 16 个，有版本差异 | 45 段 train+5 段 validation；源候选可重建 999 段 | 两边“45”来源不同，不是可迁移的最佳数据量 |
| 模型 | Qwen2.5-7B-Instruct；没有原生 Thinking 切换训练 | 已跑 Qwen3.5-4B，支持其他配置；历史跑次 non-thinking | 模型规模、结构与初始化同时变化，不能直接比较分数 |
| 模拟器 | Qwen2.5-72B-Instruct-AWQ | 4B 这轮用 Qwen3.8-27B-AWQ-INT4 | 用户行为分布与延迟可能变化；不能只凭参数量判好坏 |
| SFT mask/schema | 全轨迹 assistant-only；按两次模板渲染长度定位；训练传 `tools=None` | 全轨迹 assistant-only；显式载入工具 schema；Qwen3.5 用 generation mask | A 已改善 schema 和模板校验，没必要退回缺 schema 的训练 |
| SFT 超参 | r16/alpha32/dropout0.05，LR1e-4，5 epochs，micro2×accum4，长度10,240；配置无 eval 文件 | 相同 LoRA/学习率/epoch；micro1×accum8；4B长度24,576；5 段 loss 验证选 step24 | 小批差异主要是显存安排；数据和在线效果更值得先检查 |
| RL 超参 | 当前 YAML：每批4任务×8条，LR5e-6；报告 Vanilla200步、改进300步 | 历史4B：每批2任务×4条，LR1e-6，每个 arm3步 | 当前工程试跑规模远小于参考的训练报告 |
| 奖励 | Binary 或 outcome+0.3×手写规则过程分 | 原官方 `EvaluationType.ALL` 奖励，动态过滤不新增奖励 | B 可让同为失败的轨迹产生规则分差异；A 的过滤不能创造差异 |
| 优势 | 标准 GRPO、token 位置指数权重、再除以 sqrt(L) | GRPO 或 episode项+anchor分组step项 | 一个重塑总分/权重，一个比较相似状态的后续回报 |
| 评测 | 自写 wrapper；评所有50题，分 covered/uncovered/unseen；指标字段命名有错误 | 官方 Orchestrator+ALL；正确区分 pass@k/pass^k；task/trial清单与模型身份校验 | 不应原样拷贝 B 的指标和全体任务总分口径 |
| 已有证据 | 本地有 Base/SFT 原始评测；RL结果主要在报告中 | 本地有 SFT及4个RL短跑的原始证据/同步审计；独立成功率尚未测 | B 更接近完整实验叙事；A 当前还在工程验证阶段 |

参考联合配置为1张策略卡+独立模拟器；参考当前 Vanilla YAML 却配置2张策略卡、TP2，而文档概括为2张卡总预算。A 已测2×A800是GPU0策略/GPU1模拟器。不可把 B 全部实验的硬件口径当作已一致验证，也不能直接抄吞吐和显存数值。

## 2. 最重要的数据差异：两边都45段，但不是同一实验

### B 的数据闭环

`scripts/train/sft/collect_sft_data.py` 与配置给出：72B-AWQ 教师+独立72B-AWQ用户模拟器，每个任务尝试16次，四档温度0/0.5/0.8/1.0各4次，最多30个助手回合。只有 `traj.success` 且未被标记 contaminated 的轨迹进入候选；seen任务对应记录合并为 train.jsonl，holdout另存。

本地 train.jsonl 实数：45段、501条assistant消息、14个task ID，无独立 reasoning/thinking 字段。部分任务有多条不同轨迹，其中task38有9段。45不是代码要求采满的固定数量，而是这份成功过滤与切分后保留下来的文件数量。脚本还会按token长度继续过滤，不能把源文件行数无条件当成最终有效训练样本数。

采集 summary.json 写67段成功候选，45 train+22 holdout；实际文件分别覆盖14和3个任务。summary 的成功任务数19、报告的 covered_seen16、当前文件覆盖14不一致；split.json标记 manual_adjustment，而当前脚本默认 stride 切分，进一步说明产物与脚本存在版本差异。

SFT训练任务与RL训练任务重叠本身允许。问题是把这部分训练任务混入总体评测后，将总体收益解释成未见任务泛化。B报告的“泛化”还包括24个RL训练过的uncovered_seen；这是未被SFT覆盖的任务，不是RL从未见过的测试任务。

### A 的数据闭环

`tau3_grpo/data/sft.py` 从公开文件选每段最后一条turn记录，组合历史messages+answer恢复完整轨迹；之后进行文本近重复、长度、与blocked集合的相似度筛查。45/5被设为常量，训练入口还有 `expected_size=45/5`，所以只把JSONL扩大到几百条并不能直接训练，还需更新可配置的数量断言、manifest和预算记录。

来源中的999段不是已经由本项目独立验证成功的999段。当前45段中4段标签冲突，44段含多调用；现存构建器没有执行教师采样、实际重放或按correct/reward做质量验证。文本Jaccard/SequenceMatcher也不能证明语义隔离充分。

最值得迁移的是 B 的“在自己的目标环境采样并筛选”的方法。A 可以先审核已有来源，再从自己的200个训练任务及独立DB出发生成缺少的成功/合理拒绝示范；不能采集官方final任务后挑成功样本训练。

## 3. 提示词和工具执行：参考项目也不是无缺陷模板

B 的采集/独立评测wrapper以及GRPO parquet显式system主要只有日期锚定，未在这些入口加入完整Airline policy。在线policy API传入tools，但SFT训练传 `tools=None`。A 的RL parquet明确嵌入官方policy，SFT训练显式渲染tools；这是应保留的实现。

但 A 的SFT policy来自源轨迹，RL policy来自目标环境，不应因为都叫Airline就默认一致。需要比较取消条件、确认要求、工具行为等；不能仅替换system而保留在旧规则下产生的错误示范。

本次实际计数：

| 文件 | 含单轮多调用的对话 | 多调用assistant消息 |
|---|---:|---:|
| B train.jsonl | 9/45 | 22 |
| A 实际45段快照 | 44/45 | 76 |

B 的采集/eval wrapper逐个执行返回的所有调用，但其 veRL RL loop 同样使用 `tool_calls[:max_parallel_calls]`；因此它也有需要对齐的采集/训练执行协议。A这轮max_parallel_calls=1，与大多数源示范更明显冲突。参考成功过滤只说明其采集wrapper返回成功，不代表在另一个执行器和政策下仍是干净示范。

## 4. PRM-Lite、LATA和Tau-GiGPO到底有什么区别

### B：改变训练分数和token权重

`src/envs/tau_bench_interaction.py` 的 `_compute_reasoning_quality_score` 对工具历史算规则分：占位参数、重复查询、错误后重复调用、转人工、读写数据衔接、think调用和过长轨迹等。先平均每步分数，再加轨迹级项，截在[-0.5,0.5]；总奖励为 `outcome + 0.3*process`。

最终这个过程分被汇总成轨迹标量，agent loop在末尾放入rm_scores，再由GRPO求组内优势。因此当前代码支持“规则塑形的轨迹奖励”，不支持将报告中“每一步知道自己哪里错了”直接理解为独立的逐步监督PRM。

`verl/verl/trainer/ppo/core_algos.py` 的 LATA：先求标准GRPO标量A，再给较早token更高的指数位置权重w，最后计算 `A*w*mask/sqrt(L)`。它不按真实assistant轮次分组，也不读取每步规则分。actor默认仍是token-mean loss；代码不是简单地把原有1/L归一化替换成1/sqrt(L)，而是额外缩放优势。复现其效果时应以完整有效loss公式为准，不能只复述注释。

此外，指数权重用完整response位置索引，夹在中间且mask为0的工具/用户token会影响位置距离；它不是纯按助手有效token序号打折。该机制需要单独诊断，不能直接宣传为可靠的turn credit assignment。

B 中的think工具、普通assistant长回复统计，均不等于Qwen3.5原生Thinking模式。

### A：筛选有差异的组，在相近状态比较后续结果

动态过滤固定采样后把全0/全1组的训练mask清零，默认不补采样，不能产生新奖励，也不能从一批全失败的任务里创造学习信号。

Tau-GiGPO根据task、DB hash、已查询信息、确认标记、政策前置条件、最后观察类型建立anchor。在同anchor组内比较折扣后的最终回报，把step项加到对应assistant token区间：`A_episode + omega*A_step`；无足够分组时step项为0。

这个step项来自最终回报和状态匹配，不是PRM-Lite的手写质量分。当前structured特征也较粗：知道调用过某种读工具不代表知道查询了哪个订单，出现yes不代表确认了哪次修改；应验证分组是否把不同决策状态误合并。E2/E3 episode项与E0的标准差归一化不同，现有比较也不能只归因于新增step项。

A的自有实验臂中没有B的PRM-Lite/LATA等价路径。若用户想直接验证参考方法的迁移，应在同一A环境/SFT起点/预算下新增独立的PRM-only、LATA-only、联合对照，而非重命名已有E1/E2/E3。是否值得新增要由基础失败诊断决定。

## 5. 参考项目“+37%”的真实范围

报告宣称联合方案step250为0.240，Vanilla step200为0.175：绝对增加6.5个百分点，相对增加37.14%。这不是提升37个百分点，也不是SFT对Base的提升。

同一报告写Vanilla step150曾达到0.225。若只是算联合峰值对这个Vanilla峰值，差异为1.5个百分点、相对6.67%；这不是新的严格实验结论，而是说明选择哪个基线检查点会明显改变标题数字。原比较的训练步数也不同，未由此排除训练预算差异。

本地参考experiments只有采集、Base评测、SFT评测三类目录，没有报告中联合/RL各检查点的原始评测轨迹和训练日志，故本次不能独立复算0.240/0.175，仍应称为报告结果。

本次可复算的本地数据为：

| 本地快照 | 成功轨迹/总轨迹 | pass@1 | pass@4 | pass^4 |
|---|---:|---:|---:|---:|
| Base | 32/200 | 0.160 | 0.340 | 0.020 |
| SFT | 29/200 | 0.145 | 0.280 | 0.020 |

每个任务4次，按50任务平均。它们不支持“参考SFT必然改善总体成功率”。这只是文件中的观察，不是证明SFT一定降低能力：产物版本、checkpoint身份、采样变化和置信区间都需核对；当前SFT配置max_tokens4096，保存的split报告配置是1024，也不能把最新YAML当作历史实参。

### 指标名称错误

B 的 `pass_k_eval.py` 中 `pass_at_1` 实际统计四次至少一次成功，即本设置下pass@4；`pass_hat_4` 也用“至少一次”的公式，不是四次全成功。其 `pass_hat_1` 则确实是平均成功率，所以k>1命名错误不会自动推翻文档中的k=1数值。

例如SFT文件的 `pass_hat_4=0.28`，从逐任务结果重算四次全成功只有0.02。A已实现正确的两种公式与无效评测检查，应保留。B 的error_rate在代码里是轨迹异常比例，报告有时描述成工具错误率；per_turn content长度亦不能单独代表推理质量。

## 6. 当前项目距离用户想要的成效，主要差哪几步

1. **数据生成与质量控制。** 有公开来源与manifest，但缺少本项目目标环境中的教师采样→过滤→覆盖分析→针对失败补数据的完整闭环。
2. **可解释的基线。** 需要Base与SFT在同一固定selection上的成功率和失败类型，才能判断SFT在帮助还是损害策略；loss下降不够。
3. **有效RL预算。** 当前每arm24条/3更新是功能验证。B的配置每步32条、报告200–300步，是数千条级别。不要机械地抄300步，先按实际rollout数、生成token、有效组与成本规划可比预算。
4. **对应已观察失败的假设。** 当前最先确认的是协议不一致和漏执行，不是已证明长程GRPO发生与参考相同的训练崩溃。先修前者，再决定过程奖励、动态过滤或step credit哪种机制值得验证。
5. **周期性独立评测与消融。** 固定环境/模型起点/模拟器/预算，比较Base、清洁SFT、普通GRPO；稳定后每次只增加一项机制。新旧benchmark分数不能横向当方法增益。

建议继续在A中保留独立DB、官方verifier、正确指标、checkpoint身份和LoRA同步审计。先迁移B的数据与实验组织方法，再决定是否迁移算法。维持当前非Thinking路线即可检验这些问题；新Thinking分支属于另一个需要固定协议的实验变量。

近期顺序：统一协议 → 对自己的训练任务采集/审核示范 → 同条件Base/SFT selection → 普通GRPO有预算的小试 → 根据失败分布选择一个改进 → 相同预算消融 → 冻结后官方final。若要严格复现B的绝对数值，则需要另行固定旧tau-bench、7B/72B、原切分/数据/检查点和完整历史配置，不能与A的τ³迁移实验混为一谈。

## 7. 代码定位

下列B路径相对于第二层 `agentic-grpo-longhorizon/`；B的veRL路径在其外层。

| 功能 | B | A（相对code） |
|---|---|---|
| 采集/构建SFT | scripts/train/sft/collect_sft_data.py | tau3_grpo/data/sft.py、prepare_sft.py；缺同等在线教师采集入口 |
| SFT训练/mask | scripts/train/sft/sft_train.py、src/training/sft_dataset.py | tau3_grpo/training/sft/train.py、dataset.py、models/qwen35_template.py |
| RL任务 | scripts/train/grpo/build_grpo_parquet.py | tau3_grpo/data/dataset.py、schema.py、parquet_builder.py |
| 环境与奖励 | src/envs/tau_bench_wrapper.py、tau_bench_interaction.py | tau3_grpo/envs/session.py、interaction.py、evaluation/verifier.py |
| 算法 | 外层verl/verl/trainer/ppo/core_algos.py | tau3_grpo/algorithms/tau_gigpo.py、dynamic_filtering.py、verl_estimator.py |
| 独立评测 | src/evaluation/pass_k_eval.py | tau3_grpo/evaluation/run.py、runtime.py、metrics.py、scoring.py |
| 实验记录 | docs/ablation/ablation_diagnosis_report.md | EXPERIMENTS.md、docs/4b_results_analysis_20260911.md |

本次数据统计与源文件SHA256：`results/analysis/reference_comparison_20260911/artifact_audit.json`。这是离线计数与源码核查，不是新训练/推理测评。
