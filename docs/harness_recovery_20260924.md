# Harness 修复及 step 10–27 完整轨迹审计

日期：2026-09-24。分支：`codex/harness-recovery-20260924`，起点 `068e4bb`。

后续状态：用户随后授权单卡测试，已完成 32 次真实生成及最终 processed-logprob 复验；额外修复 vLLM 0.20.0 单请求取消接口。具体通过范围、并发文本差异和测试夹具修正见 [单卡 GPU 报告](harness_guard_gpu_20260924.md)。下文“未启动 GPU”等描述保留为首次 CPU 交付时的状态，不能覆盖后续报告；完整分布式训练和模型效果仍未验收。

用户授权的一至三项已实现：采样与终止诊断、历史全量审计及规则校准、流式重复中止与更新前批次保护。当前是 CPU 验证后的候选实现，尚未部署到远端活动源码，未启动 GPU 推理、smoke 或训练，也没有恢复成功率的实验结论。

## 1. 审计范围和输入身份

读取两个既有运行的 step 10–27 全部原始 rollout，共 36 个 JSONL 文件；没有新采样，也没有用解码文本重新 tokenize 冒充训练输入。

|运行|轨迹|assistant 轮次|生成 token|原始 length 轮次|
|---|---:|---:|---:|---:|
|MT-GTPO `mt_gtpo_5090_a45/v4-explore-u30-01`|1,152|10,649|2,497,843|145|
|GRPO 对照 `grpo_5090_a45/20260921-u30-v4`|1,152|11,088|2,103,946|6|
|合计|2,304|21,737|4,601,789|151|

每份文件核验 64 条轨迹、8 个任务组且每组 8 条，以及轨迹身份不重复、原始 assistant span 与 generation mask 精确覆盖。全部轮次自动检查并回放重复规则；重点异常文本另行复核。这里的“完整审计”不等于人工逐句评判所有回复的业务正确性。

用于解码的 tokenizer.json、tokenizer_config.json、chat_template.jinja 与远端该运行 step 30 保存的 tokenizer 三个文件 SHA256 一致。原始 JSONL 文件、token 序列、tokenizer 的哈希保存在结果清单中。实际更新 batch 的检查直接在远端 CPU 执行，没有下载整个 tensor 集合，没有修改运行源数据。

证据目录：`results/analysis/harness_recovery_20260924/`，其中：

- `raw/rollouts10-27.tar.gz` 及其解包目录：原始 36 份 rollout。
- `audit-r32/summary.json`、`turns.jsonl`：保守规则的全量结果及逐轮精确 token 身份。
- `audit-r16/summary.json`、`turns.jsonl`：敏感度对照，不替换 r32 结果。
- `remote-tokenizer-hashes.json`：远端 tokenizer 身份。
- `batch-replay10-27.json`：18 份原始 MT-GTPO 更新 batch 的哈希、张量键及重放误差。
- `execution-evidence.json`：最终源码和审计产物哈希索引，后于审计生成，不冒充原始运行时记录。

## 2. 1024 边界与崩塌出现的位置

21,737 个 assistant 轮次中，**没有单轮生成超过 1024 token**，最大值恰为 1024。多轮对话、工具 observation、模板和策略生成的总量可以超过单轮上限；字符数量也不是 token 数量。

原逻辑把普通文本的 `finish_reason=length` 终止映射为 `context_window_exceeded`，同时服务的归一化 `stop_reason=completed` 包含原始 stop 和 length。这两个名字均不足以证明模型自然答完或确实耗尽总上下文。原始长度限制生效，但诊断没有明确指明是哪层预算。`parsed=false` 也可能只是预算检查在 parser 前终止，并不证明解析器发生错误。

下表每步均为 64 条候选；length 是轮次数，重复是轨迹数。在本数据中每条重复轨迹恰有一个命中轮次。成功指已记录的官方全成功结果。

|step|成功轨迹|length 轮次|r32 重复轨迹|length 且未命中该规则|
|---:|---:|---:|---:|---:|
|10|22|0|0|0|
|11|22|1|0|1|
|12|38|1|0|1|
|13|22|2|0|2|
|14|40|1|0|1|
|15|37|1|0|1|
|16|19|2|0|2|
|17|36|1|0|1|
|18|38|0|0|0|
|19|28|0|0|0|
|20|31|0|0|0|
|21|43|0|0|0|
|22|18|5|0|5|
|23|34|2|0|2|
|24|38|6|0|6|
|25|21|29|7|22|
|26|16|34|24|10|
|27|2|60|58|2|

r32 规则在 MT-GTPO 命中 89 条，全部官方结果为失败；GRPO 对照 1,152 条中未命中。r16 额外命中 step 25 的 airline_981：反复出现 BOOKING CONFIRMED / RESERVATION COMPLETE 标题及大量勾号，累计 90 条。候选采用更保守的 r32，明确接受漏掉这一较轻重复样例。`length_prose` 只是“length 结束且未命中规则”，不是“内容全部正常”的认证；对照零命中也不是普遍零误报保证。

重复以 ✅ 为主，还出现 ✓、❌、✈、✨、💵 和重复句子。首次可检测前缀在 128–1016 token，median=152；这些位置之后原轨迹又生成了 62,145 token。这是离线潜在可减少的尾部 token，不能当作实际取消延迟或墙钟节省；引擎合并输出和取消竞态可能使真实停止更晚。

## 3. 奖励、优势和原始训练张量

89 个重复轮次的 advantage：67 个负、19 个零、3 个正。不能说“重复全部被正奖励强化”，也不能说“结果是失败，因此 advantage 必然负”。三个正优势样例为：

|step / task|trajectory_id / turn_index（从 0 起）|advantage|
|---|---|---:|
|25 / airline_403|074d54dac05549bf9d59e50893205906 / 7|0.8267842696405266|
|26 / airline_658|6d2e3b6d0f994e92a738c0ba85a5daa9 / 11|1.2456406093502417|
|26 / airline_658|9618b06bb73f4ffd94cb915cf85bc36c / 7|0.08747403179897284|

这三轮 immediate reward、return、最终 outcome 都为 0，但相对组内归一化仍可产生正 advantage。当前逐轮信用分配也会把整轮 token 共用优势，普通回复可能获得后续动作回报的信用。这是值得单独实验的机制风险，本轮不改奖励、优势或动态过滤公式。

远端 CPU 逐一加载 18 份原始更新 batch：10,649 个 MT-GTPO 轮次的优势均为有限值；与 rollout 中重放值的最大绝对差为 `1.1916631192931959e-07`，超过 `1e-6` 的错位为 0。此证据不支持“优势写入 span 错位导致本次重复”的判断。

batch 含 rollout/old/ref logprob、advantages、returns 等，但缺少每个 optimizer minibatch 当时的 current logprob。因此本审计没有重建真实 PPO loss 或梯度贡献；token 数、优势符号及 advantage×token 统计均只是诊断量。普通回复长度增长还受各步任务组成影响，不能据此单独确立因果。无 NaN 和无 span 错位也不能排除其他学习动力学或数值问题。

## 4. 已实现的三项改动

### 一：采样参数贯通和终止诊断

`models/generation_guard.py` 集中校验 repetition_penalty。训练 agent loop 不再硬编码 1.0；训练内验证继承训练值，支持 val_kwargs 显式覆盖。默认仍为 1.0，本次并未悄悄提高惩罚系数。

独立 token-v4 评测新增 `--policy-repetition-penalty`，实际请求与 run.json 一起记录；非 token-v4 不允许使用非默认值。比较器把新值差异视为协议不一致；两份历史结果均未记录时保留缺失证据提示，不反推其当时有效值。

共享 token harness 增加 `tau3_generation_diagnostics_v2`：每次请求记录 per-turn / response / context 剩余额度与所有同时生效的限制；结束回执记录原始 finish_reason、生成数量、详细终止原因、parser 是否被调用。区分 `not_attempted`、`parsed`、`parse_error`。工具观察和用户预算有独立 detail，完整回执进入 rollout reward_extra_info。

官方 termination enum 和历史评分口径保留，使用 detail 区分具体原因。没有增加 1024 上限，没有插入伪造 EOS，也没有把截断文本替换成“任务已完成”。

### 二：原始轨迹回放与保守阈值

新增 `analysis/repetition_audit.py`：按原始 token 前缀扫描、精确细化首个命中点，记录 token/span 身份、轮次奖励和优势。r32/r16 全量历史结果作为规则选择的依据，未修改历史结果或候选轨迹分母。

v1 仅处理普通 assistant 文本：至少生成 128 token，检查最多 8-token 或 64-character 的重复周期，要求至少 32 次重复，文本规则另要求至少 128 个规范化字符。空白和 variation selector 仅在检测文本中归一化，训练 token 原样保留。包含完整或部分 `<tool`、`<function`、`<parameter` 的轮次排除，工具参数的重复识别留待专门验证。

### 三：流式取消和更新前保护

`integrations/verl/generation_guard.py` 消费 vLLM cumulative 输出，核对 token 追加性和 sampled-token logprob 对齐。`off / observe / abort` 三种模式默认 off。abort 只取消当前 request，并明确不清理其他并发请求的 prefix cache。

已覆盖空 token 的 abort 回执、取消与自然 stop/length 的竞态、外部取消、回执缺失、token 重写、logprob 不完整及多请求隔离。中止时保留实际生成 prefix 和 logprob，不补 EOS。只有经当前 guard 证实的取消才转为模型 `agent_error` / `repetition_detected`；保留零结果候选。其他引擎或传输失败抛出异常，不冒充模型失败继续更新。

`training/rl/update_guard.py` 在 driver 上、actor/critic optimizer 更新之前作一次统一判断。统计完整候选的事实，排除 padding，不按奖励删掉失败轨迹，不修改 MT-GTPO 动态过滤、不补采样。模式为 `off / warn / halt`；支持连续步数和显式阈值。

halt 保存 pending-batch.pkl 和脱敏后的 fault.json，包含尚未更新的 batch、配置和触发比例，再抛出明确异常。故障快照不是续训 checkpoint；`last_completed_outer_step=24` 不表示磁盘已有完整 step 24 checkpoint，若最近正式 checkpoint 为 20，应以其真实恢复能力为准，不能把 pending step 25 标为已完成。

启动检查要求 shared token protocol、vLLM multi-turn 和 sampled-token logprob；重复比例 guard 必须有 observe/abort 证据源。在资源池创建前校验不支持的组合。

## 5. 候选配置与借鉴范围

候选入口：`configs/train/rl/formal50_5090_a45_mt_gtpo_guard_v1.yaml`，复用旧 v4 MT-GTPO 配置并包含 `configs/protocols/harness_guard_v1.yaml`。旧 profile 默认不开启 guard。奖励和算法保持旧 v4，repetition_penalty=1.0，generation guard=abort。

update guard 采用重复轨迹比例 `>=0.0625`（64 条中至少 4 条）、连续 1 步；未启用笼统的 length 比例阈值，避免把所有正常长输出作为停训依据。按原始历史 batch 离线判断，最早在 step 25 更新前触发（7/64），step 10–24 不触发。这是本次历史数据筛选的启发式阈值，不是独立前瞻验证结果；启用后轨迹分布可能改变。

CPU 配置入口检查示例（需要现有数据、模型/tokenizer 路径及环境配置）：

```bash
CUDA_VISIBLE_DEVICES='' python -m tau3_grpo.training.rl.runner \
  --estimator mt_gtpo \
  --profile configs/train/rl/formal50_5090_a45_mt_gtpo_guard_v1.yaml \
  --reward-version paper_env_split_v4 --uncalibrated-exploration \
  --token-protocol tau3_token_budget_v1 \
  --result-dir results/runs/harness_guard_v1/config-check --dry-run
```

配置回归已穿过真实 runner、shell dry-run、Hydra compose 和 RolloutConfig dataclass，不只是读取 YAML。该命令是配置检查，不是启动训练授权。

参考 Mercor `ApexAgents-SkyRL-Recipe` 固定快照 `8e7702f03b7464a36ab800a624fd911de0968a87`，本地来源清单位于 `/Users/apple/Documents/ChatGPT/Apexagentx/research/mercor-20260923/SOURCE.json`。借鉴原始 token 贯通、结束原因分层和基础设施错误独立处理。这里的重复检测和更新前 halt 是 Tau3 的新实现，不宣称直接复用了 Mercor 同名功能；Tau3 工具会修改数据库，不能照搬无差别重试或替换失败候选。用户轮次之后的 nudge 不能打断当前尚未结束的重复生成，未将其当作此故障的修复。

## 6. 验证、限制和后续顺序

本地 CPU 回归覆盖采样继承、原始 token/logprob 保留、取消竞态与错误、GRPO/GiGPO/MT-GTPO 原生循环、批次 halt、配置入口、独立评测参数及比较兼容。最终命令和结果见 `execution-evidence.json` 及对应日志。

|检查|结果|日志|
|---|---|---|
|本地 guard、完整 token harness、评测比较/CLI、telemetry|114 passed|final-regression-tests.txt|
|最后修改后的 guard、评测拒绝无效配置、非默认 penalty 元数据、真实 Hydra profile|39 passed|final-followup-tests.txt|
|vendor 补丁契约与清单|34 passed|final-vendor-tests.txt|
|远端最终源码快照的 guard helper|24 passed；vLLM SamplingParams CPU 校验通过|remote-cpu-final.txt|
|固定上游 tarball 对照重建 vendor inventory|errors=[]|vendor-final.txt|
|lint / diff whitespace|0 新增诊断；382 项既有 lint 债务；diff 检查通过|final-static-checks.json|

这些套件有重叠，不应相加宣称唯一测试总数；没有 skipped 的 GPU 验证被计入通过。

远端使用现有环境、隐藏 CUDA，在临时目录运行源码快照；guard helper 测试 24 passed，并用已安装 vLLM 0.20.0 的 SamplingParams / RequestOutputKind 校验 cumulative 输出与非默认 penalty。没有实例化模型/引擎，临时目录已清除，活动源码未替换。

开发中的失败日志保留：初轮测试遇到 subprocess 找不到 python，已通过 PATH 指向现有 CPU venv 修正；评测兼容测试曾因历史缺失 penalty 元数据失败，已修为显式未知，并增加回归。不能把这些旧失败日志当成最终测试结果，也不能把后续通过改写回旧日志。

剩余验证边界：

1. 真实 GPU 引擎的取消回执、取消延迟、并发隔离、吞吐开销及 driver halt 后的分布式 worker 清理尚未验证。
2. 训练和训练内验证接入 vLLM 流式 guard；独立 HTTP token endpoint 本次只贯通采样参数，尚不支持流式 guard 取消，不能宣称二者 guard 协议已完全一致。
3. 未验证 checkpoint 续训后的 guard 状态延续；当前候选连续阈值为 1，不依赖前一步累计状态。若使用更长连续阈值，进程恢复会重新计数。
4. r32 是明确、有边界的规则，不覆盖所有语义重复或工具内容异常。阈值和成功率提升须另做受控实验。

后续顺序：先在明确预算和 GPU 授权下验证真实取消与更新前 halt，再对统一 harness 的 SFT/GRPO/MT-GTPO 做小规模配对对照；若研究信用分配改变，用独立算法版本及旧 buffer 重放，避免混入这次 harness 修复。依据 AGENTS.md 第 1 节，本次未自动启动这些 GPU 工作。
