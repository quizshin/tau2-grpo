# v4 独立任务样本测试

结论：冻结的 v4 在新任务上仍不稳定，尚无充分证据用于 RL 语义分组。
12 个位置中 5 个通过协议/证据校验，第 3 步为 4/6，第 6 步为 1/6。
通过校验不代表完整语义正确；4 对实质语义比较全部弃权。
本轮没有补旧 24 个位置，没有按新样本修改候选或提示词，也没有重试失败请求。

## 独立性和运行范围

用户要求先测试独立样本。候选是上一轮最终 `airline_slots_v4`，在选样前冻结；
远程主仓库 `/root/autodl-fs/tau3-core/code`，复用持久环境，未进行本机运行测试或 GPU/RL 训练。

来源为已有 SFT selection 的真实模型对话：
`results/legacy/experiments/selection-dual-gpu/results-off/trajectories.jsonl`。
其 SHA256 为 `cac0f5bee1565cc7d2b27d30da39b882524c7822bfccb8f9d17f82d2a4b060a3`。
排除旧锁定记录、语义/决策 fixture 和决策配置中已知任务，共 23 个排除标识。
剩余 38 个满足 trial0/1 存在的任务，按
`sha256("semantic-independent-v4-20260916:" + task_id)` 排序，取前三个：
`airline_921`、`airline_490`、`airline_790`。

选样先于对话和结果审阅，不根据奖励或难易选择。固定 trial0/1、第 3/6 个决策位置，
缺失或重放失败不替换样本。实际 6/6 条轨迹完整重放通过，58 个工具响应一致，12 个位置齐全。
API 输入为动作发生前的可见历史，工具数据和政策身份经重放核对；不发送任务隐藏答案、
未来消息、终局奖励或审查标签。状态审查不使用奖励。

这里的“独立”指与已知语义/决策修复任务隔离，仍来自同一 SFT selection 数据集，
不是独立人工评测，也不是新领域或 SFT 模型选型之外的新测试集。不能据 3 个任务推断总体准确率。
配对标签和逐位置审查契约均由 Codex 在抽取前冻结。

## 实际结果

配置：`configs/analysis/semantic_independent_v4_20260916.yaml`。
GLM-5.3-Flash、增量 v1、并发 4、每请求 180 秒、8192 输出 token、调用上限 96，
诊断后继关闭、无自动重试。实际 23 次请求：22 次正常结束、1 次长度截断，退出码 0。
供应商累计用量 225,143 token；请求耗时总和 856.454 秒，不是并发墙钟耗时。
结束时核验 40 个源码/报告/配置文件及冻结输入、标签哈希，均未变化。

| 样本 | 第 3 步 | 第 6 步 |
| --- | --- | --- |
| airline_490 trial0 | 通过 | 输出长度截断 |
| airline_490 trial1 | 通过 | 订单指代询问类型不匹配 |
| airline_790 trial0 | 身份回答语法拒绝 | 上游身份回答失败 |
| airline_790 trial1 | 身份回答语法拒绝 | 上游身份回答失败 |
| airline_921 trial0 | 通过 | 通过 |
| airline_921 trial1 | 通过 | 问题引用失效 |

12 个原始前缀都不同，没有用相同初始前缀复制增加成功数。
5 对预设关系的评分程序给出 1 对通过、4 对弃权，但必须进一步区分：

- 通过的 1 对是同样对话与事实、只有工具调用 ID 不同的状态合并，属于传输标识不变性检查。
- 1 对身份询问改写后的同义状态、3 对目标/已告知价格/条件不同的状态，全部弃权。
  **实质语义比较可判定 0/4**，没有可报告的语义准确率；零错误合并不能解释为通过安全性验证。
- airline_921 第 3 步的一对在抽取前就未计分：一边明确要求查找忘记的订单号，是否等价依赖
  对程序性查找请求的约定；没有看到输出后再挑标签。

## 逐状态语义审查

按抽取前的审查契约，检查目标、条件、报价、批准、用户声明与已观察事实的区别。
5 个有效状态都含有原文保留项，审查结果如下；这是 Codex 人工式审阅，不是独立人类标签。

| 级别 | 数量 | 解释 |
| --- | --- | --- |
| 简单前缀的结构化含义基本忠实 | 1 | airline_921 trial1 step3：日期、路线、晚间改签与用户 ID 保留，未编造批准；仍有冗余原文，不能证明同义归并 |
| 主要信息保留但结构化不完整 | 4 | 两个 airline_490 step3，以及 airline_921 trial0 的 step3/6 |
| 发现已通过状态中的明确语义矛盾 | 0 | 审阅范围有限，不等于完整语义均正确 |
| 无可用状态 | 7 | 无法对该位置进行有效分组 |

具体而言，airline_490 保留了“改签”和“补偿”两个目标，但“最近一张订单”和取消/延误事实声明
主要仍依赖完整原文。airline_921 trial0 的试探性选择、未知订单号和查询请求也主要保留为回答原文；
身份未进入独立身份字段，查找目标在取得资料后仍保留。原文保护有助于避免混淆，但不等于这些
关系已被结构化理解或能够跨不同措辞合并。

## 新样本揭示的卡点

1. 普通身份回答 `Yes, my user ID is ...` 被 `unsupported_identity_answer` 拒绝，两个 trial 的
   第 3/6 步共 4 个位置受影响。不是模型没输出 ID，而是自然表达与严格身份语法不协调。
2. `Could you confirm which reservation you're referring to?` 被抽取为订单身份询问，但焦点没有
   字面的 “reservation ID”，触发 `unsupported_question_intent`。不能将这种确认理解为写操作批准。
3. 同轮多个问题用相同 intent 表示时，待回答记录相互替换；后续引用先前问题触发
   `answer_to_inactive_question`。这是新样本确认的表示/生命周期问题。
4. 一次后段复杂请求因 `finish_reason=length` 截断。原始响应保留，没有修复 JSON 或重试。

这些发现说明，继续补旧 24 个位置不足以验证泛化。下一步应先梳理“字段回答、多个同时问题、
条件与目标状态”的表示契约，再在开发集验证实现；新一轮泛化评估需要另选未参与修复的任务。
本轮任务一旦用于修复，就不能继续作为该修复的独立验证集。

## 证据和复现

远程根目录：`results/runs/semantic_independent_v4/20260916_01/`。

- `lock.json`：排除集、完整候选排名、选样规则、候选哈希。
- `inputs/`：动作前输入、重放记录、清单及与模型输入分开的 outcome metadata。
- `review_labels.json`、`state_review_contract.json`：调用前冻结的关系标签和语义审查契约。
- `source_manifest.json`、`sources.tar.gz`、`source.patch`：实际候选版本及未提交源码证据。
- `api/`、`api.log`、`api.exit`：原始响应、delta、状态、用量与退出状态。
- `score.json`、`metrics.json`、`primary_failures.json`：关系评分、调用统计及首次失败原文。
- `state_review.json`、`interpretation.json`、`completion.json`：逐状态审查、结论与交接清单。

不覆盖原目录；另行复现将产生 API 请求费用，并且不再构成首次盲测。

```bash
source /root/autodl-fs/tau3-core/activate.sh
cd "$CODE_ROOT"
python -m tau3_grpo.analysis.audit_semantic_api \
  --config configs/analysis/semantic_independent_v4_20260916.yaml \
  --output results/runs/semantic_independent_v4/NEW_RUN/api
```
