# 锚点 v2 接线、部署与验证（2026-09-14）

后续见 [语义规范化 v3 与原论文对照](gigpo_paper_semantic_audit_20260914.md)。v3 已作为独立候选接入；固定 DB 的真实对话诊断只从 16 增至 20 个非初始跨轨迹重复位置，尚未解决过度拆组，缺省版本未切换。

锚点修复已经接入真实 `current_anchor → ToolAgentLoop → anchor_ids/spans → Tau-GiGPO` 路径并同步服务器。默认新会话及新验证配置使用 v2；历史 c50 E0–E3 配置显式固定 v1。未启动 GPU、训练、模拟器或新的 SwanLab run。

## 修复的行为

v1 用累计关键词生成全局 `user_confirmed`；v2 不再生成该标记。新的状态由 task、DB、已有工具类型/前置条件特征，以及以下决策证据共同编码：

- 对话证据：规范化可见 assistant/user 文本的有序摘要，保留早先提出的条件，避免后来的一个 yes 覆盖此前约束。
- 当前方案与回应：上一条 assistant 文本及当时的读取/工具事件构成 proposal hash；用户回答有单独 reply hash。代理提出新方案后 reply 清空；新回应、暂停、条件或撤回会改变证据，不会复用旧确认的锚点。
- 读工具账本：成功读取按工具名＋完整参数建键，记录结果摘要。同一查询连续相同结果只记一次，不同查询的排列顺序、call ID、JSON 键顺序不影响摘要；同查询结果改变保留版本顺序。
- 工具事件：写操作、失败、无法配对的结果以及尚未完成的调用单独保留；错误信息、修改对象或输出变化会改变状态。

当前版本是**保守的上下文证据方案，不是完整自然语言动作解析器**。它不推断“此人已经授权某工具执行”。`reply_kind` 只是词面诊断标签，不进入确认权限判断，也不能作为语义准确率。锚点影响信用分配分组，不替代环境执行器。

新的 structured ID 带 `structured:v2:` 前缀。similarity 模式也要求 v2 的 task、DB、协议和决策证据完全相同，不能用较低 Jaccard 阈值把不同确认范围重新合并。DB-only 按消融定义继续只看 task＋DB。

所有证据来自该次动作之前已记录的 session 消息，不读取将来动作、最终奖励或最终 DB。提取函数没有可变全局会话状态；版本固定在 SessionEntry 中。

## 实际反例

测试固定相同 task/DB，比较真实回应与替换为无条件“yes”的回应：v1 得到相同锚点，v2 得到不同锚点。

|来源|上下文含义|v2 处理|
|---|---|---|
|E2 airline_1035 / trial 0 / message 9|只同意先查经济舱价格|查询方案与该回应绑定，不产生全局修改确认|
|E2 airline_1060 / trial 1 / message 13|要求先纠正退款银行卡|新回应更新证据，不复用此前确认状态|
|E2 airline_1106 / trial 2 / message 12|核验姓名、行李后再批准|保留条件和原文证据，不变成已经批准当前方案|

另有订单、操作、金额、币种、银行卡变化、撤回、早先用户条件、读取内容、工具错误、JSON/call-ID 稳定性、真实 hook 的 pre-action 时序、v1 兼容及会话隔离测试。

## 分组代价：目前不建议直接正式重训

在已有 E2/E3 的 479 条 selection 对话上调用真实 hook，合计 5,138 个 assistant 位置，其中 4,659 个为非初始位置。因为没有逐步 DB 快照，两版都使用同一个固定 DB 代理，只在同 arm/task 的四次试验间比较。

|固定 DB 诊断指标|v1|v2|
|---|---:|---:|
|唯一锚点|1,785|4,770|
|可跨轨迹重复匹配的位置|4,185|495|
|其中非初始位置|3,706|16|
|非初始重复匹配比例|79.54%|0.34%|
|平均 hook 时间（不包含真实 DB 哈希成本）|0.148 ms|0.960 ms|

**这不是历史线上训练覆盖率，更不是非零优势覆盖率或成功率。** 相同含义的不同措辞也会被 v2 的精确对话摘要拆分；当前修复减少错误合并，但同时明显减少可比较状态。不能将 v1 的较高重复率理解为有效信用，也不能忽略 v2 的低匹配率。

因此 v2 已部署、适合验证与采集证据，但不应宣称已经得到可提升成绩的状态抽象。后续需要基于人工复核的动作/对象/金额/支付方式/条件槽位进行语义规范化，将有依据的等价状态合并；在没有依据时仍保留保守隔离。不能为了提高覆盖率直接移除条件或读结果约束。

## 启用、兼容与复现

- 新会话默认 `anchor_version=v2`；`run_base.sh` 显式向 Ray worker 传入 `TAU3_GRPO_ANCHOR_VERSION`，并记录到算法配置。
- 新验证配置：`configs/train/rl/qwen35_4b_full_a800_gigpo_audit_20260914.yaml`，启用 v2、GRPO episode 归一化与信号审计。从 SFT 开始的 2-step 检查，未执行。
- 历史配置：`qwen35_4b_full_a800_c50_matched6h_common_20260912.yaml` 固定 v1，旧检查点继续按原协议恢复。
- 每个实际运行目录写入 `anchor_protocol.json`。历史目录缺少该文件时按 v1 对待；换版本须使用新目录，禁止在旧实验目录静默更换算法协议。
- CPU 通过实际 launcher 和 shell 的 dry-run，确认 Ray env 为 v2、算法元信息为 v2、归一化为 grpo、训练预算 2 step；未创建训练任务。
- 结果目录：`results/analysis/anchor_v2_20260914/`。`dialogue_partitions.json` 保留输入文件 hash、每个决策位置的新旧 ID 与诊断结果；`launch_dry_run.json` 保存解析后的命令。
- 实现：`tau3_grpo/algorithms/anchors/evidence.py`、`encoder.py`、`protocol.py`、`tau3_grpo/integrations/anchor_hook.py`、`tau3_grpo/envs/interaction.py`。
- 新测试：`tests/test_anchor_evidence_v2.py`；真实反例：`tests/fixtures/anchor_v2_real_cases.json`；真实对话审计：`env_info/a800_20260912/validate_anchor_v2.py`。

服务器完整回归 **274 passed**（208.59 秒，两个第三方弃用 warning）。服务器测试记录见同目录 `execution.json` 与 `pytest.log`。测试只用 CPU，包括 stub 并发测试；没有运行实际模型前后向。部署后的第一轮真实训练仍应是短程验证，并观察非初始跨轨迹非零信号，不能直接默认 v2 比 GRPO 更好。
