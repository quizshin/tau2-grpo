# Airline 语义协议 v2

本次新增 `slot_schema: airline_slots_v2`，保留无开关与 v1 的行为。没有修改 RL
配置或 5090 配置，没有启动 GPU。新模式的状态 key 与旧模式隔离。

## 无费用条件

`{"fee":{"kind":"none","scope":"all"}}` 表示该目标无费用，不填写币种。
`scope` 还可以是 change/cancellation/service；“不收改签费”不能泛化成所有费用免除。
金额上限使用独立 amount_bound 类型，保留金额、币种、比较符与费用范围。
USD 0 不自动归为任意币种均免费。模型给出的金额/币种必须有当前用户条款中的证据。
现有 goal_ids、requirement/preference、条件撤销与批准失效逻辑继续生效。

旧输出中 `eq 0 + currency:null` 只有在当前条款明确说 no fee 时才可迁移为 none；
没有依据的 USD 不被自动删除来修复真实输出。旧人工 fixture 中发现 3 处这种无依据
USD，已另存显式修正清单，旧 fixture 文件保持不变：
`tests/fixtures/semantic_slots_v2_reference_corrections_20260914.json`。

## 乘客范围

目标和方案条件使用 passenger_scope：all、ordinal、ids 三种不同类型。
`all` 与 `everyone` 有原文依据时可归并；单个/多个具体 ID 不自动等同于 all。
ordinal 附加由本地代码计算的精确对话前缀 guard，避免不同上下文的“第二位”被合并。
该 guard 不代表已验证乘客列表，也不将序号映射为具体 ID。需要此类身份解析时，
应另做工具列表证据验证；当前采用保守隔离。

具体 ID 必须被原文明确标为乘客标识，或来自已验证工具的 passengers/index/id
（或 passenger_id）叶子，不能把同一句话中的预订号充当乘客 ID。
删除乘客使用 remove_passenger 操作；已观察到的两种旧字段拆法可迁移到此结构。
替换目标仍保留历史目标与承诺，并使受影响旧方案的批准失效。

## 确认问题

新输出使用 intent 与 proposal_id：confirm_proposal、identity_user_id、search_permission。
确认意图保留所指方案版本，自由描述不决定状态相等性。必须有明确确认问句；
身份/搜索问题不能绑定成写操作批准。搜索许可保留精确文字 guard，未知问题采用
context/unknown，不用统一 other 标签丢弃不同问题的内容。

## 验证与接口

针对性真实验证配置：`configs/analysis/semantic_model_kimi_k3_v2_20260914.yaml`。
只请求费用偏好、目标替换、确认问题，以及两条此前没有模型输出的样例；单并发，
每次总时限 180 秒，无自动重试。HTTP 分阶段超时之外增加 asyncio 总时限，
避免持续收到部分数据时无限延长等待。记录每次请求耗时，错误不输出密钥或请求头。

```bash
.venv-cpu/bin/python -m pytest -q \
  tests/test_semantic_slots_v2.py tests/test_semantic_slots.py \
  tests/test_semantic_api.py tests/test_semantic_state.py

.venv-cpu/bin/python -m tau3_grpo.analysis.audit_semantic_api \
  --config configs/analysis/semantic_model_kimi_k3_v2_20260914.yaml \
  --output results/analysis/semantic_kimi_k3_v2_fresh_run
```

旧输出离线重放中，preference、replace_goal、changed_pending 三项均与修正后的
参考状态完全一致；这是归并修复验证，不是新模型请求，也不应算作独立语义准确率。

费用范围、比较含义、意图识别仍由模型提供，局部词法证据检查不是完整语义证明。
不支持的表达仍会弃权。后续应冻结协议，在独立真实 rollout 上盲标验证覆盖率、
错误合并和遗漏，再讨论在线信用分配或重新训练。

## 本轮结果

109 项相关 CPU 测试通过。5 个真实请求全部返回可解析的事件包：

| 样例 | 最终复验 |
|---|---|
| preference | 无费用偏好，不添加币种；与修正参考状态一致 |
| changed_pending | 新报价未批准；确认问题类型与参考一致 |
| changed_approve_synonym | 同义表达批准新方案；与参考一致 |
| replace_goal | 乘客范围、删除操作及历史目标均与参考一致 |
| stale_approval | consent_to_inactive_proposal，被本地规则拒绝 |

最后一项表明校验器阻止了失效方案批准，不代表 Kimi 自己正确识别了该错误。
本批配置是针对 5 个已知问题逐项检查，原 pair 列表在该子集内没有完整配对，
所以 API summary 的 pairs 为 0；上表来自另行逐项对照参考状态及预期拒绝原因。
不能把它包装成独立评测集的 100% 准确率。

新请求使用 v2 prompt，旧输出重放另存，未覆盖历史 v1 结果。对最终版本代码再次
离线复验，记录原始输出、参考修正清单及校验代码哈希。单请求耗时
67.6–147.9 秒，5 次合计约 514 秒；总时限 180 秒，本次没有超时。

结果记录：[semantic_slots_v2_verification_20260914.json](semantic_slots_v2_verification_20260914.json)。
完整本地输出：`results/analysis/semantic_kimi_k3_v2_final_20260914/`。
