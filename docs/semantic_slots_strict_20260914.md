# Airline 严格语义字段验证

## 已实现

`slot_schema: airline_slots_v1` 是离线 API audit 的显式开关。未传入开关时，
原来的 fixture、状态编译和 RL 行为保持不变。严格模式将字段版本写入状态 key 的
哈希输入，避免与旧模式的 key 混用；没有启用任何 RL hook。

- `booking_id` 统一为 `reservation_id`；已观察到的退款/收费金额、币种别名
  统一为 `quoted_refund` / `quoted_charge` / `currency`。
- 别名与规范字段同时存在且含义冲突时拒绝；币种冲突不强制归为同一种币种。
- 目标、方案字段使用白名单和类型检查。未支持字段不丢弃，而是导致弃权。
- 金额规范化后校验当前 assistant 报价中的原文金额和退款/收费角色。历史金额
  与工具数据库金额不能代替当前已沟通报价；没有币种证据的“免费”不能推断为 USD。
- 当前报价中的金额不能通过省略 terms 来绕过检查；预订号、用户标识及退款/收费
  账户必须出现在对应事件已经验证的原文或工具叶子证据中。
- 原始 JSON 包（包括校验失败的包）、有效包与错误分开保存，便于追查失败原因。
- audit 支持 1–4 个并发请求，并按可见前缀共享同一次请求；没有自动重试。
  每条返回立即写盘。summary 分开统计 false_merges、missed_merges、abstained_pairs。
- 无可解析输出的接口失败不会被计作“正确语义弃权”，单列 unavailable_pairs。

默认两条 smoke 配置已启用严格字段模式。16 条扩大验证配置：
`configs/analysis/semantic_model_kimi_k3_strict_20260914.yaml`，最多并发 3 个请求。
提示词附加协议：`configs/prompts/semantic_airline_slots_v1.txt`。
结果 provenance 同时记录基础提示词与附加提示词的 SHA256。

## 明确边界

这是有边界的规范词表与证据核查，仍需要模型识别真实业务含义。
当前 constraint 仅支持 fee 的比较谓词；其他条件应输出 unknown，不能删去条件。
自由文本字段保留精确值，不声称解决任意自然语言同义表达。金额角色检查是保守的
词法检查，复杂表达可能弃权，不能证明完整语义抽取正确，也不能授权工具执行。

验证样例有人工配对标签，但它们仍是开发/合同测试集，不是独立真实 rollout 评测集。
少量样例上没有错误合并不能证明实际错误合并率为零。仍需独立盲标验证以及后续
训练对照才能评价 GiGPO 收益。

## 验证

```bash
.venv-cpu/bin/python -m pytest -q \
  tests/test_semantic_slots.py tests/test_semantic_api.py tests/test_semantic_state.py
```

上次真实 API 返回的两条输出在严格模式下离线重放，已生成同一状态 key；
该重放没有新增 API 请求。真实扩大验证结果将在同目录验证记录中保存。

离线重验已保存 API 输出（不会调用外部 API）：

```bash
.venv-cpu/bin/python -m tau3_grpo.analysis.replay_semantic_api \
  --config configs/analysis/semantic_model_kimi_k3_strict_20260914.yaml \
  --source results/analysis/semantic_kimi_k3_strict16_20260914 \
  --output results/analysis/semantic_kimi_k3_strict16_offline_replay
```

输出目录必须是新目录。复验保存原始包哈希、测试集哈希及最终校验代码哈希。
传输失败记录不会被修复代码伪造成模型输出。

## 本轮真实结果

共调用 21 次：首轮 16 次（并发 3、超时 120 秒），只对无输出的 5 条补测一次
（并发 2、超时 180 秒）。补测结果不能替换首轮已有可解析语义包，避免按标签挑选输出。

| 指标 | 首轮修正统计 | 补测后最终复验 |
|---|---:|---:|
| 有效状态 / 样例 | 10/16 | 12/16 |
| 符合预期的配对 / 全部配对 | 9/14 | 11/14 |
| 无可解析输出而无法评价的配对 | 5 | 2 |
| 错误合并 | 0 | 0 |

最终 10 个配对实际进行了状态相等性比较，另 1 个授权歧义配对正确弃权。
剩余 3 个配对：`preference` 因币种 null 被拒绝，`changed_approve_synonym` 和
`stale_approval` 补测仍为 ReadTimeout。它们不能计作验证通过。
“未观察到错误合并”仅适用于这批少量开发样例。

无币种的“希望不收费”需要独立表示，不能默认补 USD 来提高通过率。
另外，`replace_goal` 的乘客范围（all/everyone）及历史删除乘客动作的字段拆法
仍不一致，虽可编译，但尚未解决这一类同义表示归并。人工 fixture 也只是开发参考，
字段完全一致并不等于完整语义准确率。

最终复验使用本轮全部校验修复，并记录代码哈希；首轮原始 summary 曾把一次
传输失败计作正确弃权，其原文件保留，正确统计以 `score_corrected` 和最终复验为准。
85 项相关 CPU 测试通过，Git diff 检查及新文件 lint 通过。未启动 GPU/RL。

机器可读记录：[semantic_slots_verification_20260914.json](semantic_slots_verification_20260914.json)。
本地完整输出：`results/analysis/semantic_kimi_k3_strict16_final_20260914/`。

下一阶段应先扩充“不收费”和乘客范围的结构化协议，补齐两条接口未完成样例，
再用独立真实 rollout 盲标评估错误合并、遗漏及弃权覆盖率。当前仍不建议启用在线 RL。
