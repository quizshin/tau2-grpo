# 增量语义抽取续接记录

续接任务 `01a08fc8-1e63-7503-a9eb-a067df4e07dc` 在 2026-09-14 中断的实现。
此次只做离线抽取、CPU 测试和真实 API 回归，没有启动 RL，没有修改 `.env` 默认模型。

## 实现边界

- 默认仍为 `full_prefix`；显式配置 `extraction_mode: incremental_v1` 才启用新路径。
- 每次输出当前消息的增量事件。本地给事件绑定消息位置并展开证据 ID，模型不能改写历史。
- 本地验证累计事件，维护目标、方案替换及批准失效；工具知识由已有规则提取，不当作用户批准。
- 相同可见前缀共用请求与缓存，不同分支返回独立副本。失败前缀的后继继续弃权，不自动重试或修补。
- `max_incremental_calls` 是整个审计实例的总预算，包含所有分支，不是每条轨迹的预算。
- 每个请求仍有独立总时限；整条前缀需要多次顺序调用，没有承诺整条前缀在 180 秒内完成。
- 新增 `finish_reason` 与白名单 token 计数，非正常结束仍拒绝。历史运行未记录的原因无法追溯恢复。
- 增量事件成功或被拒绝后立即刷新 `deltas.jsonl`，不会等全部审计结束才保存。

“增量”指输出和状态维护方式，不是只发送一句话：请求仍含先前事件和整个可见前缀的
证据目录，目录含历史文本及已观察工具事实。因此不能声称输入长度、总费用或总延迟必然下降。
整段文本证据可以定位来源，但不保证模型已完整提取其中语义；`opaque_context` 仍可能造成漏合并。

## 接手时已完成的真实 Pilot

原任务后台请求已自然完成，直接复用保存的结果，没有重复付费请求。
同一 airline_683 短、长前缀，缓存共享；两个模型各实际调用 6 次。

| 模型 | 请求耗时之和 | 总 token | 正常结束 | 有效位置 |
|---|---:|---:|---:|---:|
| GLM-5.3-Flash | 115.879 秒 | 49,609 | 6/6 | 1/2 |
| Kimi-K3 | 262.069 秒 | 48,113 | 6/6 | 1/2 |

两者都只通过短前缀；长前缀都在消息 15 的方案事件触发 `unsupported_all_passengers`。
原文按 3 位乘客计算两程总价与退款，但没有写出 `all passengers` 等规则认可的措辞，
本地按原规则拒绝 `passenger_scope: {kind: all}`。这暴露了证据规则的表达覆盖限制，
不能仅凭错误名认定模型臆造了乘客范围；也不能只凭费用乘数就证明所有情形下都可推导全员范围。
这次不是接口超时，但也不是长前缀语义验证成功；没有删除字段、放宽规则后重新计分。
请求累计耗时不能当作单次请求延迟，也不能当作并发吞吐指标。

Pilot 产物：`results/analysis/semantic_incremental_20260914/pilot_{glm,kimi}/`。

## 回归设计

使用 `configs/analysis/semantic_incremental_glm_diagnostic_20260914.yaml`，
GLM-5.3-Flash、并发 4、单请求 180 秒、max_tokens 8192、总预算 96 次，无自动重试。
选择 GLM 是因为这次 pilot 的累计请求耗时更低，不表示它已经达到生产可靠性要求。

复用旧审计的 24 个位置、18 个不同前缀，以及原先冻结的 22 对标签。
这批数据已被先前任务审阅，明确作为版本回归，不称为新留出集或独立泛化验证。
12 对相同前缀检查与 10 对真正语义比较分开计分；未因本次结果调整提示词或证据校验规则。
API 不接收奖励、配对标签、当前待执行动作或未来消息。

本轮回归完成 26 次 API 请求，24/24 位置返回，13/24 通过累计状态校验，覆盖 7/18
不同前缀；请求耗时 4.389--83.436 秒。step1 为 8/8，step3 为 5/8，step6 为 0/8。
失败主要是 `unsupported_all_passengers`（4）、`unsupported_identity_slot:reservation_id`
（3）、`unsupported_identity_slot:user_id_claim`（2）和 `unsupported_confirmation_question`
（2），均为本地严格校验拒绝，不等同于模型错误率。12 对完全相同前缀全部通过；10 对
语义配对只有 6 对可判定，其中 4 对通过、4 对弃权，2 对正例被分开，未出现误合并。
有效状态中有 9 个带 `opaque_context`。冻结样本奖励全部相同，事后分组没有可变回报组，
因此没有新增步级信用信号可测；这不是 GiGPO 无效的证据。

## 复现

输入和旧审计相同，构建方式见 `semantic_real_audit_20260914.md`。下列第一条命令会调用真实付费 API：

```bash
.venv-cpu/bin/python -m tau3_grpo.analysis.audit_semantic_api \
  --config configs/analysis/semantic_incremental_glm_diagnostic_20260914.yaml \
  --output results/analysis/semantic_incremental_fresh_run

.venv-cpu/bin/python -m tau3_grpo.analysis.score_semantic_real_audit \
  --cases results/analysis/semantic_real_audit_20260914/inputs/audit_cases.json \
  --states results/analysis/semantic_incremental_fresh_run/states.jsonl \
  --labels docs/semantic_real_audit_review_labels_20260914.json \
  --metadata results/analysis/semantic_real_audit_20260914/inputs/outcome_metadata.json \
  --output results/analysis/semantic_incremental_fresh_run/score.json
```
