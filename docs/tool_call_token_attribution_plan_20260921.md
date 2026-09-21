# 第一步：工具调用 token 归属记录设计

状态：设计已在本地实现并进行 CPU 验证，见 [实现与实验](tool_call_token_attribution_cpu_20260921.md)；未部署远程生产。只增加观测，奖励、优势、mask、提示词、工具解析结果和执行顺序保持原样。无需 GPU 即可实现并完成第一轮 CPU 验证。

## 已有接口

- `verl/verl/experimental/agent_loop/tool_agent_loop.py` 在收到 `TokenOutput.token_ids` 后保存轮级 `token_span`，在解析后进入工具执行；已有 `record_turn_facts` 开关。
- `tool_parser.py` 的 Qwen3XMLToolParser 输入实际 token IDs，返回 assistant 文本及 FunctionCall，但不返回源区间。当前解析器可能接受未闭合 XML，也可能在一个工具标签中发现多个 function；不能假定一个标签等于一个调用。
- `tau3_grpo/envs/interaction.py::record_tool_batch` 在执行前分配稳定 call ID，已有 observation/error/db_hash_after，可与 token 归属关联。
- `tau3_grpo/data/trajectory.py::trajectory_facts` 保存原 token IDs、mask、轮区间和实际保留区间。
- 本地 Qwen3.5-4B tokenizer.json 中，`<tool_call>` 和 `</tool_call>` 分别为独立 token 248058/248059。实现必须读取实际 tokenizer 映射并校验，不硬编码这些值。本轮远程只读可行性检查遭 SSH connection refused，未得到新的线上区间核验结果。

## 记录方案

1. 生成返回时，从实际 `output.token_ids` 扫描工具标签 token，记录每个原始块的半开区间 `[start, end)`；闭合块默认包含起止标签。原始生成序列不经 encode/decode/encode 重建。
2. 解析阶段提供可选的 provenance 结果：原始块索引、函数索引、实际解析调用索引及状态。旧 `extract_tool_calls` 接口和原解析结果保持兼容；优先复用同一次解析产生来源信息，不能另写语义不同的解析器后只按数量 zip。
3. 单闭合块且唯一解析调用、来源校验一致时，确认为一对一精确归属。一个块有多个函数、嵌套、孤立闭合、边界无法唯一确认等情况显式记为 ambiguous/unavailable，不给多个调用强行分配同一整块区间。未闭合但旧解析器接受并执行的调用，记为 incomplete + executed，不能误写成未执行或改变执行行为。
4. 在原 `record_tool_batch` 之后绑定稳定 call ID。按现有顺序执行并记录结果。若增加 db_hash_before，在每个调用执行前只读获取；与已有 after 配对。状态 hash 变化只证明发生变化，不是业务正确性的判定。
5. 发布轨迹时分别保存 emitted 与 retained 区间，标明裁剪/丢弃。区间统一相对最终 response 数组，并保留轮内偏移以便追踪。若发出的 token 被截去，要保留对应原始发出 IDs 或可信原始回执引用；部分保留的调用不能标为完整可训练样本。

建议新增可选命名空间 `call_attribution`，独立 schema 版本，不改写已有 v1 字段语义：

```json
{
  "schema": "tau3_call_attribution_v1",
  "coordinate_system": "response_token_offset_half_open",
  "blocks": [{"block_index": 0, "emitted_span": [120, 180], "retained_span": [120, 180], "closure": "closed"}],
  "calls": [{"call_index": 0, "call_id": "stable-runtime-id", "block_index": 0,
             "alignment": "exact", "execution_status": "executed",
             "error": true, "db_hash_before": "...", "db_hash_after": "..."}],
  "other_generated_spans": [[100, 120], [180, 181]],
  "eligible_for_call_credit": true
}
```

上面只是字段示例。`other_generated_spans` 包括文本、空白、EOS 等非调用生成 token，不能统称推理文本。它与原始工具块区间组成生成 token 的可检查分区，工具返回仍为 observation/mask=0。解析失败但存在原始块时保留该块，没有虚构 call ID。工具执行结果与边界可靠性是两个独立维度。

## 模块归属与兼容

纯 token 区间扫描和验证放在项目 `tau3_grpo/data/`，不放进优势算法。Qwen parser 只增加可选来源信息，agent loop 负责串联生成/解析/执行三个阶段，trajectory_facts 负责发布和验证。观测开关贯通现有事实记录配置，不自动启用调用级 loss。

若修改 vendor 文件，按 AGENTS 更新 vendor 补丁登记、固定 revision 对照和相关测试清单；不把算法设计放入 vendor parser。新增字段不能改变 process_reward_json 的原始奖励、turn_spans 或三种 estimator 的输入含义。

## CPU 验收

- 单调用、多调用、两次完全相同调用：按源位置和稳定 ID 区分，不按工具名或参数相等匹配。
- 文本在调用前/间/后、中文参数、EOS：区间覆盖原生成 token，互不重叠，不含 observation。
- 不完整 XML、格式错误、嵌套、一个块多个函数、解析异常：明确状态和归属覆盖率，不猜区间。
- 工具报错、未知工具、调用被策略限制而未执行、最后一轮终止：生成/解析/执行三类事实分别保留。
- 生成后裁剪、完全丢弃、并发多会话：区间裁剪真实、call ID 不串轨迹。
- 同一原始输出开关记录前后对照：解析调用、执行顺序、DB、官方评分、过程奖励、mask、MT 优势均不变；相关 GRPO/GiGPO 路径保持兼容。
- 合成测试可使用真实 tokenizer 编码测试文本作为“模拟生成输入”，明确是测试 fixture；历史数据不得重新编码后声称恢复了真实生成边界。

第一阶段交付是可审计的调用归属，不是调用级 MT 算法或提点证据。真实新采样验证另列小预算，CPU 完成不自动启动 GPU。已有原始 IDs 完整的轨迹可另做带来源的离线派生审计，不能覆盖旧记录或声称过去训练已使用调用级 mask。
