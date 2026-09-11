# 顺序多调用协议（2026-09-11）

项目采用 `airline_sequential_multicall_v1`，同时支持一轮一个或多个工具调用。

## 执行语义

RL 启动脚本传入 `actor_rollout_ref.rollout.multi_turn.tool_execution_mode=sequential`。
在这个模式下，完整的已解析调用列表按生成顺序逐个执行，`max_parallel_calls`
不限制调用列表长度。全部结果返回后才开始下一次模型生成。veRL 默认的
`parallel` 模式保留原行为，供其他项目使用；Tau3 interaction 会拒绝该模式，
防止允许多调用的提示词被配到会截断列表的执行器。

一条 assistant 消息包含整个调用列表，随后是每个调用的 ToolMessage。
每个调用有独立 ID，正常执行、参数/工具名错误、回放共用这个 ID。
一批 N 个调用计一个 assistant turn、N 次工具调用；最终允许轮次仍执行整批。

普通工具错误返回对应错误结果，然后继续执行后面的调用，与官方 orchestrator
一致。执行阶段抛出的异常属于基础设施/包装层失败，直接中止该 rollout 并清理
session；不会把未知执行状态伪装为工具错误，也不会自动重试可能已生效的写操作。
原有生成和上下文预算仍生效，超限轨迹仍标记为截断。

模型生成的原始 token/logprobs 保留；工具观察的 response mask 为 0。
整个批次仍属于同一次模型决策，不为批次内每个工具伪造额外推理或 anchor。

## 提示词

唯一生成入口为 `tau3_grpo/prompts.py::build_system_prompt()`。

- 每轮可以提交一个或多个调用，执行顺序与列表一致。
- 所有参数和必要的用户确认已明确时，才可以批量提交。
- 后一个动作需要观察或判断前一个结果时，必须等结果返回，再在下一轮生成。
- 每个数据库写操作都必须符合 Airline 的用户确认要求。
- 工具调用与面向用户的回复分轮进行。
- 一个工具出错不会阻止同批后续工具，模型必须检查所有结果。

原始 SFT 提示词既有明确单调用指令，也有放宽确认要求的文本。因此加载时使用
当前官方 Airline 业务规则，加上项目多调用协议，替换整个 system 消息。单纯追加
“允许多调用”会留下矛盾，本实现不采用这种做法。assistant、user、tool 内容保持不变。
这次没有重新判定历史示范是否满足业务规则，不能把更换提示词等同于数据清洗完成。

## 已有数据与评测

新生成的 SFT/Parquet 使用新提示词。已有 SFT JSONL 在 Dataset 加载时适配，重新
计算实际 token 长度和训练 mask；训练目录会保存 `effective_train.jsonl` 和
`effective_validation.jsonl`，SwanLab 示例也使用这些实际输入。

已有 RL Parquet 在 ToolAgentLoop 的 pending 阶段、模型输入 tokenization 前适配。
实际 prompt 超过配置上限会报错。原始数据文件和历史训练结果不被覆盖，旧 checkpoint
也不会因为修改代码自动变成按新协议训练过的模型。

项目 selection/final 评测使用继承官方 LLMAgent 的 MultiCallAirlineAgent；仅提供
相同的新系统提示词，调用执行和评分仍由官方代码负责。原始 benchmark 源码及其
policy.md 不修改。运行结果的 provenance 包含协议版本、系统提示词 SHA256 和
`benchmark_policy_modified=true`。这些属于“τ³ 任务 + 项目多调用协议”的结果，
不能与原始单调用提示词下的历史分数混为同一实验设置。

## 验证范围

本地 CPU 测试使用实际 veRL 状态机、工具解析器及 τ³ Airline 工具/严格回放，
模型生成和用户回复由确定性测试替身提供，不调用付费端点。
覆盖单调用、完整多调用的顺序、错误后继续、写后读、轮数边界、ID 对应、
mask/logprob 对齐、基础设施异常不重试、旧提示词适配及独立 session 隔离。

对现有 45 条 SFT 审计快照离线适配，45 条 system 消息均替换成功，其他消息保持一致。
未进行 GPU SFT/RL，也未测量修复后的模型成功率。下一步应固定 checkpoint、任务和
采样参数，对新协议建立 Base/SFT 成功率基线，然后再启动新的 RL。
