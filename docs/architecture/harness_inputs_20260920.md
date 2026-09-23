# Harness 输入修复、评测温度与完整工具 schema


## 2026-09-21 5090 实际使用核对

核对 A45/B100 各自训练时的 `source-snapshot.tgz`：SFT直接读取 `configs/envs/tool_config.yaml` 的原始14工具schema并交给tokenizer，不经过会丢字段的veRL注册模型。两份工具schema SHA256（sort_keys JSON）均为 `d3e56317590d4473fd220d11cd48e1c5fb210286ff86980e50c597e824514211`，包含Passenger必填first_name/last_name/dob、Payment必填payment_id/amount、数组items和$defs。

当前5090 GRPO v4的token协议在runner强制完整schema，并在ToolAgentLoop初始化时与官方schema逐项核对。前18步1152份真实训练输入及step10的240份评测输入均保留完整schema；与SFT归档的唯一字典差异是无参数工具list_all_airports的空required列表省略，嵌套字段没有丢失。此证据证明当前GPU运行已使用完整schema，不是legacy/full schema的收益消融实验；下文“仅CPU、尚无GPU”的措辞属于9月20日当时的状态。

日期：2026-09-20。本轮仅代码和 CPU 验证；没有 GPU、模型生成、API 推理、训练或官方 final50 访问。旧实验结果和旧协议保留。

## 2026-09-20 后续决定：所有活动采样默认统一为 0.7

用户明确要求训练、训练内评测、独立评测中的策略与用户模拟器均使用 0.7，登记为 `sampling_t07_v1`。这覆盖下文之前的 1.0/0.0 温度决定；旧参考、审计及运行结果保留当时参数，不改写历史。当前 formal50 与完整 schema 候选、直接 shell、评测 CLI/控制器及运行时回退均更新。显式覆盖仍可用于单独实验，比较以实际配置为准。历史日期 profile 保留旧策略覆盖；若复现旧实验，必须从该 run 的完整配置恢复用户温度等参数，不能只依赖当前共享默认。

本次变更没有 GPU 或模型调用，不证明温度 0.7 的效果优于旧配置。温度一致仅消除采样参数中的这一处差异，多轮 token/context 和其他采样参数仍需单独核对。验收与文件前像记录在 `results/maintenance/temperature-07-20260920/`。

本次收尾：本地与远程各120项CPU回归通过（同一组，不相加）；两端lint零新增、shell语法检查通过。真实selection60×4 dry-run为240条，策略/用户均0.7，未生成。22份本轮修改文件已逐项同步核验，前像与部署回执保留。

## 原决定与参考核对：按参考区分训练与评测

用户要求核对 Mercor 博客对应参考。其仓库 `Mercor-Intelligence/ApexAgents-SkyRL-Recipe` 的 `scripts/run_eval.sh` 明确设置 `TEMPERATURE=${TEMPERATURE:-1.0}`，随后同时传给 `harbor_trial_config.agent.kwargs.llm_kwargs.temperature` 和 `trainer.algorithm.temperature`。2026-09-20 下载的 main 文件与项目原参考快照 `8e7702f03b7464a36ab800a624fd911de0968a87` 的该文件字节一致。README 确认博客对应关系；博客正文直接请求返回 403，温度依据是实际仓库脚本，不能伪称来自已读正文。

本轮新协议 `tau3_eval_train_inputs_v3` 的 CLI 和控制器默认策略评测温度 **1.0**，用户模拟器独立评测温度保持本项目已有 **1.0**。新完整 schema 候选的训练内策略评测温度也显式设为 **1.0**；旧正式 profile 的 0.4 保留。训练内用户模拟器仍沿用训练配置，不把它与独立评测默认混同。参考脚本证明的是 agent 温度，不能据此声称 Mercor 要求 Tau 的用户模拟器温度。旧 legacy/v2 策略默认 **0.4**、用户默认 **1.0** 保持原样。显式 CLI 温度覆盖仍可用，实际温度记录在 run/provenance 中，比较器检查差异。程序化 `Endpoint` 保持兼容默认，调用者须显式传入实验所需温度。

训练用户模拟器温度 0 与评测温度 1 不同本身不是 bug。最初提出的 v3 强制用户温度 0 已撤回，该提案未部署到远程、未进行采样。不要据“输入对齐”强迫训练与评测的采样策略一致。

参考脚本和 SHA256 清单：`results/analysis/harness_inputs_20260920/reference/`。

## 新评测协议的修复

v3 继承 v2 的 STOP、轮数、错误上限和可见工具裁剪规则，新增：

- 从同一 `reason_for_call` 提取开场，并共用训练 parquet 的原有回退规则；不把隐藏任务目标、参考动作或 reward 传给 agent。
- 移除原生固定问候与额外的模拟开场，策略与用户模拟器同时从该用户消息开始；初始用户计入用户总数，不计 observation batch。
- 策略请求显式 `max_tokens=1024`；策略及模拟器请求显式 `enable_thinking=false`。
- 记录开场 hash，运行前验证全计划中的开场与 fresh-session 前提。任务、DB 初始化和官方评分仍走原生路径。
- 仅支持 AReaL selection；official final 在加载任务前拒绝。已有 message history/solo 不适用。

每轮 1024 token **不等于**训练累计 response/context 预算已经对齐。v3 仍通过聊天服务重新渲染多轮历史；服务侧上下文上限与训练累计 budget 仍有差别。原协议元数据保留 `token_context_equivalence=not_established`。

默认协议仍是 legacy。新比较必须给共同 SFT 起点和候选显式使用同一协议、温度及输入版本；不能把 v3 的分数接到历史排名。

只核对计划、不调用模型：

```bash
python -m tau3_grpo.evaluation.run --target selection \
  --checkpoint checkpoints/sft-merged/new-off \
  --harness-protocol tau3_eval_train_inputs_v3 \
  --policy-temperature 1 --user-temperature 1 --dry-run
```

已实测解析 selection60×4=240 条计划，两个温度均为1；没有执行真实推理。


## 发现并修复：工具 schema 在训练注册时丢失

实际路径为：官方 schema → 生成 YAML → veRL `OpenAIFunctionToolSchema` → `model_dump` → 模型模板。固定 veRL 的简化 Pydantic 模型会丢弃未声明字段，包括 `$defs`、数组 `items` 与嵌套对象描述；例如订票工具的 Passenger/Payment 结构未完整传给训练模型。工具执行验证器仍要求对应结构，因此不能把“工具照常执行”当作模型已经收到完整说明。

新增显式 `tau3_full_schema_v2`：

- YAML 的 tool config 携带完整原始 schema；原登记 schema 与 payload 的可见类型必须一致，否则拒绝。
- 项目 `integrations/verl/tool_schema.py` 保留 veRL parser 的类型访问接口，但序列化时返回完整原始 schema；执行继续由官方环境验证。
- 真正使用 veRL registry 加载 YAML 后，逐工具输出与官方完整 schema 精确一致；保留旧 `verl_legacy_v1` 默认，未改 vendor 或旧配置。
- 新候选 `configs/train/rl/formal50_full_schema_v2.yaml` 复用正式共享入口，显式设置 `TOOL_SCHEMA_VERSION=tau3_full_schema_v2`。兼容默认放在 base YAML，shell 仅验证并传递选择。

这个改动改变训练模型看到的输入，不能静默用于旧 checkpoint 精确续训。它是待 GPU 验证的独立候选，尚无策略收益证据；已有奖励配方、GTPO/GiGPO 优势及 DF 未修改。

## 真实 tokenizer 核对的范围

只复制既有 `new-off` 的 tokenizer.json、tokenizer_config.json、chat_template.jinja；不加载模型权重。使用原生训练 pending handler、实际工具 registry，以及新评测捕获的首请求，再由真实 Qwen tokenizer 渲染。

- 新完整 schema + v3：selection60 的首轮 messages、工具 schema、token IDs 均逐项相等。首轮 5,143–5,258 token，低于训练 8,192 prompt 上限。
- 旧训练 schema + 相同开场：60 条均不等；在此固定 tokenizer 下，旧 schema 比完整 schema 少 1,015 token。该数字不是效果增幅或成功率。
- 从修改前 parquet builder 前像重建：train 源 manifest 200 条与 selection60 的完整 row 输出未改变；200 条训练源的完整 schema 首轮最大 5,227 token，均低于 8,192 上限。正式 train50 池没有改成 200 条。
- 最初的 tokenizer 审计把同一份官方 schema 交给两端，60 条匹配仅证明“相同 schema 条件下”的首轮相等。加严为真实注册路径后暴露丢字段问题；原尝试及断言失败日志保留，最终结论以 full-schema 审计为准。

多轮反例也使用真实 tokenizer、固定的脚本输出和 native suffix 逻辑：

1. assistant 文本后出现新用户消息：chat 模板移除历史空 thinking 块，训练保留原始 token。
2. 合法紧凑 XML 工具调用后返回 observation：chat 重渲染增加换行，训练保留原始生成格式；native qwen3_coder parser 已确认该调用可解析。

因此仅首轮在**新完整 schema 配置**下建立匹配，多轮 token/context 完全等价仍未完成。实际服务 HTTP/parser 与模型输出分布亦未由 CPU 模板测试证明。

## 验证与证据

主证据目录：`results/analysis/harness_inputs_20260920/`。本地相关回归 141 项通过；新增 schema 注册、序列化、三算法候选配置 8 项通过，二者不重叠。后续启动入口回归与这 8 项有重叠，单独记录，不累加冒充独立测试数。远程主回归 **148 passed**（560.90秒），覆盖当时7项schema用例；新增训练内温度及MT入口的收尾配置回归两端各 **38 passed**（本地10.41秒、远程317.31秒），远程状态 `remote-final-config-status.json=passed`；与主回归有重叠，不相加。

两端 lint 零新增，保留既有 382 条债务；shell 语法及 diff whitespace 检查通过。未修改本轮 vendor。新增测试纳入分层 CPU 清单。

保留失败尝试：新测试最初未按用户模拟器原生角色翻转编写期望；候选配置测试首次漏传 TAU3_ROOT；加强 schema 审计后实际发现字段丢失；首次远程部署因非激活 shell 找不到 python3，在任何写入前停止。最终部署使用既有环境的绝对 Python 路径，并逐文件验证前像、备份、原子替换和最终 hash。

两端真实 tokenizer 审计逐行匹配：本地 Transformers 5.17.0、远程 5.5.1，60条首轮身份与token、旧schema差异及两个多轮反例全部一致；这不扩展成模型服务等价。`cross-host-verification.json` 和 `remote-final-source-hashes.json` 已核对本轮19个源码/配置/测试/脚本文件一致。

远程部署回执：`results/maintenance/harness-inputs-deploy-20260920/`。同步仅涵盖本轮列明文件，不代表两端整个 Git 工作区一致；没有提交或推送并行奖励修改。

## 仍需继续

- 设计保留原始生成 token 的统一多轮评测路径，明确生成、observation 与总上下文边界；不要通过删除历史 token 来伪造一致。
- 完整 schema 的真实 GPU 采样及收益对照需另定预算，且各算法共用相同输入协议。当前没有启动。
- 官方评分的对话/授权覆盖、实体留出泛化、MT-GTPO 信用分配风险继续开放，与本轮输入修复分开验证。


## 2026-09-20 多轮与预算后续

新增显式 `tau3_eval_token_v4` 与训练 `tau3_token_budget_v1`，直接共用原生 ToolAgentLoop，通过 token HTTP 传递原始 ID，并统一追加预算。v3 原实现和本报告历史见证保留。新实现、CPU/GPU 验证范围与启用方法见 [多轮 token 预算报告](token_harness_20260920.md)。活动温度后续已按用户要求统一为 0.7；本报告中温度 1 的描述是之前阶段的历史状态。
