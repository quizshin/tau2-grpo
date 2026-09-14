# Kimi-K3 离线语义抽取接口

这是可选的外部模型适配器，尚未接入 RL。GLM embedding 可以召回相似对话或状态，
但向量相似度不能证明目标、业务条件、报价、用户批准范围等价。
Kimi-K3 负责提出带原文证据的语义事件；现有 reducer 校验证据、更新状态、决定能否归并。
证据合法也不等于语义正确，仍需人工标注对照验证。

## 配置

在 `code/.env` 填写密钥（其余两项已配置）：

```dotenv
TAU3_SEMANTIC_BASE_URL=https://llmapi.paratera.com/v1
TAU3_SEMANTIC_MODEL=Kimi-K3
TAU3_SEMANTIC_API_KEY=
```

`.env` 被 Git 忽略，权限设为 0600。既有进程环境变量优先于文件；可用
`TAU3_ENV_FILE` 指定其他文件。语义接口变量独立于策略、模拟器及 SwanLab。
可选依赖安装方式：`python -m pip install -e '.[semantic]'`。

## 填好 key 后运行

从 `code` 目录执行（这条命令会调用外部 API）：

```bash
.venv-cpu/bin/python -m tau3_grpo.analysis.audit_semantic_api \
  --config configs/analysis/semantic_model_kimi_k3_20260914.yaml \
  --output results/analysis/semantic_kimi_k3_smoke_20260914
```

默认 `limit: 2`，只检查一对人工构造的同义对话，最多两次请求。本地不需要 GPU。
输出目录必须不存在；重复运行应换目录，以保留历史结果。通过后可将 `limit` 改成
`null` 跑全部 51 个样例（本次运行内相同前缀复用结果，包括失败结果）。下一阶段才是
对真实 rollout 做盲标、检查错误合并与过度弃权；这一步不能由两条 smoke 结果替代。

接口使用 `/chat/completions`、温度 0、`max_tokens: 8192`、单次超时 120 秒，
无自动重试，不跟随重定向。暂不发送供应商专有 thinking 参数或强制 JSON-mode 参数，
其支持情况尚未验证；因此这里的配置不代表开启或关闭了 Kimi 的 thinking。
若截断则弃权；需要依据真实响应再调整输出预算。

只发送提示词、schema、前缀哈希及当时可见的对话/工具观察；不发送配对标签、
隐藏任务描述、未来动作或回报。输出 `packets.jsonl`（通过协议校验的原始包及失败原因）、
`states.jsonl` 和 `summary.json`，记录模型、地址、提示词哈希、请求参数及 token 用量。
不记录密钥、请求头或供应商错误正文。有效 packet 编译失败时同样弃权。

## 本次验证范围

仅使用 `httpx.MockTransport` 和既有人工 fixture 测试请求协议、失败处理、证据检查
及结果保存；没有调用真实 Kimi-K3，没有评估真实模型准确率，没有启动 RL。
原 `audit_semantic_model` fixture 流程保持独立；已有模拟实验结果仍是模拟结果。

验证命令：`python -m pytest -q tests/test_semantic_api.py tests/test_semantic_state.py`。

## 首次真实接口验证（2026-09-14）

用户填入 key 后共请求 4 次：初始两条、失败样例格式诊断一次、解析修复后补测一次。
初始结果是 1/2 有效、配对弃权。格式诊断发现模型返回单个 Markdown JSON 代码块；
适配器现允许剥离完整的外层代码块，仍拒绝周围说明文字和非法内部 JSON，保留证据校验。
补测后第二条也能编译，但与第一条的状态 key 不同：两者目标字段分别为
`reservation_id` 和 `booking_id`，其余编译状态相同。该同义样例尚未正确归并。

另外，两条输出都使用 `refund_amount`/`refund_currency`，而现有金额原文核对仅针对
`quoted_refund`/`quoted_charge` 与 `currency`。因此当前校验通过不代表这些替代字段
经过金额证据校验。后续应约束 canonical slot/terms schema，并防止别名绕过检查；
在此之前不接入 RL，不把这次 smoke 当作语义有效性的证明。

原始初测与补测文件分别保留于本地
`results/analysis/semantic_kimi_k3_smoke_20260914_live1/`，未覆盖初测结果。
新增包装兼容测试后，相关 CPU 测试共 52 项通过。没有启动 GPU 或 RL。

后续已增加可选严格字段模式，默认 API smoke 配置启用 `slot_schema: airline_slots_v1`。
字段归并、金额核查、扩大真实验证及无 API 重验流程见
[严格字段验证](semantic_slots_strict_20260914.md)。上述首测数字保留为历史结果。
