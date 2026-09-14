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
