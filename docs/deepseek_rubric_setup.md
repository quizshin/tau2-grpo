# DeepSeek Rubric judge 环境配置

> 2026-09-26账本校正：用户确认前939次请求的实际累计消费为13.93 CNY，替代33.440796 CNY的高峰未缓存估算作为预算起点；历史调用及原估算保留。新的预算占用=已确认实际费用+其后逐请求估算/未结算预留，不能称为实际账单。后续Flash请求按北京时间时段与缓存usage估算；周末采用低峰价。法定节假日日历尚未接入，工作日节假日可能保守高估，不会少预留预算。详见`results/maintenance/billing_reconciliation_20260926/`。

本批已完成：80条试审，75条结构校验通过、5条未评分；161次调用，按高峰未缓存价估算5.70924 CNY。证据复核发现三个严重错误候选中存在取消原因和免费行李额度误判，当前judge不得直接用于自动筛选。详见`results/analysis/deepseek_rubric_pilot_20260925/review_report.md`。

## 2026-09-25 本批授权

用户确认使用DeepSeek V4.1 Flash，本批费用上限约100元，执行上限设为100 CNY。已读取官方中文价格页并归档：请求名为`deepseek-flash`；保守计费使用高峰、输入缓存未命中价格（输入2元/百万token，输出8元/百万token）。来源：`https://api-docs.deepseek.com/zh-cn/quick_start/pricing`。价格估算不是最终账单。

鉴权、模型列表和账户可用性检查通过。试审入口为`python -m tau3_grpo.analysis.rubric_pilot --output results/analysis/deepseek_rubric_pilot_20260925`。先做一条合成JSON连通检查，再从历史97条AReaL训练来源中挑80条试审；每条先根据用户及工具证据生成Rubric，再审核完整轨迹。4并发、最多200个请求、不自动重试；共享预算账本逐请求预留额度，失败或usage缺失保留全额预留。

这批是流程试审，未获得人工金标或独立judge验证，也不覆盖全999条及标签冲突分布。仅输出待复核标记，不自动删数据或进入SFT。执行状态以试审目录的manifest/budget/summary为准。

日期：2026-09-25。状态：独立环境配置及两遍试审入口已实现，39项远程CPU测试通过，真实JSON探针通过。80条试审状态查看产物目录；尚未建立经过人工金标验证的正式judge。

## 填写位置

- 本地：`/Users/apple/Projects/program-llm/tau3_grpo_fix/code/.env`
- 双A800：`/root/autodl-fs/tau3-core/code/.env`

只填写以下三项，保留其他已有配置：

```dotenv
TAU3_JUDGE_BASE_URL=
TAU3_JUDGE_MODEL=deepseek-flash
TAU3_JUDGE_API_KEY=
```

本批BASE_URL为`https://api.deepseek.com`，MODEL为`deepseek-flash`（官方文档对应V4.1 Flash）。地址不要包含最终`/chat/completions`路径，客户端会追加。API_KEY仅在私有.env或进程环境中填写，不写入文档或聊天。

`.env`已被Git忽略，文件权限应为0600。`TAU3_ENV_FILE`可指定另一份私有env；显式进程环境优先于文件。没有自动读取任意`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`或原`TAU3_SEMANTIC_API_KEY`的回退，避免误用另一服务凭据。

## 如何读取

以下只加载配置、创建客户端，不访问网络、不收费：

```python
from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel

judge = OpenAICompatibleSemanticModel.from_env(
    {"timeout_seconds": 120, "max_tokens": 8192, "temperature": 0},
    env_prefix="TAU3_JUDGE",
)
```

不要打印对象内部字典、API密钥或原始HTTP错误体。环境变量缺失时会明确提示`TAU3_JUDGE_API_KEY`；配置错误就停止，不切换模型。未确认供应商能力前不默认发送thinking或JSON mode扩展参数。

试审复用`extract_json`传输，已加入可见字段白名单、五维输出结构、证据ID检查、unknown处理、缓存/续跑和预算上限。旧`extract`方法校验语义状态packet，不用于Rubric评分。正式数据清洗仍需完成人工/独立复核、时间化证据核对和全源分布校准。

## 已确认信息与后续边界

官方端点、V4.1 Flash和本批100 CNY上限已获用户确认。本地用户配置已同步到远程；没有再次索取密钥。合成短输入连通检查不等于Rubric质量通过。全999条审核和teacher补数需先根据试审质量与剩余预算制定下一批范围。

本批包含CPU和外部API，不启动本机/远程GPU。项目AGENTS.md要求GPU训练/推理前单列用途和预算；完成Q100数据、token统计与配置后再提交阶段A训练/240条评测预算。

## 实施范围

现有语义分析客户端默认仍读取`TAU3_SEMANTIC_*`；新增可选`env_prefix`使Rubric使用独立凭据。传输、异常隐藏和不自动重试行为沿用原实现。环境设置不改变正式任务验证器、pass@k计算、policy模型或用户模拟器。
