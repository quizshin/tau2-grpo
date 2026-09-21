# 训练分析第二步：训练与独立评测 harness 差分核对

日期：2026-09-20。实验 ID：`harness-step2-audit-20260920`。

承接[第一步审计](training_step1_audit_20260920.md)。本次运行两套真实循环，以固定助手输出、固定用户回复替代模型调用；真实执行工具、修改各自内存数据库、调用官方评分器。无GPU、无模型/API推理、无新训练，没有修改生产代码、奖励、历史结果或final50。

**结论：工具执行和最终DB一致，不代表两套harness协议等价。** 本地及远程无卡环境各完成22个定向场景，最终DB全部一致，9个场景终止/分数不同。它们是刻意构造的边界探针，9/22不是实际运行故障率。

## 1. 核对方法与边界

训练侧运行原生 `ToolAgentLoop.run`，经过pending→generating→tools/interacting→finalization；使用真实Qwen工具语法解析器，另以Hermes交叉验证多工具场景。使用原生 `Tau3AirlineInteraction`、`TrajectorySession`、`Tau3AirlineTool`与官方终局verifier。

独立评测侧运行原生 `evaluation.runtime._run_one`，由真实 `MultiCallAirlineAgent`、`UserSimulator`、`Orchestrator`及 `run_simulation` 执行。仅在两种生成接口返回预先写好的助手和用户消息；没有把Orchestrator替换为项目自写循环，也没有强行指定最终终止原因。

每侧独立加载同一任务DB，记录每次工具调用的参数、错误、完整原始返回与前后DB hash，同时保存模型可见的工具返回、用户回复次数、最终原因和reward。评分仍按原任务规范执行。探针不用于评估策略能力或业务授权合规。

使用确定性的UTF-8 byte tokenizer和固定占位logprob检验循环、mask和边界；这不是Qwen真实tokenizer，未验证模型token预算的实际触发频率、BPE切分、服务模板或trainer/vLLM概率差。模型生成调用和网络连接在脚本中禁止。

两侧首条用户文本统一为`Please help`。这控制了本次输入，未证明实际初始用户生成分布、用户模拟器的完整历史或模型prompt逐token相同。独立侧模型响应已结构化，HTTP响应到结构化调用的服务解析不在本次差分范围内。

本轮运行时工作区另有split-v4奖励修改，与本次核对分开。参与本次对照的harness源文件以实际SHA256登记，不用Git HEAD冒充干净工作树。没有覆盖其他任务的修改。

## 2. 实际配置对应关系

从上一轮MT工程的resolved配置与D独立评测run.json读取，不从文件名猜配置：

|项目|训练/训练内评测循环|独立评测循环|
|---|---|---|
|助手预算|`max_assistant_turns=15`|`max_steps=30`，计Orchestrator角色转换step|
|另一计数|loop的`user_turns`同时计工具返回批与用户回复，cap15；Session另计真实用户消息且含初始消息|环境返回、助手、用户均参与step推进；初始用户回复占一步|
|工具错误上限|该循环不按累计10次错误提前终止|`max_errors=10`，达到上限终止|
|工具返回|65,536字符、middle裁剪|此次runtime直接传完整ToolMessage，无相同字符裁剪|
|模型上下文预算|本次记录配置prompt8192、response16384，循环显式判断response_length，达到边界可以不解析/不执行生成动作|runtime未执行相同的本地token计数，实际服务仍可能受自身上下文上限约束|
|采样温度|训练1.0；训练内验证0.4|policy0.4、user1.0|

温度差是训练与验证的配置选择，本次固定输出不分析其效果；不能因工具协议/系统prompt版本相同，就认定整个执行协议相同。

原配置和hash保存在 `results/analysis/harness_step2_20260920/recorded-protocols.json`。

## 3. 已确认一致的行为

- 普通文本→用户STOP、单工具→回复→STOP，在充足预算时终止原因及评分一致。
- 同轮写库→读回→工具错误→后续工具，按给定顺序全部执行；错误没有吞掉后续调用。Qwen/Hermes两种训练解析路径与独立评测的结构化调用一致。
- 同轮未知工具、参数错误均有各自错误返回，后续有效工具继续；原始环境返回与最终DB一致。
- 最后允许的助手轮含工具时，训练先执行当前工具，再阻止下一次生成；显式为独立侧匹配相应step边界后，两侧均以max_steps结束。
- 生成返回`finish_reason=length`但工具文本完整时，训练仍会处理该调用；本次固定消息下不会仅因为单轮length就强制终局0。此结论不覆盖实际不完整JSON或服务解析失败。
- 助手文本为`###STOP###`的探针中，两侧都继续按用户响应完成；没有复现“助手STOP被两侧不同处理”的猜测。

在相同动作实际得到执行的场景中，观察到的工具参数、顺序、错误、返回与DB均一致。22个场景最终DB全部一致；其中生成恰好撞上训练上下文边界的1个场景，训练没有执行工具，独立侧执行了一个只读calculate，因此调用序列不同但最终DB仍相同。

## 4. 会改变分数的协议差异

### 4.1 STOP在独立评测step边界被覆盖（ERR-022）

最小反例：初始用户消息→助手回复→用户`###STOP###`，共3个Orchestrator step。`max_steps=3`时，原生step已置 `USER_STOP`，紧接着 `_check_termination()` 又以 `step_count >= max_steps` 覆盖成 `MAX_STEPS`。保存的最后消息确为STOP，但最终官方fallback为0。改用4步预算的独立诊断场景则为user_stop/1。

这与“模型没有结束”不同，是终止判定优先级。当前代码先检测正常停止，随后仍执行预算覆盖；即使结果DB正确，也可能不进入官方评分。

另一个默认规模反例：14次工具轮之后，第15次助手回复并由用户STOP。训练15轮允许取得这次STOP，得1；独立max_steps30停在助手回复后，来不及让用户回复，得0；即便临时诊断改为31，用户已STOP仍被边界覆盖；32才在此固定脚本下通过。**32只是反例对照，不能直接作为通用配置修复。**

### 4.2 轮数与step不能直接等同（ERR-023）

训练的工具返回批计入loop.user_turns，而Session的真实user_turns又是另一套计数。独立侧是角色转换次数。相同数值或“15×2=30”的换算无法证明拥有相同最后回复机会。

`final_tool_at_assistant_cap`显式把训练cap设为1、独立仍为30时，工具执行完全相同，但训练max_steps/0，独立继续回复并得1。这是不同预算的预期结果，用于检验诊断能捕捉协议差，而非额外代码故障。

正式对照应明确助手生成轮数、工具调用数、工具返回批数、用户消息数和总step分别是什么；不能只保存一个含义不一致的turns。

### 4.3 工具错误累计上限不同（归入ERR-023）

10次工具错误后再回复完成：训练继续并取得user_stop，独立侧在第10次错误后终止为too_many_errors，不再取得最终回复。环境没有发生重放差异，是运行协议不同。

为排除第一步发现的无DB变化任务评分盲区，本次额外用 **airline_803真实参考写入**复验：先完成目标DB，再产生10次工具错误。两侧完整工具返回和最终DB相同，训练1、独立0。另一个airline_803探针先完成参考写入，再在第15轮结束，也复现训练1、独立0。

### 4.4 模型可见observation不同（ERR-024）

使用上一轮真实airline_574的转机查询参数，原环境返回76,736字符：

- 训练模型看见65,553字符，即首尾合计65,536字符加17字符截断标记；中部11,200字符被移除。
- 独立评测的LLMAgent看见完整76,736字符。
- 两侧原始工具执行和DB相同。这次固定后续文本使分数相同，**没有证明该截断已造成模型失败**；但不能声称模型看见相同输入。

前轮的裸环境重放修正了审计方式，本轮进一步确认这种可见输入差异确实存在于两套真实harness之间。

### 4.5 上下文边界与失败回执

两个确定性边界测试分别覆盖“生成文本恰好达到response预算，工具尚未解析”和“工具已执行，返回恰好达到response预算”。训练分别在动作之前/之后终止为context_window_exceeded；独立循环没有相同的本地预算判断，固定服务回复下继续结束。真实服务溢出的异常分类仍需独立测试，不能由byte tokenizer推断模型服务行为。

未知`print_response`失败返回在两侧一致，且DB无变化；训练facts里的 `db_hash_after`仍为null，重现第一步缺口。应为分发失败补当前DB快照及dispatch分类，不把历史null解释为DB变化，也不伪造历史字段。

## 5. 核对汇总与证据

20个主场景加2个真实写库反例，共22个本地场景。9个刻意触发协议差的场景出现终止/分数差异；22个最终DB一致，21个完整工具执行序列一致。剩余1个是训练在生成上下文边界之前阻止工具执行，符合当前代码。

远程同样完成22个场景。下载后逐项核对summary、每次原始工具返回/错误/DB hash、训练可见工具facts（忽略随机call ID）、脚本与源码/输入身份，全部与本地一致。两侧各自运行前后登记的源码/输入hash均未变化。对应回执为`cross-host-verification.json`。

|证据|路径|
|---|---|
|本地20场景完整输入、输出及源码hash|`results/analysis/harness_step2_20260920/local-matrix1/`|
|本地2个真实写库反例|`results/analysis/harness_step2_20260920/local-write-probes1/`|
|远程20场景与2个真实写库反例|同目录`remote-matrix1/`、`remote-write-probes1/`，原始产物已下载|
|主脚本与补充脚本|同目录`audit_harness.py`、`audit_harness_extra.py`|
|实际协议配置快照|同目录`recorded-protocols.json`|
|本地定向回归|同目录`regression-local.log`：61 passed，51.62秒|
|远程无卡定向回归|同目录`regression-remote.log`：61 passed，344.65秒；与本地为同一组，不累加成122个独立测试|
|远程预检|同目录`remote-source-precheck.json`：23份相关源码/manifest预检一致|
|两地完整行为比对|同目录`cross-host-verification.json`：22个场景逐项一致|

本地首个试跑因为探针直接将未正规化的无参数工具schema送给veRL而失败；按已有 `build_config()` 生成生产工具schema后通过。失败脚本及日志留在local-attempt1，属于审计脚本接线问题，没有修改生产schema，也不将它登记成生产故障。

核对已完成，状态为“完成并记录协议差异”，不是“两个harness完全等价”。本地/远程回归包含原有多工具、会话释放、执行eligibility和独立评测异常记录检查；发现的新增边界仍待后续行为修订，不能用这61项通过掩盖反例。

复现使用现有CPU/无卡环境，并指定**不存在的新输出目录**，避免覆盖证据：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python results/analysis/harness_step2_20260920/audit_harness.py \
  --output results/analysis/harness_step2_20260920/<新主矩阵目录>
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python results/analysis/harness_step2_20260920/audit_harness_extra.py \
  --output results/analysis/harness_step2_20260920/<新真实写库探针目录>
```

本地使用`.venv-cpu/bin/python`；远程使用`/root/autodl-fs/tau3-core/environment/venvs/qwen35/bin/python`。脚本读取训练manifest和指定数据库，不读取final50；需要同目录的`long-observation-fixture.json`。本次未修改运行中实例状态。

## 6. 建议的修订顺序

1. **优先修终止优先级**：为正常STOP与预算/错误阈值同一步发生的情况定义明确规则，增加边界回归。新行为使用新harness版本；如改vendor，登记固定revision补丁。旧评测仍保持原协议结果。
2. **明确匹配评测协议**：保留原benchmark协议，同时提供显式的训练匹配协议，用一致的计数语义、错误上限和字符/token预算进行比较。两个协议分开命名和出表，不能暗改同名selection结果。
3. **统一可见observation与回执**：若研究需要训练/独立评测输入一致，采用共享、版本化的截断规则，并保存raw与visible hash、原始长度、裁剪方式；给未知工具/分发失败记录当前DB hash。
4. **再补真实token/服务边界**：复用实际Qwen tokenizer和服务配置验证模板及长度；需要模型推理/GPU时另列预算。随后才适合小任务过拟合和算法消融。

本次授权是核对，以上为待实施修订。本轮没有改变当前算法、harness行为或官方评分。现有历史结果继续按原协议解释；本次没有估计历史多少条分数会改变，也没有计算修订后的算法增幅。


## 后续修复（2026-09-20）

本报告保留修复前的原始反例和结论。后续新增显式 v2 控制协议，共享可见工具裁剪并补派发失败 DB 回执；本地/远程 CPU 回归、历史 legacy 兼容及跨主机对照已通过。实际 token/context 和模型效果仍未验证。详见[修复报告](harness_fix_20260920.md)。
