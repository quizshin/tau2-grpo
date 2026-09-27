# 多轮原始 token 与上下文预算统一协议

日期：2026-09-20。用户授权实现多轮输入与上下文预算对齐。本轮仅 CPU 与配置检查，不启动 GPU、模型推理或训练，也不改写历史实验。

## 定位澄清：训练与评测不必完全相同

2026-09-20 用户追问后重新核对 Mercor 博客正文：Step 4 明确训练160k、评测256k，上下文提醒 nudge 用于训练而在消融评测关闭；Step 6 使用 OpenCode 检验跨 harness 泛化。Step 1 的 TITO 约束是 rollout 实际采样 token 与后续训练 token 对应，并不是要求独立评测复制全部训练设置。此前“对齐”的表述过宽，在此纠正。

本协议定位为 **与训练同条件的工程诊断选项**，不是所有评测必须遵守的唯一协议。正式效果比较应让共同 SFT 起点与 GRPO/GiGPO/GTPO 各候选使用同一套独立评测设置；该设置可以与训练的温度、上下文、工具交互实现不同。预算扩大或 harness 改动应明确登记，不能把它带来的增幅混称为算法增幅。跨 harness 泛化可另立实验，既不假定必须相同，也不假定任意差异一定无害。

新协议不覆盖原独立评测，不自动成为正式报告标准；正式 selection/final 协议选择与 GPU 验收仍是后续明确决策。无需为了让最终评测 token 逐值等于训练而增加额外 GPU 实验。

参考：Mercor，2026-09-01，Training frontier knowledge work agents: A 397B RL training guide with SkyRL，Step 1.3、4、6。正文地址保存在本轮 `mercor-reference.json`；本次已读正文，之前请求403的历史记录保留。

## 实现与版本

训练显式启用 `tau3_token_budget_v1`；独立评测显式选择 `tau3_eval_token_v4`。新评测直接复用 `verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop`、原生 qwen3_coder parser、官方完整 tool schema、Tau 交互与官方 verifier。评测仅将策略生成 transport 替换为严格的 token HTTP adapter；没有另外实现一份多轮状态机，不构造 optimizer、Ray cluster 或加载模型权重。

此前 v3 首轮已经对齐，但后续 Chat API 会重新渲染全部历史：空 thinking 块、紧凑 XML 的换行变化会导致真实 token 不同。v4 保留模型实际返回的 token ID，观察内容使用训练原生的追加模板。HTTP 使用 `/v1/completions` 传入整数 prompt，要求服务返回原始 prompt/output ID、逐 token logprob、finish_reason 和一致 usage。任何缺失、不一致、非有限 logprob 均使该 trial 失败；禁止 decode 后重编码冒充原 token，禁止自动补采样。

活动策略与模拟器温度继续默认 0.7；没有覆盖旧日期配方中保存的显式历史值。新评测对每个 trial 显式发送策略 seed，模拟器使用同一 trial seed；训练采样 RNG 与独立服务是否产生完全相同样本，不属于 CPU 输入等价证明。

## 统一预算与终止行为

预算来源：`configs/protocols/token_budget_v1.yaml`。训练启动参数与此声明不一致时拒绝启动该协议。

|项目|新训练与独立评测共用规则|
|---|---|
|初始策略 prompt|最多 8192 token|
|追加轨迹|最多 16384 token，含策略生成、工具返回、用户回复|
|策略上下文|最多 24576 token|
|每次策略生成|`min(1024, 剩余追加额度, 剩余上下文额度)`|
|用户模拟器|每次输出最多 1024，服务上下文 16384，non-thinking，top_p=1|
|轮次与工具投影|沿用 15 assistant / 15 observation 计数、65536 字符 middle 投影；新协议拒绝不匹配的覆盖|

完整工具批次在生成额度刚好耗尽时仍然执行，保存真实环境事件；完整用户 STOP 优先于观察溢出。观察内容在经过原有字符投影后，以 token 为单位检查能否完整追加，恰好装满允许追加，不够则整段不进入后续策略上下文，并以预算终止；原始工具/用户事件与“不保留”回执仍在。

length 结束的普通文本不冒充完整回答发给模拟器。半截 XML（含“完整调用＋半截下一标签”）禁止整批执行；平衡但原生 parser 未完整解析的调用批次也禁止部分执行。用户服务明确返回 ContextWindowExceededError 时记用户上下文预算终止，其他网络或服务异常保持未解决错误，不伪造成模型 0 分。

## 入口与兼容范围

- 公共正式 runner 在原参数上追加 `--token-protocol tau3_token_budget_v1`；支持 `grpo / tau_gigpo / mt_gtpo`，奖励/DF 选择仍由原参数管理。自动选完整 schema、保存协议来源/hash，续训拒绝更改 token 协议。
- 通用 launcher profile：`configs/train/rl/formal50_token_v4.yaml`。通过 e0/e2/mt_gtpo 三入口检查。
- 独立评测在原命令上追加 `--harness-protocol tau3_eval_token_v4`；tokenizer 来自 `--checkpoint` 的本地文件，只读取 tokenizer/config/template，不加载权重。评测前核对两服务公布的 max_model_len，并保存 tokenizer 三文件 SHA256。
- 经 `scripts/serve/policy.sh` 启动时设置 `TAU3_EVAL_HARNESS_PROTOCOL=tau3_eval_token_v4`；自动带 `--generation-config vllm --logprobs-mode processed_logprobs`。已有独立评测 controller 按协议传入 processed_logprobs。
- v4 目前只用于 fresh AReaL selection；官方 final50 在加载任务前拒绝。旧 legacy/v2/v3 和旧训练 profile 没有自动切换，不以新协议续跑旧协议的实验。
- 比较器要求 v4 的 tokenizer hash 身份完整且一致；历史评分没有重新计算。协议实现相同不代表真实服务 logprob 分布或模型权重已验收。

## 回执与验证

每条轨迹保留初始精确 token、逐次请求 hash/长度/额度、生成原始 token 与 finish_reason、每个观察 token 和 retained 标记、最终 hash/终止原因。完整 simulation、工具响应、官方分数、trajectory facts、trial/seed、计划分母及错误分别落盘。评测关闭训练锚点生成，不引入过程 shaping。

CPU 测试使用真实本地 Qwen tokenizer、真实官方工具与 verifier，策略/用户回复采用脚本替身。覆盖三算法四轮文本＋多工具 token 逐值一致、STOP/length/上下文边界、真实取消预订写入、截断批次无写入、并发 trial 与错误持久化、session 清理、HTTP 本地回环序列化、Hydra 三算法/续训身份及旧路径回归。

验收结果：本地与远程主要回归各 **256 passed**（136.62秒/665.49秒）；边界收尾后的新协议专项两端各 **42 passed**（本地60.26秒，远程436.29秒）。两组包含重叠用例，不相加。两端 lint 零新增（保留382条历史债务），vendor 清单通过；tokenizer三文件SHA256一致，27份本轮变更文件最终同步逐项核对。新测试已纳入CPU分层入口，diff空白/语法检查通过。实际日志与部署证据位于 `results/analysis/token_harness_20260920/`；远程对应 `results/maintenance/token-harness-20260920/`。远程运行显式隐藏 CUDA，离线依赖。按变更清单校验部署前 hash 并保留备份，没有整仓覆盖。

历史失败也保留：新增正式 runner 测试首次遗漏激活环境变量，5 项失败；补齐隔离 fixture 后 5 项通过。兼容回归发现旧测试直接构造原生 loop 而没有 config，6 项失败；将 evaluation-only 状态在构造时初始化，并在运行时使用兼容默认。以上不算 GPU 失败或训练结果。

## 尚未证明的部分

当前状态是共享执行实现及 CPU 验证，不是全链路 GPU 验收，也没有新训练效果结论。`token_context_equivalence=shared_implementation_live_service_pending`；HTTP logprob 标为 `server_reported_pending_live_attestation`，不会因启动参数写了 processed 就宣称数值等价。

仅当后续决定采用此候选协议时，可选择一轮验收： 2 卡（策略 1＋用户 1）、30 分钟硬上限、最多 2 个 selection 任务×2 trial；核对实际原始 ID、stop/EOS、上下文与 logprob 返回并将实测输出在共享循环回放，约束总预留不超过 1 GPU 小时。这不是继续旧协议训练的前置条件。该范围不含反向传播、参数更新或续训；超时/不兼容时保留失败，不自动追加预算。按 AGENTS.md，须用户决定后才执行。
