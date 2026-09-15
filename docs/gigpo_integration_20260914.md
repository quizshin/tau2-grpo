# GiGPO 归一化接线与信号审计（2026-09-14）

后续核验见 [原论文对照与语义规范化 v3](gigpo_paper_semantic_audit_20260914.md)：Fnorm=1 和同轨迹重复入组均为论文/官方允许行为；新 grpo episode + mean step 是用于控制归一化因素的混合变体，不是原版唯一正确实现。

本轮完成无卡实现与验证，没有启动 RL、vLLM、用户模拟器或新的 SwanLab run。已有四组 20-step 的成绩不变。本地与服务器均运行 CPU 检查。

## 已接入的实现

### 显式选择 episode 优势定义

`algorithm.gigpo.episode_normalization` 支持：

- `legacy_mean`：缺省值，保持旧 Tau-GiGPO 的 episode 项与 returns 定义，旧配置续训行为不变。
- `grpo`：直接调用 veRL 原有 GRPO 实现，遵守同一 `norm_adv_by_std_in_grpo` 参数，再加 `omega × A_step`；returns 与 outcome-only GRPO 的约定一致。未知配置值直接报错。

新模式在 omega=0 时，advantages 与 returns 都与标准 GRPO 一致，包括 singleton、空 mask、float32/float64 和关闭标准差归一化。omega>0 的混合组还用手算折扣优势核验。

旧实验配置未切换。新候选配置为 `configs/train/rl/qwen35_4b_full_a800_gigpo_audit_20260914.yaml`，显式启用 grpo 归一化、独立结果目录和审计，设为从 SFT 初始化的 2-step 检查、禁用评测和检查点保存。它只是供后续 GPU 冒烟使用的配置，本轮没有执行。若用于 E3，必须单独指定 RESULTS_DIR 与 SwanLab 实验名；它不是正式训练配置。

这次保留 gamma、omega、原锚点及 DF 策略，以便下一次对照单独判断 episode 尺度差异。旧检查点直接续训不等同于从统一 SFT 开始的单因素对照。

### 真实训练 batch 的审计接线

当 `TAU3_GRPO_SIGNAL_AUDIT_DIR` 非空且 estimator 为 tau_gigpo 时，trainer driver 在 DF 前复制策略 mask，在优势计算后写审计文件。未启用时不复制 mask、不写文件，GRPO 路径不采集。

每份审计包含 update、uid/task/session 元信息（字段存在时）、padding 标记、精确 token 奖励稀疏项、过滤前后 mask 区间、原始及解析后锚点/span、初始步骤标记、步优势、同锚点的不同轨迹数、算法配置与源码 hash，以及实际合并优势的 SHA256。每次重试生成独立文件名，先写临时文件再原子替换；审计失败会报错，不把缺失证据静默当成功。

原始 token 奖励与二值 mask 可以无损恢复；不存 optimizer/model 张量或明文完整提示词。浮点跨硬件回放可能有舍入差异，CPU 测试验证的是同 dtype/同实现的精确回放。审计没有改变优化器 batch。

新增 `gigpo_signal/*` 指标：

- 过滤前后非零步级位置及 token 数、被屏蔽的数量。
- 非初始且跨轨迹的非零步级位置、以全部非初始步骤为分母的覆盖率。
- omega 加权的绝对步优势总量与实际合并优势总量。
- 原始步级项非零计数和实际应用计数分开；omega=0 时应用计数、加权总量为 0。

这些是优势统计，不是梯度贡献、因果收益或成功率。原始 repeated-anchor 指标仍兼容保留，不应再将它解释为有效信号覆盖率。SwanLab 为新指标定义 `trainer/global_step` 横轴；本轮只验证接线，没有在线新曲线。

### 结构化真实对话核验

读取 E2/E3 已有 selection 的 **479 条结构化对话、5,138 个 assistant 决策位置**；E2 是原始 239 条成功写出的评测记录，单条补测没有混入。没有读取官方最终评测集。

人工结合上下文复核的实际反例：

|记录|用户真实含义|旧规则问题|
|---|---|---|
|E2 airline_1035 / trial 0 / message 9|“Yes, please go ahead and search for the economy price…”；前文只询问是否查询价格|同意读操作就锁存全局 user_confirmed，不能表示确认降舱变更|
|E2 airline_1060 / trial 1 / message 13|“Wait… adjust the refund method before we proceed?”；要求先纠正退款银行卡|包含 proceed 仍被视为确认，且之前的全局确认不会作废|
|E2 airline_1106 / trial 2 / message 12|“Once you confirm … I’ll approve the change”；身份和行李尚待核验|要求代理核验并保留未来批准，不是已经批准当前方案|

这里把同一现有确认函数应用于保存的结构化 selection 消息，不表示恢复了当年的训练锚点，也不表示一定造成了工具误操作。确认标记用于优势分组，不是环境的动作权限开关。

读工具核验发现：去掉 DB 后的旧特征投影有 1,785 个桶，其中 320 个包含多种读账本，加入读账本后共分成 2,197 份。这说明读取内容会进一步区分状态，但由于没有中间 DB hash，**不能称为 320 个实际完整锚点碰撞，也不能据此计算线上覆盖率**。

第一版筛查把 `###STOP###` 也命中了 stop 暂停关键词；核验上下文后已将该控制消息单独统计，最终结果使用 `dialogues_v3.json`。所有问号、否定词、身份词筛查数只是待复核候选，不能当作语义误判率。

本轮没有把样例级 ScopedConfirmation 或完整读历史 hash 接入生产锚点。真实对话需要区分查询许可、修改许可、带条件许可和方案变化；样例接口目前没有自动提取完整待确认方案的能力，直接替换会引入新的误分组风险。

## 验证证据

- 服务器主回归：193 passed，两个第三方弃用 warning；涵盖算法、真实 compute_advantage/filter 接口、padding、mask、回放、配置、SwanLab 接线及 stub 并发隔离，不涉及模型前后向。
- 最后补充控制消息排除测试，单独运行；没有修改生产算法。
- 80 个历史奖励/task 批次调用新的生产 estimator，在 omega=0 时 advantages 与 returns 最大差均为 0。mask/span 为人工构造，不能称原始在线 token batch 回放。
- 结果目录：`results/analysis/gigpo_integration_20260914/`，包含 execution.json、pytest.log、production_normalization.json、dialogues_v3.json 与本报告副本。
- 实现入口：`tau3_grpo/algorithms/verl_estimator.py`、`tau3_grpo/tracking/signal_audit.py`、`verl/verl/trainer/ppo/ray_trainer.py`。
- 复现脚本：`env_info/a800_20260912/validate_gigpo_integration.py` 和 `dialogue_anchor_audit.py`；接口测试：`tests/test_gigpo_signal_integration.py`。

## 后续建议

无卡阶段已经有可供真实更新使用的归一化与审计接线。若继续完善锚点，先为上述真实对话建立逐动作、逐对象的人工标签，验证抽取的动作、订单、金额、支付方式、前置条件及确认的撤回/作废；不能只判断整句是否含 yes。

开 GPU 后，优先做 1–2 step 检查，查看真实多轮 rollout 的审计文件，量化跨轨迹、非初始非零信号以及 DF 屏蔽量，再决定是否改变过滤规则和是否值得正式重训。归一化与审计正确不代表 GiGPO 将提高成功率；是否改善最终任务表现仍需要预算匹配的训练与独立评测。
