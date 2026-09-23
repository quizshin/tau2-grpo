# MT-GTPO 调用局部信用候选设计

2026-09-22。状态：`development_checks_passed`。这是待否证的算法扩展，不是论文原公式，也不是已经验证提点的新版本。下文保留设计约定；纯数组核心、数据连接及显式训练分支已实现，本地CPU回归通过，尚未在GPU采样或训练。[两步GPU测试入口与实现状态](mt_gtpo_call_local_gpu_test_20260922.md)。

## 1. 本轮只回答一个问题

在保留后续过程奖励、终局信用和非调用文本优势的条件下，把工具调用 token 的即时信用从“整轮调用奖励之和”改成“本调用奖励相对于调用层基线”，是否减少混合轮的信用混用，并保留有价值行为的学习信号？

已有 `TAU3_RECORD_CALL_ATTRIBUTION=1` 解决了实际生成 token 与执行 call ID 的归属，尚未改变优势。具体见 [CPU 记录验证](tool_call_token_attribution_cpu_20260921.md)。已有 [时间与调用诊断](mt_credit_comparison_20260921.md) 中的 T 和 D 均不直接搬进训练；尤其 D 沿用整轮基线，不能成为本候选的调用基线。

候选暂命名 `call_local_v1`。固定奖励版本及权重、采样协议、gamma、lambda、原 UID 组和轮位置分组，DF off，不同时修改 SFT、时间折扣或奖励类别。split-v4 冻结配方尚未通过，开发计算使用的权重只能称诊断输入；本设计不绕过正式训练配方检查。

## 2. 原公式和可复用数据

对应 `tau3_grpo/algorithms/mt_gtpo.py::compute_mt_gtpo`。对同一原始采样 UID 下第 k 个 assistant 轮：

- i：轨迹；j：该轮已记录的工具调用。
- q_ikj：`process_reward_json.turn_records[k].tool_calls[j].reward`，取实际支付的 reward，不取 candidate_reward，不按错误标志重新造奖。
- R_ik：现有 `turn_rewards[k]`。本候选第一版仅支持调用轮的 sum 聚合，即 R_ik = sum_j q_ikj；其他聚合显式拒绝。
- G_ik = R_ik + gamma * G_i,k+1，虚拟末轮 G_i,K = O_i，完全沿用现有实现。
- H_ik = G_ik - R_ik，包括后续过程奖励及折扣终局结果。
- I_gk：原组中真实存在该轮且有生成 mask 的轨迹，不把缺失轮补零，不包含 padding。
- mu_R、mu_H、mu_G、sigma_G：I_gk 上的总体均值/总体标准差，ddof=0。mu_G = mu_R + mu_H。
- E_i：原 UID 内终局结果的现有组归一化优势。

当原位置组支持充分且 sigma_G > 0，令 s = sigma_G + eps：

```text
A_old(i,k) = (G_ik - mu_G) / s + lambda * E_i
           = (R_ik - mu_R) / s + (H_ik - mu_H) / s + lambda * E_i
```

原代码把 A_old 写进整轮生成区间。因此一个错误调用与两个正奖励调用同轮时，三个调用都获得同一个值。

## 3. 候选公式：只替换调用区间的即时项

定义 J_gk 为 I_gk 中有调用且调用奖励完整的轨迹。每条轨迹先计算调用奖励均值，再在轨迹间取均值：

```text
q_bar_ik = sum_j q_ikj / m_ik
b_call  = mean_{i in J_gk}(q_bar_ik)

A_call(i,k,j) = (q_ikj - b_call) / s
             + (H_ik - mu_H) / s
             + lambda * E_i
```

等价替换式：

```text
A_call = A_old + [(q_ikj - b_call) - (R_ik - mu_R)] / s
```

这是明确的启发式比较基线，不是无偏价值估计：同一轮位置的不同轨迹可能已处于不同状态。本轮不增加调用语义匹配或学习 critic，避免引入第二个待研究因素。

基线约定：

1. 每条轨迹在 b_call 中只有一票，避免一次生成十个调用的轨迹在统计基线中获得十倍权重。这不消除它在 token loss 中的长度影响。
2. 不按第 j 个调用跨轨迹硬配对，也不按 reward_type、error 或成功标签挑选比较组。
3. 无调用轮不作为一个零奖励调用混入 J；它仍参与原 I 的长程统计。
4. 奖励完整但 token 归属不可认证的调用轮仍参与 J 的奖励统计；只禁止写入调用级 token 优势。边界状态不用于选择更有利的比较样本。
5. `min_group_size` 沿用 2，同时要求 I 和 J 达到门槛；不把多个调用当成多个独立轨迹凑支持数。

仍使用原 sigma_G，避免同时改变整个标准化方案。这样显式的同轮兄弟调用奖励之和不再放进本调用的即时分子，但组均值、分母及 H 仍可能受到其他调用影响；本方案不是完全解耦，更不代表因果信用已解决。

### Token 写入规则

- 从现有 `compute_mt_gtpo` 的输出副本开始。原 mask=0、padding、非调用文本、空白、EOS 保持原值。
- 第一版以整轮为资格单位：只有该轮所有调用一对一、完整保留、已执行、ID 可核对，且奖励/事实完整时，才替换各工具块的 retained span；不在一个归属不完整的轮内挑选部分调用。
- 工具块包含生成的起止标签、名称和参数。完整执行但工具报错的调用可以参与，边界准确不等于业务正确。
- 不可认证、未执行、未闭合、裁剪不完整等合法状态保留整轮 A_old，记录 fallback reason，不清空 mask，不删样本。
- 缺失 observation 功能时：旧模式不要求它；显式开启候选却整行缺失事实，应报错，不能把未开启记录伪装成低覆盖率。
- ID 重复/不对应、跨轨迹 UID、reward 与 call ID 无法完整连接、token 身份不一致、区间重叠或含 observation：属于损坏输入，拒绝该 batch，而非静默回退。
- I/J 支持不足或 sigma_G=0：保留 A_old（其中 episode 项仍按旧逻辑保留），记录数值回退。近零但非零的 sigma_G 可能放大修正，是必须审计的风险，不能只靠 eps 宣称安全。

工具返回始终 mask=0。不另加调用损失、不改 PPO loss 聚合、不按调用长度重加权、不强制错误优势为负，也不额外全批重新归一化。候选并不保证 token 加权均值、RMS 或 PPO 梯度与原算法一致，这些是观测指标而非守恒假设。

若 I 中所有轮都只有一个调用、归属完整且 J=I，则 b_call=mu_R，候选与原优势相同（数值容差内）。这是重要的单调用回归条件，不适用于混合了无调用轮的组。

### 一个只用于解释公式的例子

两条构造轨迹均为单轮、终局结果 0，故 H=0、E=0；调用奖励分别为 `[-0.4, +0.6, +0.6]` 和 `[-0.4]`。这是合成的 v4 式奖励例子，不是新采样，也不把参考匹配自动称为业务正确。

原 R 为 `[+0.8, -0.4]`，mu_R=0.2、sigma_G=0.6；b_call 约为 -0.066667。忽略 eps 的微小影响：

|位置|原优势|候选优势|
|---|---:|---:|
|第一条的错误调用|-0.4 对应的整轮优势 +1.000|-0.556|
|第一条的两个正奖励调用|各 +1.000|各 +1.111|
|第二条的错误调用|-1.000|-0.556|
|第一条的非调用生成文本|+1.000|+1.000|

这个例子展示了可能的分离效果，也显示第二条错误调用的负信号变弱。若所有调用奖励相同且都为负，调用中心化项仍可能全为 0；若 H/终局项足够正，错误调用最终优势仍可为正。不可把例子的符号变化当作全量改进证据。

## 4. 模块与接口：先 CPU，后接训练

第一阶段新增纯数组模块 `tau3_grpo/algorithms/mt_gtpo_call_credit.py`，复用原核心，不改 parser 或奖励判定；暂不注册新训练 estimator。

拟定接口（不是当前已有 API）：

```python
def compute_mt_gtpo_call_credit(
    outcomes, uids, turn_rewards, turn_spans, response_mask,
    *, call_rewards, call_spans, eligible_turns,
    gamma=0.9, lambda_outcome=0.3, eps=1e-6, min_group_size=2,
):
    """Return token_advantages, original_token_returns, diagnostics.

    call_rewards[i][k]: complete finite q values for that turn.
    call_spans[i][k]: same-order exact retained spans for eligible turns;
        unavailable turns use None, never invented offsets.
    eligible_turns[i][k]: validated all-or-none attribution capability.
    """
```

核心内部仍检查 reward 聚合、shape、span 顺序/非重叠、包含于父轮、mask、有限值及支持数。它不读 JSON、tokenizer、文件或业务环境，也不能仅信任调用者传入的 eligibility 而跳过数值和区间检查。

数据连接放 `tau3_grpo/data/` 的专用校验函数：将 process reward 的稳定 `id` 与 `trajectory_facts_json` 的 `call_id` 一对一连接，核对 UID、轨迹身份、实际 response IDs/mask 和轮区间。不能以同名同参数或数组长度相同代替 ID 连接。原始输入不可原地修改。

只读分析入口拟为 `tau3_grpo/analysis/compare_call_credit.py`，按原完整组计算，再按官方成功/失败/预算未评分及行为切片观察。输入缺少可信调用区间时，仅输出事件级数值诊断；不重新 tokenize 历史文本伪造训练 mask，不宣称完成 token 级反事实。

后续才在 `integrations/verl/mt_gtpo.py` 中接入显式配置分支，保留 `adv_estimator=mt_gtpo` 的既有注册与训练框架。需要同时完成：

- 配置暂拟 `algorithm.mt_gtpo.credit_mode=turn_v1|call_local_v1`，默认 `turn_v1`。目前 `settings_from_config` 拒绝未知键，所以只新增 YAML 不会生效；适配器应先取出/验证模式，再将原超参数传给旧核心。
- `call_local_v1` 显式启用驱动与 Ray worker 的调用归属记录，不能只在本机 shell export。
- 原模式直接走旧函数，旧 replay/schema 不改写；候选使用新 replay 身份，保存最终逐 token 优势和分量。`turn_advantages` 在候选中只能标作 baseline，不能冒充混合后的全轮常数。
- 返回的 `token_returns` 暂保留原 G，字段明确标作原始回报。本项目 PPO actor 使用 advantages；接通时仍需检查全部 returns 消费者，尤其禁止未经验证地接 critic/value loss。
- DF 如后续开放，必须在候选逐 token 优势完成后判断信号；第一批实验保持 off。全零 mask、padding 与各 rank 数量沿用旧约定。
- `runner.resolve` 的续训检查补充信用版本相等要求，禁止同一个 run 从旧信用方式切换到候选；快照、recipe 与 replay 明确区分奖励版本和信用版本。

## 5. CPU 验收和否证条件

实现前固定以下检查，不在看过结果后只留下有利指标：

1. 旧模式精确复现原数组；输入不被修改；GRPO/GiGPO 路径不被开启候选污染。
2. 满足前述条件的全单调用组与旧算法等价；独立纯 Python 参考计算核对候选分量，不只用实现生成期望值。
3. 混合调用、全部错误、重复调用、纯文本轮、缺失后期轮、单轨迹支持、零/近零方差、全 padding 均有明确行为。
4. 非调用生成 token 保持原优势，observation/padding 始终为零；精确调用块之外没有改写。
5. 相同 tool name/args 但不同 ID 的调用保持独立；缺失/重复/错 UID、越界、损坏 mask/ID、非有限 reward 均拒绝。合法 ambiguous/incomplete/unavailable 有显式回退计数。
6. call reward 必须与原 turn reward 的 sum 聚合一致；mean 配置、非调用奖励被偷偷加到调用轮等情况拒绝。
7. 按原组先算，再按官方评分状态切片；预算回退 0 不改称官方失败，不删掉预算样本重算组。
8. 同时报出错误调用正优势比例/量、正奖励写入的正信号损失、恢复代理、全部 token 的 RMS/p95/max、修正/原优势比例和按调用长度/数量的分层。另报 I/J 支持和回退覆盖率，不能只展示混合轮成功例子。

非调用文本原优势不变只能说明本次更新输入未直接删掉其信号，不能保证模型更新后查询能力不退化；共享参数和新采样分布仍会改变行为。正奖励参考写入也不是任务最终成功的替代指标。

结构性检查任一失败，不能进入 GPU。全量开发数据中若错误信用改善伴随大量正奖励写入信号损失、修正项极端放大或极低覆盖率，应记录候选失败/不足，回到设计；不能临时加负号钳制、裁剪、语义分组或调奖励后仍称同一单因素实验。效果门槛及允许损失需在实现后的首次正式离线比较前另存 analysis spec，本文件不虚构已通过的收益阈值。

保存输入/配置哈希、原 UID 和轨迹身份、评分资格、每轮 R/H/G、原组统计、b_call、每调用 q/ID/span、原/候选优势、fallback reason 及 token 统计。每次分析使用新目录，原数据与旧报告不覆盖。

## 6. 与当前训练计划的衔接

9月22日另一个任务已记录 A45 GRPO 30 步完成，并新增八卡 5090 的 MT split-v4 组合配置，见 [最小接入说明](mt_gtpo_5090_minimal_20260922.md)。这份配置仍是原轮级 MT-GTPO，不包含本候选，不能因为名称含 v4 就认为使用了调用级信用。

后续研究对照应匹配同一 A45 起点、train50、30 个外层 step、8×8、token harness、温度和评测协议；以原轮级 MT 为直接消融基线，GRPO 为额外算法参照。不能从 GRPO step30 续训后宣称同起点对照，也不将历史 new-off/20 步结果混为控制组。

实施顺序：纯 CPU 核心与数据契约 → 旧数据可支持的事件级审计及精确边界 fixture 检验 → 明确预算的新模型采样覆盖率检查 → 小规模实际更新/保存恢复验收 → 冻结候选后的匹配训练和独立评测。GPU 阶段另列预算与停止条件，CPU 设计和测试不代替模型收益证据。
