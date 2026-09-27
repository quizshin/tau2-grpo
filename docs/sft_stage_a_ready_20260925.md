# Stage A 数据与执行准备：2026-09-25

状态更新：用户已确认“可以启动”，Stage A 训练进程已启动；导出和交互评测待训练成功后接续。

费用更新：用户报告2026-09-25今日实际消费 **¥10.17（低峰日）**；下文¥23.34994保留为保守usage估算。尚未逐请求核销账单，不将两种口径混用。

## 已冻结的训练包

| 项目 | 实际结果 |
|---|---:|
| 完整训练对话 | 100 |
| AReaL 原始对话 / 历史真实执行补充 | 98 / 2 |
| 独立离线验证对话 | 52 |
| 训练输入 token / 参与 loss 的 token | 1,201,855 / 221,204 |
| 验证输入 token / 参与 loss 的 token | 682,134 / 138,546 |
| 当前运行时工具覆盖 | 14 / 14 |
| 源对话无独立 answer 的开场白 | 98 条，保留上下文、屏蔽 loss |
| 源目标缺失的其他 assistant 轮次 | 最终所选源对话中没有 |

本地与远程相对路径：`data/sft/staged_v2_A100_20260925/`。包括 `train.jsonl`、`validation.jsonl`、训练入口重新处理后的两个 `*_effective.jsonl` 和 `audit.json`。每个样本通过 `supervision.version=approved_assistant_v1`、`message_indices` 指明可监督位置。新配置启用 `require_approved_targets=true`，缺失/空/非法 mask 会报错。历史配置的默认行为保留。

一段完整对话只占一个样本，历史、用户和工具返回保留，loss 只覆盖批准的 assistant 输出及其结束标记；不监督 system/user/tool 或私有 reasoning。microbatch=1 时，现有 trainer 先对该对话的有效标签 token 求均值，再梯度累积，不是按轨迹长度直接给长对话更大权重。

训练文件 SHA256：`b22898c71a82910e58841ac62d7d548218d8646a3b7a806b424671d6c6a12b35`。
验证文件 SHA256：`e815cf638932175f4f914a1af656a8ee57c3b3a0ae0f071c2ef9c05b1b4708db`。

## 如何筛选

1. 沿用已核验 SHA256 的 AReaL 源文件和逐轮前缀/answer 对齐盘点；999 条源对话中，842 条通过本轮源标签、运行时 schema、工具回执顺序和留出集词面检查。
2. 在 judge 前预留 60 条未出现在历史训练集、与历史训练及当前候选无高词面相似度的离线 dev。3 条超过 24,576 token，保留排除记录，未截断；剩余 57 条中 5 条审查未解决，最终 dev 为 52。比原计划 60–80 少，此次先明确使用 52 条；不能把它说成 60 条或交互评测任务。
3. 从训练侧选 180 条源候选，13 条超长，得到 167 条；再核验 3 条历史补充的训练 task/DB 身份、工具执行回执与消息哈希，共审查 170 条训练候选。未强制保留旧 anchor。
4. DeepSeek 两遍审查加明确的格式/证据身份校验后，151 条训练候选满足暂定筛选条件，19 条留待复核。按合法工具覆盖和业务模式选 100 条；未为稀有工具虚构调用或工具返回。
5. 补充对话 `authored_airline_618`、`authored_airline_444` 来自已有 RL **train** task，保存了真实执行回执。它们与未来 RL 训练池可以重叠。原始 98 条源对话仍缺乏可靠 task/DB 对应，不能称为全部通过官方 replay。

留出保护覆盖 selection60、当前 reserve888、官方 Airline 任务及历史离线 dev。reserve120 尚未另行冻结，因此本轮保守保护全部 reserve888。词面阈值 0.88、来源 ID 和文件身份检查已有证据；**这不是完整语义无泄漏证明**。未使用留出集答案生成示范。

## 实际分布与不足

源模式：单目标正向 26，单目标拒绝/不可行 12，双目标 35，三目标 25，真实执行补充 2。源模式是任务结构标签，不等同于精确能力/难度评级。部分 soft quota 未达到，没有复制样本凑数。

| 工具 | 含该调用的独立对话数 |
|---|---:|
| get_user_details | 99 |
| get_reservation_details | 92 |
| search_direct_flight | 61 |
| book_reservation | 38 |
| get_flight_status | 36 |
| update_reservation_flights | 28 |
| update_reservation_baggages | 19 |
| cancel_reservation | 17 |
| update_reservation_passengers | 14 |
| transfer_to_human_agents | 11 |
| list_all_airports | 3 |
| search_onestop_flight | 3 |
| send_certificate | 2 |
| calculate | 1 |

稀有工具的情境多样性仍不足，尤其 calculate/send_certificate；B/C 阶段应在真实可执行训练任务中补充。Judge 给出 93 medium、7 easy，分辨率有限，不能据此宣称难度已经精确校准。

## Judge、确定性检查与费用

- 原 v1 的 80 条试审保留。v2 修正已说明取消原因、按每位乘客计算行李额度等规则，并给出完整 JSON 输出形状。
- 12 条控制复核全部返回合格结构；其中两条明确构造的错误（错误行李报价、撤回授权后取消）均被识别。控制样本仅用于审计，未进入训练。
- 真实候选与 dev 共 227 条：223 条审查结构/语义契约通过，4 条因“建议 keep 但仍存在 unknown/violated requirement”未评分；203 keep、15 repair、5 review。这些是审查建议，没有当作官方成功标签。
- 独立核对了 211 次有可见前置状态的行李操作，210 次付费行李数量吻合，1 次不吻合：`airline_dialog_171` 把 silver/economy/2 passengers/4 total bags 设为 1 个收费行李，正确值应为 0。该条未进入训练。
- Judge 仍有争议：`airline_dialog_822/817` 的“额外行李”解释，以及 `authored_airline_1036` 的 frozen rubric 把用户延误投诉当取消投诉，都需要证据复核。当前暂缓选入，不把它们永久标成错误数据。v2 没有人类金标或独立模型校准。
- 累计 **639 次请求，保守费用 ¥23.34994 / ¥100**；639 次都有 usage，没有悬而未决的费用预留。使用 `deepseek-flash`，延续原账本，计价按之前核验的峰时未命中缓存上界，不是发票金额。本次新增 478 次、约 ¥17.64070。

原始 API 请求/响应及调用回执全部留在远程 `results/analysis/deepseek_rubric_v2_*_20260925/calls/`。本地同步了复核记录、manifest、summary 和累计 ledger；v1 后的费用快照单独保留为 `budget_after_v1.json`。

## 已验证范围及一个模板风险

- 远程既有 qwen35 环境，`CUDA_VISIBLE_DEVICES=""`：四组相关测试 **69 项通过**。之后对稳定排序/政策身份的细修再次通过数据测试。变更文件的 ruff 检查通过。
- 真实 Qwen3.5 tokenizer 验证了：完整原生 token 不变、指定 assistant 才有标签、用户/工具/系统不监督、多调用及 EOS、无效/重复/缺失目标拒绝、effective JSONL 保留 mask。
- 100 train + 52 dev 已由真实 `TrajectorySFTDataset` 完整加载和重新渲染。未执行任何 GPU 前向、更新或服务启动。
- **原生模板差异**：8 条候选、126 个 assistant 位置中，13 个训练前缀与逐轮推理前缀完全一致，其余因 Qwen3.5 随最后 user 位置变化的空 thinking scaffolding 不同。按照用户要求保留完整对话训练和原生模板；这是一项记录在案的建模差异，不能断言是旧 SFT 退化的原因，也不能声称仅 mask 修复就实现逐轮前缀完全一致。

## Stage A 已批准的执行预算

配置：`configs/train/sft/staged_v2_A100.yaml`。

- 从 **Qwen3.5-4B Base** 开始，LoRA r16/alpha32/dropout0.05，lr **3e-5**，完整对话 **1 epoch**，microbatch1 × accumulation8，**13 次更新**。
- 离线验证仅作诊断，`load_best_model_at_end=false`，保留这一轮最终 checkpoint。不能用 loss 下降判定提点。
- 训练预计 45–90 分钟，1 张 A800；含离线验证，训练阶段上限暂定 90 分钟，未完成则停止并记录，不能当作完整 1 epoch。
- 合并导出后，独立交互评测 **selection60 × 4 = 240 条**。GPU0 policy、GPU1 原有 Qwen3.8-27B-AWQ-INT4 user simulator，预计 1–2 小时，评测阶段预算上限 3 小时。总上限约 **7.5 A800 GPU 小时**；不含另开 B/C/RL。
- 对齐已完成 Base：`tau3_eval_legacy_v1`、seed42、并发4、policy/user temperature0.7、max_steps30、max_errors10、相同 prompt/task/DB 和实际服务参数。当前 harness 源文件对照未发现变化；仍需在服务启动时核验模型/tokenizer/参数身份。若不一致，先报告，不能拿不同协议成绩硬比较。
- 主指标：官方 verifier 的 pass@1；同时报告 pass@4、pass^4、任务配对区间和严重错误。原 Base 131/240=54.58%；开发门槛暂定至少139/240=57.92%，但8条成功的差距本身不证明统计显著或跨 seed 稳定。
- 停止条件：mask/数据身份不符、OOM/NaN、服务身份或协议不符、评分基础设施异常、超出预算。失败记录保留；不静默删掉后重新计算高分，不自动扩展到 RL。

用户已明确批准本次预算。训练 PID=29641；接续控制器 PID=29971。训练成功并核验13次更新/完整epoch/effective mask后，自动CPU导出，再启动两卡selection60×4；接续状态见远程 `results/runs/sft_staged_v2_A100/seed42/continuation.json`。新增接续门禁的10项CPU数据测试通过；Base权重hash及105个评测相关源文件hash已匹配。B250/C400和RL尚未启动。

### 本次已启动的训练命令

```bash
cd /root/autodl-fs/tau3-core/code
CUDA_VISIBLE_DEVICES=0 /root/autodl-fs/tau3-core/environment/venvs/qwen35/bin/python \
  -m tau3_grpo.training.sft.train \
  --config configs/train/sft/staged_v2_A100.yaml \
  --model-name-or-path models/Qwen3.5-4B
```

训练产物目标：`results/runs/sft_staged_v2_A100/seed42/adapter`。首次启动前保存本次代码补丁、配置和数据 hash、实际命令及服务/硬件身份。

实际追踪：训练入口沿用默认SwanLab在线记录，run ID=`a9cd50tc`；运行命令没有通过统一launcher注入配置中的`TAU3_TRACKING_BACKEND=none`。这项运行差异已记录，不影响训练超参数和数据。
