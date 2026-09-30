# SFT 500 / dev150 收尾核验（2026-09-30）

状态：`frozen_cpu_verified_gpu_not_started`。该次收尾包为
`data/sft/curriculum_500_dev150_codex_20260930/`，完整 manifest 及逐文件 SHA256 已核验。
Git 内的小型身份清单见 [数据包清单](sft500_dev150_package_manifest_20260930.json)。

|集合|对话数|基础 / 多约束 / 多步策略|独立用户|
|---|---:|---|---:|
|train|500|109 / 284 / 107|67|
|dev|150|87 / 51 / 12|7|

累计子集 A109 / B393 / C500 精确为 109 / 393 / 500 条。
train/dev 用户交集和完整消息重复均为 0；dev 有 150 个不同目标。
dev 能力分布不均衡，不能称为每类 50 条或全面无泄漏的盲测；评测须同时报告能力切片。
旧 selection60 的开发曝光单独保留，官方 final50 不参与本轮选型。

## 审核、来源与实际渲染

训练数据由 70 条历史数据和 430 条 DeepSeek 新交互组成。
430 条 DeepSeek 初始目标文本互不重复。
原训练集 29 条语义不合格对话已用同任务的新原生交互替换。
650 条最终记录全部有消息及监督位置哈希绑定的 Codex 接受证据：
200 条精确继承历史 Codex 审核，其余 450 条在本轮直接审核。
五维为目标范围、证据、政策、算术和完成情况；机械回放本身不产生接受结论。
原生证据中 70 条为历史完整回执及独立参考执行继承，580 条为本轮源候选重放；
两种来源明确区分，未伪称全部重新执行。最终包没有 Kimi 接受替代 Codex 审核。

全部 650 条通过本地 Qwen3.5-4B 实际 tokenizer、正式完整工具 schema、
原生 chat template 和 `TrajectorySFTDataset`。保留逐条 input IDs / loss labels SHA256、
完整 token 数、显式 `approved_assistant_v1` 监督位置，非 assistant 标签为 ignore。
上下文上限 24,576，未以截断制造通过；精确统计保存在 `token_summary.json`。
650 条中最长实际上下文 15,280 token。
train 总 token 3,908,232、监督 token 354,489；dev 总 token 1,029,991、监督 token 75,693。
本地保有 tokenizer 资产，基座权重身份仍须在 GPU 执行前核实。
原始 system prompt 按哈希冻结；工具参数只做无损 JSON 解码，业务政策正文保持原文。

完整原始候选、失败、拒绝、API 回执和独立审核文件保留在
`results/analysis/sft_deepseek_500_20260927/` 及相应独立生成目录。
第 05 批因 Codex turn 中断失去进程，部分候选保留为未完成；相关已收到的 API 调用已结算，
补生成使用新目录和唯一请求 ID，没有覆盖旧对话或重复未知用量调用。

## 撤回旧包

旧 `results/analysis/sft_deepseek_500_20260927/codex_only_final_20260929/`
的 ready 声明已撤回，旧对话未改写，原声明归档在 `superseded_20260930/`。
恢复来源后旧 dev 有 39 条与训练用户重叠、135 条缺少用户及监督版本，
旧 token 索引有 580 条长度为 null。旧接受标签和 `user_stop` 不能证明可训练。
新包使用独立路径，旧 manifest 保持 false 并指向当前冻结版本。

## 配置与验证边界

三份 `configs/train/sft/curriculum_codex_{A109,B393,C500}_1epoch.yaml`
已指向当前数据及冻结 prompt 映射。每阶段 1 epoch、effective batch 8、
LoRA16、LR 3e-5、cosine、seed42，分别 14 / 50 / 63 次更新。
三个阶段共计划消费 1,002 条对话（含累计重复）和 127 次更新；
逐阶段实际监督 token 预算见 `stage_token_summary.json`。
A 从 Qwen3.5-4B 开始；B/C 必须显式传入上一阶段 merged model，
属于新优化器的阶段续训，不能声称精确 optimizer resume。
当前配置使用阶段末 checkpoint；未来任务表现选型须另行冻结协议。
硬件 profile 为 2×A800，SFT 入口使用其中 1 张策略卡。

训练入口在模型加载前核验完整包、源候选、审核证据及文件 SHA；
实际 tokenizer 渲染还必须匹配冻结的 IDs 和 loss labels。
本地相关 CPU 回归 57 项通过；三阶段数据/渲染与 launcher dry-run 回执单独保存。
这不代表远程集成、GPU 更新、导出模型或任务表现验收。
未启动 GPU、未部署服务器、未推送。

共享 API 上限 150 CNY；本轮收尾时账本占用约 78.00 CNY，
余额约 72.00 CNY。其中历史用户确认实付 13.93，
其余为 usage 估算或保守预留，不是供应商实付账单。未调用 Kimi 或子代理。


后续：保持本包及训练文件不变，已另行补齐均衡dev150（50/50/50）；当前主线转为[均衡收尾记录](sft_balanced_dev_finalization_20260930.md)中的独立数据包与配置。上面的87/51/12为本历史版本实际分布。
