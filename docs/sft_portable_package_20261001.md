# SFT 独立训练包（2026-10-01）

当前包：`data/sft/curriculum_500_dev150_portable_codex_20261001/`。
状态：`frozen_cpu_verified_gpu_not_started`。schema为`codex_reviewed_sft_package_v2`，
完整manifest SHA256为`32430e35051520fbe96bbf3e3b63f5c8b060a978edd701482c787ce052897683`。
小型身份清单见[包身份](sft500_dev150_portable_package_manifest_20261001.json)。

## 为什么迁移

原均衡v1包的启动校验读取2255个文件，其中2231个在87个历史结果目录。
它把开发过程的候选、审核过程和源码快照全部绑定到训练入口；仅部署最终数据目录无法启动。
用户明确希望原始数据与当前JSON足以支撑日常训练，因此把正式包与历史完整审计拆开。

## 当前需要保留什么

本包包含13个受保护文件及manifest，共24,106,299字节：

- `train.jsonl`、`validation.jsonl`、`train_A109.jsonl`、`train_B393.jsonl`；C500直接使用train500，去掉同字节副本。
- `review_index.json`：650条完整接受结论、五维检查、原生执行摘要、审核来源与历史证据哈希；逐字节保留原账本。
- `token_mask_audit.json`：逐样本实际token IDs与监督labels哈希及计数。
- `frozen_prompt_protocols.json`、`tool_config.yaml`：冻结提示协议和工具schema。
- `token_summary.json`、`stage_token_summary.json`：数据及阶段预算。
- `selection.json`、`rejected_index.json`：均衡dev的确定性选择与拒绝索引。
- `source_manifest.json`：原2255文件的身份快照，历史路径仅用于溯源，不再打开这些文件。

原始数据库保留在`data/raw/`，用于生成、重放和任务评测；不作为SFT梯度训练的启动依赖。
模型、真实tokenizer和环境仍为正常运行依赖，不放进数据包。
完整历史候选、失败尝试、完整DB回放、API回执及实验结果本次没有删除；
这些属于独立的研究归档。历史细节重审或完全重放仍需相应归档，而日常训练不需要它们。

## 校验语义

`tau3_grpo/data/compact_sft.py`导出时先执行原v1完整文件/审核/阶段校验，才创建独立v2目录。
新manifest只包含包内相对路径，禁止路径穿越及逃出目录的软链接；复制到任意位置都可验证。
运行时校验包内文件哈希、原manifest身份和接受账本的精确继承、用户/消息隔离、
监督契约、累计子集和实际token/labels。改写接受账本并仅重算当前哈希不能绕过原来源身份。

这是一次已验收结论的冻结迁移，不是新语义审核；训练启动不再逐次重查完整历史原始证据。
`source_manifest.json`中的路径和哈希是出处记录，而非远程待下载文件清单。
需要复核原生工具或完整状态时另用历史归档，不把摘要冒充完整历史重放能力。

## 配置与结果身份

新增`curriculum_codex_{A109,B393,C500}_portable_dev_1epoch.yaml`，均指向本包。
训练/验证消息、审核账本、token/mask及A/B累计文件逐字节不变；C500等于原train_C500。
model、LoRA、学习率、epoch、更新次数、提示与监督选项均与均衡v1配置一致。
A/B/C仍为14/50/63次更新；B/C需显式传入上阶段merged model并新建优化器。
仅更新数据/工具文件路径及独立输出目录，旧包和旧配置原样保留。

本地验证回执放在`results/maintenance/compact-sft-20261001/`：
真实包隔离复制、三阶段验证、全650条实际tokenizer/labels重渲染全部通过，相关CPU测试68项通过。
变更代码Ruff通过，全仓lint未引入新诊断（仍有381项已登记历史债务），vendor inventory通过。
本次不启动GPU、不进行API采样、不部署远程或推送；远程和GPU验收另行记录。
