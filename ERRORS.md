# 错误与风险回顾索引

## 2026-09-20 本轮收尾状态

本轮所有有界GPU工程验收已通过；原失败尝试仍保留，未覆盖为成功。ERR-017的CPU/GPU参数化混跑已通过隔离复验，ERR-018端口预检修复有15项回归；三算法完整采样身份/生成logprob和活动文本权重、MT两组重放及GiGPO真实恢复均有回执。DF on真实两批均无零信号组，因此只证明开关/重放及正常更新路径，未证明真实剔除收益。最终范围与剩余独立研究见[接口报告](docs/architecture/interface_acceptance_20260919.md)。

下表历史状态按对应实验时间解读，不把后续通过回写成原失败不存在。

更新：2026-09-19。这里集中导航与结论，原始失败日志、修复细节仍保留在各报告；不复制一份失去同步的故障全文。修复实现、回归通过与新实验效果分别记录。

|编号|问题及影响|状态 / 回顾依据|
|---|---|---|
|ERR-001|实际执行与 gold replay 共用可变 DB，可能制造虚高成功率|已隔离并重评分；历史旧分数不能继续作为有效基线。[记录](docs/sft_comparison_20260908.md)|
|ERR-002|SFT 梯度累积重复除法、最佳 checkpoint 标记问题|已有修复及对应 SFT 对照；不能沿用文件名推断真实最佳模型。[记录](docs/sft_comparison_20260908.md)|
|ERR-003|工具 observation 截断到 256 字符，模型无法看到完整结果|提高上限并审计参考返回；reward=1 仍不保证回复业务文本完全正确。[记录](docs/observation_budget_20260909.md)|
|ERR-004|全词表 logprob OOM / 磁盘满中断|未完成更新不能记作“模型成功率 0”；分块与续训分别验收，tracking FINISHED 不是成功依据。[记录](docs/rl_validation_comparison_20260909.md)|
|ERR-005|Qwen3.5 导出重复 language_model 前缀，推理加载失败|修复格式导出并精确验证 724 个张量；训练、导出和评测状态分开。[记录](docs/post_rl_selection_20260914.md)|
|ERR-006|E2 独立评测一条上下文异常|原 239/240 仍无有效总排名；补测是单独新尝试，不覆盖原失败。[记录](docs/post_rl_selection_20260914.md)|
|ERR-007|过程奖励参数匹配未对齐实际转换；排序列表改变真实 DB 语义|使用新版本环境语义匹配，保留旧版本回放，不能用排序掩盖状态差异。[记录](docs/mt_gtpo_environment_fix_20260918.md)|
|ERR-008|提前终止的记录 0 分与已完成官方评分混淆，查询奖励跨任务相关性误导|读取器与分层诊断已修；新 split 奖励只有开发验证，未证明新模型收益。[读取修复](docs/mt_gtpo_reward_transfer_fix_20260918.md)、[split](docs/mt_gtpo_split_reward_revision_20260918.md)|
|ERR-009|重复 anchor 不等于跨轨迹有区分性的 GiGPO 信号；DF 前后位置会改变可用信号|历史审计事实已记录；因果收益与新行为修复待独立实验。[信号审计](docs/training_signal_audit_20260914.md)|
|ERR-010|语义状态与直接判等误合并/弃权|候选仍用于研究，不默认上线；双向一致不能保证语义正确。[记录](docs/semantic_pair_direct_v1_20260916.md)|
|ERR-011|rollout 原始 stop/length 被归并 completed|已增加原始原因及全算法事实；GiGPO 两步又覆盖 3 个真实 length，精确 token/span/mask 对齐。单轮 length 与轨迹终止分开；不宣称所有长度边界已覆盖。[新增证据](docs/architecture/gigpo_acceptance_20260919.md)|
|ERR-012|完整 process-turn 记录只走 MT-GTPO 分支；旧评测缺 task DB / harness source hash|新正式 runner 统一事实层，新评测自动记录任务/DB/源码哈希；旧记录保持原状，缺失身份仍披露。9月20日增强A三算法实际采样身份、完整生成logprob和活动文本权重检查全部通过；MT更新后的接口仍待B。[接口证据](docs/architecture/interface_acceptance_20260919.md)|
|ERR-013|续训曾只校验收据列出的文件，可能漏掉优化器/rank 状态；本批统计曾依赖全局 last_stats|统一完整恢复文件集合、路径/大小/marker 校验；trainer 使用本批显式 diagnostics。CPU 回归及GiGPO真实GPU step2→3恢复已通过，四rank状态/同云端run 1/2/3/数据16→24核对一致；不宣称与不中断训练逐位等价。[恢复证据](docs/architecture/interface_acceptance_20260919.md)|
|ERR-014|完整 GPU 工程验收受冷启动、更新与全状态保存耗时限制|9 月 19 日完成 A 和 B-GRPO 后，剩余约 34 分钟低于下一组约 56 分钟的实测推算，主动收尾；原控制器 SIGTERM/failed 回执保留，汇总明确为预算收尾，并非 GRPO 数值失败。其余 B/C/D 未完成，5 卡释放。[记录](docs/architecture/gpu_acceptance_20260919.md)|
|ERR-015|CI push 触发仅列 master/dev，遗漏实际默认分支 main|已补 main 与手动触发，远程 CPU 检查通过；未发布本批代码，尚无托管运行回执，不能视为 CI 已验收。[收尾记录](docs/architecture/cleanup_20260919.md)|
|ERR-016|单组验收控制器重复检查 step1 边界，可能在 step2 已开始后按“下一步预算不足”误停|改为每个完成边界只判断一次，9 项控制器 CPU 回归通过；运行中通过有 PID/创建时间记录的 CPU 守护接管监督，原训练未重启、未延长截止时间。两步完成，原执行快照和接管过程保留。[记录](docs/architecture/gigpo_acceptance_20260919.md)|
|ERR-017|GPU数值补验混入CPU参数化用例，GPU可见时CPU tensor误进Triton|原尝试保留failed（17通过/4 CPU失败）；隐藏CUDA的CPU完整回归已通过，GPU-only另目录17项全部通过。未修改生产计算逻辑。[记录](docs/architecture/interface_acceptance_20260919.md)|
|ERR-018|独立评测结束后增强A的端口bind预检瞬态失败，队列又因无阶段回执掩盖原错误|未开始新采样；保留原日志。预检使用SO_REUSEADDR但仍拒绝监听占用，15项控制器回归通过；新v3队列保留原退出码/日志入口。[记录](docs/architecture/interface_acceptance_20260919.md)|

## 新错误模板

- ID / 首次发现日期 / 实验 ID / 当前状态 / 负责人或后续任务。
- 触发条件与症状：task/trial/seed、step、算法/reward/harness 版本、错误日志和最小复现。
- 原因：已证实事实与假设分开；说明是算法、环境、数据、服务、数值、资源还是记录问题。
- 影响：哪些分数/检查点/实验结论失效，哪些可以继续使用；是否涉及数据曝光。
- 修复：变更源码、配置版本、兼容与旧结果保留方式。
- 验证：CPU / GPU 更新 / 续训 / 独立评测各自证据，未验证项明确列出。
- 关闭条件：对应风险的验收证据；不要用“单测通过”替代真实训练/推理结果。
