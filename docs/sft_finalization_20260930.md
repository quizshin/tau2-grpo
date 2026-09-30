# SFT 500 / dev150 收尾核验（2026-09-30）

旧目录 `results/analysis/sft_deepseek_500_20260927/codex_only_final_20260929/`
的可训练声明已撤回；原 manifest、报告与审核索引归档在其
`superseded_20260930/` 子目录，对话文件保留。

从原候选及消息哈希恢复完整来源后，验证集有 39 条与训练集共用用户：
`james_adams_7486ed` 20 条、`steven_flores_e5cc0e` 19 条。
135 条验证记录没有来源用户，且监督信息缺少 `approved_assistant_v1`。
真实 `TrajectorySFTDataset` 拒绝该验证文件；旧 token 索引有 580 条实际长度为 null。
仅统计缺失字段、接受标签或 `user_stop` 不能证明可训练。

本轮新增 `tau3_grpo/data/reviewed_sft.py`，校验两份数据的精确条数、来源用户、
消息与监督位置哈希绑定、五维审核完成状态、原生执行证据、实际 token 数和
跨集合用户/对话去重。该函数只校验已有审核记录，不产生语义接受结论。
对应回归测试归入 CPU core。

当前收尾证据位于
`results/analysis/sft_deepseek_500_20260927/finalization_audit_20260930/`，
逐条语义决定位于 `finalization_reviews_20260930/`。
工具回放通过的候选仍标为 `mechanically_checked_semantic_pending`，必须另有
来源绑定的 Codex 审核决定。错误候选保留；重新生成使用独立目录及既有
150 CNY 预算账本，不调用 Kimi 或子代理。

冻结新版本前必须完成缺口替换、用户隔离、全量 tokenizer 与 loss mask 核验。
此处不声明新 500/150 已冻结；未启动 GPU、未部署服务器、未推送。
