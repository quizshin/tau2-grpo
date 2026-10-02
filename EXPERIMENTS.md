# 实验索引

当前主线及历史结论见 [研究摘要](docs/research-summary.md)，入口见 [RL](docs/rl.md) 与 [SFT](docs/sft.md)。

旧实验逐次日志已保留在[归档标签的完整索引](https://github.com/quizshin/tau2-grpo/blob/archive/pre-a800-focus-20260929/EXPERIMENTS.md)。新实验按独立run记录协议、身份和回执，再在此添加简短索引；不将计划写为结果。

2026-10-02：接入本地 `a141dd4` 的数据包与代码整理。当前train500/dev150 portable包及
A109/B393/C500配方已冻结，GPU未开始；详见[SFT](docs/sft.md)与[包身份](docs/sft500_dev150_portable_package_manifest_20261001.json)。
原开发提交保留在Git历史，发布树只保留当前指南及必要技术规格。

2026-10-02：新增五维轨迹诊断链路 `airline_five_dimension_v1`，任务细则事前冻结，
模型输出事后独立审查，成功率和训练奖励保持原协议。历史rubric pilot归档兼容。
CPU/模拟裁判验证不等于GPU模型评测或真实裁判校准；使用与待完成的数据边界见[评测](docs/evaluation.md)。
