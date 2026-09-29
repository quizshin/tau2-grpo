# 评测

区分代码测试与模型评测：core/benchmark/verl通过不代表模型任务成功率提高。

- dev37及审核冻结后的dev150：用于课程开发和阶段选择。
- selection60：已有开发曝光的历史回归面板，不重新命名为盲测。
- 官方final50：模型与协议冻结后才用于最终评测，不参与训练或选型。

同起点SFT、GRPO、ARPO、MT-GTPO比较必须一致使用任务、用户模拟器、奖励、温度、预算及终止协议。完整保留异常和分母，不截取成功子集。

实现入口：`tau3_grpo.evaluation.run`、`tau3_grpo.evaluation.compare`；统计与冻结规则见 [独立评测](independent_evaluation.md)。
