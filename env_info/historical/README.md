# 历史实验工具

这里保留从旧独立实验目录发现的有用控制/诊断脚本，供审计和后续复用；各目录名对应原实验来源。
新正式实验使用 `python -m tau3_grpo.training.rl.runner`，评测使用 `tau3_grpo.evaluation.run`；`scripts/` 提供兼容入口。`env_info/a800_20260912/` 的控制器用于旧实验复现、诊断和续训，不再是新正式实验的推荐入口。历史脚本可能仍包含当时的硬编码路径，执行前按 `docs/code_history.md` 中的新路径映射检查参数。
旧代码树与独立 source archive 均保存在 history/* Git 引用中，不应重新创建长期维护的代码副本。

2026-09-19 引用审计：33 个日期 profile／旧 run、launch 控制器中，31 个有直接引用；其余两个由 `tests/test_formal_components.py` 动态拼接引用。保留全部文件，不根据日期或零字面命中删除。范围与证据见 `docs/architecture/cleanup_20260919.md`。
