# 当前实验与开发入口

更新日期：2026-09-20。这里记录已核实的实验事实与开发状态，不作为实时进程看板。原 9 月 12 日进度原文保存在 [历史快照](CURRENT_EXPERIMENT_snapshot_20260912.md)。

## 当前协议

正式对照继续采用 Qwen3.5-4B `new-off` SFT、非 thinking、全语言参数 RL、train50、seed42、20 个外层 step，每 step 8×8 条候选轨迹，学习率 1e-6、KL 0.01。每 10 step 在 selection60×4 上评测，并保存完整续训检查点；核验新检查点后才轮换旧检查点，保留最新 1 份。SwanLab 每步在线，同一实验续训沿用原 run 和真实 step。

原“E0 六小时后确定共同 N”已得到 20 step 预算，不再按各算法速度重新确定 N。算法、奖励和 harness 可以继续修改；每次变化明确版本、对照和验证，不覆盖旧实验身份。官方 final50 只在模型名单及该次评测协议确定后使用。

## 已有结果与候选

|项目|已核实状态|证据|
|---|---|---|
|E0–E3|四组各完成 20 step；共 5,120 条训练候选记录|[信号审计](training_signal_audit_20260914.md)|
|独立 selection|SFT、E0、E1、E3 各 240 条完成；E2 原始 239 条完成 + 1 条异常，原汇总无效|[独立评测与补测规则](post_rl_selection_20260914.md)|
|MT-GTPO reference_write v3|有 1,280 条训练记录及 step10/20 训练内评测；不能与独立 selection 混为一套成绩|[运行记录](mt_gtpo_v3_execution_20260917.md)|
|paper_env_split_v3|历史数据开发验证通过；尚无新训练或独立效果结论|[奖励修订](mt_gtpo_split_reward_revision_20260918.md)|
|语义 anchors|v1 是历史正式路径；后续候选尚未证明在线 RL 收益|[直接判等审计](semantic_pair_direct_v1_20260916.md)|
|架构第一批|配置来源、公共 runner、adapter 分层、严格比较器与记录标准已实现；验证回执独立登记|[第一批交付](architecture/batch1_20260918.md)|
|架构后续批次|稳定组件配置、统一检查点、共享轨迹事实、评测身份、配方公共接口和本批统计已落地；CPU 交付完成，GPU 状态见下一行|[后续交付](architecture/batch2_20260918.md)|
|架构 GPU／接口验收|增强 A 三算法共 12 条；GRPO/GiGPO 两步更新、GiGPO 四 rank 恢复、724 张量导出与独立 8/8 评测、MT-GTPO DF off/on 各两步及重放全部通过。文本参数审计范围单独登记|[最终接口报告](architecture/interface_acceptance_20260919.md)|
|架构 CPU 收尾|兼容默认与 arm 统一、历史引用审计、回执草稿生成已交付；71 组命令相同、核心 364 passed；本批无 GPU、托管 CI 未运行|[收尾记录](architecture/cleanup_20260919.md)|
|MT-GTPO DF 对照工程验收|off/on 各 128 候选、两次非零更新、云端 1/2、过程奖励／优势／过滤重放通过。DF on 两批均有有效信号、剔除 0 组，不构成过滤增幅证据|[最终接口报告](architecture/interface_acceptance_20260919.md)|

历史数据已经被用于开发分析，不能重新命名成盲测或独立校准留出集。结果、错误、计划分别查看 [实验索引](../EXPERIMENTS.md)、[错误索引](../ERRORS.md)、[消融方案](architecture/ablation_plan_20260918.md)。

## 当前源码与入口

远程唯一主仓库 `/root/autodl-fs/tau3-core/code`；环境入口 `/root/autodl-fs/tau3-core/activate.sh`。本地编辑路径 `/Users/apple/Projects/program-llm/tau3_grpo_fix/code`。每次运行的 Git HEAD、dirty patch 和源码哈希共同标识实际源码。历史验收使用当时的源码快照；后续 Git 发布不改写历史身份。

- 配置驱动的一般入口：`python -m tau3_grpo.launch`；SFT 继续使用原入口。
- 新的单实验正式控制器：`python -m tau3_grpo.training.rl.runner`，支持 `grpo/tau_gigpo/mt_gtpo` 和 DF 开关。本轮已约定工程验收完成；不同模型／奖励配方／硬件与并行模式不能自动继承该结论。
- 旧 `scripts/train/rl/run_mt_gtpo_formal.py` 转发到同一实现；旧 E0–E3 队列保留历史预算发现逻辑，不作为新 20-step 对照的默认入口。
- 活动配置及候选状态：`configs/experiments/catalog.yaml`；正式基础：`configs/train/rl/formal50_a800.yaml`。
- 独立评测控制器：`python -m tau3_grpo.evaluation.controller`；旧 `env_info` 路径转发。它仍管理历史 E0–E3 队列，单模型评测使用下面的通用入口。
- 已有独立评测：`python -m tau3_grpo.evaluation.run`；离线重评分：`evaluation.rescore`；严格比较：`evaluation.compare`。
- 实验／错误草稿：`python -m tau3_grpo.experiments.review`，人工审核后归入现有索引，使用新的输出目录保留历史批注。
- `configs` 保存参数，`scripts` 保存入口，二者按用途形成相似目录是有意设计；公共实现放 `tau3_grpo`，不在两个目录复制。

本轮工程验收已结束。后续 GPU 实验仍需先明确目的、预算与停止条件；本次 Git 发布不启动实验。2026-09-20 按用户授权退役 GRPO 工程 step2、MT-GTPO DF off/on 各自 step2，共释放 151.90 GiB，清理后持久盘可用 219.31 GiB；五份正式 E0–E3/MT-v3 step20 检查点保留。容量是该次维护快照，不是实时读数。删除节点明确放弃精确续训，轨迹和验收证据继续保留。完成范围与正式研究待办见 [剩余工作](architecture/remaining_work_20260919.md)。
