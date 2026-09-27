# 当前实验与开发入口

更新日期：2026-09-20。以下为已核验状态，旧尝试历史见研究报告。

## 四模型统一 selection 已完成

北京时间23:00:08完成；四组各240条，0执行异常/0缺失，16条预检单列。协议为tau3_eval_legacy_v1、完成STOP优先修订、两侧温度0.7、30 Orchestrator step、10工具错误、raw返回、不设1024请求限制。

Base/SFT/GRPO/MT pass@1为50.00%/45.00%/41.25%/39.58%。MT为reference_write/v3 DF off step20，GRPO为E0 step20。SFT−Base、GRPO−SFT、MT−GRPO差值的95%任务配对区间均跨0；本轮未看到提点，不作跨seed因果结论。历史0.4/1.0成绩不可混合。

完整960条轨迹、60任务表、538条失败/预算索引与10条定向案例复核已留存。下一步只建议研究写操作业务参数与目标状态一致性，先复用历史训练buffer离线核对，不把selection参考直接用于训练；未启动训练/新候选/final50。并行共享服务的校准采样属于其他用户授权任务，结果单独记录。

含准备2.283小时、11.416预留GPUh；本次服务和后代已清理，GPU计算进程为空。051实例没有关机，平台计费仍可能继续。本任务收尾后暂停heartbeat。

[完整结果与案例](../results/analysis/grpo_improvement_evidence_20260920/execution-stop-fixed-20260920/final-analysis/report.md)、[研究过程](grpo_improvement_evidence_20260920.md)。

## 2026-09-21 badcase 诊断已完成

全量960条CPU重放复现官方结果，未改变分数。主要观察是写操作终态错误；已分别记录支付映射、额外写入、模拟用户偏离和乘客数组顺序差异。历史训练信号复核支持下一步只研究写操作参数/目标一致性反馈；未启动新实验。详见[诊断报告](selection_badcase_analysis_20260921.md)。

## 2026-09-21 训练信号链审计完成

已按MT→SFT→GRPO核对：MT原始奖励与优势精确复现，发现同轮混合信用/相对未来贡献风险；SFT实际输入和渲染一致，E0标量优势与20步日志一致。未发现已证实的反号或mask实现故障。替代前一轮直接增加状态奖励的初步建议，优先对MT信用粒度做离线单因素设计，再处理SFT数据缺陷。未改公式、启动采样或训练。[完整报告](training_credit_chain_20260921.md)。

## 当前协议

正式对照继续采用 Qwen3.5-4B `new-off` SFT、非 thinking、全语言参数 RL、train50、seed42、20 个外层 step，每 step 8×8 条候选轨迹，学习率 1e-6、KL 0.01。每 10 step 在 selection60×4 上评测，并保存完整续训检查点；核验新检查点后才轮换旧检查点，保留最新 1 份。SwanLab 每步在线，同一实验续训沿用原 run 和真实 step。

原“E0 六小时后确定共同 N”已得到 20 step 预算，不再按各算法速度重新确定 N。算法、奖励和 harness 可以继续修改；每次变化明确版本、对照和验证，不覆盖旧实验身份。官方 final50 只在模型名单及该次评测协议确定后使用。

## 已有结果与候选

|项目|已核实状态|证据|
|---|---|---|
|预算截断状态审计|307条重放终态/调用数一致；4条截断时达标、3条曾一致后偏离、167条变化但未达标、133条未变化；未改官方分数或训练|[状态审计](mt_gtpo_truncation_state_20260920.md)|
|E0–E3|四组各完成 20 step；共 5,120 条训练候选记录|[信号审计](training_signal_audit_20260914.md)|
|独立 selection|SFT、E0、E1、E3 各 240 条完成；E2 原始 239 条完成 + 1 条异常，原汇总无效|[独立评测与补测规则](post_rl_selection_20260914.md)|
|MT-GTPO reference_write v3|有 1,280 条训练记录及 step10/20 训练内评测；不能与独立 selection 混为一套成绩|[运行记录](mt_gtpo_v3_execution_20260917.md)|
|paper_env_split_v3|旧1280条在当前源码下精确复现；新增256条工程轨迹固定诊断发现GENERIC转人工混入state_change、分区支持不足及已评分error优势方向问题；未冻结／新训练|[奖励修订](mt_gtpo_split_reward_revision_20260918.md)、[9月20日诊断](mt_gtpo_engineering_reward_diagnosis_20260920.md)|
|paper_env_split_v4|已分离GENERIC转人工，1,536条固定权重对照完成，旧版完整回放一致；类别支持/信用方向仍未通过，未冻结或新训练|[v4修订与对照](mt_gtpo_split_v4_20260920.md)|
|split v4错误信用|1,536条分解完成：工程23个已评分失败正错误轮均来自相对未来贡献、无后续gold_write；恢复成功与预算截断分开报告；未修改公式|[信用诊断](mt_gtpo_error_credit_v4_20260920.md)|
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


## 2026-09-21 离线信用对照完成

固定原MT-v3 buffer的时间单因素候选将正优势错误轮381降到68，并保留成功参考写入信号；同时242个无即时奖励成功轮失去正信号，同轮混合调用仍在。调用级事件诊断不直接作为训练公式。下一步是CPU设计局部与长程信用作用范围及真实调用边界记录，尚未实施；无新GPU/训练授权执行。[完整报告](mt_credit_comparison_20260921.md)。
