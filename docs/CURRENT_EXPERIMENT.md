# 当前实验与开发入口

更新日期：2026-10-01。当前本地提交中的主线是独立 SFT 数据包与三阶段课程训练。历史实验状态只表示各自归档时的事实。

## 当前正式数据与配置

- 数据包：`data/sft/curriculum_500_dev150_portable_codex_20261001/`。
- train500：基础109、多约束284、多步策略107；67个来源用户。
- dev150：基础50、多约束50、多步策略50；16个来源用户。
- train/dev来源用户和完整消息交集0；全量接受记录、token/mask及来源身份已冻结。
- 包内13个保护文件加manifest，约24.1MB；训练不读取`results/`历史候选和审核目录。
- SFT配置：`configs/train/sft/curriculum_codex_{A109,B393,C500}_portable_dev_1epoch.yaml`。
- 入口：`scripts/train/sft/run.sh --config CONFIG`；B/C须通过原生参数显式传入上一阶段的merged model。
- A/B/C分别1epoch、14/50/63次更新；LoRA16、LR3e-5、effective batch8、seed42。
- B/C加载上阶段权重并重建优化器，属于阶段续训；当前选择阶段末checkpoint。

状态：本地CPU包验证完成，GPU训练、模型导出、新课程交互评测未启动。具体GPU预算须另行决定。SFT-only/GRPO/ARPO的新课程对照仍为计划；旧repair72配置不能直接当作课程RL配置。

[独立包迁移记录](sft_portable_package_20261001.md) · [当前包身份](sft500_dev150_portable_package_manifest_20261001.json) · [数据语义审核和均衡来源](sft_balanced_dev_finalization_20260930.md)。

## 历史结果与旧版本

|项目|归档状态|证据|
|---|---|---|
|repair72 GRPO|20step完成；selection终局奖励均值起点/10/20为44.58%/48.33%/49.17%；051在9月28日的关机回执保留|[执行记录](repair72_rl_and_curriculum_plan_20260927.md)|
|A45 GRPO 8×5090|30step完成；step30为130/240，54.17%|[验收](grpo_5090_a45_u30_20260921.md)|
|MT split-v4未校准探索|30step完成；step30为6/240，2.50%，严重退化|[执行与异常](mt_gtpo_v4_exploration_20260922.md)|
|四模型统一selection 9/20|Base/SFT/GRPO/MT各240条；pass@1为50.00%/45.00%/41.25%/39.58%|[研究记录](grpo_improvement_evidence_20260920.md)|
|E0–E3与MT reference_write v3|各20step；旧E2独立selection有一条异常，原汇总无效|[信号审计](training_signal_audit_20260914.md)、[异常口径](post_rl_selection_20260914.md)|
|三算法GPU接口验收|限定配置的采样、更新、保存、恢复、导出及独立评测证据已登记|[验收范围](architecture/interface_acceptance_20260919.md)|
|SFT 9/30均衡包v1|最终JSON及身份保留；本地历史构建依赖已清理，旧配置不可启动，当前使用独立v2包|[均衡收尾](sft_balanced_dev_finalization_20260930.md)、[清理记录](sft_portable_package_20261001.md)|
|SFT 9/30初版dev|87/51/12，保留为历史分布版本|[初版收尾](sft_finalization_20260930.md)|
|9/29旧ready包|ready声明已撤回，不能用于训练|[撤回原因](sft_finalization_20260930.md#撤回旧包)|

不同起点、奖励、温度、harness和评测版本不能混合排名。selection60已有开发曝光，作为历史回归面板；final50在名单和协议冻结后使用，不用于课程选型。

## 代码与记录入口

- 代码架构：[layout](architecture/layout.md)；新增工作：[开发标准](architecture/development_standard.md)。
- 活动/候选配置：`configs/experiments/catalog.yaml`；唯一实验索引：[EXPERIMENTS](../EXPERIMENTS.md)；故障索引：[ERRORS](../ERRORS.md)。
- RL公共runner：`python -m tau3_grpo.training.rl.runner`，支持GRPO/GiGPO/MT-GTPO/ARPO的显式分支。
- 独立评测：`python -m tau3_grpo.evaluation.run`；离线比较：`python -m tau3_grpo.evaluation.compare`。
- 独立包导出：`python -m tau3_grpo.data.compact_sft --source-manifest SOURCE --output NEW_DIR`，只做已有证据迁移，不生成新接受结论。
- 原始数据、模型、冻结包和运行产物独立管理；本地当前版本不意味着服务器已同步或GPU已验收。
