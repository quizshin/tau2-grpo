# 代码组织与调用关系

|目录|职责|
|---|---|
|tau3_grpo/data|数据、划分、模板和轨迹事实|
|tau3_grpo/envs|每轨迹会话、工具与模拟用户|
|tau3_grpo/algorithms|GRPO辅助逻辑、ARPO、MT-GTPO与兼容算法|
|tau3_grpo/integrations/verl|框架注册、batch与rollout接入|
|tau3_grpo/training|SFT与RL公共控制器、服务生命周期|
|tau3_grpo/evaluation|奖励验证、独立评测、比较|
|tau3_grpo/analysis|离线审计和分析|
|configs|显式实验参数；公开入口见experiments/catalog.yaml|
|scripts|薄启动入口、维护工具和已有回归依赖的研究辅助脚本|
|env_info|固定依赖、A800部署、上游补丁身份|

RL调用：`training.rl.runner → launch → scripts/train/rl → training.rl.train → veRL → agent loop / env session → verifier → algorithm → 参数更新`。

SFT调用：`scripts/train/sft → training.sft.train → dataset / tokenizer监督模板 → trainer → 导出与独立评测`。

`scripts/a800_research`和`scripts/engineering_checks`保留少量回归测试依赖的旧控制器/离线诊断，不作为公开生产入口。核心实现不放入env_info；vendor目录保留固定来源和必要补丁，不裁剪上游许可证。

运行数据放results/runs，固定输入放data/manifests。没有模型权重、tokenizer或本地机器环境快照进入Git。

数据层的能力特征、可见消息、证据编码和参考动作配方分别位于
`data/capability_features.py`、`data/messages.py`、`data/sft_evidence.py`、`data/outcome_recipes.py`；
离线analysis复用这些函数，数据构建不反向导入analysis。
Teacher rollout共享一个实现，通过split和预算参数选择行为；旧validation入口仅薄转发，
保持预算默认值、账本锁和train/validation隔离。历史结果中的源码快照不作为当前程序入口。

外部存储根由 `paths.runtime_environment` 在launcher与正式runner中统一解析；
原始模型/数据和运行结果独立管理，源码发布不迁移资产。
