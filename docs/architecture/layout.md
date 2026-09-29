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
