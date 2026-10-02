# 三阶段 SFT

当前方案使用已冻结的 **train500/dev150 portable v2** 包；CPU数据和监督校验通过，GPU训练未开始。
精确文件身份见[数据包清单](sft500_dev150_portable_package_manifest_20261001.json)，当前配置统一从
`configs/experiments/catalog.yaml` 的 `sft.profiles` 选择。

|阶段|累计对话|更新次数|初始化|
|---|---:|---:|---|
|A109|109|14|Qwen3.5-4B基座|
|B393|393|50|默认A阶段最后一个checkpoint导出的merged model|
|C500|500|63|默认B阶段最后一个checkpoint导出的merged model|

每阶段1 epoch，LoRA r16/alpha32、学习率3e-5，有效batch为8。当前SFT使用GPU 0，
每卡batch 1、累积8；双A800部署不意味着这份配方进行双卡SFT。
B/C必须通过`SFT_MODEL_NAME_OR_PATH`显式提供前阶段模型，重新建立优化器；
不能默认为再次从基座训练，也不能把权重衔接当作原优化器状态的精确续训。

入口为 `scripts/train/sft/run.sh`，实现为 `tau3_grpo.training.sft.train`。
三份配置是 `configs/train/sft/curriculum_codex_{A109,B393,C500}_portable_dev_1epoch.yaml`。
完整dev150按基础／约束／策略各50条冻结，所有阶段使用同一完整开发集检查保留能力。
新课程默认以C500最后一个checkpoint导出的merged model作为同尺寸GRPO/ARPO的共同起点。
后续可根据同一课程评测挑选表现较好的merged checkpoint，记录所选模型身份后再冻结RL起点。
实际阶段模型尚待训练、导出与验证，新课程RL入口衔接及dev150交互评测接入仍需执行验收；
现有repair72 GRPO/ARPO配置是旧起点模板，不自动成为新课程实验。

数据包只依赖包内13个受保护文件及manifest，约24.1MB。启动校验保护文件哈希、
来源身份、审核结论、train/dev用户与对话隔离、累计子集及真实token/mask。
历史路径仅用于溯源，不再逐项打开旧候选/审核目录。旧v1包依赖的构建历史已在本地清理，
旧配方不作为当前入口；历史配方的训练语义保留为测试夹具，原提交仍可追溯。

模型、tokenizer及完整数据包独立部署，见[资产说明](data.md)。
监督模板和mask说明见[thinking与监督](sft_thinking.md)。
