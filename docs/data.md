# 数据与资产

已跟踪的基准数据库和冻结任务输入见 `data/SHA256SUMS.json`。这些JSON不是模型tokenizer。
40/50任务清单统一位于 `data/manifests/rl_curriculum40_seed42/` 和
`data/manifests/rl_curriculum50_seed42/`；清单内容与迁移前逐字节相同。
results只保存运行产物，正式selection parquet需在训练前按冻结协议准备。

当前SFT包为 `data/sft/curriculum_500_dev150_portable_codex_20261001/`，
精确身份见[清单](sft500_dev150_portable_package_manifest_20261001.json)。
包内13个文件加manifest约24.1MB，包含train/dev JSONL、A/B累计子集、审核结论、
token/mask校验、提示/工具协议和来源身份；C500直接使用train.jsonl。
开发集150条按基础／约束／策略各50条冻结，不参与梯度训练。

完整包和真实tokenizer不上传Git，由持有资产的本地或部署环境提供。
`source_manifest.json`中的历史路径只是来源记录，不是待下载文件列表；
旧候选的完整历史重放资料已清理，原始数据库和当前包仍保留。
包文件可独立复制，运行时不再依赖历史results目录；有身份清单不代表资产已经安装。
旧500/37身份清单仅作为历史记录。

`TAU3_DATA_ROOT`、`TAU3_MODEL_ROOT`、`TAU3_RUN_ROOT`、`TAU3_CACHE_ROOT`
指定独立存储根目录；相对根路径以代码根目录解析。训练包中的相对路径仍限定在包内。
数据生成、清单筛选和扩展入口要求新输出位置，避免覆盖已冻结输入。

真实tokenizer测试使用已有 `models/Qwen3.5-4B/tokenizer.json`，不自动下载。
部分harness测试接受 `TAU3_TEST_TOKENIZERS` 指向已有tokenizer目录；
框架测试还需要其余固定型号的资产。测试用小型合成数据不代替真实包/tokenizer验收。
