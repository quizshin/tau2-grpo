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

完整包通过 [GitHub Release](https://github.com/quizshin/tau2-grpo/releases/tag/sft-train500-dev150-20261001) 提供；
真实tokenizer按[模型资产说明](model-assets.md)下载。数据包不作为Git源码文件提交。
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

包内五维审核记录仅说明冻结样本的接受状态；模型新回复使用[五维轨迹评测链路](evaluation.md)，
另行输出证据判断，不能复用样本 accepted 结论。

## 下载和校验完整 train500/dev150 包

[直接下载压缩包](https://github.com/quizshin/tau2-grpo/releases/download/sft-train500-dev150-20261001/curriculum_500_dev150_portable_codex_20261001.tar.gz)。
压缩2,037,186字节（约2.04MB），解压24,106,299字节（约24.1MB），14个原始文件。
固定压缩包SHA-256：`52a5224092d49cdc9285013407a736209eff3ba80830999a0162ce291d5c4653`。
下载地址、大小与哈希也记录在[机器可读发布清单](sft500_dev150_release.json)。

在仓库根目录执行（Python标准库，不需要Torch、tokenizer或GPU）：

```bash
python scripts/maintenance/fetch_sft_package.py
# 下载后可重复执行离线校验：
python scripts/maintenance/fetch_sft_package.py --verify-only
```

默认安装到 `data/sft/curriculum_500_dev150_portable_codex_20261001/`；
自定义资产根目录用 `--data-root /absolute/path/to/data` 或 `TAU3_DATA_ROOT`，
并确保训练也使用相同根目录。相对根目录以仓库根解析。
已存在的包只核对文件，不覆盖；不匹配时先保留并检查原文件。
安装器校验压缩包、manifest及全部13个受保护文件，拒绝未知文件、重复文件、链接及越界路径，
验证通过才放入最终目录。

GitHub下载不通时，可先在其他机器下载上面的固定压缩包，复制到执行机器后运行：

```bash
python scripts/maintenance/fetch_sft_package.py --archive /absolute/path/to/curriculum_500_dev150_portable_codex_20261001.tar.gz
```

文件名与冻结内容保持原样；历史路径仍只是溯源记录，不需要下载旧results目录。
包提供train500、dev150、A109/B393子集、审核与token/mask证据、工具及提示协议；
C500直接用train.jsonl。dev150仍是SFT会话开发集，不包含已完成的150任务交互评测部署。
数据来源与原许可见[第三方来源](../THIRD_PARTY_SOURCES.md)及包内source_manifest.json；
发布完整包不修改任何样本、审核结论或冻结哈希。
