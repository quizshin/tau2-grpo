# 数据与资产

保留已跟踪的基准数据库及训练任务输入；版本与哈希见 `data/SHA256SUMS.json`。这些JSON是测试/实验输入，不是模型tokenizer。

固定课程清单统一位于 `data/manifests/rl_curriculum_20260912/` 和 `data/manifests/rl_curriculum50_20260912/`。results只放新运行产物，不再承担固定配置输入。

SFT500/37的数据包身份见 [清单](sft500_package_manifest_20260928.json)。完整数据、审核回执和模型资产由部署环境提供，不因清单存在就认定已经安装。保持用户、任务来源和重复样本隔离，开发集不参与梯度训练。

真实tokenizer测试使用已有 `models/Qwen3.5-4B/tokenizer.json`，不自动下载、不上传Git。部分harness测试接受 `TAU3_TEST_TOKENIZERS` 指向已有tokenizer目录；框架测试还需要其余固定型号的资产。
