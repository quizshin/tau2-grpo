# Qwen3.5模型支持

当前训练主线是Qwen3.5-4B语言工具智能体，双A800；框架版本见env_info/qwen35-constraints.txt。共享模型兼容代码/离线tokenizer测试可覆盖其他尺寸，这不等于提供它们的部署环境。

策略4B和9B权重及tokenizer的固定下载命令、目录和校验见[模型资产准备](model-assets.md)。SFT使用LoRA监督模板；RL通过配置选择策略初始化与LoRA参数，不能由activate脚本静默覆盖。

训练入口见[sft](sft.md)、[rl](rl.md)。旧模型实验与兼容历史见归档标签。
