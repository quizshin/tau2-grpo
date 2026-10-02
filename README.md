# 双 A800 工具智能体训练

面向 Airline 多轮工具调用的 **Qwen3.5-4B 三阶段 SFT → GRPO / ARPO（可选 MT-GTPO）**。部署目标统一为两张 A800 80GB；研究算法与执行环境分层维护。

## 当前范围

- SFT 课程：累计 **A109 → B393 → C500**，每阶段从前阶段选定 checkpoint 继续。train500/dev150 已冻结为独立 portable 包，开发集基础／约束／策略各50条。三阶段配置与CPU校验已完成，GPU训练未开始。
- RL：同一 SFT 起点分别运行 GRPO、ARPO，可选 MT-GTPO；不要把 GRPO 结果当作 ARPO 的共同起点。
- RL双卡共享：策略使用 GPU 0、1；模拟器在 rollout 阶段使用 GPU 1，并在策略计算阶段休眠。
- 日志、独立评测、完整检查点、续训和停止边界沿用公共 runner。

**验证边界：**旧 repair72 GRPO 的双卡20步实验已完成；新三阶段课程对照尚未执行。SFT当前在GPU 0运行，不能仅因部署有两卡就改变有效batch与更新预算。ARPO和新的双卡MT-GTPO组合不能继承GRPO的GPU验收结论。

## 从哪里开始

|文档|内容|
|---|---|
|[双 A800 安装](docs/setup-a800.md)|唯一支持的部署路径与环境组件|
|[数据](docs/data.md)|train500/dev150完整包下载、哈希校验与train/dev隔离|
|[模型资产](docs/model-assets.md)|4B与9B策略基座、tokenizer、共享量化模拟器的固定版本下载|
|[三阶段 SFT](docs/sft.md)|课程及阶段衔接|
|[GRPO / ARPO / MT-GTPO](docs/rl.md)|双卡配置、算法边界、启动前检查|
|[评测](docs/evaluation.md)|开发集、历史selection、官方final|
|[运行维护](docs/operations.md)|检查点、恢复、日志与停止|
|[架构](docs/architecture/layout.md)|模块职责和调用链|
|[开发验证](docs/development.md)|CPU CI与真实资产集成测试|
|[研究与历史](docs/research-summary.md)|结果边界及旧版本位置|

## 代码布局

```text
tau3_grpo/  算法、数据、环境、训练、评测的公共实现
configs/    双A800入口、算法与奖励组件、冻结研究协议
scripts/    启动和维护入口
tests/      核心、基准环境和框架集成测试
env_info/   固定依赖、部署脚本与vendor补丁清单
data/       必需冻结输入和资产索引
docs/       当前使用说明及保留的技术规格
verl/       固定版本的训练框架及项目补丁
tau2-bench/ 固定版本的基准环境
```

冻结 train500/dev150 完整包已提供 [GitHub Release 下载](https://github.com/quizshin/tau2-grpo/releases/tag/sft-train500-dev150-20261001)，解压约24.1MB；模型与tokenizer按[资产说明](docs/model-assets.md)从固定来源下载。凭据和运行产物不上传Git。上游目录不代表本项目支持其中的所有硬件/示例。GiGPO等共享兼容代码继续保留，但不作为当前部署主线。

## 获取实验资产

克隆仓库后，在仓库根目录下载并逐文件验证冻结 SFT 包（只需 Python 标准库）：

```bash
python scripts/maintenance/fetch_sft_package.py
```

默认安装到 `data/sft/curriculum_500_dev150_portable_codex_20261001/`，既有文件仅校验、不覆盖。
模型准备见[4B / 9B 与共享模拟器](docs/model-assets.md)；三阶段课程训练和9B的GPU验收状态仍以上述范围及对应实验说明为准。

## 开发检查

```bash
bash setup.sh cpu-test
source .venv-cpu/bin/activate
python -m pip install ruff==0.16.6
python scripts/maintenance/check_lint.py
python scripts/maintenance/check_cpu.py --suite core --report-dir results/maintenance/core-check --require-no-skips
```

CPU检查不会启动训练，不能代替GPU验收。具体运行前先准备资产与预算，见上述文档。

## 历史与升级

本次收敛前的完整文档、5090及旧多卡部署工具保存在标签 [archive/pre-a800-focus-20260929](https://github.com/quizshin/tau2-grpo/tree/archive/pre-a800-focus-20260929)。本次发布不更新已有本地code和服务器checkout。以后升级会应用文件删除和路径迁移，须显式检查差异。许可证和上游来源见 [THIRD_PARTY_SOURCES.md](THIRD_PARTY_SOURCES.md)。
