# Tau3-GRPO：训练、评估与实验记录

> 当前本地环境需要通过 setup 重建。2026-09-10 核对时，历史 `.venv-qwen35-local` 不存在，不能直接使用旧激活命令。安装 Python 3.12 后，在本项目根目录执行 `bash setup.sh cpu-test`，创建的是 `.venv-cpu`，然后执行 `source .venv-cpu/bin/activate`。这是本地 CPU 开发环境；A800 Qwen3.5 训练环境使用 `bash setup.sh a800-qwen35`，见 env_info。 本次仅更新状态说明，未安装环境或重新运行训练。

在 veRL 上进行多轮工具智能体训练，支持 Qwen2.5、Qwen3.5 系列的文本策略与用户模拟器，以及全量语言参数 / LoRA SFT 和 RL。具体模型和硬件组合的验证状态见 [实验记录](EXPERIMENTS.md)，配置存在不代表该组合已经完成真实 GPU 训练。

## 从哪里看起

| 你要做什么 | 看哪里 |
|---|---|
| 看实验进展、效果与问题 | [EXPERIMENTS.md](EXPERIMENTS.md) |
| 看最新 RL 验证、观察修复和 full / LoRA 对照 | [2026-09-09 对照记录](docs/rl_validation_comparison_20260909.md) |
| 接手训练、SSH 连接、恢复服务器 | [2026-09-09 接管记录](docs/training_handover_20260909.md) |
| 改模型、学习率、预算和显卡配置 | [configs](configs/) |
| 启动训练、模拟器或评估 | [scripts](scripts/) |
| 理解实现与调用关系 | [架构说明](docs/architecture/layout.md) 与 [tau3_grpo](tau3_grpo/) |
| 看数据准备和完整实验流程 | [操作流程](docs/workflow.md) |
| 配置 SwanLab 指标和对话案例 | [SwanLab](docs/swanlab.md) |

## 目录

```text
code/
├── README.md / EXPERIMENTS.md      说明与实验记录
├── pyproject.toml / setup.sh      业务包安装与环境入口
├── .env.example                  凭据和存储路径模板
├── configs/
│   ├── train/sft/                full / LoRA SFT 参数
│   ├── train/rl/                 full / LoRA RL 参数
│   ├── experiments/              E0～E3 算法设置
│   ├── models/                   策略模型配置和固定版本
│   ├── simulator/                模拟器服务与下载配置
│   ├── hardware/                 GPU 分配
│   ├── envs/                     工具、交互配置
│   └── tracking/                 SwanLab 默认项目
├── scripts/                      按 data、train、eval、serve、maintenance 分类
├── tau3_grpo/
│   ├── training/sft/             SFT 数据集、训练、导出
│   ├── training/rl/              注册算法并启动 veRL
│   ├── algorithms/               动态过滤、Tau-GiGPO、状态锚点
│   ├── envs/                     任务、会话、工具和用户交互
│   ├── evaluation/               官方评分、独立评估、报告
│   ├── data/                     数据准备和切分隔离
│   ├── models/                   模型兼容、下载与服务检查
│   ├── tracking/                 SwanLab、指标和真实对话案例
│   ├── integrations/             veRL 回调与补丁检查
│   ├── experiments/              运行清单与最终模型锁定
│   ├── launch.py                 读取配置、启动子进程
│   └── paths.py                  统一路径解析
├── tests/ / docs/ / env_info/     测试、文档、环境版本
├── verl/                         veRL 源码及本项目适配
└── tau2-bench/                   官方 Tau3 环境；Python 包名仍为 tau2
```

本机的 `data/`、`models/`、`results/`、`.venv*` 和缓存是被 Git 忽略的运行资产。它们也可以完全放在源码目录外。项目只安装 `tau3_grpo` 包，依赖源码各自安装，避免把数据目录当作 Python 包。

## 环境与路径

Qwen3.5 Linux/A800 环境：`bash setup.sh a800-qwen35`。Qwen2.5 的固定环境：`bash setup.sh a800`。两种环境分开安装；不要将 Mac 虚拟环境复制到 Linux。仅做 CPU 检查可用 `bash setup.sh cpu-test`。

运行前激活对应环境，并从 `.env.example` 创建本机 `.env`。密钥不提交、不上传源码附件。可设置：

```bash
export TAU3_DATA_ROOT=/root/autodl-tmp/tau3/data
export TAU3_MODEL_ROOT=/root/autodl-tmp/tau3/models
export TAU3_RUN_ROOT=/root/autodl-tmp/tau3/runs
export TAU3_CACHE_ROOT=/root/autodl-tmp/tau3/cache
```

没有覆盖时默认使用 `code/{data,models,results,.cache}`。Python 路径由 `paths.py` 解析，shell 入口由 `scripts/lib/paths.sh` 设置。

## 常用入口

以下相对配置路径从 `code` 根目录解析。`--dry-run` 只展示启动配置，不训练、不启动服务、不创建 SwanLab 实验。先用它确认模型与存储路径。

```bash
# 全量 / LoRA SFT：相同入口，选择不同配置
bash scripts/train/sft/run.sh --config configs/train/sft/qwen35_full.yaml --dry-run
bash scripts/train/sft/run.sh --config configs/train/sft/qwen35_lora.yaml --dry-run

# 独立 4B 用户模拟器：默认 GPU 1
bash scripts/serve/simulator.sh --config configs/simulator/qwen35_4b.yaml --dry-run

# RL：默认 GPU 0，短程工程预算；先检查 SFT 起点路径
bash scripts/train/rl/run.sh --config configs/train/rl/qwen35_lora.yaml --experiment e0 --dry-run
bash scripts/train/rl/run.sh --config configs/train/rl/qwen35_full.yaml --experiment e0 --dry-run
```

确认后移除 `--dry-run` 执行。Qwen2.5 配置位于相同目录，前缀为 `qwen25`。full / LoRA RL 比较应指定相同 `MODEL_PATH`（SFT 起点）；默认 Qwen3.5 RL 指向本轮已保留的 LoRA SFT 合并模型。新 SFT 的输出若不同，应显式覆盖路径。

`--experiment e0/e1/e2/e3` 选择 GRPO、GRPO+DF、Tau-GiGPO、Tau-GiGPO+DF。后接原生参数可覆盖底层设置，例如 `-- actor_rollout_ref.actor.optim.lr=2e-6`；SFT 后接 `-- --output-dir ...`。实际训练指标和完整参数继续由原训练入口记录到 SwanLab。

本次 Qwen3.5 RL YAML 是 2×A800 的 **3 步工程配置**；正式预算另见 `configs/train/rl/base.yaml`。全量 RL、9B 和新的模拟器组合需单独验证。视觉模块冻结，Qwen3.5 使用已适配的原生 padded forward。

## 数据与评估边界

独立评测以 **pass@k** 为主，可追加 **pass^k**；冻结模型服务、结果文件和离线重算命令见
[独立评测说明](docs/independent_evaluation.md)。

训练和内部选型使用固定版本 AReaL Tau2 风格 Airline 数据。官方 Tau3 的 50 个任务只用于模型锁定后的最终评估，不能用于训练或选择 SFT/RL 方法。评分沿用官方逻辑；详见 [运行边界](docs/architecture/runtime-boundary.md)。

开发检查：激活本机环境后运行 `pytest -q tests`。已有算法、模板、最后轮工具执行、评分和 SwanLab 回归均随本次目录重构迁移。上游版本与本地补丁见 [THIRD_PARTY_SOURCES.md](THIRD_PARTY_SOURCES.md)。
