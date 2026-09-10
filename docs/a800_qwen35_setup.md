# AutoDL 环境、模型与存储布局

> 本文保留 2026-09-08 首次安装时的状态。当前目录、SFT/RL 进展及连接方法以 [2026-09-09 接管记录](training_handover_20260909.md) 和 [实验记录](../EXPERIMENTS.md) 为准。

准备日期：2026-09-08。当前实例处于无卡模式，CPU 配额 0.5 核、内存上限 2GB。
代码、依赖和模型已准备完成；GPU 训练与推理尚未验收。

## 目录和环境

| 内容 | 服务器路径 |
|---|---|
| 标准 veRL 项目代码 | `/root/autodl-fs/tau3_grpo_fix/code` |
| 独立 Python 解释器 | `/root/autodl-fs/tau3_grpo_fix/runtime/python` |
| 新 venv 环境 | `/root/autodl-fs/tau3_grpo_fix/runtime/venvs/qwen35` |
| 项目内环境入口 | `code/.venv-a800-qwen35`，链接到上述 venv |
| 基础模型主副本（持久存储） | `/root/autodl-fs/tau3_grpo_fix/model_store` |
| 基础模型工作副本（数据盘） | `/root/autodl-tmp/tau3/models` |
| 复用的原始/处理后数据 | `/root/autodl-fs/tau3_grpo_fix/data` |
| 当前运行产物 | `/root/autodl-tmp/tau3/runs` |
| 后续需保留的结果 | `/root/autodl-fs/tau3_grpo_fix/results` |
| 缓存与临时文件 | `/root/autodl-tmp/tau3/cache`、`tmp` |
| 本次安装和迁移验证记录 | `/root/autodl-fs/tau3_grpo_fix/runtime/validation` |
| Linux 依赖锁和解释器来源 | `/root/autodl-fs/tau3_grpo_fix/runtime/locks` |

`code/{data,models,results}` 分别链接到上述数据、模型工作副本、
当前运行产物目录。模型主副本保留在文件存储，供更换服务器后恢复；训练读取数据盘副本。
原 `code` 中的旧源码已替换，临时的 `code_qwen35` 目录已并回正式路径。
按用户要求删除旧 `code_agent`、独立 `verl-agent` / `verl-main` 副本、旧源码归档、
全部旧实验 checkpoint / adapter / 日志 / 结果 / 实验配置及其归档，
以及 Qwen2.5-0.5B、7B 基础模型和 Qwen2.5-1.5B 下载缓存。旧环境、旧安装缓存
和旧 FlashAttention wheel 也已清理。代码内可复用的配置模板和训练数据保留。
系统盘的 AutoDL 基础 Conda/Jupyter 环境保留，当前项目 venv 不依赖该解释器。

新 SSH 交互会话会自动激活新环境；已有会话可以执行：

```bash
source /root/autodl-fs/tau3_grpo_fix/activate.sh
cd /root/autodl-fs/tau3_grpo_fix/code
```

使用根目录 `activate.sh`，可同时把 Hugging Face、pip、uv、vLLM、Triton、Torch、
Ray、W&B 缓存和临时产物定向到数据盘；仅激活 venv 不会设置这些路径。

Python 3.12.14；Torch 2.11.0+cu130；Transformers 5.5.1；vLLM 0.20.0；
PEFT 0.18.1；Accelerate 1.12.0；NumPy 2.2.6；SciPy 1.14.1；TensorDict 0.10.0。
安装器包含 `stats` extra，确保报告所需的 SciPy 被安装。
Python 来自官方 `python-build-standalone` 20260901 发行版；独立解释器、标准库、
venv 包和 `uv` 均在文件存储中，包文件不依赖已删除的数据盘缓存硬链接。

## 换服务器后恢复

把同一份文件存储挂载到 `/root/autodl-fs`，使用兼容的 Linux x86_64 镜像：

```bash
source /root/autodl-fs/tau3_grpo_fix/activate.sh
bash /root/autodl-fs/tau3_grpo_fix/restore_models.sh
cd /root/autodl-fs/tau3_grpo_fix/code
```

`restore_models.sh` 将主副本复制到新节点数据盘，并逐片计算 SHA-256。
当前服务器已完成复制和校验，无需重复恢复。新节点可把上述 `source` 命令加入其
`~/.bashrc`；当前节点已配置。venv 使用固定的绝对路径，迁移时应保持相同挂载位置。
GPU 驱动由新节点提供，仍需执行下方开卡检查。

每次训练结束，将选定的结果复制到文件存储的 `results/` 后再释放数据盘，例如：

```bash
# 替换为实际需要保留的相对运行目录名；此操作不会自动删除数据盘副本。
RUN_NAME=qwen35_0.8B
rsync -a "/root/autodl-tmp/tau3/runs/${RUN_NAME}/" \
  "/root/autodl-fs/tau3_grpo_fix/results/${RUN_NAME}/"
```

结果持久化需要主动复制；当前项目 `results` 链接指向数据盘。

## 已下载的模型

| 用途 | 模型 | 权重大小 | 固定 Hugging Face revision |
|---|---|---:|---|
| 默认策略模型 | Qwen3.5-0.8B | 1,746,942,600 bytes | `2fc06364715b967f1860aea9cf38778875588b17` |
| 默认模拟器 | Qwen3.5-4B | 9,319,828,096 bytes | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

权重来自 Qwen 官方 ModelScope 仓库，其 SHA-256 与上述 Hugging Face
固定版本的 LFS 清单逐项匹配。下载后再次计算了全部三片权重的 SHA-256。
每个模型的配置、tokenizer、权重索引及预处理配置也与固定版本匹配。
索引检查覆盖 0.8B 的 488 个张量和 4B 的 738 个张量。
各模型目录的 `tau3_source_revision.json` 记录来源和权重校验值。

## 已完成的验证

- 文件存储上的独立环境全量项目测试：**375 passed、3 skipped**，122.58 秒。
  跳过的是未下载的 9B tokenizer 和已删除的 Qwen2.5 tokenizer 相关检查。
- `pip check`、Ruff、veRL 补丁契约及 `tau_gigpo` 注册检查通过。
- Torch CPU 运算、Transformers、PEFT、vLLM 以及 veRL FSDP/rollout 模块可导入。
- 45 条训练、5 条验证 SFT 数据与本地冻结文件哈希一致，Qwen3.5 标签审计通过；
  最大长度分别为 16,473 和 16,702，使用 24,576 的配置上限。
- Qwen3.5 E0 启动命令 dry-run 通过；SSH 默认环境切换已验证。
- 远程 1,383 个源码和文档文件与本地最新工作区逐项匹配。

首次安装记录保存在 `runtime/validation/bootstrap-20260908/`，包括测试、补丁契约、
模型下载及数据审计。迁移到文件存储后的安装与重新验证记录保存在
`runtime/validation/migration-20260908/`；依赖锁为 `runtime/locks/qwen35-linux.txt`。
以上相对路径均以 `/root/autodl-fs/tau3_grpo_fix` 为根。

## 开两张卡后的第一步

当前驱动版本为 580.105.08，但无卡模式下 `/usr/lib64/libcuda.so.1` 是占位文件，
vLLM CUDA 扩展尚不能加载。开卡后需要确认平台挂载了真实驱动库，并执行：

```bash
cd /root/autodl-fs/tau3_grpo_fix/code
source /root/autodl-fs/tau3_grpo_fix/activate.sh
nvidia-smi
python -c 'import torch; import vllm._C; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); print(torch.ones(1, device="cuda"))'
python -m tau3_grpo.models.check_qwen35 \
  --model models/Qwen3.5-0.8B --gpu
```

之后按 [模型运行指南](qwen35.md) 执行 SFT、合并、模拟器服务和短程 E0–E3
验收。默认 GPU 0 承担策略训练/rollout，GPU 1 运行 Qwen3.5-4B 模拟器。
无卡模式下没有启动模型服务，也没有进行真实预训练权重的 SFT/RL。
