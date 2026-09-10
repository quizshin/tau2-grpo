# AutoDL 三盘布局与恢复

更新：2026-09-09。固定持久根目录 `/root/autodl-fs/tau3_grpo_fix`，运行根目录 `/root/autodl-tmp/tau3`。

| 位置 | 内容 | 换服务器时 |
|---|---|---|
| 系统盘 `/root` | 系统服务、SSH、基础 Conda 等 | 不依赖旧系统盘中的项目文件 |
| 文件存储 `autodl-fs/tau3_grpo_fix/code` | 最新源码、配置、文档、私有 `.env` | 挂载同一文件存储后保留 |
| 文件存储 `runtime/python`、`runtime/venvs` | 独立 Python、Qwen3.5 训练环境、Qwen3.8 推理 overlay | 保持相同挂载路径；要求兼容的 Linux x86_64、NVIDIA 驱动 |
| 文件存储 `data`、`model_store` | 数据主副本，0.8B、4B、27B 完整模型 | 无须重新联网下载 |
| 文件存储 `results`、`reports` | 已同步的 SFT 成品/评估、SwanLab 本地记录、历史诊断与清理回执 | 持久保存 |
| 数据盘 `runtime/code`、`runtime/qwen35` | 源码及训练环境的运行副本 | 节点丢失后可重建；环境也可直接使用 FS 版本 |
| 数据盘 `models`、`data` | 常用 0.8B/4B 模型和工作数据 | 可从 FS 恢复 |
| 数据盘 `runs` | 当前训练检查点、轨迹、评估 | 训练结束后手动同步到 FS，再更换/释放实例 |
| 数据盘 `cache`、`tmp` | vLLM/Triton/HF 缓存、Ray 临时文件 | 停止任务后可清理；下次首次启动会重新编译 |

27B 量化模型占约 19.6 GiB，默认直接从 FS 加载。50 GiB 数据盘不默认缓存它，给训练输出留空间。模型校验记录保存在模型目录的 `tau3_source_revision.json`；27B 另有 `download-manifest.json`。

## 当前节点使用

```bash
source /root/autodl-fs/tau3_grpo_fix/activate.sh
cd "$TAU3_CODE_ROOT"
```

激活脚本优先使用数据盘已有的源码/训练环境，缺失时回退到 FS。数据、模型、结果、缓存通过 `TAU3_DATA_ROOT`、`TAU3_MODEL_ROOT`、`TAU3_RUN_ROOT`、`TAU3_CACHE_ROOT` 传给训练入口。缓存与临时文件写到数据盘，不写系统盘。

27B 模拟器入口（实际启动需要 GPU）：

```bash
bash /root/autodl-fs/tau3_grpo_fix/code/scripts/serve/simulator_qwen38.sh
# 只检查生成的启动命令：
TAU3_DRY_RUN=1 bash /root/autodl-fs/tau3_grpo_fix/code/scripts/serve/simulator_qwen38.sh
```

模拟器使用 FS 的 `runtime/venvs/qwen38-sim`；该 overlay 的 Transformers 5.8.0 共享 FS 中 Qwen3.5 环境的 Torch/vLLM，两个目录都要保留。

## 更换服务器

1. 挂载同一份文件存储到相同路径，选择兼容 CUDA 驱动的 Linux x86_64 实例。
2. `bash /root/autodl-fs/tau3_grpo_fix/restore_node.sh`：恢复源码、数据和 0.8B/4B 模型；模型通过 SHA256 校验。默认保留至少 12 GiB 数据盘空间。
3. `source /root/autodl-fs/tau3_grpo_fix/activate.sh`。新节点可以直接使用 FS 环境，首次导入会较慢。已有节点的运行环境不重复拷贝。
4. 根据实验配置，从 FS `results/sft_compare_20260908` 复制需要的 SFT 起点到数据盘对应路径，或将实验的模型路径指向 FS 成品；不默认复制所有历史权重。
5. 检查 GPU、运行环境及训练配置，再启动训练。无卡验证不替代 GPU 训练验证。

单独检查模型恢复所需空间：

```bash
bash /root/autodl-fs/tau3_grpo_fix/restore_models.sh --dry-run
# 只恢复一个模型：
bash /root/autodl-fs/tau3_grpo_fix/restore_models.sh Qwen3.5-0.8B
```

训练停止后、释放实例前同步成果：

```bash
bash /root/autodl-fs/tau3_grpo_fix/sync_results.sh
```

该脚本使用 checksum 比较，只增补/更新 FS，不删除 FS 独有记录。同步不是后台自动执行的。

## 清理记录

本次清理回执位于 FS `reports/storage-20260909/`。旧零奖励 RL 检查点、系统盘重复模型、废弃下载/部署包和编译缓存删除；小型轨迹、审计、旧维护脚本归档。保留全量与 LoRA 的 SFT 成品及评估。

根目录四个维护脚本链接到 `code/env_info/autodl/`，以后从源码维护，避免根目录脚本与仓库各有一份。归档内的旧绝对路径仅供查阅，不能再作为启动入口。
