# 远程训练服务器整盘清理 · 2026-09-09

用户明确要求整盘清理后，扫描 AutoDL 系统盘、数据盘和持久盘，完成受控去重、失败残留与缓存清理。本次范围仅为远程训练服务器。

| 磁盘 | 清理前可用（十进制 GB） | 清理后可用（十进制 GB） | 增加 |
|---|---:|---:|---:|
| 系统盘 | 32.12 | 32.16 | 0.04 |
| 训练数据盘 | 15.83 | 29.64 | 13.80 |
| 持久盘 | 128.70 | 134.10 | 5.40 |

实际可用空间增加合计约 **19.24 GB**（df 前后差）；文件逻辑大小合计 19.23 GB，两者因文件系统分配计量而不同。`df -h` 使用二进制口径，清理后数据盘约 28G、持久盘约 125G；不能混用单位比较。

## 已执行

- 数据盘 RL-003 / RL-005 的 step2、step3 共四个检查点，以及 SFT 的 `full/sft_full_seed42`、`lora/sft_lora_seed42`，合计 13,431,316,547 字节，逐文件与持久副本核对 SHA256 后去重。原路径改成指向持久盘的目录软链接，六处所有文件仍可通过原路径访问。
- 删除持久盘 RL-007 的不完整 `e0_seed42/global_step_2`，5,397,662,888 字节。该检查点曾在优化器保存时失败，不能用于恢复。保留当时日志、训练 batch、历史 SHA256 与失败文件清单；RL-007 的完整 step1 和 RL-008 的完整 step3 均保留。
- 清理已经停止训练的 Triton、vLLM/torchinductor 编译缓存、Ray 会话缓存及 pytest 临时目录，文件逻辑大小约 0.40 GB。下次训练可重建，编译阶段可能更慢。

保留 0.8B / 4B / 27B 模型、运行环境、训练数据、两组 SFT 合并模型、LoRA 适配器和检查点、完整 full 检查点及实验日志/轨迹。系统盘 overlay 仅约 1% 使用率；du 可见的基础镜像层并非等量可回收的可写层，没有删除平台、CUDA 或系统 Python。

## 清理后验证

六处链接目标和所有文件大小检查通过。RL-008 持久检查点清单的 17 文件仍存在且大小相符，latest marker 为 3。最终模型 SHA256 重新核对为 `82ffd79cb718d894e7806ef32cc601e199554057fac17dd248b5ee4f5444704d`；RL 使用的 SFT 起点重新核对为 `36c9c10fc6da92b4babc929012b80aedba844815aff653b8718ff7531edae9a0`。运行 venv 中 `tau3_grpo`、`verl`、`tau2` 均正常导入且指向运行源码。

后续恢复仍使用 RL-008 的完整 `global_step_3`。数据盘上的去重链接依赖当前持久盘挂载；换机时应恢复同一路径布局。原 `preservation.json` 与 relocation 回执保留历史快照含义，本次清理回执才反映删除后的库存；不要再把 RL-007 不完整 step2 视为仍保留。历史复制脚本未为本次目录软链接重写，不应直接重跑，以免覆盖原清单或重复复制。

## 证据

本地目录：`results/validation/server_cleanup_20260909/`；同名证据完整归档于 `/root/autodl-fs/tau3_grpo_fix/results/server_cleanup_20260909/`。

- [交互式清理报告](../results/validation/server_cleanup_20260909/report.html)
- `scan-before.json`：三盘扫描与空闲 GPU 检查。
- `cleanup-plan.json`：具体路径、逐文件 hash 对比及删除清单。
- `cleanup-receipt.json`：逐项处置、前后 df、实际释放空间。
- `post-cleanup-verification.json`：链接、模型 hash、最终检查点与环境导入验证。
- `cleanup-plan.json` 的 `failed_checkpoint.files`：删除前的失败文件大小与原归档 SHA256。
