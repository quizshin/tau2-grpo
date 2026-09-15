# 远程主仓库与历史代码 — 2026-09-15

唯一工作代码：`/root/autodl-fs/tau3-core/code`，工作分支 `main`。
Python 环境：`/root/autodl-fs/tau3-core/environment`。

```bash
source /root/autodl-fs/tau3-core/activate.sh
cd /root/autodl-fs/tau3-core/code
```

## 合并范围与协议

- 原持久盘提交 `30a68b06f4de703d4d64d3b7542f8b230e79029b` 是远程 RL 目录当前提交的祖先；两者相差 23 个提交。其 5090 配置和 loss projection 修改已在本地历史中保留。
- 本地代码 `08c05411fb4582e80236a1daa6f19a85bea3850c` 与远程 RL 代码 `f0ee91ee8d40fbf566a3256faf365842f5ab8912` 在 `792d70d` 后分叉，本次使用真正的双亲 Git 合并整合。
- 保留远程 RL 的续训、checkpoint 导出、评估、SwanLab 连续性、可选 GRPO 归一化与 signal audit；保留本地 5090/loss projection、v4、语义 API 和增量语义审计。
- 8 个冲突文件逐项解决：默认 anchor 仍为 v1；历史 E0–E3 配置明确固定 v1；v2/v3/v4 显式启用。GiGPO 默认 `legacy_mean`，新对照实验显式选择 `grpo`。不以默认值变化改写历史实验语义。
- GDN/FLA 独有诊断工具并入 `env_info/a800_20260912/`；旧实验控制脚本保留在 `env_info/historical/`。这些诊断脚本未被宣称已完成 GPU 数值验证。
- veRL 三个未提交的 VLA 工具文件保留在原包路径。旧 thinking 文档独有版本只是缺少后来的验证段落，保留较新的正文，旧版本存在历史分支。

远程 `f0ee91e` 是目录整理前的版本，包含四次 RL 之后的修改，不能把它当作四次训练启动时完全相同的源码快照。实际实验协议和启动版本以各次结果中的 `launch.json`、`experiment_manifest.json`、`source-snapshot.json` 为准。

## 各代码副本的历史分支

历史内容存在同一 `.git` 内，不保留多份工作目录。`source-*` 分支是源文件快照，包含原有未提交源码；排除凭据、运行数据、缓存、模型和生成的包元数据。运行数据另按下表保存。

| 原目录 | 历史分支 | 快照提交 |
|---|---|---|
| `/root/autodl-fs/tau3-core-20260912/code` | `history/source-persistent-dirty-20260915` | `a0751abae54b` |
| `/root/autodl-tmp/tau3-perf-20260912/code` | `history/source-remote-rl-20260915` | `946b75d8b5e7` |
| `/root/autodl-tmp/tau3-local-20260915/code` | `history/source-local-snapshot-20260915` | `adbd974478f7` |
| `/root/autodl-tmp/tau3-gdn-diagnosis-20260912/code` | `history/source-gdn-diagnosis-20260915` | `615a553f7226` |
| `/root/autodl-tmp/tau3-5xa800-20260911/code-before-thinking-integration` | `history/source-before-thinking-20260915` | `ea8c1e9fd0af` |
| `/root/autodl-tmp/tau3-5xa800-20260911/integration-test` | `history/source-integration-test-20260915` | `8d4d5d77e819` |
| `/root/autodl-fs/tau3-core-20260912/experiments/multicall-aa267bb/selection/code` | `history/source-multicall-selection-20260915` | `8aca39fcb7be` |

另外保留完整开发历史入口：`history/local-20260915`、`history/remote-rl-20260915`。

查看历史文件示例：

```bash
git show history/source-remote-rl-20260915:scripts/train/rl/run_qwen35.sh
git diff history/local-20260915 history/remote-rl-20260915 -- tau3_grpo/
```

不要直接在生产目录切换旧版本进行训练；先明确实验协议与环境需求。

## 代码压缩包和 bundle

这些源码归档均已纳入主仓库历史。三个 tar 源码树没有发现不在已保存 Git 对象中的新源码内容。

| 原文件 | 历史引用 | 提交 |
|---|---|---|
| `/root/autodl-tmp/tau3-cpu-signal-20260914.bundle` | `history/bundle-tau3-cpu-signal-20260914` | `7ec39e7e4a27` |
| `/root/autodl-tmp/tau3-gigpo-integration-20260914.bundle` | `history/bundle-tau3-gigpo-integration-20260914` | `b1723977ca0e` |
| `/root/autodl-tmp/tau3-gigpo-audit-report-20260914.bundle` | `history/bundle-tau3-gigpo-audit-report-20260914` | `fe4ab6b2c459` |
| `/root/autodl-tmp/tau3-anchor-v2-20260914.bundle` | `history/bundle-tau3-anchor-v2-20260914` | `6189636ed973` |
| `/root/autodl-tmp/tau3-anchor-v2-report-20260914.bundle` | `history/bundle-tau3-anchor-v2-report-20260914` | `c224f8bae1ed` |
| `/root/autodl-tmp/tau3-semantic-v3-20260914.bundle` | `history/bundle-tau3-semantic-v3-20260914` | `72fc81341deb` |
| `/root/autodl-tmp/tau3-perf-20260912/tau3-perf-20260912.bundle` | `history/bundle-tau3-perf-20260912` | `3a6f7c3dc26e` |
| `/root/autodl-tmp/tau3-perf-20260912/tau3-matched50-sync.bundle` | `history/bundle-tau3-matched50-sync` | `1e4ae80ca2ab` |
| `/root/autodl-tmp/tau3-perf-20260912/tau3-swanlab-sync.bundle` | `history/bundle-tau3-swanlab-sync` | `d3d129e7c5ce` |
| `/root/autodl-fs/tau3-core/code/results/legacy/experiments/multicall-aa267bb/sync/source.bundle` | `history/bundle-source` | `aa267bba7f58` |
| `/root/autodl-fs/tau3-core/code/results/legacy/experiments/multicall-aa267bb/sync/test-fix.bundle` | `history/bundle-test-fix` | `d2dcfcdb1dc0` |
| `/root/autodl-tmp/tau3-gdn-diagnosis-20260912/baseline.tar.gz` | `history/archive-gdn-baseline-20260915` | `cab42dcb6d88` |
| `/root/autodl-fs/tau3-core/code/results/legacy/experiments/multicall-aa267bb/sync/code-before-sync.tar.gz` | `history/archive-multicall-before-sync-20260915` | `3042ed0202fb` |
| `/root/autodl-fs/tau3-core/code/results/legacy/experiments/multicall-aa267bb/sync/source.tar` | `history/archive-multicall-source-20260915` | `646245422b0a` |

## 模型、数据、checkpoint 与结果位置

以下均在主仓库目录内落盘。模型和运行产物被 Git 忽略，不能用 `git clean -fdx` 清理生产目录。

| 内容 | 主仓库中的位置 |
|---|---|
| Qwen3.5-4B 与 Qwen3.8-27B 基础模型 | `models/` |
| 冻结原始数据、训练划分、parquet、SFT 输入 | `data/` |
| 其他型号的 tokenizer 测试数据（不含模型权重） | `data/tokenizers/` |
| SFT 模型、旧/新 SFT 权重 | `checkpoints/`；四份合并权重均在 `checkpoints/sft-merged/{old-off,old-on,new-off,new-on}/` 实际目录 |
| 四次 RL 及其训练 checkpoint | `results/runs/rl-c50-matched6h-a800-20260912/` |
| RL 后评估 | `results/runs/post-rl-selection-20260914/` 和 `post-rl-selection-recovery-20260914/` |
| 其他持久盘历史实验结果 | `results/legacy/experiments/` |
| GDN、性能诊断及五卡试验结果 | `results/legacy/gdn-20260912/`、`perf-20260912/`、`five-a800-20260911/` |
| 各源码目录自带的 outputs/results | `results/legacy/source-*/` |

基础模型复制后的全部普通文件内容与远程原模型一致。三份 SFT 归档也做了全文件内容比对。原持久盘的大型 results/checkpoints 通过同盘 rename 迁移，目录 inode 保持不变。

`models/Qwen3.5-4B/tau3_source_revision.json` 使用远程完整模型的 ModelScope 来源记录；本地同步版本仅标记 `tokenizer_only: true`。两份记录的模型名及 revision 完全一致（`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`），用户确认只保留远程完整模型；tokenizer-only 同步记录及其留档已删除，未复制第二套权重。

E2/E3 原入口指向已失效的 `/dev/shm`。完整 checkpoint 已同盘移动至 `results/runs/rl-c50-matched6h-a800-20260912/e2_seed42/` 和 `e3_seed42/` 实际目录；validation、rollouts、SwanLab 等辅助产物也在各 arm 内实际落盘。检查了完成标记中的所有文件尺寸，并核对每个 arm 的 23 个小型元数据 SHA256。大分片未重复计算 SHA256；它们通过同盘移动保留。历史 launch/receipt 正文不改写。

## 环境及实际路径

统一入口不设置通用 MODEL_PATH，避免覆盖正式 RL 配置中的 SFT 起点。统一入口脚本模板已纳入 `env_info/autodl/activate_canonical.sh`。三个 venv 的 editable 导入均指向主代码；FLA 在 `environment/overlays/fla`，FA2 诊断依赖独立放在 `environment/overlays/fa2-overlay`，不加入默认训练路径。

2026-09-15 已移除项目兼容软链接：`tau3-core/{runs,checkpoints,models,experiments,migration}` 及旧 `tau3-core-20260912`。当前配置和 A800 启动工具直接使用 `code/` 内实际路径。项目目录及产物的软链接已清零（扫描排除 environment、Git 和缓存）；Python 环境内部的标准运行时链接保持原样。平台软链接 `/root/autodl-fs → /autodl-fs/data` 保留，未作修改。

历史 launch/receipt、源码快照和既有迁移清单保留原文，反映当时状态。旧路径映射见 `docs/migration/20260915/remove-symlinks/path_map.json`；原日期前缀先替换为 `tau3-core`，再按最长前缀应用映射，直至路径不再变化。历史诊断脚本仍须根据其实验所需的输入、GPU 和运行条件使用，不代表本轮重新完成了历史 GPU 实验。

凭据存放在被 Git 忽略的 `.env` 与 `.private/credentials/`，不会出现在本文件或源码历史中。

## 验证

第一轮合并相关回归：253 passed，11 skipped（GPU 数值测试），1 warning。
已验证在不设置 PYTHONPATH 时，tau3_grpo、verl、tau2 仍从新的主仓库导入。
清理后的完整远程 CPU 回归：**932 passed，15 skipped**，3 warnings，耗时 402.28s (0:06:42)。原默认配置测试继承部署变量的问题已通过测试环境隔离修正。该轮输出见 `docs/migration/20260915/final-tests.txt`。

7 份源码副本已归并，26 个旧目录/文件路径已清理。首次清理时确认只有一份工作代码、产物软链接均有效；本轮已进一步改为实际文件和目录；E0–E3 的 step 20 checkpoint、优化器分片和 validation 入口均已核对。日志与完整迁移清单位于 `results/maintenance/20260915/`。
本次为无卡远程验证，不能据此宣称多卡 FSDP 或真实 GPU RL 已重新跑通。

834 MiB 的上游完整 SFT 原始语料保留于 `data/raw/areal_tau2/tau2_sft_train.jsonl`，不提交进 Git；其 SHA256 记录在迁移产物清单中。

移除项目软链接后的回归：首轮 928 passed、4 个旧路径断言失败、15 skipped；修正断言后 4 项复测全部通过，最终 932 项通过。实际目录、路径映射及验证记录见 `docs/migration/20260915/remove-symlinks/README.md`。
