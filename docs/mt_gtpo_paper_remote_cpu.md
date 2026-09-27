# MT-GTPO paper_v1 远程无卡验证

结论：代码与配置链路通过远程 CPU 验证；真实历史训练 buffer 的 IRC 校准未通过，没有导出冻结配方，没有启动训练、模拟器或新增 rollout。CPU 验证不包含 GPU/FSDP 更新、续训或算法效果验收。

## 环境、同步与证据

- 远程主仓库：`/root/autodl-fs/tau3-core/code`；复用 `environment/venvs/qwen35/bin/python`，PyTorch `2.11.0+cu130`。
- 显式设置 `CUDA_VISIBLE_DEVICES=""`、`NVIDIA_VISIBLE_DEVICES=void`；运行时 `torch.cuda.is_available() == False`、设备数为 0。
- 同步前检查双方工作区、相关依赖及远程运行状态；只同步 14 个论文修复相关文件，备份 6 个已存在文件，逐文件核验 SHA256。另有 10 个相关依赖与本地一致。随后对航班查询分类补修的 2 个文件再次备份与核验。
- 证据目录：`results/maintenance/mtgtpo-paper-cpu-1789661265079335246/`，远程保存完整备份；本地同名目录保存测试、配置预检、重算报告和源码清单。
- 首轮 148 项测试全部通过，无跳过，耗时 316.95 秒。唯一 pytest 警告来自 Ray 的旧导入路径弃用提示。
- 航班查询补修后，重跑奖励、IRC、Hybrid 与 veRL 适配相关 69 项测试全部通过，耗时 130.63 秒；ruff 与差异检查再次通过。两轮测试有重叠，不能相加为独立用例数；补修轮警告为 audioop/Ray 弃用提示。
- ruff、veRL 补丁契约、`git diff --check`、正式入口 `--reward-version paper_v1 --updates 20 --dry-run` 均通过。
- dry-run 核验了 50 个训练任务、60 个 selection 任务、20 step、1,280 条候选轨迹与历史任务调度一致，并正确标记 `irc_calibrated=false`。dry-run 不启动训练。
- DF 开关、联合优势后的过滤、三种 estimator 共存、旧奖励回归、冻结配方与续训限制均在相关测试中覆盖。

## 真实数据重放发现的分类遗漏

首轮 IRC 报告发现 45 条轨迹含未知工具类别，共 87 个调用，全部为 `get_flight_status`。基准的 `AirlineTools.get_flight_status` 明确标注 `ToolType.READ`，paper 分类遗漏了该查询。

已仅在 paper 分类分支补齐只读查询，保留 error/gold/duplicate/soft 的优先级；旧 audit/reference_write 版本含义不变。新增回归验证首次查询为 read_only、重复查询为 duplicate、执行错误为 error、gold 查询仍可获得 gold 奖励，以及旧 audit 结果不变。

修复后的同一 buffer 重算显示 unknown 归零。由于原 unknown 与 read_only 权重均为 0，本次补修不改变该 buffer 的逐轮奖励和 Hybrid 优势；只改变类别覆盖与 IRC 的未知类别检查。没有降低阈值或修改 soft/gold 的校准要求。

## IRC 结果

输入仅为已完成 v3 训练的 `results/runs/mt_gtpo_reference_write_v3/20260917_s42_df0/rollouts/1.jsonl` 至 `20.jsonl`，共 1,280 条。先逐 update 验证原始奖励、优势和过滤 mask 可重算，再按 paper 分类重算候选。

50 个训练任务按固定哈希划分为 40 个校准任务（960 条轨迹）和 10 个留出任务（320 条）。没有使用 selection60/final50 拟合。20 个 update 合为一轮 buffer；补修前后是同一 buffer 的两次工程验证，不是两轮新策略采样，也不是 IRC 迭代收敛证据。

| 检查项 | 校准集 | 留出集 | 结论 |
| --- | --- | --- | --- |
| gold_exact 出现轨迹数 | 955/960 | 318/320 | 未出现样本仅 5/2，低于支持数要求 20，不能稳定估计类别相关性 |
| soft_match 与成功的相关系数 | -0.151786 | -0.182029 | 与预设正向语义冲突，不能把负权重建议当成验收通过 |
| 候选平均过程奖励与终局的相关系数 | 0.078403 | 0.184206 | 校准集未超过阈值 0.1 |
| duplicate 平均 Hybrid 优势 | -0.305122 | +0.027035 | 留出集违反负向要求 |
| unknown 出现轨迹数 | 0 | 0 | 分类补修后通过 |

soft_match 在校准集的候选平均 proxy/Hybrid 优势均为负；留出集 proxy 为负而 Hybrid 为正，进一步说明即时奖励方向和最终信用分配不同。其余失败项详见 `real-buffer-irc-flight-fix/report.json` 的 `issues`。

最终状态 `needs_new_buffer_or_reward_revision`，退出码 2；这是预期的校准拒绝结果，程序没有崩溃。没有生成 `frozen-recipe.json`，候选负 soft 权重没有应用到正式训练配置。

该验证证明了工程流程可运行，也暴露了奖励定义/统计支持/优势方向尚未满足要求。历史结果标签固定，离线改分无法证明新策略成功率提升；下一步应先处理这些校准问题，再另行开展 GPU 与同预算效果对照。
