# MT-GTPO split-v4 并发续采（2026-09-20）

用户在确认 SFT 评测完成后，明确要求重启服务并开始采样。本次目的为补充训练池上的 IRC 校准轨迹，不进行参数更新。

- 实例：051，原独立评测四个模型均完成 240/240，错误数均为 0；原控制器完成清理后才启动新服务。
- 运行：`results/runs/v4_shared_irc/20260920_sft_c4_resume/`；sampler 输出在其 `sampling/` 子目录。
- 服务：GPU 1 上原 SFT new-off，GPU 4 上 Qwen3.8-27B AWQ 模拟器；共两张 A800。SFT max-num-seqs 从 4 改为 8，模拟器仍为 16；客户端并发 4。
- 预算：服务加载计入 1 小时墙钟，2026-09-20 23:03:22 至 2026-09-21 00:03:22（北京时间）。提前 15 秒停止采样；GPU 服务到期关闭。仅 CPU 校准另有最多 260 秒收尾时间，不关闭平台实例。
- 目标：保持原 50 训练任务 × 每组 8 条，共 400 条。已完成的 20 条原样带入，只排队缺失的 380 个 task/trial 身份。首先补 airline_739 的 trial 4–7。
- 每个 worker 有独立 ToolAgentLoop、interaction 和原生私有 session。沿用原 seed、温度、native legacy training harness、上下文与轮数上限；v4 指奖励版本，未启用额外 token harness。
- 配方仍为此前未通过的候选，DF 关闭。仅完整 8 条同组轨迹进入 IRC；不足 50 个完整组时明确使用 development-only，不冻结训练配方。

## 验证与运行证据

远程 CPU 检查通过：20 条身份无重复、380 条待采无重叠、两个旧完整组精确回放、不完整组被拒绝、奖励实现身份未变。当前 1,021 个源码文件哈希与此前评测/共享服务期间的源码快照全部相同。

运行脚本在 `results/analysis/v4_shared_sampling_20260920/`：`collect_resume.py` 复用原 `configure_loop` 与 `replay_group`；`supervise_resume.py` 记录并管理仅本次启动的子进程。原串行运行与旧评测结果不覆盖。

原始轨迹保存真实 token IDs、mask、服务 logprobs、官方评分和过程奖励。服务 logprobs 未声明与训练 processed-logprobs 等价。保留先前尝试和本次 provenance，已审阅过的任务划分不称为新盲测。

状态以 `controller-status.json`、`sampling/status.json` 及回执为准。只读查询：`python3 results/analysis/v4_shared_sampling_20260920/fetch_resume_status.py`。启动服务不等于完成校准，校准通过前不宣布可以正式训练。

## 23:53 实际运行验收

累计 104/400 条，其中本轮新增 84 条，13 个完整组，0 个采样错误。原 20 条逐条内容完全一致；84 个新轨迹 session 身份无重复，4 个 worker 均有完整产出。823 个 assistant turn 的实际 prompt IDs、生成 token IDs 和 logprobs 与服务回执精确匹配。审计回执：`results/analysis/v4_shared_sampling_20260920/resume-live-audit.json`（远程运行目录 `live-audit.json`）。采样仍在进行，00:03 到期自动收尾；此状态不是 400 条采集完成或校准通过。
