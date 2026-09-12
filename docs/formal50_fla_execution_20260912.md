# 50 任务四组正式实验：接入已验证的 FLA 优化

用户在读取优化任务 `01a09471-81fb-7131-9219-50114e1fcf23` 后授权启动正式实验。本工作树为 `perf/a800-e0-20260912`，远程目录为 `/root/autodl-tmp/tau3-perf-20260912/code`。优化任务的工作树和原始主工作树保持独立。

## 本次采用的计算路径

从优化任务移植 `qwen35_fla_ieee.py`，以及模型构建前的 runtime 设置和构建后的 GDN 安装入口。三个文件与其在线实验源文件 SHA256 一致。前次完整实验及适用限制见 [FLA 在线验证](fla_online_validation_20260912.md)；原始小型证据复制到 `env_info/a800_20260912/fla_adoption_evidence/`。

- FLA 0.5.2；Triton FP32 和 triangular solve 均使用 IEEE，关闭 TF32。
- Actor/reference 参数和计算均为 FP32；保留原生 GDN norm，FP32 SDPA 显式重复 KV heads。
- 每卡 microbatch=1，裁去单条轨迹外部 padding，不合并轨迹，不复用递推状态。
- Actor 参数与优化器不 offload；reference 参数 offload。
- 保留 compact checkpoint head，chunk=256，以及 singleton padding guard。
- 不接入外部 FA2；`use_remove_padding=false` 保留，因为本实现通过 compact 路径裁剪，而非 veRL 通用 packing；`use_fused_kernels=false`、`bypass_mode=false`。

前次完整在线 step 18.83 分钟、actor 更新 6.74 分钟，仅为一次测量。前次采用旧 40 池的八个任务，当前正式实验采用冻结的 50 池。此前 64 条轨迹通过，不保证所有新长轨迹均不会 OOM；不根据单次 29/64 采样成功率声称 RL 提升。

## 四组协议

|组|算法|共同起点|
|---|---|---|
|E0|GRPO|new-off 非 thinking SFT|
|E1|GRPO + DF|同一个 SFT，独立初始化|
|E2|Tau-GiGPO|同一个 SFT，独立初始化|
|E3|Tau-GiGPO + DF|同一个 SFT，独立初始化|

任务池固定 50 个，每 step 8 个任务 × 每题 8 次；不减少任务或采样。Manifest SHA256 为 `641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae`。四组调度使用相同前缀，学习率 1e-6、KL 0.01，多工具按顺序执行。

E0→E1→E2→E3 依次运行。E0 从模拟器启动计时，约六小时后在完成的 step 观察进度，向上取整至 10 的倍数 N 并完成该节点；其他组均训练 N 步。50 步不是硬性目标，E0 探索上限为 100。保存和评测耗时计入预算；不能按 18.83 分钟简单保证六小时结束。

每 10 step 保存完整恢复状态并评测 selection60×4。每组只保留最新完整检查点，新检查点的四 rank 模型、优化器、随机状态及 dataloader 元数据完整后才删除旧目录。各节点评测结果、轨迹、日志保留。SwanLab 每 step 实时上传，续训同一 run ID，恢复 30 后下一点为 31。

运行根为 `/root/autodl-fs/tau3-core-20260912/runs/rl-c50-matched6h-a800-20260912`。正常停止写入该目录的 `STOP_AFTER_BOUNDARY`，完成下一整十节点的保存与评测后暂停整个队列。启动前归档原先的 `HOLD_UNTIL_USER_START`；这次用户授权允许启动。

## 启动和容量保护

`bash env_info/a800_20260912/launch_matched50.sh dry-run` 展开四组配置；`verify_fla_formal.py` 将实际计算参数与此前通过的在线配置逐项对照，检查源文件、导入路径和 50 任务数据哈希。正式入口为同一脚本的 `run` 参数。

GPU 0–3 为策略，GPU 4 为模拟用户，端口 8100；启动前拒绝占用，不终止其他任务的进程。启动器只引入已有 FLA overlay，不安装或更改共享环境。

控制器每 15 秒将整卡显存与利用率写入 `gpu-memory.log`，用于辅助观察长轨迹和保存阶段的显存压力；采样最大值不等于连续测量峰值。

启动检查时 fs 配额 200 GiB、空闲约 153 GiB，tmp 空闲约 19 GiB。前次完整检查点约 50.6 GiB，四组共约 202.4 GiB，超过当前持久存储可用量。因此保留每组开始前至少 110 GiB 的轮换空间检查；E0 可先执行，后续可能在 `waiting_for_storage` 暂停。不会删除其他组 optimizer 或将内存盘当作唯一恢复存储来绕过保护。

## 接入验证

本地在显式指定本工作树的 PYTHONPATH 后，89 项配置、课程调度、预算、checkpoint 续训、补丁和 launcher 测试通过。本地 CPU venv 缺少 SwanLab，相关 15 项在远程正式依赖环境运行。第一次本地测试未显式设置 PYTHONPATH，误导入主工作树 veRL；该失败不作为本次实现的验证证据。

服务器 104 项回归测试通过，包含 15 项 SwanLab 测试。回归结果与配置核对结果保存在 `env_info/a800_20260912/fla_adoption_evidence/`。本次未重复前次的 64 条数值回放；采用完全相同核心源文件及计算配置，正式 E0 用于进一步观察连续多步表现。
