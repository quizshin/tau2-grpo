# 搬迁 SFT 模型并接续 E1–E3

用户授权搬迁三个合并 SFT 模型并启动 E1，随后明确允许 E2/E3 的检查点暂存 300 GiB 内存盘。

## 已完成搬迁

持久盘原目录 `/root/autodl-fs/tau3-core-20260912/checkpoints/sft-merged/` 下：

|模型|新位置|大小|
|---|---|---:|
|old-off|`/root/tau3-sft-archive-20260913/old-off`|8.47 GiB|
|old-on|`/root/tau3-sft-archive-20260913/old-on`|8.47 GiB|
|new-on|`/root/autodl-tmp/tau3-sft-archive-20260913/new-on`|8.47 GiB|

逐文件复制并比较 SHA256 后，将原目录替换为软链接，再释放已校验的旧副本。全部校验通过；持久盘空闲从 101.41 增至 126.83 GiB。回执为远程 `migration/sft-archive-20260913.json`，本地副本 `env_info/a800_20260913/sft-archive-receipt.json`。

共同起点 new-off、LoRA 原件、E0 step20 完整续训检查点保持原位。系统盘上的合并模型可由保留的 LoRA 和基座重建，不把不可重建的 RL 完整状态迁入系统盘。

## 接续运行

入口 `bash env_info/a800_20260912/launch_matched50.sh e1`，dry-run 为 `e1-dry-run`。启动前核对 E0 completion、四 rank 完整检查点文件大小、两个 240 条评测和已核验的 SwanLab 20 步，拒绝覆盖已有 E1/E2/E3。E0 不重跑。

E1、E2、E3 均从 new-off 独立初始化，各 20 step，保持 50 任务、8×8、相同调度前缀、FLA IEEE 与 SwanLab 连续性设置。E1 写持久盘；E2/E3 的运行目录链接至 `/dev/shm/tau3-matched50-20260913/e2_seed42` 和 `e3_seed42`。软链接根目录下的整十检查点发布和旧 checkpoint 清理已增加测试。

每组完成后核对 10/20 步评测、完整检查点和 SwanLab 云端步数。模拟器由同一控制器管理，队列结束或失败时释放自己启动的进程。正常主动停止仍使用 `STOP_AFTER_BOUNDARY`，整十保存评测结束后暂停整个队列。

## 内存盘与持久化

实查 `/dev/shm` 上限 300 GiB，容器 cgroup RAM 上限 600 GiB，迁移后空闲运行时占用约 68 GiB。内存盘与训练共享 RAM；启动 E2/E3 前检查内存盘至少 110 GiB 空闲、cgroup 至少 160 GiB 余量。E2/E3 各最新完整检查点约 50.6 GiB，加一次轮换约 152 GiB，实际占用随训练监测。

E2/E3 完成后，最终四 rank 模型权重（每组约 19.28 GiB）、配置、评测和运行记录复制到持久盘运行根的 `persistent-models/<arm>_seed42`。权重副本再次核对 SHA256；归档不含 optimizer，不复制完整 checkpoint 完成标记，`archive.json` 明确标注这一限制。持久模型归档不宣称能恢复原 Adam 状态。

E2/E3 完整模型、optimizer、RNG 和 dataloader 仍保留在内存盘；要精确续训，关机、重启或销毁实例之前必须导出这些完整状态。控制器不会自动关机或删除内存盘检查点。

预算估算：持久盘 126.83 GiB，E1 最终检查点约 50.6 GiB，E2/E3 持久模型归档合计约 38.6 GiB，结束后预计仍余约 37 GiB，另扣运行日志与小文件。E1 新旧检查点共存期间，预计持久盘仍余约 25 GiB。
