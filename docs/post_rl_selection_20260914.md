# E3 完成后自动独立评测

用户已授权执行。新增独立接续控制器，不修改运行中的 RL 控制器或训练算法。

## 顺序与 GPU

CPU 阶段提前校验 new-off SFT，导出 E0/E1/E2 step20 的 BF16 Hugging Face 推理模型，并将 E2 完整续训状态复制到持久盘。原 FP32 分片保留。

接续器每 15 秒检查一次完成信号，不依赖每小时巡检。只有 RL controller-state=complete、controller.exit=0、原控制器退出、四组 step20 完整检查点/评测/云端步数验收通过，且 GPU0–4 已释放，才准备 E3 并启动 GPU 评测。训练暂停、失败或 GPU 被其他进程占用时不会强行接管。

| GPU | 任务 | 服务端口 |
|---|---|---|
| 0 | SFT new-off | 8200 |
| 1 | E0 step20 | 8201 |
| 2 | E3 step20 | 8202 |
| 3 | E1 step20，完成后接 E2 step20 | 8203 |
| 4 | 27B INT4 用户模拟器 | 8210 |

每个模型先独立运行固定 4 个 selection 任务各 1 次 smoke，然后执行 selection60×4。共 20 条 smoke 和 1,200 条正式轨迹，分别保存，smoke 不混入成绩。smoke 验证端点/协议/完成性，不按成功率挑选模型。

每模型最多并发 4 条轨迹，总并发上限 16。策略统一 BF16、关闭 thinking、顺序多调用、temperature=0.4、max model len=24576、engine seed=42；忽略各 checkpoint 自带 generation_config，统一使用 vLLM 默认生成配置。模拟器沿用相同 27B、temperature=1.0、16384 上下文。每任务 4 次种子为 42/43/44/45，官方 Orchestrator 最多 30 step。

这些独立评测与训练内评测分别留存，不能把两者结果混为同一次评测。共享服务下的实际吞吐尚须 GPU smoke 验证，不承诺 8 点前全部完成；不在固定钟点杀掉未完成评测。

## 存储和运行入口

2026-09-13 晚上持久盘已扩为 500 GiB。完整备份写入训练根目录 `persistent-checkpoints/e2_seed42` 与 `e3_seed42`；包含最新 checkpoint 的模型、optimizer、extra state、data.pt、完成标记及顶层训练配置/日程/SwanLab 身份。逐文件 SHA256 校验后原子发布，原内存盘副本保留。完整对话等其他产物继续由已有 persistent-models 归档保留。

启动入口：

```bash
bash env_info/a800_20260912/launch_post_rl_eval.sh dry-run
bash env_info/a800_20260912/launch_post_rl_eval.sh prepare
bash env_info/a800_20260912/launch_post_rl_eval.sh watch
```

- dry-run：校验冻结的 60 任务及 DB 哈希、保存计划，不启动 GPU。
- prepare：CPU 模型导出和完整备份，不启动 GPU。
- watch：先完成/核验 CPU 准备，等待 E3 正常结束后自动评测。

运行目录：`/root/autodl-fs/tau3-core-20260912/runs/post-rl-selection-20260914`。

主要证据：`plan.json`、`software.json`、`state.json`、`controller.pid`、`controller.exit`、各模型 `*-export.json`、`e2-backup.json`、`e3-backup.json`、`*-attestation.json`、`selection/<model>/{run,summary}.json` 与完整 trajectories/errors 文件。服务、smoke、正式评测各有独立日志。

控制器使用进程锁防止重复启动；已有正式评测目录时拒绝自动覆盖或重跑。每个输出的 task/trial/seed 必须与冻结计划相同，任何未评分试验都会保留错误并使该模型评测无效，不取剩余任务排名。只结束自己启动的进程组，不执行全局 ray stop。

本轮只执行 selection，不访问官方 final、不创建 winner lock、不追加 RL、不自动关机。GPU 评测实际 smoke 在 E3 完成后执行；CPU 单测或 dry-run 不表示 GPU 服务已实测通过。
