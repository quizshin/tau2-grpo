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
| 3 | E1 step20 | 8203 |
| 4 | 27B INT4 用户模拟器 | 8210 |

E2 step20 排在候补队列，由最先完成评测并释放服务的策略卡接续；不会固定等待某一组较慢的评测。

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

## 2026-09-14 导出修复与补评

第一次接续中，四组 RL 均已完成 step20；SFT 独立 selection60×4 完成，pass@1=43.75%，pass@4=68.33%。E0–E3 在 vLLM 加载阶段失败，尚未开始任何 smoke 或正式评测。FSDP 源检查点的 724 个张量名称正确；导出时 Transformers 的反向格式转换把 `language_model` 前缀重复插入。修复在 Qwen3.5 merger 中显式设置 `save_original_format=False`，并以重建的 BF16 FSDP state 对导出文件逐张量进行精确比较。CPU header 检查会提前拒绝重复前缀、缺少关键权重或损坏的分片索引。

用户已明确授权继续修复和执行。补评输出使用新目录 `post-rl-selection-recovery-20260914`，旧目录中的错误日志和完整 SFT 结果保留。重新从原始 step20 分片导出四组模型；E2/E3 完整续训备份复用已有持久盘副本并重新校验，不复制第二份。

```bash
export TAU3_POST_OUTPUT="$TAU3_RUN_ROOT/post-rl-selection-recovery-20260914"
bash env_info/a800_20260912/launch_post_rl_eval.sh watch \
  --reuse-sft-from "$TAU3_RUN_ROOT/post-rl-selection-20260914"
```

补评的 GPU0/1/2/3 分别运行 E0/E1/E2/E3，GPU4 保持相同模拟器。每组先 smoke4，再 selection60×4，本次新增16条 smoke 和960条正式轨迹。任务、trial seed、生成配置和评测源码保持一致；`sft-reuse.json` 保存原 SFT 结果路径和文件哈希，不重跑、不混入 smoke、不将训练内成绩代替独立评测。复用入口会拒绝协议变化、SFT结果不完整、评测源码变化或 RL 已开始评测的情况。

## 2026-09-14 清理回执

四组新导出均通过 724/724 张量精确相等校验，E0 实际 GPU 加载和工具调用预检通过，E2/E3 完整持久备份重新核验完成后，按用户授权删除两项：旧补评目录 `post-rl-selection-20260914/models` 中的错误导出，以及 `persistent-models/{e2,e3}_seed42/global_step_20/actor` 中共 8 个重复模型分片。清理前核对分片 SHA256 与完整备份回执，并检查没有进程打开待删文件。

实际释放 77.19 GiB；清理后 500 GiB 持久盘已用 264.12 GiB、可用 235.88 GiB。四组 step20 完整续训检查点、新 BF16 导出、SFT 初始化及评测、轨迹、SwanLab 记录均保留。E2/E3 旧归档新增 `model-weights-location.json` 指向完整持久备份。逐项记录见新补评目录 `storage-cleanup.json`；内存盘副本不在本次清理范围。

## E2 单条补测

用户授权仅补跑原 E2 `airline_802 / trial=3 / seed=45` 的上下文超限记录。入口 `retry_one_selection_trial.py` 校验原失败身份、评测源码、提示词协议、checkpoint hash 与实际服务，再执行一次原 `_run_one`。保留原 task/trial/seed、模型、生成温度、非 thinking、多调用、30 轮和 24,576 上下文限制；不自动扩大预算或重复尝试到成功。

结果独立写入新补评目录 `supplemental/e2-airline802-trial3-attempt1`，保存原失败、run/manifest/source hash、完整新轨迹或异常，以及 summary。原 `selection/e2` 的轨迹、错误和汇总不修改。补测是新的尝试，不是旧轨迹的精确续接；原始 E2 汇总有效性不会因补测自动改变。若补测完成，任何包含它的后续汇总都须显式标注重试来源。
