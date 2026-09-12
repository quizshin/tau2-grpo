# 本地主目录合并正式训练版本

本地 `tau3_grpo_fix/code` 通过 Git 合并正式训练提交 `792d70db61c7e905dfda27d231d54c0ecfc5a74c`。合并前，本地 43 个文件的未提交工作已保存为 `33fe284`，可从历史恢复。没有覆盖远程工作目录、修改远程训练或推送 GitHub。

## 合并结果

- 引入正式 50 任务四组配置、FLA IEEE、padding 修复、compact head、工具解析修复、预算控制、完整检查点与 SwanLab 续接功能及对应测试。
- 保留 36 个本地独有文件，包括 5090/Paratera 配置、环境脚本、数据筛查工具和 loss projection 实现。
- 保留模拟器 `TAU3_USER_TP` 参数；默认值仍为 1。
- `dp_actor.py` 同时保留两套投影实现。正式配置启用 `VERL_QWEN35_COMPACT_HEAD`，本地低显存配置启用 `VERL_QWEN35_LOSS_ONLY_LOGITS`。同时开启会明确报错，不静默选择其中一套。
- `curriculum40` 恢复对应 40 任务；本地 50 任务草案保留在独立 `curriculum50` 配置。当前正式 50 任务运行使用 `c50_matched6h` 四组配置，两边课程测试均保留。
- 课程方案保留本地历史内容，并注明现用正式执行入口。

该结果是保留本地功能的合并版本，不是与正式提交逐字节相同的副本。原训练的精确版本仍以 `792d70d` 标识。

## 验证

Mac CPU 环境中针对课程配置、正式四组配置、预算、检查点恢复、SwanLab 离线记录、XML 工具解析、launcher、补丁契约和两套投影实现的测试：**112 passed，15 skipped**。跳过项均需要 CUDA；没有在正在使用的远程 GPU 上重复运行合并版本的数值测试。另有一条既有 Ray API 弃用提示。

本地测试环境补装项目指定的 `swanlab==0.10.0`。CPU 环境为 Torch 2.8.0、Transformers 5.17.0，与正式服务器版本不同；本次测试不能替代合并版本的 A800 全流程验证。

## GitHub 交接边界

源码已通过 Git 整合，但运行资产仍在 Git 之外：`data/`、`models/`、`results/`、虚拟环境、凭据与检查点没有纳入提交。

本地现有两份课程 manifest 的内容已核对：

- 40 任务：`results/analysis/rl_curriculum_20260912/manifests/areal_airline_train_seed42.jsonl`，SHA256 `79e317a824e8b55f7ce338c9f30a5811d83f2dce17dd896397eaffb003f5efbb`。
- 50 任务：`results/analysis/rl_curriculum50_20260912/manifests/areal_airline_train_seed42.jsonl`，SHA256 `641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae`。

这两份文件仍被 `results/` 忽略规则排除，Git clone 不会自动获得。交接运行时需单独提供课程 manifest、对应 sidecar、基础数据及模型，并核对哈希。正式启动脚本仍引用服务器激活脚本和 FLA overlay 路径，需要按目标机器配置；本次合并没有更改已经验证的远程入口。
