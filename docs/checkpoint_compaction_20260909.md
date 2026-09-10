# 检查点精简记录 · 2026-09-09

用户授权保留必要模型参数，删除冗余检查点。清理前持久存储盘约116GiB，数据盘约24GiB。本次不删除代码、训练数据、运行环境、日志、奖励及审计记录。

第一批于21:53前完成，53个冗余文件已删除，两个原始基座完成两盘去重，18条保留参数/配置hash记录重新核验通过。持久盘实际使用56.75GiB，释放59.39GiB；数据盘使用23.19GiB。下表原计划60.27GiB包含数据盘重复文件，和持久盘单独释放量不同：

| 项目 | 计划删除量 GiB | 保留内容 |
|---|---:|---|
| 旧0.8B LoRA RL-003/005 | 9.14 | 各自最终适配器、配置及精确SFT基座 |
| 4B E0/E1 RL-009/010 | 21.46 | 各自最终适配器、配置及共同SFT基座 |
| 旧全参RL-007/008 | 15.34 | RL-008唯一最终完整模型、模型配置 |
| SFT冗余精度副本及优化器 | 3.99 | 已评估BF16全参SFT导出、最佳LoRA及合并模型 |
| 原始0.8B/4B基座两盘去重 | 10.35 | 数据盘逐文件hash一致副本，持久路径改为软链 |

最终LoRA均先与完整检查点的LoRA张量逐元素比较，确认一致且有限；旧全参SFT的FP32副本先验证转换至已有BF16导出后完全一致。基座去重比较所有非缓存文件SHA256。删除逐项记账，完成后重新核验保留参数的SHA256。

执行回执`initial-receipt.json`记录completed=true、protected_hashes_verified=true；精确两盘占用记录在`initial-disk-state.json`。必要27B INT4模拟器、SFT合并基座、最终适配器与旧RL-008最终全参参数均保留。原始0.8B/4B去重后依赖本实例数据盘副本，后续迁移到其他实例时须同时迁移数据盘模型或重新下载，再修复对应链接。

执行脚本：`results/validation/checkpoint_compaction_20260909/compact_checkpoints.py`。远程脚本位于`/root/autodl-tmp/tau3/runs/compact_checkpoints.py`；计划与执行回执保存在两盘的`checkpoint_compaction_20260909`目录，本地小证据镜像位于`results/validation/checkpoint_compaction_20260909`。

精简后只保证推理/评估所需模型参数，完整优化器续训状态已退役。历史`preservation.json`和完整检查点审计不覆盖精简后的库存；不得据旧清单重跑完整归档或恢复。当前库存以新增`inference-only.json`及清理回执为准。

E2、E3全套训练、审计与本地证据hash核验完成后，已执行`--phase final --apply`。第二批删除18个冗余文件，持久盘再释放20.73GiB，最终实际使用46.57GiB，低于50GiB目标；数据盘使用22.86GiB。最终LoRA与完整检查点逐张量一致，保留参数hash复核通过；14项清理证据已同步本地并校验hash。最终回执为`final-receipt.json`，占用为`final-disk-state.json`，当前模型库存为各run的`inference-only.json`。

两批合计删除71个冗余文件并去重两个原始基座，持久盘累计释放约80.13GiB（期间E3新增检查点，因此清理前后占用差不等于累计释放量）。原始最佳SFT适配器及小型最佳step24证据保留。

全部训练、验收、精简和本地证据同步已完成。23:35已从AutoDL平台核实051（实例ID`857546be50-4fdafb0e`）与SSH端点`root@connect.nma1.seetacloud.com:28713`一致，执行正常关机并回读“已关机”，磁盘保留。状态记录为套件目录`shutdown-status.json`。平台提示连续关机15天后自动释放实例，届时数据盘会被清空；数据盘上的原始基座副本依赖该实例保留或后续迁移。
