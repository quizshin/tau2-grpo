# 本地代码整理（2026-10-02）

范围：当前本地checkout；保留算法、训练参数与正式数据身份，不启动GPU/API或部署远程。
每批独立验证并提交，实际实验事实仍归`EXPERIMENTS.md`。

## 1. 当前配置和失效入口

catalog增加`current_sft`，唯一当前课程候选为portable train500/dev150 A109/B393/C500。
两个旧v1候选标为historical，保留冻结时CPU验收事实，同时声明本地历史依赖已清理，
不可按旧配置启动；原配置参数和冻结manifest未改。旧输入保留不是旧入口可运行性证明。
验证：三个当前profile可经launcher解析，包守卫验证三个阶段，未启动训练。

## 2. Teacher rollout公共实现

生成、原生工具执行、失败保留和共享预算统一在`data/teacher_rollout.py`。
新增显式`--split train|validation`及`--limit-cny 100|150`；公共默认train/100。
三个旧validation模块仅作薄转发，保留validation/100或validation/150默认值。
既有账本禁止改变总预算，增量10元及请求/轮数上限不变；混入其他split在模型调用前拒绝。
源码快照统一从包目录定位，记录公共实现及实际旧入口，吸收retry版路径修正。

两份原未提交v4文件不是整份导入Git；核对确认仅有150元上限及快照路径差异，
将差异参数化后以薄兼容入口提交。原530行文件前像保存在本地维护回执中。
原文件SHA256（重构前身份）：

- `teacher_rollout.py`：`f74472304d2fac693035cc8629300451a277c85e292c2941b69f4bdc17c42bcb`。
- `teacher_rollout_validation.py`：`75beec74c17771ffed6340b1a7fcfbaee23016413b206bf7154051e2603fc222`。
- `teacher_rollout_validation_v4.py`：`22d1446317d249386ecd11e298ade0b3d159be9da06caca49661a3aeb329c715`。
- `teacher_rollout_validation_v4_retry.py`：`663d0469aabc5c0d3a7a6197f9f6fb4b83692e0fea091f4567c24121e0bd8769`。

验证：teacher rollout及冻结prompt相关32项通过，包含原生多工具执行、私有信息隔离、
4个旧入口默认预算/来源快照、跨split拒绝和账本预算禁止变更；没有付费请求。

## 3. 数据层公共逻辑

`data/messages.py`管理审核可见事件；`data/sft_evidence.py`管理前缀证据及历史digest编码；
`data/capability_features.py`管理能力代理特征和分布汇总；`data/outcome_recipes.py`管理既有
证据审核的参考动作变体/修复。奖励/策略判定仍从evaluation公开接口读取。
5个数据构建模块不再导入analysis；4个原analysis模块的公开函数转发到同一函数对象。
迁移函数AST逐项与原实现相同，不改变消息allowlist、审核判断或历史hash编码。

验证：相关111项通过，含原生参考动作执行、冻结数据/ledger、私有信息隔离、
新解释器阻断analysis导入、旧/新函数对象一致性。首轮暴露selection_repair误从
新recipe模块读取evaluation函数的问题，修正为直接读取evaluation后全部通过。
没有重新生成数据或重写已接受结论。

## 4. 正式路径与数据清单

40/50任务清单及sidecar逐字节复制到`data/manifests/rl_curriculum{40,50}_seed42/`，
正式组件和当前数据构建/审计读取新路径；四份旧路径文件仅保留作历史输入，旧日期profile参数不改。
`data/SHA256SUMS.json`登记25份输入身份，全部核验；50任务hash仍为
`641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae`。
正式selection parquet路径改为数据根目录，当前本地未生成该parquet，训练前仍需按冻结协议准备。
共享`paths.runtime_environment`统一launcher和formal runner的外部存储根默认/相对路径解析。
模拟器默认模型/环境/cache使用当前根目录，显式旧环境覆盖仍生效。
旧curriculum脚本解除`../code_pytrio`依赖，扩展入口强制新输出目录，不覆盖冻结输入。

验证：相关135项最终通过（首次重跑121项通过，修正路径比较测试的类型断言后14项通过），
覆盖GRPO/GiGPO/MT-GTPO/ARPO、实际Hydra、DF、恢复防护、外部data root、模拟器dry-run。
与旧Hydra对照仅允许已登记的温度和数据存储路径差异，其余配置逐项相同。
首轮发现formal runner未初始化TAU3_DATA_ROOT，新旧启动器改用同一公共解析函数后修复。
未启动任何服务、采样或GPU；远程资产未迁移。

## 5. 结果目录中的脚本

逐文件登记770份Python历史产物的路径、SHA256、大小和直接引用：350份源码证据、
192份维护/验证证据、7份有直接引用的历史辅助脚本、221份未发现直接引用的历史辅助脚本。
直接字符串检索不证明无动态依赖，因此这些分类不用于批量删除。
完整索引位于本地`results/maintenance/local-code-cleanup-20261002/results-script-inventory.json`。

可复用的源码库存统计迁入`analysis/sft_source_inventory.py`，两个课程构建工具迁入
`data/rl_curriculum_screen.py`和`data/rl_curriculum_extend.py`；scripts目录仅保留薄CLI。
当前review配方位于configs/data，旧脚本目录的配方JSON仍保留其历史身份。
三个公共工具均要求新输出位置，不覆盖旧输入/报告。

已退役results中的inventory_source.py及一份与原v4完全一致的teacher副本；
原字节均有维护回执前像，新公共入口见retired-scripts.json。剩余768份原产物明确登记
为历史证据，保留原文件身份，不再把它们列为当前执行入口。没有删除实验轨迹、指标或日志。
验证：能力/新工具11项通过，源码库存的旧脚本与新函数在固定输入上完整JSON一致；
三份CLI在checkout之外可运行help，未采样、未重建正式清单。

## 6. 文档与最终验证

当前入口/资产与实验状态归CURRENT_EXPERIMENT和catalog；历史实验事实归EXPERIMENTS；
README、layout、环境入口说明同步这些索引。本报告记录代码维护，不另造竞争的实验总表。
本轮最终检查回执位于`results/maintenance/local-code-cleanup-20261002/`，包括干净CI依赖
环境的core，以及既有资产环境的benchmark/verl CPU分层回执。各层结果以receipt.json为准；
CPU条件跳过的CUDA验证、远程服务/环境及GPU训练未执行，不能声称全部平台通过。
全仓lint仍有381项已登记债务，本轮不得引入新债务；模型和数据身份独立校验。
