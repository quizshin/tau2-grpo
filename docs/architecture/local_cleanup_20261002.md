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

待完成。

## 6. 文档与最终验证

待完成。
