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

待完成。

## 5. 结果目录中的脚本

待完成。

## 6. 文档与最终验证

待完成。
