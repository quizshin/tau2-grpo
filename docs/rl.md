# 双 A800 强化学习

公开入口索引：`configs/experiments/catalog.yaml`。

|算法|profile|状态|
|---|---|---|
|GRPO|`configs/train/rl/repair72_grpo_2xa800_shared32_full_eval.yaml`|旧repair72双卡20步已验收|
|ARPO|`configs/train/rl/repair72_arpo_2xa800_shared32.yaml`|CPU接入；GPU验收待执行|
|MT-GTPO（可选）|`configs/train/rl/mt_gtpo_2xa800.yaml`|双卡组合待GPU验收；奖励配方需要显式冻结|

这些模板默认仍涉及旧repair72起点和selection60。新课程起点必须显式替换模型、数据与评测协议并核对生效配置，不能把旧运行成绩当成新课程收益。

仅生成配置与运行快照的示例：

```bash
python -m tau3_grpo.training.rl.runner --estimator grpo \
  --profile configs/train/rl/repair72_grpo_2xa800_shared32_full_eval.yaml \
  --result-dir results/runs/grpo-review/seed42 --updates 20 --dry-run
```

`--dry-run`不启动GPU，但会写入指定结果目录。ARPO需显式选择`--estimator arpo`和对应profile；MT-GTPO选择`--estimator mt_gtpo`，奖励版本与recipe必须匹配，不能静默跳过校准门槛。

共享双卡配置使用策略0,1、模拟器1，并通过sleeping simulator manager错开计算阶段。GRPO工程预算4组×8候选、每10步保存；ARPO loss归约和种子策略可能不同，正式对照前显式对齐。

规格：[ARPO](architecture/arpo_tau_v1.md)、[token harness](architecture/token_harness_20260920.md)。MT-GTPO保留版本化过程奖励、调用信用候选和回归测试；候选组合不等同于已证明有效算法。

公开CLI必须显式传入`--profile`，避免落入旧多卡默认值；Python resolve保留历史回放兼容性。MT-GTPO模板使用native终局协议以匹配现有IRC配方，与GRPO passenger-multiset协议不直接等价，正式算法对照前须冻结一致协议。
