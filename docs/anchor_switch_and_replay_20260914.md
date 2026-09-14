# code 目录的锚点开关与真实 DB 重放（2026-09-14）

后续实现见 [v4 状态更新与弃权机制](decision_state_v4_20260914.md)。v4 通过独立版本开关启用，真实覆盖检查未通过，缺省仍为 v1。

本轮在 `tau3_grpo_fix/code` 实施，迁入 perf 工作树的 v2/v3 锚点、审阅样例和 CPU 诊断工具，并增加真实 Airline 工具重放器。生产缺省仍为 v1，训练未启动。没有迁入 perf 分支的其他训练优化、归一化变化或硬件配置。

5090 配置原有版本保存在独立分支 `config/5090-20260914`，指向本次修改前的 `893c903`。这是现有代码和配置的版本快照；本次提交不修改 5090/Paratera 文件，也不删除当前分支原有的配置、冻结数据和模型适配。

## 锚点配置开关

统一入口是环境变量 `TAU3_GRPO_ANCHOR_VERSION`，可在 YAML 的 `launch.environment` 中设置，也可显式通过 shell 环境覆盖。环境变量优先于 YAML 默认值。

|值|行为|当前用途|
|---|---|---|
|v1|原有 task+DB+信息掩码及关键词确认特征|默认、历史正式实验兼容|
|v2|完整可见对话/读结果/工具事件证据|保守候选，存在过度拆组|
|v3|在 v2 上加入有限句式语义规范化|候选，尚未解决真实长句拆组|

版本进入每个 session、Ray worker 环境和算法运行元信息。未知版本在创建 rollout session 前拒绝。运行目录通过 `anchor_protocol.json` 固定协议；历史 v1 检查点目录不能静默切换到 v2/v3。

已有正式 c50 common 配置显式固定 v1。两个独立候选配置：

- `configs/train/rl/qwen35_4b_full_a800_anchor_v2_20260914.yaml`
- `configs/train/rl/qwen35_4b_full_a800_anchor_v3_20260914.yaml`

它们设为 2-step、禁用评测及检查点保存，并使用新结果目录，仅供后续明确决定开卡时检查。继承旧归一化，**没有**加入 perf 分支的 `episode_normalization=grpo` 混合变体。因此可以单独考察锚点版本。若改跑 E3，必须另设 `RESULTS_DIR` 和 SwanLab 实验名；不要复用 E2 目录。

只检查配置、不启动训练：

```bash
TAU3_ROOT=/path/to/runtime python -m tau3_grpo.launch rl \
  --config configs/train/rl/qwen35_4b_full_a800_anchor_v3_20260914.yaml \
  --experiment e2 --dry-run
```

本轮没有实现完整的“目标＋方案＋批准条件”状态更新器；v3 仍是有限规则。不能把配置已迁入理解为信用分配问题已经解决。

## CPU 真实工具重放

入口：`python -m tau3_grpo.analysis.replay_decisions`。
配置：`configs/analysis/decision_replay_20260914.yaml`。

- `enabled: false`：直接返回，不加载数据/环境、不创建输出目录。
- `enabled: true`：显式执行离线重放；不挂在训练启动路径中。
- `anchor_versions`：比较的锚点版本。
- `comparison: json`：只忽略 JSON 键顺序和空白，保留列表顺序、字符串、布尔和数值表示差异；也支持严格的 `exact_text`。
- `task_ids`：首批 1035、1060、1106；`[]` 表示该冻结 selection manifest 中的全部任务。

从 code 目录执行，输入为已有轨迹文件，输出目录必须是新的：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=.:verl:tau2-bench/src TAU3_GRPO_TEXT_ONLY=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-cpu/bin/python -m tau3_grpo.analysis.replay_decisions \
  --config configs/analysis/decision_replay_20260914.yaml \
  --input /path/to/e2/trajectories.jsonl \
  --input /path/to/e3/trajectories.jsonl \
  --output results/analysis/decision_replay_20260914/new_run
```

重放器校验 manifest 的 task 指纹、DB 文件 SHA256、任务/来源/split 和记录 policy。只接受 AReaL selection 的 half-duplex 文本轨迹；不加载官方最终任务，非空 initial_state 暂拒绝。

每条轨迹使用 fresh official Airline Environment，按生成列表顺序执行所有工具调用，包括修改操作；这些修改只作用于隔离的内存 DB。每个返回都检查 ID、error 标志和内容。调用缺失、乱序、重复 ID 或返回不匹配时停止认证后续状态，先前已验证的前缀仍单独标记为可信前缀，不把整条轨迹报成功。

每次 assistant 动作之前保存实际 DB hash、可见前缀 hash、policy hash 和 v1/v2/v3 锚点。当前尚未执行的动作、后续回复、奖励不进入该快照。原始消息不写入摘要，凭输入文件 SHA256 与 message index 可定位前缀。所有输出记录本地环境/锚点源码 hash。

这认证的是**在固定本地实现下可复现的工具状态及可见前缀**，不恢复用户模拟器隐藏状态，不证明语义等价，也不是在线训练优势回放。

## 首批实测

输入为此前保存的 E2/E3 selection 轨迹，task 1035/1060/1106，各 4 trials，共 24 条。结果：

- 24/24 条轨迹全部重放验证通过。
- 108 次工具返回一致。
- 248 个动作前决策位置，224 个非初始位置。
- 实际 DB 下，v1 有 182/224 个非初始位置跨 trial 重复（81.25%）；v2/v3 均为 0/224。

这里按同一实验、同一 task 的 evaluation trials 统计，未恢复原训练采样组。v1 高匹配不代表分组正确，v2/v3 零匹配也不等于其实现没有接通：它与此前的过度拆组诊断一致。

结果保存于 `results/analysis/decision_replay_20260914/`。原轨迹文件仍是 Git 外运行资产；审阅用的 47 对 fixture 已迁入 `tests/fixtures/decision_state_pairs_20260914.json`，来源及人工/模型审阅限制保留在文件内。

## 验证与下一步

CPU 回归 **225 passed, 2 warnings**（15.03 秒）；两个 warning 为第三方弃用提示。包括真实 DB 的同轮写后读、结果不匹配截断、缺失/乱序/重复调用、失败返回、动作前无未来信息、关闭分析开关、v1/v2/v3 传递至 shell/Ray、旧版兼容、并发隔离和优势算法回归。没有模型前后向或 GPU 训练。

下一步可直接用已验证的中间 DB/前缀来审阅“改舱确认后”的状态，并开发有限范围的目标/方案/批准更新器。解析不清楚的状态应退出步级比较，而不是退回 v1 强行合并。现有 v3 不建议直接重训正式实验。
