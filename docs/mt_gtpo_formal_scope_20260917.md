# MT-GTPO 首次正式训练：奖励范围与运行协议

用户授权：确认过程奖励范围后，在远程按 `AGENTS.md` 启动训练。本次使用独立实验目录，未改动官方评分或历史 E0–E3 奖励。

## 冻结奖励范围

`algorithm.process_reward.mode=reference_write`、`version=v2`。这是带终局成功条件的参考写操作奖励实验，不是原论文完整复现，也不声称解决全失败组的正向奖励稀疏问题。

- 官方终局成功由原 `DB + COMMUNICATE` 验证器给出，不添加 `ACTION`，不改变对外成功率。
- 正奖励仅适用于任务包含 DB 评分、官方 outcome=1、工具执行无错误、工具为数据库写操作且完整参数精确匹配尚未消费的参考动作。
- 数据库写操作白名单：`book_reservation`、`cancel_reservation`、`send_certificate`、`update_reservation_flights`、`update_reservation_passengers`、`update_reservation_baggages`。转人工不在正奖励白名单中。
- 每个参考写动作首次匹配获得 `1 / 参考写动作总数`，一条轨迹正奖励总量不超过 1。执行失败不消费匹配机会；重复匹配不重复支付。
- 查询、普通对话、软匹配、非参考写操作及未知类别均为 0；明确工具执行错误为 -0.1。不对“未匹配参考”本身扣分。
- v2 严格比较完整参数，不让 `compare_args` 隐藏错误的写入参数；不模糊匹配 ID、不改变列表顺序、不将数字字符串自动转为数字。等价但不同的合法路径可能漏奖，这是当前范围限制。
- 不引入 LLM judge、语义抽取器或额外模型；奖励在终局之后计算，参考动作不进入策略上下文。
- 终局成功门控不是独立的授权／政策验证，仍继承官方评分的盲点。不能据此声称每次受奖动作都取得了正确用户授权。
- `audit/v1` 和 `conservative/v1` 保持原行为；`get_flight_status` 的只读分类修正在 v2 中启用。

旧 conservative/v1 在本次 50 个任务上无法给正奖励，因为这些任务均不含 ACTION。这是本次新增显式 v2 配方的原因，不将 v1 默认为 v2。

## 检查与证据

启动前使用 `tau3_grpo.analysis.audit_mt_gtpo_scope` 检查整个训练池及历史 E0 前三个训练批次。不使用 selection 或 final 标签拟合权重。

参考动作在新建的官方数据库实例中顺序执行，分别以合成 outcome=1/0 检查奖励范围与上限；这不是重新测得的真实任务成功率。历史训练文本使用生产 XML 参数解析器，要求工具调用计数和初始／最终 DB hash 一致后，才纳入调用级奖励统计；不一致的记录单列。

旧记录缺少可靠轮次 span，因此此次历史审计不声称重算了当时的逐轮优势，也不声称每条 observation 或用户授权均已验证。新训练保存完整 turn/span/奖励/过滤记录，供首步结束后精确重算。

远程审计输出：`results/analysis/mt_gtpo_scope_20260917/audit.json`。测试包括 v1 兼容、v2 正奖励门控与预算、重复／错误／软匹配、worker-trainer outcome 一致性、DF 两种模式和离线重算。

## 本次训练协议

- `mt_gtpo`；gamma=0.9，lambda_outcome=0.3；DF 关闭，仍可通过独立配置开启。
- Qwen3.5-4B `new-off`，非 thinking，全参数；50 任务，seed42，每步 8×8 条，20 step／1,280 条候选轨迹。
- GPU0–3 策略、GPU4 独立 27B 模拟器；沿用经过验证的 FLA FP32/IEEE、padding 裁剪和 compact head 配置。
- 每 10 step 保存完整恢复状态，并评测 selection60×4；完整新检查点校验后清理旧检查点，历史评测结果保留。
- SwanLab 在线，每步按真实 global_step 记录；明确 resume 路径后使用同一 run ID。正常暂停通过结果目录中的 `STOP_AFTER_BOUNDARY` 请求。
- 入口：`scripts/train/rl/run_mt_gtpo_formal.py`；正式 profile：`configs/train/rl/qwen35_4b_full_a800_mt_gtpo_formal_20260916.yaml`。
- 每次运行保存实际展开配置、代码 SHA256、Git HEAD、未提交差异及源码归档。文件存在及 CPU 测试通过不等于 GPU 训练完成。

```bash
source /root/autodl-fs/tau3-core/activate.sh
cd "$CODE_ROOT"
python scripts/train/rl/run_mt_gtpo_formal.py \
  --result-dir "$CODE_ROOT/results/runs/mt_gtpo_reference_write_v2/20260917_s42_df0" \
  --dry-run
```

去掉 `--dry-run` 启动单次正式实验；它拒绝占用中的 GPU、已有训练状态和不足以轮换完整检查点的空间。`--dynamic-filter` 仅用于新的独立 run。续训使用同一目录、`--resume-from` 指定完整整十检查点、`--updates` 指定更晚的整十目标，任务调度前缀保持不变。
