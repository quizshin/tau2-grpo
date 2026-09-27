# 奖励重算与分层诊断修复（2026-09-18）

## 完成本轮的范围

新增可复现的 `tau3_grpo.analysis.audit_reward_transfer`，把上一轮一次性的字段盘点推进到真实轨迹重算。
支持已有 MT-GTPO process 记录，以及 SFT／E0–E3 独立评测的 simulation messages。
修复的是离线数据读取和诊断不足；没有声称奖励公式或模型效果已经修好，也没有把验证集上低相关性改成通过。

读取器按工具 ID 和执行顺序恢复同轮多调用，保留明确的 error、observation 及截断标记；缺返回、重复 ID、乱序返回、跨轮未完成调用均拒绝。
失败重试不消费 gold，状态变化后的重复读取继续由 `paper_env_v2` 判定。
原始 MT-GTPO 过程记录必须先通过原配方重算验证，才进入候选配方诊断。
模拟消息使用显式标注的合成回合区间，只计算奖励；不会导出伪造的训练 token mask、原始优势或 mt_gtpo_replay。

独立评测中的“轨迹写入成功”不等于“完成官方 DB 评分”：固定版本官方 evaluator 在提前终止时返回 reward=0、reward_basis=null。
新入口保留这个 0 分用于原口径，同时单独记录 scored=false、提供仅官方已评分轨迹的敏感性统计。
不把 null basis 当作有效 DB 失败，也不将这类轨迹静默丢掉。
初版读取器因 null basis 拒绝的 128 条在修复后全部恢复；初版输出留存为调试证据，不作为结论来源。

新增读／写 gold 分项、工具类别统计、回合数、总体与任务内去均值相关性。
奖励分项求和必须重构原总奖励；连续奖励特征明确分别统计正、零、负值，避免把负惩罚误报为“没有出现”。
任务内去均值统计仅用于诊断，没有替换原 IRC 的 eta 门槛或修改验收规则。

## 数据与验证

远程 CPU 最终相关回归 63 项通过，ruff 通过。覆盖现有 IRC／环境匹配／信用诊断与新增读取器，包括独立 Python 进程启动 CLI，避免测试初始化掩盖入口问题。
最终实数重算结果：

- 训练：1,280 条 MT-GTPO 历史轨迹，全部重算成功。
- 验证／独立评测：480 条 MT-GTPO 加 1,199 条 SFT／E0–E3，合计 1,679 条，全部重算成功。
- E2 原独立评测的 1 条运行异常不在这 239 条轨迹中；原补测仍单列，未合并成伪造的 240 条完整原结果。
- E0–E3 的 5,120 条旧训练文本仍不能直接做严格结构化回放；新入口显式拒绝该格式，不凭文本猜测工具错误状态或 token 区间。

所有运行设置 CUDA 不可见。未调用 LLM judge、未新增采样、未启动 GPU 训练、未读取 final50、未改已有权重。

## 检查结果

始终使用上一轮固定候选：gold=1、soft/duplicate=0、state_change=-0.20961006222627246、error=-0.35298915110170204。

下表中的相关系数以各数据集原来记录的终局分数为准，包括提前终止的 0 分；各组单独报告，不混算评测协议。

| 数据集 | 条数 | 完成官方评分 | 总过程奖励相关性 | 任务内去均值相关性 | gold 查询分项相关性 | gold 写入分项相关性 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MT-GTPO validation step10 | 240 | 226 | 0.1679 | 0.3596 | -0.0256 | 0.5823 |
| MT-GTPO validation step20 | 240 | 231 | 0.0749 | 0.3082 | -0.1347 | 0.5752 |
| SFT 独立 selection | 240 | 219 | 0.1626 | 0.3279 | -0.0526 | 0.6498 |
| E0 独立 selection | 240 | 213 | 0.1738 | 0.3785 | -0.0047 | 0.5371 |
| E1 独立 selection | 240 | 211 | 0.0715 | 0.4405 | -0.1065 | 0.5071 |
| E2 独立 selection（原239条） | 239 | 211 | 0.2421 | 0.4928 | 0.0585 | 0.6378 |
| E3 独立 selection | 240 | 217 | 0.1884 | 0.3965 | -0.0019 | 0.5792 |

只看 step20 中完成官方评分的 231 条，总奖励相关性仍为 0.068290，任务内去均值为 0.357050，gold 查询分项为 -0.151349，gold 写入分项为 0.593556。
因此，提前终止标签并不能解释掉 step20 低于 eta=0.1 的结果。

step20 有 955 次 gold 查询和 171 次 gold 写入，各自都领 +1。成功轨迹的平均查询奖励为 0.446769，失败轨迹为 0.517481；平均写入奖励分别为 0.138648 与 0.041790。
这一分解支持“查询奖励缺少区分度，且跨任务组成影响整体指标”的解释；它不是查询导致失败的因果证明。
任务内相关性明显更高也说明不能把一个全局相关系数直接当作训练梯度错误或算法无效的证据。

当前没有发现这批奖励重算的数值实现错误。问题更接近奖励语义／统计适配：把高频参考读取与真正修改任务状态的调用用一个固定 gold=1 档位覆盖。
不能因此直接把所有查询奖励删掉，也不能以任务内指标替换原全局门槛来宣称通过。
参考写入与成功的相关性也不等于正确授权或唯一合法解；最终仍由原基准终局评价决定。

## 后续修订边界

若继续改奖励，应先在训练数据上明确分开 gold-read 与 gold-write 的作用、支持数和奖励预算，建立新的独立配方，并与原候选保留对照。
本轮已查看的 selection 数据只能视为开发诊断来源；不能边看边调后再把同一批结果称作独立通过。
不恢复旧错误的列表排序、不改 gamma/lambda、不靠动态过滤隐藏奖励问题。
当前训练门禁保持原样，没有冻结的新配方。

## 复现和证据

远程根目录 `/root/autodl-fs/tau3-core/code`，激活现有项目环境后运行：

```bash
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
python -m tau3_grpo.analysis.audit_reward_transfer \
  --recipe-report results/analysis/paper_env_conservative_20260918/development/report.json \
  --manifest results/runs/post-rl-selection-20260914/manifests/areal_airline_selection_seed42.jsonl \
  --split selection \
  --input results/runs/mt_gtpo_reference_write_v3/20260917_s42_df0/validation/20.jsonl \
  --output-dir /path/to/new-audit-directory
```

结果目录：`results/analysis/reward_transfer_20260918/selection-final/` 和 `train-final/`，包含 report.json 和不含原始对话文本的逐轨迹 features.jsonl。
`checks-final.json` 保存最终源码 SHA、精确命令与退出码；`tests-final.log` 保存测试输出。
每个报告保存候选报告、manifest、输入文件、奖励实现和审计实现的 hash。旧实验结果未覆盖。
