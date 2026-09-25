# Airline badcase 分析修复与验证（2026-09-25）

## 改动边界

`tau3_grpo/analysis/airline_badcases.py` 是分析实现，`scripts/analysis/analyze_airline_badcases.py` 仅为兼容 CLI。分析不改变 reward、训练算法、credit assignment 或环境行为，不启动 GPU/模型服务。新的分类定义见 `configs/analysis/airline_capability_taxonomy.yaml`。

修复内容：

1. 读取真正的 `simulation.messages` 和 nested `function` 参数，同时支持训练的 `trajectory_facts_json`。结构化工具调用与整数计数分开；缺失 telemetry 记 unknown/None，不能当 0。读取 tool observation 的 error 标记。
2. 复用项目 `evaluation.eligibility`：`agent_error`、`max_steps`、length/context 等 completed fallback-zero 轨迹保留为失败，不能因 `scored=False` 或 `officially_scored=False` 就丢掉。明确区分 official score、complete fallback 和 unresolved infrastructure。工具/agent 错误不是 simulator 故障。
3. 数据库 hash 改变只标 `db_changed_on_failure`；write 调用只标 `write_attempt_on_failure`。不再自动推断“正确写入”“部分完成”“错写 DB”。相同调用签名可能是必要重读，不自动断言重复冗余。工具 schema、policy 合规、首错因果仍需另外验证。
4. 保留 task / group UID / trajectory ID / trial / seed / step。结果与 errors sidecar 同源分析；任一 unresolved 尝试保留在记录中并使该 source/task 的成功率不可用。重复 trajectory ID 或同组 trial 拒绝。单独提供文件的率仅覆盖已提供记录，不能证明全部计划预算完成。
5. 输出一次生成 `records.jsonl`、`summary.json`、`report.md` 和 manifest，发布前校验。`--verify-dir` 不仅对 checksum，也从 records 重算 summary/report；修改摘要再更新 checksum 仍会失败。输入哈希固定、长读取前后核验，现有输出目录一律拒绝覆盖。

## 旧 v1 的状态（保留原件，不覆盖）

目录 `results/analysis/airline_badcase_v1/` 的原件哈希登记在 `env_info/airline_badcase_repair_20260925.json`。

| 数据源 | duplicate_call：旧 summary | 同目录 records 实际计数 |
| --- | ---: | ---: |
| GRPO eval10 | 185 | 25 |
| GRPO eval20 | 189 | 23 |
| GRPO eval30 | 202 | 46 |

这些差异说明旧聚合与明细不同步，不能把一方当作修复后的真值。旧 SFT A/B eval 各 240 条明细均为 0 次工具调用，系读取层级错误。对应完整原始 A/B eval 和 GRPO eval10/20/30 此次本地不可用，远程认证也未成功，**未声称已修复重建这五组历史报告**。不使用损失信息的 v1 records 伪造缺失原轨迹。

## 本次真实 raw replay

新目录：`results/analysis/airline_badcase_v2_20260925/`。

- 2026-09-25 本地 Base/SFT100 selection60，各 4 trials、seed 42–45、240 条；从 `/private/tmp/tau3_compare` 复制到项目忽略目录 `results/analysis/review_repairs_20260925/inputs/` 后校验哈希，避免临时目录成为唯一来源。另存的原始 summary 为 complete 240、failed 0、missing 0、metrics_valid true；逐任务 trial/seed 集合也已独立核验。
- 历史 GRPO 与 MT-GTPO 训练 step10–27，每步 64 条，共 36 个独立 source、2,304 条。不将不同 step 合并成稳定策略评测率，不与 selection eval 混为一谈。
- 总计 **38 sources / 2,784 records / 19,482 structured tool calls**；其中 **490** 条 complete fallback zero，unresolved 0。全部 38 个 source 的工具调用数和成功数从原始输入独立统计，与 v2 一致。

| 当前本地 eval（不是旧 A/B） | Base | SFT100 |
| --- | ---: | ---: |
| 全成功 / 240 | 131（54.58%） | 98（40.83%） |
| official scored | 213 | 204 |
| complete fallback zero | 27 | 36 |
| 实际工具调用数 | 1,586 | 1,774 |
| 观察到工具错误的轨迹 | 20 | 36 |
| 相同调用签名重复的轨迹 | 18 | 39 |

表仅重新解析已存在本地 raw，不是新跑实验、完整模型来源审计或因果归因。不要把本次 Base/SFT100 与历史 A/B 名称互换。

## 可重复执行

原始数据不新增到 Git；小型输入索引 `configs/analysis/airline_badcase_replay_20260925.json` 固定每个 source 的路径（相对 manifest）、SHA256、expected_records 和 scope，方便在有相同资产的机器上重跑。

```bash
python -m tau3_grpo.analysis.airline_badcases \
  --input-manifest configs/analysis/airline_badcase_replay_20260925.json \
  --out results/analysis/airline_badcase_v2_new_run
python -m tau3_grpo.analysis.airline_badcases \
  --verify-dir results/analysis/airline_badcase_v2_new_run
```

标准 `trajectories.jsonl` 同目录存在 `errors.jsonl` 时自动一并读取；重命名文件必须在 manifest 显式给 `errors_path`（以及可选 `errors_sha256`）。`expected_records` 是两者合计，漏记录则 fail closed。该检查不替代 task/trial/seed 与原始 run plan 的完整预算核验。

本次输出文件、代码、eligibility 实现、source manifest 和历史 v1 哈希登记于 `env_info/airline_badcase_repair_20260925.json`。所有旧结果未覆盖。

## 测试与剩余问题

- badcase 回归：**37 passed**，登记 core 层；覆盖 nested messages、工具错误、所有 fallback termination 的真实 scored=False 格式、未知/非有限 outcome、demonstration、schema 缺失、重复轨迹、结果/error 合并、输入 SHA 漂移、数量缺口、拒绝覆盖，以及聚合/报告与 checksum 防漂移。
- guard 定向 CPU：**213 passed**（第一提交）；clean14 + SFT 定向 **27 passed**，benchmark 全层 **534 passed**。
- 本地 core 全层两次均 **482 passed / 1 failed**：`test_training_services.py::test_launch_log_and_idempotent_stop` 在 `training/services.py:61` 的 `os.killpg(group, 0)` 报 `PermissionError`；该文件单独运行 **3 passed**。相关源码和测试相对任务起点 `068e4bb` 无变化，不能断言由当前改动引入，也不能掩盖 suite 失败。移除新 badcase 测试模块后的原 core 列表为 **446 passed**；因此尚不能排除时序/测试组合影响，未定位，不声称这是已证实的平台故障，留给单独的进程生命周期排查。
- 先前审查的完整 verl 层还有既存 FSDP 失败：`test_qwen35_fsdp.py::test_lora_fsdp_accumulates_multiple_microbatches_with_tied_embeddings[False]`，`Expected [768], got [32,3,2,2,2]`。当时 604 passed / 1 failed / 16 skipped，定向复现；有关文件本次未改，本次不重复跑完整 verl，也不声称已修复。当前本地依赖与远程 pinned 环境不同。
- Ruff 基线 **0 新增诊断**（既存 382 条）；`git diff --check` 通过。以上均为本地 CPU/offline 验证，不替代远程 pinned 环境或 GPU/FSDP 验收。
