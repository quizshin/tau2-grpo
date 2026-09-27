# MT-GTPO v3 正式运行记录（2026-09-17）

用户已确认采用本方案并启动训练。实施与验证状态按下方实际记录更新。

## 奖励定义

- `algorithm.process_reward.mode=reference_write`，`version=v3`。
- 官方终局评分保持 DB + COMMUNICATE，不向官方成功率添加过程奖励。
- DB 任务中，首次匹配未消费的参考数据库写操作且执行无错误，奖励 `1/M`；M 为参考数据库写操作总数。
- 删除 v2 的终局成功门控；失败或达到轮数上限的轨迹也可获得已经完成的局部动作奖励。
- 每条轨迹正过程奖励总量不超过 1；工具错误 -0.1；重复、软匹配、其他写操作、普通查询、对话及转人工为 0。
- 参数等价只复用固定版本工具的实际转换：订票的 FlightInfo/Passenger/Payment、改航班的 FlightInfo、改乘客的 Passenger。
- 顶层键、ID、列表顺序严格比较；嵌套数据的额外字段和类型转换完全由上述官方模型决定，不做额外模糊匹配。保留原参数及有效参数以便重算。
- 不使用 LLM judge，不额外判定自然语言授权。参考匹配不能证明政策遵守，合法替代方案仍可能漏奖。
- v1/v2 保留原配置；v2 历史说明不代表本次实际配方。原重复动作测试暴露的 Python 对象别名导致记录覆盖问题已修复，各调用事件独立保存。

## 依据与边界

历史 E0 前三批次 192 条，191 条通过调用数量和初末 DB hash 核验，1 条未验证。保持 v2 原始严格匹配，仅移除终局门控，正奖励轨迹由 48 增至 72，正奖励调用由 65 增至 118，新增 24 条失败轨迹的局部正反馈。
证据：`results/analysis/mt_gtpo_scope_20260917/ungated_comparison.json`。
这属于历史调用级奖励分析，不是新模型结果，不是逐轮优势重算，也不是独立授权验证。
50 个任务中 26 个有 1 个参考写操作，22 个有 2–5 个，2 个仅转人工；全部 communicate_info 为空。

## 正式协议

- 新目录 `results/runs/mt_gtpo_reference_write_v3/20260917_s42_df0/`。
- Qwen3.5-4B new-off，全参数、非 thinking；固定 train50、seed42、历史一致任务调度。
- 20 个外层 step，每步 8 组 × 8 条，合计 1280 条候选轨迹。
- MT-GTPO gamma=0.9、lambda_outcome=0.3；动态过滤默认关闭，可独立配置开启。
- 4 张 A800 策略卡、1 张 Qwen3.8-27B-AWQ-INT4 模拟器卡；沿用正式基线计算路径。
- SwanLab online，每步记录；step10/20 保存完整续训检查点并评测 selection60×4；新检查点完整校验后仅保留最新一份。
- 入口 `scripts/train/rl/run_mt_gtpo_formal.py --reward-version v3 --result-dir <上述目录>`。
- 入口仍可显式 `--reward-version v2`；同 run 续训拒绝变更奖励版本和 DF 开关。
- 不使用 selection/final 数据校准过程奖励；不在同一次正式训练中自动改奖励。

## 执行状态

- 本地已实现 v3、等价参数与真实工具效果测试、失败组信号及 DF 双模式重算测试。
- 本地 97 项通过（39.11 秒），远程同组 97 项通过（135.88 秒）；ruff 与 diff 检查通过。正式 Hydra 预检通过，任务调度与历史 E0 一致。
- v3 在 191 条已核验历史轨迹上产生 84 条正奖励轨迹、145 次正奖励调用，其中 31 条为失败轨迹；参数规范化额外覆盖 12 条轨迹。证据为同目录 v3_coverage.json。
- 远程部署前备份：results/maintenance/mtgtpo-v3-1789627817952883407。
- 正式控制进程 PID 7403 已启动，session 为 controller-session-1789628354432173099。模拟器健康检查通过；4 卡 FSDP 和 rollout 模型完成加载，真实多轮采样已开始。
- SwanLab run_id：0c4c047eef6d4fd7bdb87；run_path：quizshin/tau3-grpo-pytrio/0c4c047eef6d4fd7bdb87。
- 当前为首次采样，尚未核验 step1 数值云端上报、更新后的权重同步或整十检查点；训练后台继续，未宣称完成或效果提升。
