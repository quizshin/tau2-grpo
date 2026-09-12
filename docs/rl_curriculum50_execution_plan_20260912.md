# 50 任务、E0 约 6 小时确定四组共同步数

用户最终选择：不减少 50 个任务，不减少每 step 的 8 个任务 × 每任务 8 次采样。E0 先运行，以约 6 小时实际完成的训练进度向上取整到 10 的倍数 N，完成 N 的保存和评测；E1/E2/E3 各自从相同 SFT 起点运行到 N。不是四组分别限时，也不再强求 50 step。

## 停止和对照规则

- E0 从模拟器启动前开始计时，包含初始化、训练、保存和 selection 评测。训练循环在一整个 step（以及该 step 应有的保存/评测）结束后检查时间，不中途打断更新。例如检查时完成 17→目标20，29→30，30→30。目标一经确定不再移动。
- E0 的探索上限为 100 step，避免无限运行；若提前到达上限，统一 N=100。其他组只运行到 E0 确定的 N，不继承 E0 的模型或 optimizer。
- E0/E1/E2/E3 任务顺序 seed42 相同，每 step 8×8=64 条轨迹。DF 固定采样预算，过滤不补采。每组训练总轨迹 64N，四组共 256N。
- 每 10 step 对固定 selection60 每任务采样 4 次（240 条），温度 0.4。完整评测轨迹、分数与训练日志独立保存。训练/selection ID 不重叠，final 集合本轮不使用。
- 每 10 step 保存完整 model/optimizer/extra state，每组只保留最新一份；新检查点写完才清理该组旧检查点。旧 step 的评测结果、轨迹、日志保留，不存 10 份历史权重。

## 四组配置

共同起点：new-off 非 thinking SFT，Qwen3.5-4B 全语言参数训练，4 张 A800 policy + 1 张 27B AWQ 模拟器，顺序执行同轮全部工具调用。

|组|算法|DF|目标|
|---|---|---|---|
|E0|原生 GRPO|关|约6小时后向上取整为N|
|E1|GRPO|开，过滤不补采|N|
|E2|Tau-GiGPO|关|N|
|E3|Tau-GiGPO|开，过滤不补采|N|

GiGPO 参数 omega=1.0、gamma=0.95、fnorm=1.0、min_anchor_group_size=2。相同 LR=1e-6、KL 系数0.01；microbatch=1、native kernels、compact checkpoint head、GDN padding guard、max_num_seqs=16，关闭 bypass 和 padding trim。

配置：`configs/train/rl/qwen35_4b_full_a800_c50_matched6h_{common,e0,e1,e2,e3}_20260912.yaml`。文件默认50仅用于配置/日程预检，实际控制器明确覆盖为 E0 上限100、其余组实际N。入口 `env_info/a800_20260912/run_matched50.py`，预算实现 `tau3_grpo/integrations/matched_budget.py`。

## 数据冻结与时长依据

50任务 manifest SHA256：`641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae`，保留原40，新增10组合任务。全部来自原train200，与selection60 ID隔离，DB哈希一致。这是身份和配置检查，不是语义正确性或可解性证明。

10step覆盖全部50个任务（30个出现2次，20个出现1次）；25step每任务4次，50step每任务8次。所有step内无重复，短长schedule前缀一致。

旧40池首批64条实测32.15分钟，不包含初始化、保存和独立评测。新50池尚不能保证同速；约6小时更可能落在10或20step，29→30只是停止逻辑示例，不是吞吐预测。四组相同步数，实际用时可以不同。

已验证显存优化 compact head 降低回放峰值，速度收益约3%；max_num_seqs=16 的单引擎前缀测试提高生成吞吐，但没有完整训练A/B证明总加速倍数。microbatch=2 虽约快12%，梯度相对差0.224–0.451，未采用；TF32检查通过但没有速度收益，维持默认。FLA/trim/fused CE 未通过数值检查，均不进入正式配置。

另外修复已观察到的 XML 参数 set 序列化失败：JSON/安全字面量解析替代 eval；非法结构保留原始字符串交工具验证，不将集合猜成业务列表，也不因该错误丢失整轮调用。

## 空间和执行边界

单个完整恢复状态此前约50.6GiB。四组最新状态约202.4GiB；当前共享盘空闲约153GiB不足。为保留写新检查点期间的旧恢复点，启动每组前要求至少110GiB空闲。E0有空间可运行；后续容量不足时控制器记录 `waiting_for_storage` 并退出，保留已完成结果，不擅自删除其他组optimizer，不把唯一恢复状态放到易失内存盘。

控制器顺序 E0→E1→E2→E3，每组结束校验最终完整四rank模型/optimizer/RNG分片、目标step及所有240条评测记录；存在旧实验目录时拒绝自动覆盖/重启。仅管理本控制器启动的进程，禁止全局 ray stop。需要恢复或继续因容量暂停的队列时先检查当前结果，再制定明确存储方案。

新50预检、停止/配置测试和实际启动状态以远程 `runs/formal50-preflight-20260912/`、`runs/rl-c50-matched6h-a800-20260912/controller-state.json` 及训练日志为准。本方案替代此前100步研究预算和减少为2×8/1×8的限时建议。
