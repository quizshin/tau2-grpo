# SFT 冷启动 A45 / B100 对照

目的：检验补充工具和场景示范是否改善 RL 前的冷启动行为。本轮只运行 SFT 与独立评测，不接 RL，不能据此宣称后续 RL 收益。

A 为原45条训练对话；B为100条，包含全部原45条、52条新增公开 AReaL SFT 对话和3条根据 AReaL RL 训练任务编写并实跑工具的示范。原5条验证集独立保留。B200方案已被用户改为最多100条。

公开 SFT999条没有 send_certificate 调用。公开 RL1148个 Airline 任务中53个参考动作含该工具：train12、reserve33、selection8。8个selection任务不进入训练。3条补充示范来自 airline_618/444/1036；对话文字是编写的，工具返回来自隔离数据库上的真实执行，并与独立执行参考动作的最终数据库一致。这不等于教师模型采样成功，也不冒称官方独立评测成功。

B覆盖全部14工具；其中 send_certificate3、calculate2、list_all_airports5、search_onestop_flight5条。检查了数据身份、原45条包含关系、原5条隔离、工具参数与返回配对、词面近重复、原生模板与长度。原始公开SFT未完成独立任务回放与充分语义去污染。

两组从同一个 Qwen3.5-4B 固定快照初始化，non-thinking、LoRA r16/alpha32/dropout0.05、lr1e-4、有效batch8、seed42，统一30次更新并评测第30步最终权重。更新预算一致，实际监督token数不完全一致，必须记录；本实验比较的是数据扩充与工具覆盖方案整体，不单独归因于数量。

统一评测：tau3_eval_token_v4，selection60，每题4次，两组policy温度0.7，用户模拟器温度0.7，相同seed42及任务/试次计划。主评测共480条轨迹，另有启动联调每组4条。使用现有严格比较器输出配对任务置信区间、任务成功率、工具错误和调用诊断。正式final50不参与。

运行目录：`/root/shared-nvme/tau3/runs/sft_coldstart_ab/20260921-AB100-v1`。A使用GPU0，B使用GPU1，每组3小时上限；训练后自动合并，policy评测服务使用GPU2/3，共享模拟器使用GPU6/7；每组正式评测4小时上限。失败保留日志并停止对应后续阶段，不静默补跑或删除失败轨迹。

当前证据：远程数据/扩充相关CPU测试12通过，原生tokenizer复查6通过；两组GPU均已产生非零参数更新。尚无完成或效果结论。

配置：`configs/train/sft/coldstart_a.yaml`、`coldstart_b.yaml`；数据审计和来源搜索：`results/runs/sft_coldstart_ab/20260921-prepare/`。

## 13:03 检查与恢复

A完成30次更新，训练墙钟767.33秒（约12.8分钟），实际训练225次对话。B在第8次更新后反向传播CUDA OOM：请求4.62GiB，设备剩余4.42GiB，另有4.64GiB PyTorch保留未分配显存。原评测控制器正确停止，未产生评测结果。

恢复运行：`/root/shared-nvme/tau3/runs/sft_coldstart_ab/20260921-AB100-v2`。保留A完成权重，B从同一基座重跑；仅设置`PYTORCH_ALLOC_CONF=expandable_segments:True`，数据、loss、LoRA及30更新预算不变。B尚无可续训的10步检查点，未冒称从第8步续训。旧失败目录保留，新B训练上限170分钟。此时尚不能确认分配器调整已解决峰值显存问题。

## 15:00 评测恢复

A/B均已完成30次更新与合并；val_loss分别0.42261523/0.41305286，不视为任务成功率。首次评测仅A的2条smoke完成，余6条报错，正式480条未开始。主要失败为veRL动态导入提前写入sys.modules，线程并发读取未完成初始化的Tau3AirlineTool；B另有120秒token端点超时，是否冷启动延迟尚未证实。

项目侧修复通过正常import锁预加载工具与交互类，不修改vendor。新增显式token_request_timeout，默认120秒保持，恢复配置使用600秒且不重试。旧失败记录保留，新恢复目录计划为v2/evaluation-recovery-v1。远程CPU回归进行中；未重启GPU评测。

15:04：远程66项回归通过（含三算法原生token loop），冷启动并发测试子进程原为exit0但并发stdout解码失败，修复测试诊断解码后独立1项通过（46.80秒）。六个部署文件SHA256一致且原文件已备份。恢复控制器PID12106已启动，复用冻结权重，新目录evaluation-recovery-v1，尚待smoke与formal验收。

15:10：新恢复运行A/B各4条启动smoke全部完成，零执行错误；两组正式selection60×4均已启动，旧失败记录未合并。此前读到的首批5条smoke轨迹约48–66秒，仅作运行耗时估计，正式比较仍待完整480条。


## 2026-09-21 16:48 Final evaluation complete

A completed at 16:46:42; B at 16:40:04. Recovery-v1 controller status complete, exit 0. Each arm scored all 240 selection60 trials with zero execution errors or missing trials. Policy/user temperatures both 0.7. Strict protocol/identity comparison passed. A45 pass@1=45.83% (110/240), B100=41.25% (99/240); B-A=-4.58 pp, paired-task bootstrap 95% CI [-10.83,+1.67] pp. Pass@4: 76.67% vs 68.33%; pass^4: 23.33% vs 16.67%. No cold-start benefit observed from this expansion; one training seed and the primary CI do not establish general degradation or downstream RL impact. Tool error flags: 2.58% of 1742 calls vs 8.01% of 1911 calls; these are distinct from infrastructure execution failures.

Remote evidence: /root/shared-nvme/tau3/runs/sft_coldstart_ab/20260921-AB100-v2/evaluation-recovery-v1. Local comparison: results/runs/sft_coldstart_ab/20260921-prepare/final-comparison/comparison.{json,md}. Recommend retaining A45 as the present RL starting point; no formal RL launched based on this result.
