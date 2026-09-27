# selection60 覆盖审计

> 后续状态（2026-09-26）：60题Rubric共665项及Base/1epoch失败分类已补齐，详见`selection60_checklists_20260925.md`；本页“尚未完成”是当时快照，不能据此说现在没有Rubric。三组成绩、分布限制及Shopping划分方法修订见`shopping_alignment_and_sft_audit_20260926.md`。有检查表不等于已测完全部模型的所有能力。

日期：2026-09-25。依据固定任务 manifest、两模型全部原始轨迹与评分代码，只做离线分析。

## 选择方式

`tau3_grpo/data/manifest.py:build_airline_splits` 将1148条Airline任务按seed=42、ID和内容指纹哈希排序，取前200训练、接着60开发评测、余888保留。不是按工具、五维能力或语义任务族分层。三个split任务ID无交集；这不证明语义任务族完全独立。

## 14工具覆盖

计数单位是独立task；4个trial不算4个任务。参考动作不是强制调用清单；实际出现也不是正确使用证明。

|工具|参考动作涉及任务/60|Base实际调用涉及任务/60|SFT实际调用涉及任务/60|
|---|---:|---:|---:|
|get_user_details|57|60|57|
|get_reservation_details|56|56|56|
|search_direct_flight|38|42|45|
|search_onestop_flight|4|12|19|
|list_all_airports|0|3|0|
|get_flight_status|10|14|14|
|calculate|20|2|1|
|book_reservation|7|8|10|
|update_reservation_flights|33|33|33|
|update_reservation_baggages|14|12|13|
|update_reservation_passengers|3|4|6|
|cancel_reservation|12|13|16|
|send_certificate|8|10|9|
|transfer_to_human_agents|3|15|9|

参考动作覆盖13/14，缺list_all_airports。机场代码已知时不调用该工具可以合理；Base偶然调用不证明评测设计覆盖了机场识别。calculate参考出现20任务，但可以通过正确心算得到同一终态，不能用调用率衡量计算正确性。

改航班33/60、订票7/60、乘客更新3/60、补偿8/60、取消12/60、行李14/60，标签可重叠。HKEG34同订单族占7/60，该族成功次数从16/28到7/28；其余53任务合计115/212到114/212。本轮降分集中，但不能据此删去这个任务族。

## 五维能力测量现状

|五维|现有证据|尚未完成|
|---|---|---|
|目标与约束理解|多子目标、航班偏好、支付选择任务和轨迹|60任务固定Rubric与统一逐项评分|
|证据与参数准确性|参数、观察、工具error、终态|关键条件分支覆盖保证，例如先订单后补查资料|
|动作选择与合规|政策、gold和实际写操作|资格、确认、依赖顺序的统一检查|
|结果与任务完整性|DB终态和整体成功率|子目标完成率与严格名单顺序敏感性审计|
|终止与效率|终止原因、轮数、token统计|冗余、过早结束、合理拒绝的Rubric评估|

SFT的215条正常完成轨迹reward_basis是DB+COMMUNICATE，但communicate_checks全部为空；另25条预算/上下文终止按协议记0。当前总成功率主要反映数据库终态和预算内完成，不等同于五维评分。不能声称五种能力已被充分测量。

## 使用边界

selection60适合既定开发对照，不能称为均衡完整的14工具/五能力测试或最终盲测。保留Base、1epoch和已启动3epoch相同协议，不因看到降分而换题。后续覆盖矩阵应在评测前冻结，按任务族、业务、条件分支与合理拒绝区分，并审计用户模拟器/gold冲突。独立确认集在候选与协议冻结前保持封存；改协议后的比较需所有候选一致执行。具体selection答案不进入训练。
