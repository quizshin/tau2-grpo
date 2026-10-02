# 评测

区分代码测试与模型评测：core/benchmark/verl通过不代表模型任务成功率提高。

- 当前冻结dev150：用于课程开发和阶段选择；dev37属于历史数据身份。
- selection60：已有开发曝光的历史回归面板，不重新命名为盲测。
- 官方final50：模型与协议冻结后才用于最终评测，不参与训练或选型。

同起点SFT、GRPO、ARPO、MT-GTPO比较必须一致使用任务、用户模拟器、奖励、温度、预算及终止协议。完整保留异常和分母，不截取成功子集。

实现入口：`tau3_grpo.evaluation.run`、`tau3_grpo.evaluation.compare`；统计与冻结规则见 [独立评测](independent_evaluation.md)。

## 当前 rubric：五维诊断 v1

新模型回复统一使用 `airline_five_dimension_v1`，不要再使用历史 pilot 的 0/1/2 总分。

|维度|评什么|
|---|---|
|scope|是否理解并遵守用户任务、乘客范围和限制|
|evidence|回复、ID、工具参数是否来自作出决定前已知的证据|
|policy|业务规则、工具 schema、确认和执行顺序是否合规|
|arithmetic|金额、差价、费用和行李额度的输入与计算是否正确|
|completion|实际操作是否完成，是否核对工具结果并如实告知|

每维输出 `satisfied / violated / unknown / not_applicable`，附理由和轨迹事件引用。
**不计算综合总分**。效率、工具调用数、token 和终止原因另列；未知不算零分，N/A不算通过。
成功率沿用独立评测的原有官方或显式冻结 outcome contract，rubric 不修改 GRPO/ARPO/MT-GTPO 的训练奖励。

SFT 包内同名五维是**已审核样本的接受记录**，`reviewed_sft` 检查其身份、审核状态、token/mask等，
不会为模型新回复打分。新轨迹必须走下面的评测链路，不能把训练数据的 accepted 结论复制过来。

## 使用顺序（SFT、GRPO、ARPO、MT-GTPO 共用）

以下命令展示接口，**不是已经完成的模型实验**。先导出待评测 checkpoint；所有模型使用同一任务、
数据库、任务细则、模拟用户、采样预算和 harness。GPU 采样和外部裁判调用分别按预算授权。

1. 从将要使用的 selection manifest 生成草稿（离线）：

   ```bash
   python -m tau3_grpo.evaluation.rubric draft \
     --manifest data/manifests/areal_airline_selection_seed42.jsonl \
     --output results/rubric/task-draft.json
   ```

   草稿只提供五维定义及实际任务/DB/政策/工具schema身份。逐任务核对原任务、数据库和政策，
   补充各维的任务细则、`capabilities`、`bucket` 和真实的 `reviewed_by`，再冻结：

   ```bash
   python -m tau3_grpo.evaluation.rubric freeze \
     --draft results/rubric/task-draft.json --output results/rubric/task-frozen.json
   ```

   reviewer字段只能表示记录的审核人，不是程序对审核质量的证明。能力标签使用
   `rubric_contract.CAPABILITIES` 中的九类（意图、状态读取、schema、工具选择、参数绑定、
   调用顺序、数据库修改、结果验证、终止）；标签在看模型结果前确定，不按模型动作反推分母。
   `bucket` 可选 foundation/constraints/strategy/unclassified，不能自动猜测课程难度。
   当前 CLI 草稿入口针对 selection；final50 必须先锁定模型，不用作课程细则调优集。

2. 在现有独立评测命令上增加 `--rubric-bundle results/rubric/task-frozen.json`。
   `--dry-run` 也检查任务/DB/政策/工具身份；不匹配时在采样前拒绝。
   评测目录保存冻结包、完整计划、结果、异常，以及自动生成的 `rubric_requests.json`。
   **不会自动调用裁判**。请求只含可见会话、执行工具证据、冻结政策/schema和细则；
   不含原始 reward、gold、参考答案或单独的 reasoning 字段。

3. 经预算批准后，显式调用裁判（需要 `.[semantic]` 和 `TAU3_RUBRIC_API_KEY/BASE_URL/MODEL`）：

   ```bash
   python -m tau3_grpo.evaluation.rubric judge \
     --requests results/evaluation/RUN/rubric_requests.json \
     --output results/evaluation/RUN/rubric-judge \
     --max-calls 240 --max-tokens 4096 --execute
   ```

   `240` 仅是 selection60×4 的示例，实际限制须覆盖请求数并符合预算。
   每请求最多一次调用，发送前持久化预约记录；超时、无效JSON、错误引用留作失败，
   无自动重试。同一个输出目录不能重跑。输入token仍收费，max-calls/max-tokens不是货币硬限额。
   中断后可以直接报告已有响应，未完成部分保持 unjudged，不必重跑GPU。

4. 离线生成成功率、五维分布、能力/课程切片和带证据的 badcase：

   ```bash
   python -m tau3_grpo.evaluation.rubric report \
     --run-dir results/evaluation/RUN --bundle results/rubric/task-frozen.json \
     --reviews results/evaluation/RUN/rubric-judge \
     --output results/evaluation/RUN/rubric-report
   ```

   输出 `report.json`、`report.md`、`badcases.jsonl`。失败任务、成功但疑似违规、证据不足和
   未审查案例都会保留。能力切片可重叠，不加总；已知判断通过率是条件指标，完整任务宏平均
   在有未覆盖任务时不输出。证据ID检查不能证明裁判的语义判断正确。

5. 比较两个模型：

   ```bash
   python -m tau3_grpo.evaluation.rubric compare \
     --baseline results/evaluation/SFT --treatment results/evaluation/GRPO \
     --baseline-reviews results/evaluation/SFT/rubric-judge \
     --treatment-reviews results/evaluation/GRPO/rubric-judge \
     --bundle results/rubric/task-frozen.json --output results/rubric/comparison.json
   ```

   必须同任务/种子计划、原评测协议、冻结包、裁判配置与prompt，且完整覆盖；
   rubric比较还要求五维都能确定判断，避免unknown/N/A改变分母后虚增差值。
   结果是描述性诊断，尚无人工标注校准的裁判不能宣称能力评测准确率已验收。

已有轨迹可用 `prepare --run-dir ... --bundle ... --output ... --retrospective` 补审，
生成报告时也须加 `--retrospective`。历史缺少的工具schema身份保持未验证，不补造；
缺任务/DB/政策身份的轨迹拒绝补审。回溯诊断不能混入事前冻结的正式模型对比。

当前 dev150 是冻结 SFT 会话开发集，**还不是已接通的交互任务评测包**。
需要恢复相应任务与初始DB、审核任务细则，才可进入这条交互评测链路。
本次实现提供可执行链路和模拟/CPU验证；未伪造dev150任务审批、裁判校准或模型能力结论。

## 历史 pilot 归档

旧实现位于 `tau3_grpo/analysis/legacy/rubric_pilot.py` 和 `rubric_batch.py`。
旧的同名公共CLI已退役并提示新入口；兼容import继续转发，以保留历史测试和重放。
历史重放需要显式 `--legacy-review`。公共请求/预算工具归 `tracking/judge_budget.py`，
可见消息工具归 `data/messages.py`，现行其他审查工具不再依赖pilot。

旧 intent/evidence_arguments/action_compliance/completion/termination_efficiency 的0/1/2分
只解释旧报告，不能换名字后映射为当前五维。旧结果、账本和数据审核证据不删除、不重写。
