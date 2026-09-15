# GiGPO 无卡验证 1–3（2026-09-14）

本轮只做 CPU 诊断与候选方案验证，未修改生产算法，未启动 RL 或模型服务。服务器设置 CUDA_VISIBLE_DEVICES 为空；实测 cuda_device_count=0、cuda_initialized=false，计算张量位于 CPU。

103 项测试通过（76.26 秒），历史 80 个 update 的奖励与任务分组计算通过。整个执行约 142 秒。测试通过包括成功复现旧规则缺陷，不表示缺陷已修复。

## 1. omega=0 与 GRPO 的归一化对齐

使用四组共 5,120 条历史奖励及任务分组，配人工 token mask/span，直接调用当前 veRL 的真实 GRPO 函数对照。不是原始训练 token batch 的精确回放。

|比较|80 批结果|
|---|---|
|现有 Tau-GiGPO omega=0 vs 默认标准 GRPO|80/80 不一致；72 批最大绝对差 1.5998667342，8 批为 0.8701816746|
|现有 Tau-GiGPO omega=0 vs GRPO 关闭标准差归一化|80/80 最大差 0|
|对齐候选 omega=0 vs 默认标准 GRPO|80/80 torch.equal 精确一致，最大差 0|

原因：当前 Fnorm=1 的 episode 项只减组均值；默认 GRPO 还除以组内样本标准差加 epsilon。该差异可以是算法设计选择，但意味着 E2/E3 相对 E0/E1 同时改变了 episode 优势尺度与步级信用分配，不能把组间成绩差单独归因于 GiGPO 的步级项。

诊断候选直接复用 GRPO 的 episode 项，然后叠加步级项，验证了 omega=0 的退化条件。候选尚未接入训练器；omega>0 的步级尺度与权重仍需独立审计。float32/64、混合/全成功/全失败组、singleton、空 mask 与 span 均有测试。

## 2. 状态锚点是否真的代表可比较的决策状态

现有确认关键词规则在 6 个刻意选择的案例中有 5 个不满足动作确认语义：

|输入|现有 user_confirmed|应有含义|
|---|---|---|
|I do not confirm this cancellation.|true|否定确认|
|Yes, that is my name.|true|仅身份确认|
|What happens if I confirm?|true|假设询问|
|Please wait, do not proceed.|true|要求暂停|
|Yes, please go ahead.|true|明确同意（仍需要对应待确认动作）|
|先同意，再说 Wait, I withdraw my consent.|true|撤回后应失效|

5/6 不是线上误判率，只是缺陷反例集。固定同一 task 与 DB 时，还复现两种锚点碰撞：查询不同订单仍同锚点；同订单但返回价格不同仍同锚点。工具类型 mask 没有编码具体查询对象与观察值。

候选验证：

- 将确认绑定 operation/reservation_id/amount/currency，修改对象、金额、币种、操作后确认失效；否定、撤回及不同 session 隔离测试通过。
- 成功读工具的有序账本包含工具名、参数、返回值，可区分上述两种碰撞；JSON 键顺序与 call ID 不影响同一内容的指纹。

边界：确认候选依赖样例预先提供结构化待确认动作，尚无真实对话到 proposal 的自动提取。读账本使用精确历史，可能把语义等价状态过度拆分，降低可比较组覆盖率；尚未测实际覆盖率。因此这两项只是候选接口和样例级验证，不能宣称语义状态建模已解决。

## 3. 步级信号与过滤的交互

通过实际算法函数构造受控样例，结果如下。

|样例|过滤前→过滤后|解释|
|8 条全成功、长度 2–9、gamma=0.95|非零步级位置 8→0；token 加权绝对优势总量 0.6537507→0|等终局奖励不保证等折扣步回报，DF 会屏蔽该信号|
|相同样例、gamma=1|非零位置 0→0|前一项信号来自折扣与长度差|
|全失败且奖励全零|非零位置 0→0|GiGPO 无法从全零终局奖励凭空生成信用信号|
|单轨迹内两步重复同锚点|现有步优势 [-0.025,+0.025]；跨轨迹候选 [0,0]|现有组大小至少 2 不保证来自不同轨迹|
|4 成功＋4 失败，每轨迹两个共享锚点|非零步骤 16→16；非零 token 32→32|混合组保留信号，其中 8 个非初始步骤具有跨轨迹非零优势|

全零样例重复锚点覆盖率仍为 18.18%，说明重复覆盖不等于非零信号覆盖。上述绝对优势总量不是梯度贡献，也不是任务成功率。全成功样例只有初始锚点共享，没有非初始跨轨迹信号；其长度偏好不能直接当作有用动作信用。

先前历史审计定位 E3 的 15 个全成功过滤组（120 条轨迹）均有长度差，但缺少在线 anchor_ids、anchor_spans 与原始策略 mask，因此本轮不能算出历史实际损失信号比例，也不能证明该机制导致 E3 成绩下降。

## 对当前实验结论的影响与后续顺序

已有 selection 结果：E0 47.50%、E1 48.75%、E2 46.25%、E3 45.42%。此前任务级配对 Bootstrap 的组间差值置信区间包含 0。这轮 CPU 诊断未产生新模型或新的成功率结果，不能证明 GiGPO 有效或无效。

建议先继续无卡工作：把 episode 归一化设为显式配置并接入真实训练路径，确保 omega=0 对齐；补充在线审计 payload 与跨轨迹、非初始、非零信号统计；对真实对话验证确认作用域与状态表示，再考虑改锚点。DF 是否保留某类等回报组，应依据实际合并优势及信号含义决定，不能只因存在非零值就保留。

上述实现完成后，再进行 1–2 step GPU 冒烟测试以核对真实更新 batch、mask 与优势接线；最后用相同 SFT 初始化、任务/采样预算的训练对照判断收益。旧 20-step 检查点已经受原算法影响，不能将直接续训当作只改变一个因素的干净对照。本轮没有启动这些后续工作。

## 文件与复现

- 诊断代码：env_info/a800_20260912/cpu_signal_validation.py
- 新增测试：tests/test_cpu_signal_validation.py
- 本地及服务器结果：results/analysis/cpu_signal_validation_20260914/
- 主要结果 validation.json；执行记录 execution.json；pytest.log；validation.log；输入 historical_snapshot.json。
- execution.json 保存 Python 版本、源码 SHA256、测试列表；pytest_exit=0、validation_exit=0。
- 旧审计：docs/training_signal_audit_20260914.md。

服务器代码目录 /root/autodl-tmp/tau3-perf-20260912/code，Python 为 /root/autodl-fs/tau3-core-20260912/environment/venvs/qwen35/bin/python。设置 CUDA_VISIBLE_DEVICES=''、OMP_NUM_THREADS=2、OPENBLAS_NUM_THREADS=2、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、TAU3_GRPO_TEXT_ONLY=1、PYTHONPATH=.:verl:tau2-bench/src 后执行：

```sh
python -m pytest -q tests/test_cpu_signal_validation.py tests/test_tau_gigpo.py tests/test_dynamic_filtering.py tests/test_anchors.py tests/test_rollout_concurrency_stress.py
python env_info/a800_20260912/cpu_signal_validation.py --snapshot results/analysis/cpu_signal_validation_20260914/historical_snapshot.json --output results/analysis/cpu_signal_validation_20260914/validation.json
```

这里 python 指上述完整路径。并发测试是 stub 环境下的 4/16/32 并发隔离测试，不是模型在线并发实测。唯一 warning 为 audioop 弃用提醒。
