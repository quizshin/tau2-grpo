# Harness 修复：版本化控制流与可见工具返回

日期：2026-09-20。依据[第二步原生循环审计](harness_step2_audit_20260920.md)。本次为 CPU 修复与对照，未调用模型、启动 GPU 或重训；不代表算法增幅。

## 修复范围和版本选择

新增 `tau3_eval_train_control_v2`，保留 `tau3_eval_legacy_v1` 作为兼容默认。新研究评测应显式选择 v2，并用同一个版本重新评测共同 SFT 起点与全部候选；不能把新协议分数直接接到旧协议排名。

|控制项|legacy v1|train_control v2|
|---|---|---|
|终止优先级|固定上游逻辑，预算检查可能覆盖已收到的 STOP|已经完成的 STOP/错误终止优先，预算检查不覆盖|
|轮数单位|30 个 Orchestrator 角色转换 step|最多 15 次策略生成；最多 15 批工具/用户观察；真实用户消息最多 15 条，含初始用户|
|最后一轮工具|原始规则|整批按顺序执行，再阻止下一次策略生成|
|最后一轮文本|可能来不及接收 STOP|用户预算允许时可接收该轮用户回复|
|工具错误|累计 10 次即终止|与当前训练一样，无单独累计错误上限；仍受 15 轮硬边界约束|
|正常工具返回|独立评测看到完整原文|与训练共用 65,536 字符 middle 规则；截断标记在预算之外|
|官方任务、工具、终局评分|固定版本|保持同一固定版本；不重写官方 reward|

v2 是**固定的当前训练控制配置**，不是对所有自定义训练预算的自动适配。`--max-steps/--max-errors` 属于旧协议；v2 若传入与兼容默认 30/10 不同的值会在加载数据/调用服务前拒绝，而不是静默忽略。它们在 v2 下不充当实际预算，实际规则记录在 `provenance.harness_protocol` 中。需要不同的轮数、错误策略或裁剪形态时，应扩展显式协议并增加对照；这不冻结后续行为。

v2 仅支持有用户模拟器、无已有消息历史的 Airline fresh-session 路径；不把计数规则套用到已初始化对话或 solo 模式。任务初始化 DB/action 与消息历史是不同概念，前者仍由官方环境执行。

## 实现位置

- `evaluation/harness.py`：协议身份、实际控制规则及非法组合校验；不依赖模型服务。
- `envs/orchestrator.py`：项目侧原生 Orchestrator 子类，只调整终止检查和计数；未改固定 tau2 上游源码。
- `envs/observations.py`：共享字符裁剪及 raw/visible SHA256、长度、截断方式回执。保持正式训练的原有裁剪输出（含标记长度）。
- `envs/agent.py`：仅给模型请求制作裁剪后的副本；持久 agent state 与官方 trajectory 都保留完整 ToolMessage。后续请求中的历史工具消息使用相同投影。
- `envs/interaction.py` 与 veRL tool loop：记录工具派发失败之后的真实数据库 hash；不把空缺旧回执补造成历史事实。已开始执行后的基础设施异常仍向外传播，不重试可能已写入 DB 的调用。
- `evaluation/runtime.py`：每次 run 和每条 simulation 记录协议；独立评测的可见消息回执写入 `simulation.info.tool_observation_receipts`。该列表描述实际送给 agent 的工具输入，不补造终止之后未发生的请求。
- `evaluation/run.py`、`evaluation/controller.py` 与两个 eval shell 入口：协议选择贯通到实际执行；比较器拒绝不同协议或已记录协议与未知旧协议之间的增幅计算。两份均缺版本的旧记录仍按原比较规则检查，披露版本证据缺失。

veRL 补丁登记已更新；从原固定 revision 的两个上游归档重建 `env_info/vendor_patches.json`，未升级 vendor。补丁测试与局限见 `env_info/vendor_patch_notes.json`。

## 如何使用

只解析任务和协议、不调用模型的检查：

```bash
python -m tau3_grpo.evaluation.run \
  --target selection --checkpoint /path/to/exact/merged \
  --harness-protocol tau3_eval_train_control_v2 --dry-run
```

已授权推理时，同一个参数可用于 `scripts/eval/run_qwen35.sh`，或 `python -m tau3_grpo.evaluation.controller`。两个 shell 入口也读取 `TAU3_EVAL_HARNESS_PROTOCOL`。历史入口缺省仍选择 legacy；新的配置必须显式写明版本，不凭文件名推断。

selection 评测的 task/trial/seed、服务权重身份和原有 final50 winner lock 检查继续生效。本次没有打开 final50，也没有运行该命令的真实模型推理部分。

## 验证记录

证据目录：`results/analysis/harness_fix_20260920/`。本地与远程同一套相关回归分别 **129 passed**（本地 63.77 秒，远程 536.83 秒），另各通过 **1 项两次真实写入之间的失败回执测试**（本地 8.73 秒，远程诊断尝试 66.09 秒），合计各 **130 项独立用例**。两端 lint 均 **0 新增**，vendor 清单核验均通过。包含三算法共享 rollout 事实、未知工具/非法参数回执、基础设施异常传播、多工具完整执行、原生新旧终止边界、CLI 和协议比较防护。130 = 129 项共同回归 + 1 项追加状态快照测试；不与远程相同用例重复相加，也不把下述诊断见证数量加成单测数。

本地最终新旧两协议各运行 22 个见证。兼容检查确认 legacy 的 22 例终止/分数/调用计数、原始工具事件/DB、完整训练 token 记录与修复前一致。新增字段没有改变这些既有输出。

两个 `airline_803` 真实写入见证中，旧评测分别以 max_steps / too_many_errors 记为 0，新协议与训练都为 user_stop / 官方 1；21 次和 18 次工具执行及最终 DB 完全相同。这是控制流反例的修复，不是模型成功率提升。真实长查询 76,736 字符，两边可见 65,553 字符且 SHA256 相同；官方 simulation 中仍保留 76,736 字符原文。未知工具派发失败的 DB hash 与现场工具事件核对一致。

原 22 个审计见证（20 个控制流/观察案例 + 2 个真实 DB 写入案例）在独立新目录运行，旧审计和旧评分保持原样。v2 的固定预算与原本故意设置为 1 轮的训练探针仍不相同；另有两例 token/context 边界尚不等价。这些差异不得从报告中剔除后宣称完全等价。

初轮回归的 2 个失败是旧测试替身缺少新增回执属性，改为允许替身无回执；次轮 1 个失败是新增断言未开启事实记录，已修正测试设置。原日志保留，没有把失败运行覆盖为成功。第一次远程部署在修改前因平台持久盘软链接路径的安全断言停止，改为先解析根目录后再次校验；没有发生部分覆盖。

远程同步采用逐文件 preimage SHA256 校验、备份、原子替换和同步后 hash 复核；21 个本次源码/测试/脚本文件已完成。回执与前像位于 `results/maintenance/harness-fix-deploy-20260920/`，不代表整仓库 Git 状态相同。远程完整对照、lint/vendor 均完成。61 份主证据文件已下载并逐份 SHA256 核验；压缩包 SHA256 为 `7aa10766114a42098c08c913ca87e51770ae536d73a16dc01ac50baae0dc4421`。追加测试文件另有前像和同步回执。最终 21 个部署文件与本地相同，不代表整仓库 Git 提交/工作区相同。

## 最终核对与失败尝试保留

- 两端各运行 legacy 22 例 + v2 22 例。44 组跨主机摘要、原始工具事件、DB hash、训练 byte-token 记录、策略可见消息均匹配；每个分组的源码与任务/DB输入 hash 相同。
- 跨主机比较排除原生消息顶层的 `timestamp` 元数据，内容、工具参数和调用 ID 不裁掉。第一次直接比较含运行时间的消息序列失败，保留 `cross-host-attempt1.json`，没有把时间戳差异误当作业务内容差异。
- v2 的 22 例均得到一致最终 DB；18 例终止/分数也一致。其余 4 例为两例刻意保留训练 1 轮而新评测 15 轮，以及两例 token/context 上限差异。不是随机样本，18/22 不能解读为真实一致率或模型成功率。
- 追加状态快照测试第一次远程尝试在 300 秒到期前没有产生 pytest 结果，退出 124，原空日志和超时记录保留。独立 600 秒诊断尝试开启栈定时输出、强制本地 LiteLLM 价格表、关闭无关 pytest 插件自动加载，66.09 秒通过，退出 0。没有足够证据把首次延迟归因于某一个因素；未修改生产逻辑以绕过测试。
- `local-evidence-verification.json` / `remote-evidence-verification.json` 验证历史 legacy 输出不变、长返回 raw/visible 身份及派发失败现场 DB hash；`cross-host-verification.json` 保存跨主机断言结果。`check_evidence.py` 与 `check_cross_host.py` 可复核这些文件，无需模型服务。
- 实际 selection CLI dry-run 解析了 60×4=240 条计划并显示 v2 控制规则，没有启动采样。train 源 manifest 200 条与 selection60 均无初始消息历史，符合新协议的 fresh-session 限制；这不是把正式训练池由 50 改成 200。

## 仍未解决的研究边界

1. 真实 Qwen tokenizer、生成 token 上限、总 context 上限、HTTP/parser 行为、开场 prompt 与用户历史尚未证明训练/独立完全等价。确定性脚本使用 byte tokenizer 和伪 logprob，只证明控制流与工具/DB行为。
2. 新协议不能证明裁剪提高或降低模型成功率，也没有证明调大训练预算会带来净收益；以后需一致配置下的有限推理对照。
3. ERR-020 的无 DB 变化任务评分覆盖、ERR-021 的共享实体泛化边界继续开放。应保留官方 reward，同时独立版本化补充任务/对话指标，不能直接改旧官方分数或删任务。
4. 过程奖励配方、GTPO/GiGPO 优势和动态过滤未在本次变更中修改。其他任务正在进行的 split-v4 工作保留独立边界。
