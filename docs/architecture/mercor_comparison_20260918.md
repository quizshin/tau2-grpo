# Mercor 配方架构对照与 code 优化补充方案

日期：2026-09-18。状态：源码对照与提案，未实施运行时代码改动。

本文补充 `refactor_plan_20260918.md`。已读取任务 `01a0b3cb-5b12-7652-b3dd-c31af71a1c32`（“解读 Mercor A397B 训练指南”）、本仓库 AGENTS.md，以及该任务产出的本地优化计划。主要判断重新对照实际源码；其他任务的运行声明不算本轮验证结果。

## 1. 对照基线与证据范围

- 参考仓库：`Mercor-Intelligence/ApexAgents-SkyRL-Recipe`。
- 本轮拉取的 revision：`8e7702f03b7464a36ab800a624fd911de0968a87`。结论限于这个快照。
- 仓库来源：`https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe`。
- 本地只读副本：`/tmp/apex-recipe-architecture-20260918`；临时副本不是永久交付依赖，以仓库身份、revision、文件路径定位证据。
- 目标为本地 `tau3_grpo_fix/code` 当前工作树，包含未提交修改；不能用 HEAD 代替真实源码身份。
- 远程对照沿用同日约 20:26 的 `audit_snapshot_20260918.json`，本轮未重新连接远程。该快照的 1,828 个共有文件内容和执行权限一致，不是当前进程状态或 GPU 可用性的持续保证。
- 没有安装或执行参考仓库代码，没有运行目标项目测试、训练或模型推理。

## 2. 参考仓库实际上如何分工

```text
ApexAgents-SkyRL-Recipe/
├── pyproject.toml / uv.lock
├── scripts/                         # smoke、两种大模型训练配方、eval
└── apex_agents_skyrl_recipe/
    ├── entrypoints/
    │   ├── common.py                # 配置、数据集、generator 与 Ray 接线
    │   ├── main_tito_harbor_train.py
    │   ├── main_tito_harbor_fully_async.py
    │   └── main_tito_harbor_eval.py
    ├── dataset.py                   # task directory -> dataset item
    ├── tito_harbor_generator.py     # trial -> SkyRL GeneratorOutput
    ├── agents/
    │   ├── archipelago.py           # agent 多轮执行、MCP 工具交互
    │   ├── tito.py                  # 原始 token / mask / logprob 状态
    │   └── llm.py / ...
    ├── ecr_modal_env.py             # Harbor Modal 环境扩展
    ├── harbor_trial_config/archipelago_tito.yaml
    └── metrics_helper.py
```

任务包、训练框架、沙箱服务是这个目录以外的依赖，不能拿其顶层文件数直接比较我们的完整研究仓库。

| 职责 | 参考系统中的归属 | 关键含义 |
| --- | --- | --- |
| 环境 | Modal sandbox + ECR world image + MCP servers | world 提供文件和工具；不同 trial 获得自己的环境 |
| 任务 | 外部 Harbor task directory | 包括 instruction、task.toml、world 元数据、隐藏 verifier/reference |
| harness | Harbor Trial + ArchipelagoAgent + trial config | 规定启动、消息、工具执行、预算、终止、验证与回收 |
| 轨迹 | TITOAgentState | 同步维护 token IDs、loss mask、采样 logprobs，追加 observation |
| 可验证结果 | Harbor verifier_result 与 trial artifacts | 任务评测代码随任务包分发；README 描述了 LLM rubric judge |
| 训练接入 | TITOHarborGenerator | 将执行结果转换为 tokens、reward、mask、stop reason 等训练输入 |
| 参数更新 | 外部 SkyRL trainer、actor、vLLM | 训练循环、loss、分布式更新、权重同步不是本仓库重写的 |

调用链为：训练数据选 task path → generator 发起 trial → Harbor 启动环境并调用 agent → agent 请求推理、执行 MCP 工具 → verifier 判分 → generator 整理 token 数据及有效性 → SkyRL 计算优势、loss 并更新权重。

它的关键设计是“执行一次任务”和“更新模型”通过清晰的数据输出连接。`tito_harbor_generator.py:316` 的 GeneratorOutput 包含 `prompt_token_ids / response_ids / rewards / loss_masks / stop_reasons / rollout_logprobs`；`agents/tito.py:267` 检查三组逐 token 数组的长度一致。

但该仓库也有明确代价：generator 约 700 行，agent 约 1,464 行，两份大模型启动脚本各约 255 行；`ecr_modal_env.py` 为补扩展点复制了一个与 Harbor 版本耦合的私有方法；依赖配置含机器路径、定制 wheel 和多组 override。因此借鉴其职责边界，不复制全部目录与部署方案。

## 3. 我们已有对应能力，需要加强哪些接口

| 参考概念 | 当前 tau3 实现 | 优化重点 |
| --- | --- | --- |
| TaskDataset | `data/`、包内 `experiments/` 的 manifest 与调度 | 固化 task、采样组、数据库和协议身份 |
| Trial / sandbox | `envs/session.py`、`tools.py`、`interaction.py` | 复用现有会话与数据库隔离，明确生命周期 |
| Agent / harness | veRL ToolAgentLoop；独立评测的 Orchestrator + MultiCallAirlineAgent | 固定动作输入下做差分验证，共享可复用规则 |
| TITOState | ToolAgentLoop 的 prompt_ids、response_mask、response_logprobs | 统一所有算法的事实记录及 token 对齐检查 |
| Verifier result | `evaluation/verifier.py` 中已有 TerminalReward | 保留 reward、breakdown、scored、failure_category，不再造同义结果类型 |
| Generator adapter | agent_loop 输出、interaction/reward 接口、训练 batch 转换 | 项目相关转换逐步集中到 `integrations/verl/` |
| Trainer | veRL + `training/rl/train.py` + 正式控制器 | 框架计算继续留 veRL，项目运行管理迁入公共 runner |
| Research algorithms | `algorithms/` 的 GRPO/GiGPO/MT-GTPO | 保留独立纯计算；适配器、记录与过滤决策边界清楚 |

### 已有基础不能误判为缺失

`ToolAgentLoop` 直接追加生成的 `output.token_ids`，工具和模拟用户 observation 的 mask 为 0。项目不是每轮把完整生成历史转成文本再重新编码，已经具备 TITO 的关键基础。

`response_logprobs` 与 `rollout_log_probs` 的传输能力也存在，但 vendored 默认配置 `calculate_log_probs: False`；本轮搜索 `configs/`、`scripts/` 和项目包没有发现显式开启。环境变量、CLI 和远程 resolved config 仍可能改变它，因此实际实验是否保留行为概率尚待确认。

现有 veRL 已注册 `dppo_tv`、`dppo_kl`。如后续引入 Mercor 式训练目标，工作是核对概率来源、mask、reduction 与实际 actor 路径，不是再实现一份同名 loss。

### 事实记录必须独立于算法

目前 `record_process_turns` 由 `adv_estimator == mt_gtpo` 决定；GRPO/GiGPO 有其他记录，但缺少统一逐轮事实接口。建议拆开：

- 基础事实：任务/组/轨迹身份、token spans、工具参数和结果、原始结束原因、可见 observation、策略/模板/协议版本。
- 奖励解释：官方终局结果、过程奖励及冻结 recipe 身份。
- 学习解释：advantages、过滤前后 mask、loss 配置、需要时保存的精确重放 tensors。

基础记录由三条算法共享；开启记录不能额外调用过程奖励模型，不能自动改变 shaping。沿用现有回放工具，增加共同 schema 和版本化解码器。缺少字段的旧日志应明确标记不能精确重放。

### harness 需要版本与一致性证据

harness 是模型完成任务时的执行规则集合，包括提示和模板、工具解析/顺序、用户交互、预算、重试、终止、验证调用与回收。它不是模型权重，也不等于工具清单。

当前训练和独立评测由两个执行循环承担。不要先强行合并两种运行时：先把固定 assistant 输出与用户回复送入两者，对比消息、工具顺序、数据库变化、轮数/长度预算、终止分类与 reward，再抽取共同规则。各自保存 `harness_id` 和协议 hash，训练内评测与独立评测仍分别报告。

本轮源码确认 vLLM adapter 将 `finish_reason=stop` 与 `length` 都映射为 `completed`，损失了原始逐轮截断类别。应先兼容保存原始原因，再单独决定续写/重试/判分行为。不能把 Mercor 的 length 强制 reward=0 直接复制过来。

### Reward、无效样本和动态过滤分别管理

参考 generator 对 length 改 reward，对基础设施错误/超时按规则 mask 个体或整组。这些是该配方的训练选择，不能由“放进了 generator”推断其通用于 tau。

我们的建议分工：

1. `evaluation/` 保存官方 TerminalReward 原始结果和独立的过程奖励明细；verifier 失败与任务真实失败可区分。
2. 运行有效性策略明确哪些 infra/截断轨迹允许训练、是否重试、如何计候选预算，并保存原因。
3. `algorithms/` 负责按优势信号做 DF；MT-GTPO 保持联合优势后过滤、固定候选预算不补采样。
4. veRL adapter 合成最终训练 mask，保存各阶段来源；policy loss 内部 gate 不与 DF 混成一个含义不明的过滤开关。

可靠 reward 需要可追踪的任务、初始状态、实际动作、verifier/recipe 版本和重算证据。接入 Harbor 或使用 LLM judge 本身并不保证判分正确；保留 tau 官方验证与已有隔离/重放工作更合适。

## 4. 精简目标结构：沿用现有包，只收拢职责

```text
configs/                         # 稳定组件 + 当前实验组合；最终配置有来源
scripts/                         # 薄入口
tau3_grpo/
  configuration.py              # 解析、校验、配置来源（拟议）
  data/trajectory.py            # 身份、事件、token 区间（拟议，复用已有类型）
  envs/                         # session、tools、interaction；共享执行规则
  evaluation/                   # 官方 verifier、过程奖励、独立评测
  algorithms/                   # 优势/credit assignment/DF/anchors 纯逻辑
  integrations/verl/            # batch、注册、rollout hooks、框架转换（拟议）
  training/rl/runner.py          # 公共正式运行生命周期（拟议）
  training/rl/checkpoints.py     # 项目级恢复校验与轮换（拟议）
  training/services.py          # 服务所有权、启动与回收（拟议）
  experiments/                  # 调度、身份、运行 manifest
  tracking/ analysis/ models/   # 复用现有职责
verl/ tau2-bench/               # 固定版本及登记补丁，保持当前位置
env_info/                      # 部署环境记录；解除活动依赖后归档旧控制器
```

不要求新增顶层 `harness/`、`generator/`、`core/` 三套抽象。先用接口和少量模块形成边界；不把整个 veRL ToolAgentLoop 复制进项目包。必要的调用点继续留在框架补丁中，将项目专属逻辑交给适配层。

`configs` 和 `scripts` 的镜像分类合理：前者定义“用什么参数”，后者定义“调用哪个入口”。需要消除的是 YAML、shell、历史控制器同时拥有参数默认值和算法分支。正式实验应组合稳定组件，摆脱对历史 pilot 配置的继承。

当前 `env_info/.../run_matched50.py`、`scripts/train/rl/run_mt_gtpo_formal.py` 与独立评测控制器中的服务/快照/校验逻辑应复用；训练、评测各自保留业务流程，共享明确的服务与资产接口。

## 5. 配置要区分的训练维度

| 维度 | 回答的问题 | 例子 |
| --- | --- | --- |
| Reward recipe | 任务完成或过程表现得到什么分 | 官方 reward、版本化过程 shaping |
| Advantage estimator | 分数如何分配给动作 | grpo / tau_gigpo / mt_gtpo |
| Sample eligibility | 哪些执行结果可以进入学习 | infra、timeout、length 的明确规则 |
| Signal filtering | 有没有可学习的优势信号 | 既有 DF，按算法定义先后顺序 |
| Policy loss | 用优势如何更新概率 | 当前基线、候选 DPPO-TV |
| Loss reduction | 不同长度/任务组如何加权 | 历史语义、候选 prompt_group_mean |
| Harness protocol | 模型实际看到什么、如何行动 | 模板、工具、轮数、停止/重试 |
| Runtime profile | 如何部署与执行计算 | GPU、模拟器、精度、服务端点 |

因此 DPPO 不是与 GRPO/GiGPO/MT-GTPO 并列替换的一个 estimator。参考大模型脚本实际分别设置 `advantage_estimator=grpo`、`policy_loss_type=dppo`、`loss_reduction=prompt_mean`。

如果开展 prompt-group reduction，先定义组为同一次采样的 sample_group_uid，而不是全局 task_id；在明确 optimizer minibatch 范围内，对组内有效 token loss 求均值，再对有效组平均。全空组、DF 和 loss gate 对分母的影响，以及跨 microbatch/rank 缩放必须定义并验证，不能只改配置名称。

对于 DPPO，要区分采样行为概率、训练端重算旧概率和当前概率。现有实现的可用性不等于已经接成 Mercor 配方；实际 actor 快捷分支、bypass、概率修正需要专项审查。它们是后续行为实验，不属于文件整理。

## 6. 对原方案的优先级补充

原方案 P0–P5 是结构迁移路线。本次增加一条“先证明接口语义”的验收路线，不要求等待所有目录迁移完成后才处理训练可观测性。

| 顺序 | 交付 | 验收与边界 |
| --- | --- | --- |
| 第一批 | 冻结源码/有效配置基线；统一轻量逐轮事实，保留原始结束原因 | GRPO/GiGPO/MT-GTPO 同一记录协议；保持历史行为，记录不引入 shaping；基础 token/mask 对齐 |
| 第二批 | 稳定配置组件与公共正式 runner | 各旧入口有效参数可比；checkpoint、评测、停止、SwanLab 续训规则不变 |
| 第三批 | 共享轨迹接口、框架适配边界、训练/评测差分用例 | 固定输入下动作、DB、终止与奖励差异可解释；既有奖励/优势可重算 |
| 第四批 | 依赖解除后归档历史文件；第三方补丁和 CI 分层登记 | 当前运行无历史脚本依赖；上游 revision/补丁/回归可追踪；两端按源码清单发布 |
| 独立研究批次 | 验证真实 reduction、行为概率，再实验 prompt-group / DPPO | 固定 tensor 与实际 FSDP 验证后，使用新实验配置和共同基线；不覆盖历史含义 |

训练语义的审计可与结构整理交替推进，但提交和验收要分开。已知 UID 作用域、归一化或停止策略的行为修改不夹带在路径迁移中。

暂不迁移 SkyRL/Harbor/Modal，不照搬全异步、长上下文和集群拓扑；先在现有部署测量 generation、用户模拟器、工具、验证、actor update、权重同步、保存/评测耗时，再决定性能投入。`rollout.mode=async` 本身不能说明 trainer 已是全异步训练。

正式实验继续遵循 AGENTS.md 的共同 SFT、20 step、8×8 候选、固定 train50/selection60/final50 边界、KL、保存/评测和 SwanLab 恢复约定。新机制在 tau 上能否提高效果，本轮没有运行证据。

## 7. 源码定位

参考仓库（revision 见开头）：

- `README.md`：Architecture、Data format、Grading、Dependency pinning。
- `entrypoints/common.py`：配置加载、TITOHarborExp、dataset/generator 注入。
- `entrypoints/main_tito_harbor_fully_async.py`：委托外部 FullyAsyncRayPPOTrainer。
- `dataset.py`：任务目录适配；`harbor_trial_config/archipelago_tito.yaml`：agent/environment/verifier 配置。
- `agents/tito.py:65`、`:148`、`:267`：状态、追加、长度契约。
- `tito_harbor_generator.py:316`、`:333`、`:571`：输出、trial 执行、失败组处理。
- `ecr_modal_env.py`、`pyproject.toml`：环境扩展与依赖/版本耦合。
- `scripts/run_qwen35_397b_fully_async.sh:193`：estimator、policy loss、reduction 分别配置。

目标仓库：

- `verl/verl/experimental/agent_loop/tool_agent_loop.py:178`、`:442`、`:624`、`:660`：记录条件、token 追加及 observation mask。
- `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:592`：finish reason 映射。
- `verl/verl/trainer/config/rollout/rollout.yaml:216`：默认 logprob 开关。
- `verl/verl/trainer/ppo/core_algos.py:1370`、`:1451`：已有 DPPO 注册。
- `tau3_grpo/evaluation/verifier.py:45`：已有 TerminalReward。
- `tau3_grpo/evaluation/runtime.py:64`：独立评测执行链。
- `tau3_grpo/training/rl/train.py`：现有框架注册与转发入口。

另一个任务的详细训练语义提案在 `/Users/apple/Documents/ChatGPT/Apexagentx/tau3-fix-optimization-plan-20260918.md`，可作为专项审查清单；其中未验证的运行条件不能直接提升为已确认训练故障。
