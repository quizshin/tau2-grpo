# code 架构审查与渐进优化方案

审查日期：2026-09-18（Asia/Shanghai）。状态：已开始分批实施；本文件的静态审查部分仍保留当时事实。第一批范围和未完成项见 [交付记录](batch1_20260918.md)。

依据：完整读取本地 `AGENTS.md`；本地目录、AST import、配置与入口审查；远程只读源码哈希、Git 状态、资产路径、进程及 editable 安装检查。远程状态采样时间为 20:26 左右。原审查阶段未启动训练、未改远程、未迁移资产。实施状态以交付记录为准，不能把本方案所有拟议路径当作已实现功能。

## 1. 审查结论

保留现有单仓库和 `tau3_grpo` 包，沿用环境/验证、算法、训练的职责边界。优先消除配置与运行控制的多处定义，建立可回放的数据接口，再迁移历史文件。此次不建议改成微服务、重写训练框架、整体切换 `src/` 布局、移动第三方目录或创建另一份长期维护的代码副本。

核心算法已有纯 NumPy 实现，环境具有独立会话，训练/最终任务隔离、轨迹记录、检查点与 SwanLab 续训机制已经存在。这些应作为迁移基线保留。问题主要是跨模块职责、历史运行脚本进入当前依赖链、配置优先级不统一及证据清单更新不足。

同日补充：参考 Mercor ApexAgents-SkyRL-Recipe 的实际源码后，增加 [Mercor 架构对照](mercor_comparison_20260918.md)。除下面的结构迁移路线外，应在早期统一跨算法事实记录、保留原始结束原因，并建立训练/独立评测 harness 差分验收；无需等全部目录迁移完成。DPPO、prompt-group reduction 和新停止策略作为独立行为实验，不并入结构重构。

## 2. 本地与远程核对

| 项目 | 本地 | 远程 |
| --- | --- | --- |
| 位置 | `/Users/apple/Projects/program-llm/tau3_grpo_fix/code` | `/root/autodl-fs/tau3-core/code` |
| Git HEAD | `08c05411fb4582e80236a1daa6f19a85bea3850c` | `1f58472f7961f760938ed8ceb3126fe4a9cb938a` |
| 审查时工作区状态条目 | 165：44 修改、121 未跟踪 | 92：23 修改、69 未跟踪 |
| 源码/配置/文档共有文件 | 1,828 个，内容和执行权限全部一致 | 同左 |

未跟踪数是 `git status` 条目数，部分条目可以代表目录，不能解释为精确文件数。核对范围是 `tau3_grpo/configs/scripts/tests/env_info/docs/verl/tau2-bench/.github` 和 9 个根文件；排除模型、数据、检查点、results、秘密文件、缓存和软链接。扫描仍包含本地 veRL egg-info 与远程历史 swanlog 配置，因此总数不等于纯源码数。

仅本地存在：`docs/mt_gtpo_formal_scope_20260917.md` 以及 5 个 `verl/verl.egg-info/` 安装元数据。仅远程存在：`env_info/historical/experiments/multicall-aa267bb/sft/{off,on}/swanlog/.../files/config.yaml` 两份历史运行配置。没有共有文件内容差异。证据见同目录 `audit_snapshot_20260918.json`。

远程本次未发现匹配的训练、vLLM、Ray worker、pytest 进程；`/dev/nvidia[0-9]*` 为空。这是当前快照，不代表以后仍空闲。未执行 GPU 验证。

远程 Python editable mapping 指向当前主仓库；veRL 与 tau2 的 `.pth` 也指向当前主仓库对应目录。激活脚本的 PYTHONPATH、环境位置与 AGENTS 登记一致。AGENTS 内容在哈希核对范围内，双方一致。

六个登记的模型/检查点资产软链接及目标均存在。本地 `models` 是目录，远程是资产软链接，本地没有同等完整运行资产布局；这些是部署差异，不应以目录同步消除。本次只检查存在性与链接目标，没有读取/校验全部模型权重。

## 3. 当前架构地图

统计排除 `.DS_Store`、pycache 等缓存；按实际可见文件计数，不等于 Git 跟踪数。

| 层 | 现有位置 | 规模与职责 | 判断 |
| --- | --- | --- | --- |
| 核心业务包 | `tau3_grpo/` | 116 个 Python 文件，约 16,026 行 | 保留包名，按职责渐进整理 |
| 参数与协议 | `configs/` | 97 文件，RL 配置 48 个；含 prompts | 配置组合与当前/历史分类需要整理 |
| 用户入口 | `scripts/` | 26 文件；3 个 Python 实现共 959 行 | Shell 薄入口保留，较重逻辑迁入包 |
| 数据 | `tau3_grpo/data/`、根 `data/` | 数据适配、划分、SFT、manifest | 区分输入清单与运行产物，保持内容身份 |
| 环境 | `envs/`、`prompts.py` | 独立 session、工具、模拟用户、协议 | 业务执行与 veRL 接口逐步分开 |
| 验证与奖励 | `evaluation/` | 官方 verifier、过程奖励、独立评测、指标 | 官方结果和 shaping 保持不同接口 |
| 信用分配 | `algorithms/` | GRPO 接入、GiGPO、MT-GTPO、DF、anchors | 保留纯算法，移出框架适配职责 |
| 训练 | `training/` | SFT 较完整；RL 入口主要负责注册和转发 | 正式运行管理应收拢到这里 |
| 实验身份 | `experiments/`（包内） | 任务调度、运行清单、winner lock | 继续负责可复现身份，避免扩成万能模块 |
| 模型与研究 | `models/`、`analysis/`、`anchors/` | 模型兼容、语义抽取、离线重放和 IRC | 消除运行模块反向依赖分析命令 |
| 框架接入 | `integrations/`、第三方补丁 | hooks、边界保存、预算、补丁验证 | 明确 callback 和运行管理的界线 |
| 记录 | `tracking/` | 指标、样例、SwanLab 续训、算法审计 | 记录消费结构化结果，减少私有算法依赖 |
| 部署和历史操作 | `env_info/` | 119 文件，79 个 Python 文件约 8,943 行 | 混有当前控制器、一次性诊断和历史证据 |
| 测试 | `tests/` | 81 Python 文件及 6 份 fixture | 分层执行与 CI 覆盖需要跟进 |
| 文档 | `docs/` | 113 文件 | 当前入口、实验记录、历史叙述需区分 |
| 第三方依赖 | `verl/`、`tau2-bench/` | 固定版本源码和项目补丁 | 不为减少目录数量而裁剪或升级 |

根 `experiments/` 本次只有 Finder 元数据，没有实质文件；不要与 `tau3_grpo/experiments/` 混淆。`outputs/`、缓存、egg-info 等也应在文档中注明是生成物，不加入源码发布清单。

当前训练链：

```text
Shell 入口 / 历史正式控制器 / MT-GTPO 正式控制器
  -> launch.py + YAML includes + 环境变量
  -> run_qwen35.sh -> run_base.sh
  -> training.rl.train -> veRL trainer
  -> ToolAgentLoop <-> interaction/session/tools/user simulator
  -> verifier + optional process reward
  -> estimator / filtering / actor update
  -> checkpoint / evaluation / telemetry / SwanLab / stop boundary
```

## 4. 有源码证据的问题

### 4.1 配置解析与覆盖规则分散

`launch.prepare()` 默认保留已存在的环境变量；正式控制器却先用 profile 覆盖部分环境变量。直接入口和控制器入口的有效参数来源不完全相同。`GROUP_SIZE`、`GROUPS_PER_UPDATE`、`TOTAL_UPDATES` 在 YAML 与多个 Shell 层重复设置，算法 arms 也同时存在 YAML 与 shell case。

最新 split 奖励配置依次继承 `c50_matched6h_common -> curriculum40_fast_pilot -> curriculum40_pilot -> formal`，另加 MT-GTPO overlay。当前正式实验因而依赖历史 pilot 配置。不能直接删掉这些旧配置。

优化：一个有类型的实验配置解析器，产出完整配置及每项来源；Shell 只处理激活和传参。通过稳定的模型、硬件、算法、奖励、数据、训练协议组件组合实验，当前配置不再继承历史实验。

拟定优先级：公共默认 < 模型/硬件组件 < 实验配置 < 显式 CLI。激活脚本只指定机器路径、环境和端点，不能隐式覆盖已选算法、SFT 初始化、任务或预算。旧入口保留旧优先级用于复现；新入口有明确版本，不能静默改旧命令语义。冻结奖励配方和续训身份字段受到额外校验，不能靠 CLI 无声覆盖。

### 4.2 正式运行管理没有统一归属

E0–E3 在 `env_info/a800_20260912/run_matched50.py`；MT-GTPO 在 `scripts/train/rl/run_mt_gtpo_formal.py`；独立评测控制器又在 `env_info`。它们重复进行服务生命周期、输入预检、源码快照、检查点校验、SwanLab 云端读回和完成回执。

优化：公共 RL 控制器放 `training/rl/runner.py`，服务生命周期放 `training/services.py`，运行 manifest 继续放 `experiments/`，检查点管理放 `training/rl/checkpoints.py`。独立评测仍由 `evaluation/` 负责，只复用服务与资产校验。算法只提供配置和 estimator，不再各写一套正式控制器。

保留明确状态：prepared/running/checkpoint_saved/evaluation_complete/verified/failed；训练结束、导出结束和评测完成分别记录。沿用既有预算、SwanLab 身份及边界停止协议。

### 4.3 轨迹格式分散，训练和离线工具依赖松散字典

存在官方消息、`tau3_turn_v1`、`tau3_process_reward_v1`、`mt_gtpo_replay_v1` 及 GiGPO 自己的回放格式。不同格式有必要，但身份、轮次、工具结果和 token 区间的公共定义缺少集中接口。

优化：在 `data/trajectory.py` 定义少量共享类型：TrajectoryIdentity、ToolEvent、AssistantTurn、TerminalResult、训练 token spans。区分 task_id、sample_group_uid、trajectory_id、trial、seed。保留原始消息和参数，另存执行规范化结果。环境实际状态 hash 与基于模型可见历史的 anchor 表示分开，隐藏答案不能进入可见状态。

训练轨迹要求精确 token spans/masks；独立推理评测不一定有这些信息，schema 必须允许明确缺省并标出能力，禁止为旧日志伪造字段。JSON 字符串运输可以继续留在 veRL adapter，核心逻辑使用结构化对象。旧格式提供只读解码器，不覆盖历史文件。

### 4.4 纯算法与 veRL 接入仍混在 algorithms

`tau_gigpo.py`、`mt_gtpo.py` 的纯计算适合保留。`verl_estimator.py` 与 `mt_gtpo_verl.py` 则包含 torch 转换、注册、解析奖励 payload、mask 修改、审计序列化及模块级 `_LAST_STATS`。

优化：这两份 adapter 迁至 `integrations/verl/`，算法接收数组/结构化记录，返回 advantages、returns、filter decision、diagnostics。数据交换和写 batch 由 adapter 负责。优先用每批显式结果传递替代全局最后一次统计；框架暂时要求全局注册的部分保持局限范围。

不在本次结构迁移中改变 gamma、归一化、UID 作用域、DF 先后顺序或奖励版本。已知旧 GiGPO UID 分组边界作为独立行为修复配置处理，保留历史可回放含义。

### 4.5 少数依赖方向和私有接口不合理

- `models/semantic_extractor.py` 导入 `analysis/replay_decisions.visible_message`。
- `data/prepare_sft.py` 导入 `training/sft/dataset._render_ids`。
- `tracking/signal_audit.py` 导入 estimator 的多个下划线私有 helper。
- 正式 MT-GTPO 控制器从 `analysis/calibrate_paper_rewards` 加载冻结配方。

这些静态依赖并不证明发生运行时循环导入，但使工具迁移和复用困难。

优化：可见消息投影进入 `data/trajectory.py` 或 `envs` 的公共纯函数；token 渲染归模型模板层；冻结配方读取/身份验证归 `evaluation/rewards/recipe.py`；校准分析只负责拟合与报告。审计接收 estimator 显式产生的结果，不再重新调用私有适配器逻辑。仅抽取已有多处共用行为，不新增笼统的 common/core 大包。

### 4.6 在线算法与语义候选的状态不清楚

规则 anchor、语义状态 v1–v4、API 抽取、增量方案和直接判等散落在 anchors/models/analysis/configs。文件存在不代表已经在线启用或通过独立验证。

优化：首先添加实验目录索引与状态（historical/development/validated），注明实际入口、配置、证据和默认是否启用。运行可用的 anchor 仍在 algorithms；API 客户端仍在 models；离线审计在 analysis。确有版本策略需求时通过显式注册表选择，禁止把未通过的语义候选自动接入训练。不要仅因名字带 v1/v2 就删除。

### 4.7 第三方补丁与验证证据清单需要统一

`THIRD_PARTY_SOURCES.md` 的静态补丁清单、`patch_contract.py` 的 requirement 清单与当前实际补丁范围不完全覆盖。仅扫描标记也会漏掉没有统一标记的模型导出、数值路径或 verifier 改动。现有部分契约检查只检查字符串出现，不能证明运行行为正确。

优化：机器可读补丁登记，记录 upstream revision、路径、补丁目的、关联回归与支持的硬件路径；文档由登记生成。保留少量 veRL 调用点，把项目逻辑委托给 integrations。先不重构 Qwen3.5/FLA/compact head 数值实现，不升级上游。

### 4.8 测试、文档和源码交付滞后

CI 显式选择的 CPU 测试没有包含新 MT-GTPO 和奖励校准测试；有些回归需要远程框架环境，不能仅以默认 CI 绿灯视为全部通过。README/CURRENT_EXPERIMENT 中也有历史状态和默认参数描述。Git HEAD 不足以表明当前真实运行源码，因为重要新增文件仍未跟踪。

优化：按 pure CPU、benchmark integration、veRL CPU、GPU update/resume 分层；新算法相关测试纳入相应层。近期不必先移动 81 个测试文件，可先用标记和集合统一执行。文档明确采样时间和当前入口；保留历史记录原文。建立已提交源码基线与单向发布清单，避免依靠聊天里的同步回执作为唯一来源。

## 5. 目标结构（示意，非要求一次建立全部目录）

```text
code/
├── configs/
│   ├── models/ hardware/ simulator/ tracking/ envs/ prompts/
│   ├── algorithms/ rewards/ data/       # 稳定可组合参数
│   ├── train/{sft,rl}/                  # 当前入口配置
│   ├── experiments/                     # 实验选择、状态和来源索引
│   ├── analysis/                        # 离线候选配置
│   └── historical/                      # 解除活动依赖后的旧配置
├── scripts/{train,eval,serve,data,analysis,maintenance}/ # 薄入口
├── tau3_grpo/
│   ├── configuration.py                 # 配置校验/来源/解析；先保持单模块
│   ├── launch.py                        # 唯一配置启动 facade
│   ├── data/                            # 数据与共享轨迹结构
│   ├── envs/                            # benchmark 会话、工具、模拟用户
│   ├── evaluation/
│   │   ├── verifier.py runtime.py metrics.py ...
│   │   └── rewards/                     # shaping 版本与冻结配方接口
│   ├── algorithms/                      # 纯信用分配、过滤、anchor
│   ├── integrations/verl/               # 框架注册、rollout hooks、batch adapter
│   ├── training/
│   │   ├── services.py
│   │   ├── sft/
│   │   └── rl/{train,runner,checkpoints}.py
│   ├── experiments/                     # 任务调度、run 身份、winner lock
│   ├── models/ analysis/ tracking/ utils/
│   └── paths.py prompts.py
├── env_info/                            # 环境锁定、激活、部署；历史另归档
├── docs/{architecture,operations,history}/
├── tests/
├── verl/ tau2-bench/                     # 原位置、固定版本
└── data/ models/ checkpoints/ results/   # 原逻辑资产路径
```

不要求用新名字替换所有旧模块。每次抽取必须有具体复用者或明确职责边界。旧路径如果是已发布入口，可保留有到期计划的转发脚本；不保留第二份实现，不恢复长期目录软链接。

## 6. 分阶段实施与验收

| 阶段 | 改动范围 | 必须通过的验收 |
| --- | --- | --- |
| P0：记录基线 | 记录两端 dirty 状态、源码内容/权限、实际入口/配置/任务/奖励身份；确定包含未跟踪文件的提交基线 | 重要新增文件均可恢复；两端差异逐项解释；不改资产与旧结果 |
| P1：配置与入口 | 建当前索引；抽稳定基础配置；集中解析；旧入口转发或历史冻结 | E0/E1/E2/E3、MT-GTPO DF 开关、SFT full/LoRA 的有效配置与命令逐项对照；模型/tokenizer/任务顺序/奖励/预算/精度/轮数完全保持；只允许事先声明的日志路径差异 |
| P2：运行管理 | 公共 runner、服务所有权、检查点/续训/停止、完成回执；迁入原控制器逻辑 | 远程 CPU 场景覆盖正常完成、边界停止、云端领先、分片缺失、已有 run 拒绝覆盖；不启动长训练验证目录迁移 |
| P3：接口与 adapter | 共享轨迹结构、过程奖励接口、框架 adapter、审计结果对象 | 旧轨迹奖励/优势/mask 精确重算；GRPO/GiGPO/MT-GTPO 切换无泄漏；多工具顺序、数据库隔离、终止分类不变；旧日志缺字段明确拒绝或降级 |
| P4：归档与发布 | 解除引用后迁移历史配置/操作脚本；补丁登记；CI 与文档；发布清单 | 活动源码不依赖 historical；旧证据不改；发布前后 hash/权限/真实 import 路径一致；测试层与跳过原因明确 |
| P5：真实集成 | 有卡后独立 smoke 与完整续训验证 | 新轨迹、非零更新、权重同步、保存/加载、续训步数及数据调度、SwanLab 同一 run；CPU 成绩不能替代 |

P1–P4 按可回退的小提交执行，不把算法修复与目录迁移放同一提交。配置变更与生命周期变更分别验收，不以“能 import”替代训练行为验证。GPU smoke 使用独立 run，预算/保存频率的例外明确记录，不能混入正式对照曲线。

正式协议继续保持 AGENTS：共同 new-off SFT、非 thinking、全参数、train50、seed42、20 外层 step、8×8 候选、LR 1e-6、KL 0.01；每 10 step selection60×4 与完整续训保存，仅保留最新完整检查点；SwanLab 每步在线、续训同 run 真实 step；主动停止完成下一个整十节点。新检查点通过对应完整性校验后才删除旧检查点，保留历史评测和轨迹。

## 7. 发布与回退

远程主路径保持 `/root/autodl-fs/tau3-core/code`，复用其现有环境；不移动六个资产链接、不改平台 `/root/autodl-fs` 链接。以文件清单发布源码，排除 `.env`、环境、模型、数据、检查点和结果，禁用整个项目根目录的删除式同步。

执行重构前建立包含现有有效未提交改动的可审查分支/提交；两端历史不同，不能用 reset/强推消除差异。发布记录同时包含 Git revision、必要 dirty patch、源文件哈希和运行环境身份。若不能当次统一历史，至少先让同一发布清单可核对；统一 Git 历史另作显式迁移。

更新前再次检查远程进程，避免修改运行中进程的源码。迁移 Python 入口或安装元数据时，在原环境重新安装所需 editable 包并核验实际模块路径。回退使用前一源码提交/清单，不覆盖运行产物；遇到不兼容 checkpoint 或 recipe 应拒绝启动，不能静默转换后宣称精确续训。

## 8. 原审查阶段交付边界

本轮完成静态架构审查及远程只读核对，仅新增本方案与审查快照；没有改算法、配置、启动脚本、依赖、远程源码或运行资产，没有执行训练、模型推理或回归测试。该方案优先推荐实施 P0 与 P1，后续按每批实际配置快照推进公共运行管理。


## 9. 实施约定补充

用户随后明确授权实施，要求当前行为不永久冻结，并在需要 GPU 时停止等待决定。P0 记录可回退源码身份，不能解释成禁止修改算法/harness。第一批先完成明确的公共边界和严格比较入口；历史配置解除依赖、全算法 trajectory 契约与原始结束原因等独立批次状态见交付记录，不以搬文件宣称验收完成。

## 10. 后续实施状态

后续 CPU 批次已继续完成稳定配置组件、统一 checkpoint、独立评测 controller、全算法 trajectory facts、原始结束原因、公开 recipe、显式 batch diagnostics、评测 provenance、补丁清单与分层验证入口。逐项证据见 `batch2_20260918.md`。

P4 不按日期批量搬走仍有复现引用的配置/脚本；它们已有 historical 状态和唯一活动入口。后续删除以引用解除为条件。Git 工作区原有未提交历史未被重写；本次通过两端哈希发布清单与源归档交付，尚未把整个既有 dirty 工作区统一成一个新提交。P5 的真实 GPU 更新/同步/恢复/独立推理验收保持待用户决定。
