# 开发与实验记录标准

生效于 2026-09-18。规则约束代码职责和证据记录，不冻结未来算法或 harness 行为。对具体实验记录精确配置，是为了可比较与可回放。

## 职责与依赖

|模块|负责|新增代码要求|
|---|---|---|
|configs|模型、硬件、数据、协议、算法参数|先复用 includes；新增 profile 必须能到达真实入口；区分有效配置与输入 YAML|
|scripts|命令入口|调用包内公开 API；不增加算法或另一份运行管理实现|
|configuration / launch|组合配置与启动|记录包含链和哈希、环境来源、CLI 覆盖及最终 Hydra；不得静默改变旧优先级|
|envs|任务状态、会话、工具、模拟用户|实际执行和 gold 验证独立数据库；不同轨迹隔离；同轮多调用完整按序对应|
|evaluation|官方验证、过程奖励、独立评测|终局官方 reward 与训练 shaping 分开；奖励变化使用显式版本|
|algorithms|优势、信用分配、过滤、anchors|尽量纯数组输入输出；不启动服务、读凭据或依赖离线 CLI|
|integrations/verl|注册、torch/batch 转换、框架回调|接入三种 estimator；批次身份与 mask 不丢失；上游补丁尽量委托到此|
|training|SFT/RL 运行流程、服务与续训|共享 controller 与服务所有权；运行失败不能被 tracking 的 FINISHED 代替|
|data / models|可见消息、数据契约、模板和模型兼容|可见投影不带 gold/reward 元数据；保留原始 token 身份，禁止重 tokenize 冒充训练 token|
|analysis / tracking|只读重放、诊断和记录|消费公开接口；不能成为模型运行层的反向依赖|

公开配方接口为 `evaluation/rewards/recipe.py`；`analysis/calibrate_paper_rewards.py` 只拟合、导出并兼容旧 import。新配方使用 v2 身份；v1 仅接受已登记的迁移前校准源码 SHA 且数值算法、奖励、基准工具身份全部一致，未知身份拒绝。旧文件原样保留。

trainer 消费 `batch.meta_info["tau3_estimator_diagnostics"]` 的本批结果；`last_stats` 仅兼容旧外部调用。全算法事实记录不等于过程奖励：只有 MT-GTPO 把 turn records 送入 shaping。精确 token、生成/观察 mask、发出/保留 span 和逐轮原始结束原因由 `data/trajectory.py` 定义，不能把未提供的 logprob/UID/seed 填成猜测值。

## 新功能的最小交付

1. 写明改动的研究假设、模块归属及可观察结果。已有功能优先扩展配置，不按实验日期复制整个控制器。
2. 新参数必须贯通 YAML、CLI、实际入口及配置快照；给出默认值、非法值处理、与旧版的差异。
3. 修改共享入口验证 GRPO、GiGPO、MT-GTPO 三条路径。修改奖励/分组/DF 必须能在旧 buffer 上重算，不能只看单个成功样例。
4. 只增加能捕捉实际风险的测试：协议差异、丢轨迹、多工具、数据库隔离、空 mask、跨算法泄漏、续训身份或服务所有权。纯文档无需 GPU。
5. 更新本索引中的职责、必要实验记录与错误条目，说明本地/远程 CPU/GPU 分别验证到了哪里。
6. 发布使用变更文件清单、哈希和备份；保留工作区既有变更。模型、环境、数据和历史结果不进入源码同步。

兼容转发只保留入口，不保留第二份实现。删除旧入口前检索源码、配置、测试和历史复现说明；至少一个迁移批次证明无活动依赖，再记录替代入口与删除原因。

## 每次实验应保存什么

`results/runs/<experiment_id>/<run_id>/` 保存运行事实；`EXPERIMENTS.md` 是唯一实验索引，详细报告放 `docs/`，不另建互相竞争的实验总表。

- 身份：Git revision、dirty patch/源码归档与哈希，依赖环境，模型与 tokenizer 身份，数据/任务/DB 哈希，算法/奖励/harness 版本。
- 配置：输入 YAML 包含链、CLI、环境变量来源（凭据脱敏）、最终 Hydra、实际模型服务配置。`launch_inputs_not_final_hydra` 不能当作完整生效配置。
- 预算：外层 step、候选轨迹数、实际 actor 更新数、过滤后有效 token/组数、模拟用户调用、墙钟时间/GPU 小时及统计口径。
- 轨迹：task/sample_group/trajectory/trial/seed、原始消息与工具请求/响应、终止原因、reward 明细；训练额外保留精确 token span、mask、old logprob 与行为策略版本，缺失能力必须标出。
- 产物：checkpoint 和可恢复性回执、模型导出及加载回执、独立评测计划与错误、SwanLab run/step、完成或失败状态。

允许 `planned / development_checks_passed / cpu_verified / gpu_smoke_verified / trained / independent_evaluated / blocked / failed / historical` 等有证据支撑的状态。训练结束、导出成功、评测完整是不同事实。

## 分数和增幅

官方全成功判定沿用 `1±1e-6`，报告 pass@1/2/4 与 pass^k；过程 reward、训练平均 reward、独立 pass@k 分开。先核验完整 task/trial/seed 和协议，再按任务配对。任何未评分尝试保留在分母计划中并使总指标不可用；不取交集、不把 endpoint 异常直接当模型 0 分、不用补测覆盖原失败。

增幅写绝对百分点与相对变化；基线 0 时相对变化为空。CI 按任务配对 bootstrap；它不覆盖训练 seed 差异，也不自动控制多算法/多指标选择带来的偏差。单个 seed 的点估计不能宣称稳定提升。

成本必须带字段覆盖率。工具成功应有执行响应与 error flag，不能仅凭模型写出调用；token 使用服务 usage，缺失不是 0；累计轨迹耗时不是总墙钟/GPU 时间。旧记录的 hash 缺失要披露，不从当前代码补造历史身份。

## 验证分层与 GPU 边界

- 本地：静态审查、语法、源码对照；不冒充远程集成。
- 远程 CPU：同一既有环境、`CUDA_VISIBLE_DEVICES=""`、离线依赖；纯算法、benchmark 集成、veRL CPU 配置/适配器与离线重放分项记录。
- GPU：新采样、非零更新、actor→rollout 同步、保存/恢复、真实 SwanLab 连续性必须单独授权后实测。

用户于 2026-09-18 明确要求在需要 GPU 的地方停下来。提交实验目的、卡数、轨迹/step 预算、预计成本范围与停止条件后等待决定。CPU 通过不自动授予启动模拟器或评测推理的权限。

## 可执行维护入口

- 活动配置索引：`configs/experiments/catalog.yaml`。新正式配置通过协议、模型、硬件、数据、模拟器、奖励组件组合；不继承日期命名的 pilot。
- `python scripts/maintenance/check_cpu.py --suite core|benchmark|verl|all` 使用 `env_info/cpu_test_suites.json`；新增测试文件必须归类，未归类使入口失败。`--list` 只列出用例。该入口将 OpenMP/MKL/OpenBLAS/NumExpr 线程数限制为 1，避免小模型 CPU 检查启动数百线程；不修改正式训练配置。
- `python scripts/maintenance/check_lint.py` 拒绝新 lint 诊断；历史债务显式登记在 `env_info/lint_debt.json`，不得因测试通过声称全仓库 lint clean，也不得无理由扩展豁免。
- `python -m tau3_grpo.integrations.vendor_inventory` 离线核验已部署 runtime/packaging 清单。修改 vendor 后先补 `env_info/vendor_patch_notes.json` 的目的、测试和 GPU 局限，再用指定 revision 的上游 tar 包重建清单；不能仅改当前 hash 掩盖未知补丁。
- checkpoint receipt 证明文件集合、路径和字节数结构完整；GPU 加载、参数映射及精确续训是单独的验收，不能由收据替代。

## 兼容默认与记录草稿（2026-09-19）

- 基础 shell 默认由 `training/rl/runtime_defaults.py` 读取 `configs/train/rl/base.yaml`；Qwen3.5 的兼容参数在 `configs/runtime/rl_qwen35.yaml`。新增可调默认先写对应 YAML，再补入口映射；不在 shell 重新复制同一值。正式协议的显式覆盖与历史兼容默认分别保留，不能为去重改变旧命令。
- arm 与 ablation 使用 `configuration.resolve_arm`；启动快照的 `runtime_configuration_sources` 记录兼容参数来源。原生 Hydra 覆盖最后追加，修改后验证三算法的最终命令或完整 Hydra。
- `python -m tau3_grpo.experiments.review --receipt KIND=PATH --experiment-id ID --run-id ID --output-dir NEW_DIR` 从明确回执生成实验／错误草稿；KIND 为 `gpu_acceptance`、`formal_controller` 或 `evaluation`。新目录防止覆盖人工批注；审核后归入已有唯一索引。草稿仅转录事实，不推断因果、算法增幅或真实 GPU 验证完成。
