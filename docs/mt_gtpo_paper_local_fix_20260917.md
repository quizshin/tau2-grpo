# MT-GTPO paper_v1 本地修复与使用

2026-09-17 本地修复记录。下文验证状态为当时记录；后续已同步远程并完成无卡验证，发现并补修航班查询分类遗漏，但真实 buffer 的 IRC 校准未通过。最新证据与边界见 [远程无卡验证](mt_gtpo_paper_remote_cpu.md)。没有启动新训练或生成校准通过的冻结配方。

依据与边界见 [论文一致性审计](mt_gtpo_paper_audit_20260917.md)。此实现补齐论文描述的奖励层与离线IRC流程；作者未公开细节仍采用下列显式约定，不声称与作者源码逐行一致。

## 实际修改

- `evaluation/paper_reward.py`：独立的论文奖励分类、参数规范化和逐轮计分。
- `evaluation/process_reward.py`：新增 `mode=paper, version=paper_v1` 分支，旧audit/conservative/reference_write v1/v2/v3含义保持不变。
- `algorithms/mt_gtpo_verl.py`：接入paper版本的终局一致性校验；总过程奖励指标使用实际turn_rewards，另记调用奖励总和，以区分可选mean聚合。
- `analysis/calibrate_paper_rewards.py`：完整训练buffer重算、按任务划分校准/留出、迭代调权、双重优势方向检查、不可覆盖报告与冻结配方导出。
- `scripts/train/rl/run_mt_gtpo_formal.py`：支持paper_v1与冻结配方；实际正式启动需要校准通过，续训锁定同一配方、Hybrid参数和DF开关。
- 新增独立训练profile和IRC参数文件；gamma=0.9、lambda=0.3、按uid/轮次分组保持不变。

## 奖励与未公开细节的实现约定

初始化权重：gold_exact=1、soft=0.5+0.5×重合比例、read=0、state=-0.1、error=-0.1、duplicate=-0.2、message=0、unknown=0。不限制只奖励DB写操作，不采用1/M或整条正预算1。

分类优先级：执行错误 → 尚未消费的精确gold → 重复成功调用 → 同工具软匹配 → 普通读取 → 状态改变 → 未知。
gold每个出现次数可消费一次；错误尝试不消费，不阻止合法重试。成功调用的规范化签名用于重复识别；重复gold以外的调用也适用。多个相同gold出现次数优先分别消费。

规范化：递归去掉None/空字符串/空列表/空字典条目；数字字符串按精确Decimal转换；字典列表排序，普通列表保持顺序；布尔值不同于数字；额外非空参数保留。完整比较参数，不使用compare_args忽略字段。这与环境执行等价模式不同：数字ID转换和列表排序可能放宽语义，必须在实际任务中审计。

soft比例按规范化后期望的顶层参数键值计算，默认得分0.5+0.5×比例。分类优先级与重合解释均是显式工程约定。部分字段相同但关键ID错误仍可能成为soft，不能把软匹配当作正确性证据。

`paper_options.aggregation=sum`为默认，同轮调用先求和，整轮共享Hybrid优势；可显式设mean。没有擅自改成call-level信用分配或语义状态分组。

校准迭代采用Algorithm 1的类别权重r_c，明确把`soft_scoring`从初始化的`overlap`切换为`constant`，避免对已校准r_c再乘一次重合比例。该变化记录进候选/冻结配方。

## 离线IRC

默认参数在 `configs/analysis/mt_gtpo_irc_paper_20260917.yaml`，alpha/delta/eta、支持数、留出比例等是本项目显式取值，不冒充论文公布的超参数。

每轮执行：

1. 输入必须是完整训练update的JSONL，含原始process和mt_gtpo_replay。先重算原奖励、优势、mask，拒绝缺样本/篡改记录。
2. 读取train-only manifest，拒绝selection/final标签与训练池外task_id。固定seed按任务划分20%留出；同任务各组/各轮始终属于同一侧，组内统计不跨任务。
3. 重新按paper模式计分，按轨迹中的类别出现与二元终局结果计算point-biserial，支持数不足/零方差记为不通过。
4. 仅用校准侧数据提议 `r_c=alpha*rho`（低于delta则0），固定中性类别保持0。预期方向由配置固定，不能跟随rho反转后自行宣称对齐。
5. 重算Algorithm 1即时奖励代理优势与正式Eq.(2) Hybrid优势；校准与留出两侧分别检查类别平均方向，以及平均过程回报与终局的关联是否超过eta。
6. 中性即时奖励不要求最终优势逐token为0，报告真实方向分布；未知工具类别出现会阻止冻结。混合好坏调用轮和执行错误获得正优势的覆盖范围额外报告，不冒充论文的逐调用保证。
7. 保存每轮候选、阈值、输入hash、支持数、关联、优势方向与失败原因。最后一轮通过才输出冻结配方，否则退出码2并仅输出report。

每个 `--round` 对应一轮新采集的训练buffer；重复文件hash会被拒绝，不能反复计算同一批数据制造“多轮收敛”。脚本不调用模型、不自动训练或采样。新配方需要新训练轨迹时，由明确的独立采样/训练任务提供下一轮buffer。

使用示例（从code目录运行；替换占位路径，当前未执行真实校准）：

```bash
python -m tau3_grpo.analysis.calibrate_paper_rewards \
  --train-manifest results/analysis/rl_curriculum50_20260912/manifests/areal_airline_train_seed42.jsonl \
  --config configs/analysis/mt_gtpo_irc_paper_20260917.yaml \
  --round /path/to/training-buffer/1.jsonl /path/to/training-buffer/2.jsonl \
  --output-dir /path/to/new-irc-report
```

多轮时追加 `--round /path/to/new-buffer/1.jsonl ...`。每轮根据前一候选重新评估，留出侧只用于验收，不参与权重估计；反复使用留出侧做人工选配方仍有选择偏差，需要额外独立评测。

通过产物 `frozen-recipe.json` 记录完整奖励/Hybrid参数、训练manifest hash、任务划分、验收报告、源码hash和配方hash。源码、配方或训练池变更必须重新审计。

## 训练入口与动态过滤

本次只实现和验证入口，没有执行以下训练命令：

```bash
python scripts/train/rl/run_mt_gtpo_formal.py \
  --reward-version paper_v1 \
  --reward-recipe /path/to/new-irc-report/frozen-recipe.json \
  --result-dir /path/to/new-paper-run \
  --updates 20
```

DF默认关闭；加 `--dynamic-filter` 开启，仍是联合优势之后的固定rollout整组过滤、不补采样。DF可选，不承担奖励语义校验职责。

正式控制器无冻结配方时拒绝启动paper_v1；`--dry-run`允许检查尚未校准的初始化配置，但会记录uncalibrated。通用底层launcher仍能用于显式的未校准算法研究，不应将其称为IRC完成的正式对照。

续训必须提供与原run相同的冻结配方；检查配方hash、完整奖励配置、Hybrid参数、DF设置，以及原有完整检查点和SwanLab身份。沿用20step预算、每10step保存与selection60×4评测、最新完整检查点保留1份的既有规则。

## 已验证与未验证

本地测试覆盖：额外取消的惩罚、嵌套比较/数字/空值/列表、重复与合法重试、soft计分、sum/mean、多调用对象隔离、旧模式回归；校准/留出隔离、负相关soft不得伪装对齐、零方差/低支持拒绝、完整buffer重算、重复buffer拒绝、冻结hash与失败报告；paper worker载荷、DF开关、shell/Hydra参数传递和冻结配方续训约束。

验收：相关测试共148项通过。整组首次147通过、1项旧shell测试因PATH缺少虚拟环境Python而失败；使用`.venv-cpu/bin`加入PATH后单独重跑该项通过（0.11秒）。新增与算法核心67项早先单独验证通过；上述数字有重叠，不相加。ruff、Python编译、git diff --check及veRL补丁契约通过。

测试集合：`test_paper_reward.py`、`test_paper_irc.py`、`test_process_reward.py`、`test_mt_gtpo.py`、`test_mt_gtpo_verl.py`、`test_algorithm_coexistence.py`、`test_rollout_boundaries.py`、`test_configs.py`、`test_formal50_profiles.py`。

没有真实训练buffer的paper配方校准通过结论，没有远程/GPU更新、跨rank或效果验证。
错误写入虽然扣分，同轮总奖励/优势仍可能为正；支持不足或方向冲突时流程会明确失败。后续应先用真实训练轨迹校准，再进行同预算消融，不能把本地测试通过当作成功率已改善。
