# 架构完成范围与剩余工作（2026-09-20收尾）

本轮主架构整理与已约定工程/接口验收全部完成；正式效果研究仍待开展；源码发布和托管CI另见[发布记录](git_publication_20260920.md)。具体逐项证据见[最终接口报告](interface_acceptance_20260919.md)，实验与失败分别记录于[EXPERIMENTS](../../EXPERIMENTS.md)和[ERRORS](../../ERRORS.md)。历史清单前像完整保存在`results/maintenance/interface-acceptance-20260919/docs-final-preimages/`，远程同步也保留前像。

## 已交付

- 活动配置分层、公共训练入口、服务生命周期、完整checkpoint校验、算法与veRL适配边界；configs存参数、scripts存薄入口，不再按同名目录误判重复。
- 三算法统一真实轨迹事实：环境/任务/DB/harness身份，真实token、mask、逐轮结束原因、多工具执行与观察，采样组/trial/data-seed语义及生成分布logprob。旧轨迹缺失字段保留缺失，不补造。
- 官方结果、过程奖励、学习优势、执行资格与算法DF分开；资格策略版本化，基础设施失败不混成普通零分；预算截断明确未评分，固定候选不补采样。
- 全活动文本权重映射逐值/SHA256审计，覆盖GRPO/GiGPO/MT初始同步及MT两次更新后状态；不覆盖vision/TP>1/量化/LoRA。
- GPU更新、完整保存、SwanLab真实步数、GiGPO四rank真实续训与数据进度、724张量导出、真实加载和独立评测均通过。MT DF off/on各128候选的过程奖励、优势、过滤离线重放通过。
- pass@k/pass^k、完整性检查、对照增幅与配对置信区间接口，以及实验/错误记录标准；接口实现与工程小样本不等于已证明算法提升。
- CPU 1302通过，GPU-only17通过，控制器补丁15通过；增量lint零新增。集合有重叠，不能相加。

## 借鉴边界

参考`Mercor-Intelligence/ApexAgents-SkyRL-Recipe`固定快照`8e7702f03b7464a36ab800a624fd911de0968a87`的任务执行/更新分离、具名harness协议、真实轨迹、验证与学习信号分离、可重算实验和消融协议。保留本项目veRL、tau环境、GRPO/GiGPO/MT-GTPO与现有算法配方；没有迁入SkyRL/Harbor/Modal或复制参考算法。行为允许继续演进，改动需新配置/版本/证据，不冻结。

33个历史入口仍有引用（31直接、2测试动态引用），已登记用途与替代关系，应保留。兼容默认已收敛到YAML、共享arm解析器，71组命令对照一致；不为目录简短删除复现依赖。

## 后续独立任务

|任务|范围与完成条件|
|---|---|
|正式效果与增幅|匹配SFT起点、任务调度、seed和20step/1280候选预算，统一独立评测，输出增幅和配对置信区间；本次未额外启动|
|多seed与机制消融|按已登记消融计划验证终局/过程信号、GiGPO分组、DF等，不用两步smoke推断因果效果|
|DF真实剔除覆盖|本次DF on两批共16组都有信号、剔除0；CPU边界已覆盖，真实筛除收益需要另选协议与实验|
|Git发布记录|2026-09-20 按用户要求分为运行实现与文档提交；运行提交 `d0667f5`。补入三份被忽略的上游源码，742份 vendor 文件与 Git index 校验一致。发布版本与回执见[发布记录](git_publication_20260920.md)|
|托管CI|工作流执行 core CPU、lint 和 vendor 清单；本地发布前 core 379项、配置与三算法相关72项通过，集合不相加。GitHub 执行状态单独核对，见[发布记录](git_publication_20260920.md)|
|历史lint|382条已登记旧债，相关模块修改时逐步处理；不宣称全仓lint-clean|
|harness扩展|现有确定性差分与真实工具链不等于所有上下文/终止/采样协议完全等价；行为修改需新证据|
|按需扩展|通用跨实验队列、强类型全配置、DPPO/IS/RS、prompt-group reduction及全异步不属于此次重构完成条件|

## 资产与关机

当前保留五份正式 E0–E3/MT-v3 step20 完整检查点，清理前后结构校验通过。2026-09-20 经用户授权，GRPO 工程 step2 及 MT-GTPO DF off/on 各自 step2 共三份已退役，释放 151.90 GiB，持久盘可用 219.31 / 500 GiB；78份验收证据 hash 未变。三份工程节点明确放弃精确续训，失效 latest 指针已移除。回执见 `results/maintenance/engineering-checkpoint-retirement-20260920/receipt.json`。GiGPO临时step3此前已在C/D通过后退役，D推理导出保留；四份E0–E3可再导出的推理权重此前已删，源hash和恢复说明保留。缓存本轮未清理。

最终关机结果以本地`results/maintenance/interface-acceptance-20260919/shutdown-015.json`的verified_off为准；仅关闭015，不释放/删除。AutoDL页面提示连续关机15天会自动释放，系统盘/数据盘资产不是永久备份。正式研究不随关机自动启动。

2026-09-20 清理时只读核对：015已由外部重新以无卡模式运行，本任务未执行开机或GPU实验；03:25关机回执仅代表当时状态。本次按当前清理授权保留无卡运行状态。源码同步核对覆盖1,912份文件，运行资产与环境独立管理。
