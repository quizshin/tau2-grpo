# 5090 空闲四卡 RL 可运行性检查

用户授权：在A/B SFT评测继续运行时，使用剩余GPU0、1、4、5测试GRPO和MT-GTPO能否运行。不得占用评测GPU2、3、6、7及端口8202、8203、8210。

策略使用0、1、4三卡，GPU5运行独立27B AWQ模拟器（TP1，端口8230，2路并发，16K上下文，0.85显存比例）。GRPO然后MT-GTPO顺序运行，共用模拟器。各2次更新，每次3任务组×4条=12条，共48条训练轨迹；总墙钟上限90分钟，每组训练最多40分钟。该上限含独立服务启动；不自动延长。

两组均从A45最终30步合并权重初始化，Qwen3.5-4B、non-thinking、全参数（lora_rank=0）、seed42、lr1e-6、KL0.01、温度0.7。保留tau3_token_budget_v1及完整工具schema。FP32 FLA/IEEE及compact head继承现有已验证路径；适配31GiB显存使用参数和优化器CPU卸载。与A800正式配置的差别是设备/卸载和小预算，不作为速度或模型效果对照。

训练池来自目标机现有train200冻结清单，使用现有确定性调度挑选6个任务位置，两个算法顺序一致。未使用selection或final作为训练输入；验证禁用，满足框架结构的val_files指向同一训练计划，不冒称独立验证。MT-GTPO使用paper_env_split_v4默认初始化权重，没有加载校准recipe，明确只做工程检查。短测仅本地console/JSONL记录，不声称SwanLab云端、完整恢复或收敛验收。

预期检查：真实多轮采样、GRPO/MT-GTPO优势计算、有限非零梯度、优化器更新、更新后下一轮rollout权重同步、step2完整checkpoint结构。现有权重审计覆盖卷积；不冒称所有语言权重均经过逐张量独立核对。若实际梯度为0、更新跳过、保存不完整或发生错误，分别报告。

目标目录：`/root/shared-nvme/tau3/runs/rl_5090_feasibility/20260921-spare4-v1`。配置为`configs/train/rl/smoke_3x5090_token.yaml`与`smoke_3x5090_mt_gtpo_token.yaml`。实验接线脚本复用`launch.prepare`、`experiments.prepare`、现有`evaluation.controller.Controller`的命令/服务所有权管理和`training.rl.checkpoints.validate_checkpoint`，没有复制训练实现。Ray使用独立短临时路径，每组最多24个CPU，不连接已有集群，不执行全局ray stop。

15:26：两组真实shell展开与Hydra配置检查通过，包含estimator、全参数、3卡、2步、save2、无评测、token协议、温度及MT版本。控制器PID13639已启动，独立模拟器加载中；尚无RL更新成功证据。

15:36：首次GRPO未开始采样即在actor→rollout初始权重传输失败：新smoke配置的512MiB bucket无法容纳FP32词嵌入约2425MiB。该错误是测试配置问题，并非已证明的5090容量不足。已停止本次所有自有RL服务（A/B评测不受此停止影响），恢复原项目2560MiB默认bucket；旧目录保留。新`20260921-spare4-v2`增加CPU最大张量/bucket容量检查后重跑，90分钟总截止沿用原运行，约16:56:53，不因重试重新计时。

15:55：v2初始权重同步通过，真实GRPO首批12条采样、old/ref打分与优势计算完成，但首次backward在FLA Triton自动调优内CUDA OOM；没有完成optimizer step，MT尚未开始。首批12条response mask共23850 token，非零优势6347 token，优势均有限。失败目录和batch保留，所有自有RL服务清理。参数/优化器CPU offload是阶段卸载，update_actor仍把优化器状态加载回GPU，不等于CPU优化器。

16:02：单独GPU0、相同FP32 FLA内核在257和4096长度下反向通过，所有输入梯度有限；首次含编译/自动调优约143秒，PyTorch峰值分别0.323/1.159 GiB，不能当作整卡峰值。证据v2/fla-free-memory-probe.json。这支持尝试预热缓存，尚不能证明训练共驻时能通过。新v3仅复用预热后缓存，预算截止仍16:56:53，先做CPU配置预检再运行；未改变精度、模型、数据或任务长度。

16:07：v3两算法CPU预检通过，控制器PID27712已启动，当前独立模拟器加载；策略仍0/1/4，模拟器5，评测2/3/6/7。没有启动40-step正式RL。用户请求40步耗时推算：暂以64轨迹/步，40步2560训练轨迹，另每10步selection60×4共960评测轨迹。给排期的条件范围18–28小时/算法，假定后续完整step约20–30分钟（纯训练13–20小时），再加评测/启动/保存；此为情景估算，尚无5090完整optimizer step可实测支撑，8卡具体策略/模拟器布局及batch整除也须正式启动前验证。历史4策略A800+1模拟器的18.83分钟单步只作背景，协议和硬件不同，不作为5090速度证据。

同时16:07:40刷新A/B评测：A150/240、B159/240，共309/480（64.375%），两组执行错误0；按近阶段吞吐还需约35–50分钟，不保证固定完成时刻。
