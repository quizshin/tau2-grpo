# 当前实验：50 任务上的四组工具智能体 RL 对照

此页是当前实验的统一入口。历史记录中的 40 任务、固定 50/100 step、其他硬件试验均不能替代下述协议。方案更新时应同步本页，并保留变更记录。

## 研究问题与范围

在相同 SFT 起点、训练任务和匹配的更新步数下，动态过滤（DF）和 Tau-GiGPO 的步骤信用分配，能否改善多轮工具任务的独立评测成功率？同时观察工具错误、长对话截断、奖励有效组比例和训练成本。

最初的长期目标包含 thinking 模型训练；**当前正式对照使用非 thinking 的 `new-off` SFT 起点**，不改变 thinking 开关，不据此回答“thinking 是否更好”。先固定现有起点检验 RL 方法，避免同时更换数据、推理模式与算法。

## 已确定的协议

|项目|当前设置|
|---|---|
|策略|Qwen3.5-4B，`new-off` 合并 SFT 权重，thinking 关闭|
|参数更新|全语言参数训练，冻结视觉分支；不是 LoRA RL|
|任务池|冻结的 50 个 Airline 训练任务；来自原 train200，与 selection60 无任务 ID 重叠|
|每步采样|8 个任务 × 每任务 8 条轨迹，共 64 条|
|E0|GRPO|
|E1|GRPO + dynamic filtering|
|E2|Tau-GiGPO|
|E3|Tau-GiGPO + dynamic filtering|
|共同起点|每组都从同一个 SFT 独立初始化，不继承前一组 RL 权重|
|执行顺序|E0 → E1 → E2 → E3|
|学习率 / KL|1e-6 / 0.01；训练采样温度 1.0|
|资源|5 × A800 80GB，四卡训练/策略推理，一卡 Qwen3.8-27B AWQ INT4 模拟用户|
|加速|FLA 0.5.2 FP32/IEEE、单轨迹外部 padding 裁剪、compact head；不使用外部 FA2|
|状态隔离|每卡 microbatch=1，轨迹独立，不复用递推状态，不做 sequence packing|

50 任务 manifest SHA256：`641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae`。数据 ID 隔离不等于已经证明全部任务语义无泄漏，也不代表奖励器完全覆盖业务正确性。

### 预算与评测

E0 从模拟器启动开始观察约六小时预算，包含初始化、保存和评测。在完整 step 后观察预算，向上取整到整十步 N 并完成该节点；E1/E2/E3 使用同一个 N。六小时不是强制杀进程的截止时间，50 步不是硬目标，E0 探索上限为 100。

四组匹配更新步数与配置，DF 的额外采样/过滤成本需结合实际日志比较，不能把相同 N 直接解释成相同总成本。

每 10 step 评测 selection60×4（采样温度 0.4）并保存完整恢复检查点。每组保留最新完整状态，确认新状态齐全后才清理旧状态。正常主动停止在整十节点完成保存与评测后停止队列。SwanLab 每 step 记录，续训使用原 run ID 和实际 global step。

效果判断以固定 selection 的独立评测及成本为依据；训练 batch 的 reward 仅作为训练诊断。官方 final50 留到模型锁定后再评估，不参与本次训练或选型。

## 当前进度快照

**核对时间：2026-09-12 20:41（Asia/Shanghai）。本节是快照，不是实时状态。**

- E0 正在运行，日志已有 5 个完整 step，SwanLab 最后记录 step=5；训练进程仍存活。
- 第 4/5 步完整耗时分别约 14.42 / 17.41 分钟。尚未到第 10 步评测节点，不能宣布训练后成功率提升。
- E1/E2/E3 尚无本次对照结果，最终匹配预算 N 尚待 E0 确定。
- [E0 SwanLab 实时监控](https://swanlab.cn/@quizshin/tau3-grpo-pytrio/runs/f7ef00e3d47b4742ba90b)。访问权限取决于 SwanLab 设置，GitHub 私有仓库权限不自动授予看板权限。

已验证的工程结论：先前独立单步实验完成 64 条新轨迹，完整 step 18.83 分钟、actor 更新 6.74 分钟，相比旧在线批次 32.15 / 19.97 分钟明显缩短；数值、有效梯度、会话隔离及权重同步检查通过。新旧在线批次不同，不能将全部提速归因于 FLA 单项，也不能将该单批速度当作每批保证。详见 [FLA 在线验证](fla_online_validation_20260912.md)。

仍未确定：四种 RL 方法的效果排序、稳定收益、最终 N 和所有组的实际总成本。长轨迹显存余量有限，完整检查点占用较大；后续组可能因存储空间不足暂停，不能据队列配置宣称四组已经跑完。

## 代码版本与唯一推荐入口

- **服务器当前训练版本：`792d70db61c7e905dfda27d231d54c0ecfc5a74c`。**
- GitHub 交接版基于本地合并提交 `458d649`，整合上述训练版本并保留 5090/Paratera 功能；后续说明文档提交不代表服务器训练代码已更新。
- 正式配置：`configs/train/rl/qwen35_4b_full_a800_c50_matched6h_{e0,e1,e2,e3}_20260912.yaml`，共用 `c50_matched6h_common`。
- 控制器：`env_info/a800_20260912/run_matched50.py`。
- 启动入口：`bash env_info/a800_20260912/launch_matched50.sh dry-run`；完成数据、环境及资源预检后才使用 `run`。
- [正式执行细节](formal50_fla_execution_20260912.md)、[本地合并与测试记录](local_training_merge_20260912.md)。本地合并版回归 112 项通过，15 项 CUDA 测试跳过；未在当前占用的训练卡上重跑合并版本。

## 接手运行前必须准备的资产

Git 仓库现在包含源码、配置、测试、原始 AReaL 数据、train/selection/reserve 划分、服务器实际 SFT 输入以及 40/50 任务课程 manifest 和 sidecar。见 [数据说明与 SHA256 校验](../data/README.md)。当前 50 任务 manifest 保留在 `results/analysis/rl_curriculum50_20260912/manifests/`，已通过精确例外规则纳入 Git；clone 后先校验哈希。**模型权重、完整 checkpoint、虚拟环境和凭据仍不包含在仓库中**，需要另行准备。

正式服务器使用 Python 3.12、Torch 2.11/cu130、Transformers 5.5.1、vLLM 0.20.0 和 FLA 0.5.2；启动脚本依赖服务器激活脚本、FLA overlay 及模型路径。迁移到新机器需重建环境并配置路径，不能复制 Mac 虚拟环境到 Linux。凭据在目标机器单独设置。

运行目录：`/root/autodl-fs/tau3-core-20260912/runs/rl-c50-matched6h-a800-20260912`。GitHub 不自动同步其中的实时日志、状态或 checkpoint；进度以带时间戳的服务器记录和 SwanLab 为准。
