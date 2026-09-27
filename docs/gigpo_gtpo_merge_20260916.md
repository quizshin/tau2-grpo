# GiGPO 与 MT-GTPO 代码基线合并

日期：2026-09-16。远程主仓库 `/root/autodl-fs/tau3-core/code`；本地对应仓库 `/Users/apple/Projects/program-llm/tau3_grpo_fix/code`。

## 合并范围

统一文件内容，保留双方 Git 历史和已有未提交工作。合并前本地 HEAD 为 `08c05411fb4582e80236a1daa6f19a85bea3850c`，远程 HEAD 为 `1f58472f7961f760938ed8ceb3126fe4a9cb938a`。没有整体覆盖仓库或同步模型、数据、环境、检查点和旧实验结果。

远程已有的 GiGPO episode 归一化选项、训练信号审计、SwanLab 日志修复、Qwen3.5 导出校验、统一存储路径及维护历史均保留；本地 MT-GTPO 的算法、过程奖励、逐轮记录、过滤、IRC 重算、配置和测试合入远程。

前次清单的 113 个“远程独有文件”指 Git 可见清单差异。其中 3 个 VLA 文件本地实际存在且 SHA256 与远程一致，只是受 `verl/.gitignore` 中 `ENV/` 的大小写匹配影响被忽略。本轮补充实物核验，未将它们误作缺失后覆盖。其余 110 个文件与 21 个同名差异文件从远程引入本地；共享的 `ray_trainer.py` 单独三方合并。

## 三条算法路径

| estimator | 奖励/分组 | 过滤与审计 |
| --- | --- | --- |
| `grpo` | 标准终局 GRPO | 保留现有 E0/E1；GiGPO/GTPO 私有逻辑不进入该路径 |
| `tau_gigpo` | 终局回报、anchor 步级项；episode 归一化显式选 `legacy_mean` 或 `grpo` | 保留 E2/E3 过滤、过滤前 mask 及 `gigpo_signal` 审计 |
| `mt_gtpo` | 同一 uid 的同轮次回报；audit 或 conservative 过程奖励 | 可选按最终优势过滤，保留逐轮重算记录及空 batch 处理 |

历史 GiGPO 配置仍默认 `legacy_mean`，新增选择不隐式改变旧实验；GTPO 默认配置不替换 E0～E3。新语义比较器仍为离线候选，未因本轮合并启用在线 LLM 语义分组。

## 冲突处理

`verl/verl/trainer/ppo/ray_trainer.py` 有一处文本冲突：优势计算之后，本地需要记录 GTPO 过滤结果及 `mt_gtpo_replay_json`，远程需要执行 GiGPO 的 `audit_update`。

合并为按 estimator 选择的两个分支，保留双方逻辑。数据 payload 同时支持两种自定义算法；`norm_adv_by_std_in_grpo` 仅向 GiGPO 透传。过滤前的 GiGPO mask 捕获继续受算法与环境开关约束；GTPO 先算优势再过滤，普通 GRPO 不使用这两个专属分支。

新增 `tests/test_algorithm_coexistence.py`：在同一进程内按不同顺序切换三种 estimator，使用全失败但含工具错误的批次，检查 GTPO 过程信号得以保留、旧 GiGPO 过滤行为不变、普通 GRPO mask 不受影响，以及指标/审计记录不串入其他算法。

## 验证与回执

远程 22 个相关测试文件共 **379 passed、0 failed、0 skipped，608.60 秒**。涵盖三种 estimator 的共存与切换、GTPO 数学及奖励重算、多工具调用与终局 payload、过滤及 padding、GiGPO 信号审计、启动配置、Qwen3.5 小型随机模型前后向及导出回载、SwanLab 离线日志。唯一 warning 为 Ray state API 弃用提示，不影响通过结果。

`python -m tau3_grpo.integrations.verify_patches --check-registration` 通过，明确回读 `tau_gigpo` 与 `mt_gtpo` 注册成功。新增共存测试的 ruff 检查、相关 shell 语法及 Git diff 空白检查通过；新增测试最初的 import 排序提示已修正，未改变运行算法。

所有项目执行均使用远程既有 qwen35 环境，并隐藏 CUDA 设备；本地只做编辑、源码合并与文件核验。本轮没有启动真实策略训练任务、模型 rollout 或分布式 FSDP 验证；小型随机模型 CPU 回归不能替代这些验证，也不证明算法效果提升。

证据目录：`results/maintenance/merge-gigpo-gtpo-20260916_01/`。其中保存双方原始清单、源码归档、补丁、三方合并结果、部署清单及测试日志。部署前逐文件核对旧 hash，避免覆盖同步期间的其他写入。`source_verification.json` 记录源码 SHA256、可执行权限和本地/远程差异，`completion.json` 记录完成状态及最终归档哈希。核验范围不包括模型、数据、环境、结果和 `.git` 内部文件；Git HEAD 与双方历史保持原样。
