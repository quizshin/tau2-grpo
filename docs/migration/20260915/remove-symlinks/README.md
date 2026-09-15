# 项目软链接改为实际路径 — 2026-09-15

主代码 `/root/autodl-fs/tau3-core/code`、环境 `/root/autodl-fs/tau3-core/environment` 和激活入口不变。平台 `/root/autodl-fs → /autodl-fs/data` 保留，字面目标仍为 `../../autodl-fs/data`。

- 15 个产物目录通过同盘移动替换链接；包括 E2/E3 checkpoint 及辅助产物、三份 SFT 合并模型。
- 128 个附件链接替换成普通文件，逐个 SHA256 相符。移动时核对了 4,270 个既有普通文件的 inode、设备和尺寸。
- 移除项目根目录 runs/checkpoints/models/experiments/migration 兼容链接、旧日期根入口、4B tokenizer 别名。配置及启动路径直接引用 code 内实际文件；4B tokenizer 直接读取完整模型。
- 扫描项目层软链接为 0；扫描排除 environment、.git 及缓存。Python 标准运行时链接保持不变。
- E0–E3 均为 step 20，每组 13 个 checkpoint 文件尺寸相符，包含四份优化器分片；每组两份 validation JSONL 均存在。
- 三个 venv 无 PYTHONPATH 时仍解析到主仓库；qwen35 的 torch 导入和 CPU 张量运算成功。E0–E3 启动 dry-run 成功，起始模型为 code/checkpoints/sft-merged/new-off，anchor 为 v1。

远程完整 CPU 测试首轮：928 passed、4 failed、15 skipped；四个失败均为 formal profile 断言仍期待旧兼容路径。修正这四个期望后，该文件复测 4 passed，最终 932 个不同测试通过、15 个 GPU 测试跳过。首轮后仅修改测试期望及文档，没有进一步改动运行逻辑。CPU FSDP 为 world_size=1 / NO_SHARD，不构成 GPU 多卡训练验证。

完整日志、原始链接清单和文件校验信息在 `code/results/maintenance/remove-symlinks-20260915/`。本目录保存小型审计结果及路径映射。历史实验 launch/receipt 和旧迁移清单保留当时原文；其中路径按 path_map.json 的最长前缀迭代转换。源文件历史分支仍在同一仓库。
