# 故障索引

当前维护约束见 [开发验证](docs/development.md) 和 [运行维护](docs/operations.md)。

- CI依赖必须声明，真实tokenizer验证不属于GitHub无模型core套件。
- 保留兼容导出；lint自动整理后须运行完整相关集成测试。
- 本地Torch2.8不等同于Qwen3.5固定Torch2.11环境，先核对版本再判断FSDP故障。
- 训练checkpoint、云端step和实际更新须分别验收；保存成功不代表可恢复。

历史故障的影响、修复及实验身份见[归档完整记录](https://github.com/quizshin/tau2-grpo/blob/archive/pre-a800-focus-20260929/ERRORS.md)。

- SFT包的ready标记不能替代文件、审核和实际token/mask校验；旧v1历史依赖清理后使用portable v2包，不能仅改路径绕过验收。

- macOS子进程退出期间，僵尸进程组的signal 0探测可能返回权限错误。清理循环继续回收并等待组消失；TERM/KILL的真实权限失败仍抛出，不能提前标记停止成功。
