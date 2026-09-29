# 双 A800 部署

唯一维护的训练部署目标：Linux、Python3.12、2×A800 80GB、Qwen3.5-4B。使用 `env_info/qwen35-constraints.txt` 的固定框架版本；不再提供旧Torch2.8或5090安装路线。

策略训练与Qwen3.8量化模拟器仍是两个隔离组件，这是依赖兼容边界，不是两种硬件部署方案；不要把两个环境盲目合并。CPU开发环境只承担CI，不是另一套训练平台。

```bash
bash setup.sh a800-qwen35
# 模拟器组件的固定安装入口：
bash env_info/setup_qwen38_simulator.sh
```

安装前阅读两个脚本的路径变量，并指向既有环境目录；不要在服务器原环境上未经核对重装。发布清理没有在服务器安装、运行或验证这些命令。

使用 `env_info/autodl/activate_canonical.sh` 设置标准资产路径；双卡共享profile指定策略0,1和模拟器1，启用模拟器休眠管理。单独source环境不会启动服务，也不能证明资源可共用。

旧服务器外部activate.sh不会随GitHub发布变化。迁移时核对其内容，不能假定它已采用新默认值。

详细模型边界：[Qwen3.5](qwen35.md)、[模拟器](qwen38_simulator.md)。
