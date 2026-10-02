# 双 A800 部署

唯一维护的训练部署目标：Linux、Python3.12、2×A800 80GB、Qwen3.5-4B。使用 `env_info/qwen35-constraints.txt` 的固定框架版本；不再提供旧Torch2.8或5090安装路线。

策略训练与Qwen3.8量化模拟器仍是两个隔离组件，这是依赖兼容边界，不是两种硬件部署方案；不要把两个环境盲目合并。CPU开发环境只承担CI，不是另一套训练平台。

安装策略环境前指定目标目录。模拟器安装工具需要 `uv`，先按[uv官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)
准备可执行文件；下面通过 `command -v uv` 取其路径，不修改训练环境里的依赖。

```bash
export TAU3_BASE_VENV="$PWD/.venv-a800-qwen35"
export TAU3_VENV_DIR="$TAU3_BASE_VENV"
bash setup.sh a800-qwen35
source "$TAU3_BASE_VENV/bin/activate"
# 模拟器组件的固定安装入口，显式覆盖脚本中的历史目录默认值：
export TAU3_SIM_VENV="$PWD/.venv-qwen38-sim"
export TAU3_UV="$(command -v uv)"
test -x "$TAU3_UV"
bash env_info/setup_qwen38_simulator.sh
export TAU3_SIM_PYTHON="$TAU3_SIM_VENV/bin/python"
```

安装前阅读两个脚本的路径变量，并指向既有环境目录；不要在服务器原环境上未经核对重装。发布清理没有在服务器安装、运行或验证这些命令。

使用 `env_info/autodl/activate_canonical.sh` 设置标准资产路径；双卡共享profile指定策略0,1和模拟器1，启用模拟器休眠管理。单独source环境不会启动服务，也不能证明资源可共用。

旧服务器外部activate.sh不会随GitHub发布变化。迁移时核对其内容，不能假定它已采用新默认值。

完成环境安装后，按[模型资产准备](model-assets.md)下载4B、9B策略及共享量化模拟器，
按[数据说明](data.md)下载完整train500/dev150包。9B资产可独立准备，当前冻结训练profile仍默认4B。

详细模型边界：[Qwen3.5](qwen35.md)、[模拟器](qwen38_simulator.md)。
