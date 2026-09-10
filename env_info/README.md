# Environment contract

> 当前本地环境需要通过 setup 重建。2026-09-10 核对时，历史 `.venv-qwen35-local` 不存在，不能直接使用旧激活命令。安装 Python 3.12 后，在本项目根目录执行 `bash setup.sh cpu-test`，创建的是 `.venv-cpu`，然后执行 `source .venv-cpu/bin/activate`。这是本地 CPU 开发环境；A800 Qwen3.5 训练环境使用 `bash setup.sh a800-qwen35`，见 env_info。 本次仅更新状态说明，未安装环境或重新运行训练。

This directory records reproducible requirements, not a snapshot of somebody
else's machine.

- Python 3.12 is used for both local CPU tests and the official τ³ runtime.
- `a800-torch28-cu128.md` records the remote training contract and preflight.
- `a800-constraints.txt` keeps veRL and vLLM 0.10.2 on Torch 2.8.
- `versions.lock` freezes source revisions before experiments.
