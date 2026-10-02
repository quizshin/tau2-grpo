# Environment contract

> 2026-10-02本地CPU开发使用现有`.venv-cpu`（Python 3.12）；新机器可执行`bash setup.sh cpu-test`创建。最终core另在只安装CI声明依赖的临时干净环境检查，证据见`results/maintenance/local-code-cleanup-20261002/`。本地环境不代表远程训练环境；远程按对应host的既有激活脚本与环境登记运行。

This directory records reproducible requirements, not a snapshot of somebody
else's machine.

- Python 3.12 is used for both local CPU tests and the official τ³ runtime.
- `a800-torch28-cu128.md` records the remote training contract and preflight.
- `a800-constraints.txt` keeps veRL and vLLM 0.10.2 on Torch 2.8.
- `versions.lock` freezes source revisions before experiments.
