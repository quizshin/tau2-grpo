# Environment contract

This directory records reproducible requirements, not a snapshot of somebody
else's machine.

- Python 3.12 is used for both local CPU tests and the official τ³ runtime.
- `a800-torch28-cu128.md` records the remote training contract and preflight.
- `a800-constraints.txt` keeps veRL and vLLM 0.10.2 on Torch 2.8.
- `versions.lock` freezes source revisions before experiments.
