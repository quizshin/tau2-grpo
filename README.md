# Tau2 → Tau3 GiGPO monorepo

This repository deliberately mirrors the four-directory layout of the reference
`agentic-grpo-longhorizon` repository while replacing its legacy τ-bench runtime
and experiment code.

```text
.
├── tau3-grpo-longhorizon/  # project-owned adapters, algorithms, configs and tests
├── env_info/               # reproducible Mac and A800 environment contracts
├── tau2-bench/             # vendored Sierra τ³-bench v1.0.1 source (package: tau2)
└── verl/                   # vendored veRL v0.7.1 source plus documented local patches
```

The naming is intentional. Sierra publishes τ³-bench in the repository named
`tau2-bench`, and its Python package/CLI remain `tau2`. AReaL τ²-style Airline
tasks are used for training and internal selection. The 50 official Airline tasks
from τ³-bench v1.0.1 are opened only after model selection for final evaluation.

## Quick start

```bash
bash setup.sh cpu-test
source .venv-cpu/bin/activate
pytest -q tau3-grpo-longhorizon/tests
```

On the A800 host:

```bash
bash setup.sh a800
bash tau3-grpo-longhorizon/scripts/smoke/check_installation.sh
```

See `tau3-grpo-longhorizon/README.md` for data preparation, training arms and
the frozen-final evaluation workflow.
