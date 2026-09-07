# Tau3-GRPO on veRL

Long-horizon tool-agent reinforcement learning on Tau3's Airline environment,
implemented on a pinned veRL/vLLM stack. The repository trains and selects on
the frozen AReaL Tau2-style synthetic pool, then opens the 50 official Tau3
Airline tasks only for the final frozen winner.

## Repository layout

```text
.
├── tau3-grpo-longhorizon/  # project-owned adapters, algorithms, configs and tests
├── env_info/               # pinned source revisions and A800 environment contract
├── tau2-bench/             # Sierra Tau3 Bench v1.0.1 source (Python package: tau2)
└── verl/                   # veRL v0.7.1 source plus five documented local patches
```

The `tau2-bench` name is intentional: Sierra publishes Tau3 Bench in that
repository, while the Python package and CLI remain named `tau2`. Training and
internal selection use AReaL's Tau2-style Airline records; official Tau3 tasks
are protected from training and selection by source guards and a frozen-winner
lock.

## Experiment arms

- E0: vanilla GRPO
- E1: GRPO + fixed-rollout Dynamic Filtering
- E2: Tau-GiGPO with structured state anchors
- E3: Tau-GiGPO + fixed-rollout Dynamic Filtering

All four arms have completed one real optimizer update in the 8×A800 validation
topology (six policy GPUs and two user-simulator GPUs). This verifies rollout,
tool execution, user simulation, official reward, advantage computation,
optimizer update and actor-to-vLLM weight synchronization. It is infrastructure
validation, not evidence of a benchmark improvement; formal 7B multi-update,
multi-seed, selection and final Tau3 experiments remain outstanding.

## Quick start

Deterministic CPU checks use Python 3.12:

```bash
bash setup.sh cpu-test
source .venv-cpu/bin/activate
pytest -q tau3-grpo-longhorizon/tests
```

On an A800 host with the pinned Torch 2.8/CUDA 12.8 stack:

```bash
TAU3_VENV_DIR=/root/tau3-venv-a800 bash setup.sh a800
source /root/tau3-venv-a800/bin/activate
bash tau3-grpo-longhorizon/scripts/smoke/check_installation.sh
```

See `tau3-grpo-longhorizon/README.md` for data preparation, SFT, E0–E3
launchers and the frozen-final evaluation workflow. Source revisions and local
patch boundaries are documented in `THIRD_PARTY_SOURCES.md` and
`tau3-grpo-longhorizon/docs/implementation_status.md`.

## Collaboration

`master` is the stable branch and `dev` is the integration branch. Create
focused feature branches from `dev` and merge through pull requests. See
`CONTRIBUTING.md` for checks and repository-boundary rules.

Project-owned code is released under the MIT License. Vendored `tau2-bench/`
and `verl/` retain their upstream MIT and Apache-2.0 licenses respectively.
