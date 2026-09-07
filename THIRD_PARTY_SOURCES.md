# Third-party sources

This repository vendors exact benchmark and training-runtime revisions so that
the experiments remain reproducible. Project-owned code is under
`tau3-grpo-longhorizon/`; third-party trees keep their upstream licenses.

| Component | Local path | Upstream revision | License | Local treatment |
|---|---|---|---|---|
| Tau3 Bench | `tau2-bench/` | `sierra-research/tau2-bench` tag `v1.0.1`, commit `fc0055dc4e0a316c3f83133267fbd6faaa770992` | MIT | Vendored benchmark/runtime source; no project patches |
| veRL | `verl/` | `volcengine/verl` tag `v0.7.1`, commit `bec9ef74768dd201881cd4e54cd0385e87caae27` | Apache-2.0 | Vendored source with five Tau3-GRPO patch files |
| AReaL synthetic data | downloaded outside Git | `inclusionAI/AReaL-tau2-data`, revision `86971dc03da6e7c1a7933295e05b84aab8215386` | See upstream dataset card | Immutable training/selection input; raw files and generated artifacts are Git-ignored |
| Layout reference | not vendored | `qiqihezh/agentic-grpo-longhorizon`, commit `2004fcbc747b9b282bc0a8ce0c006683c7c42751` | See upstream repository | Directory-layout reference only |

Machine-readable pins live in `env_info/versions.lock`.

## veRL patch boundary

The five modified veRL files are:

1. `verl/experimental/agent_loop/tool_agent_loop.py`
2. `verl/trainer/config/algorithm.py`
3. `verl/trainer/ppo/ray_trainer.py`
4. `verl/utils/config.py`
5. `verl/workers/actor/dp_actor.py`

Local changes are marked with `Tau3-GRPO local patch` and guarded by
`tau3-grpo-longhorizon/tests/test_patch_contract.py`. Algorithm semantics and
runtime responsibilities are described in
`tau3-grpo-longhorizon/docs/implementation_status.md`.

The root MIT license applies only to project-owned work. It does not relicense
the vendored source trees.
