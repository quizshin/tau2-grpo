# Third-party sources

This repository vendors exact benchmark and training-runtime revisions so that
the experiments remain reproducible. Project-owned code is under
`tau3_grpo/`; third-party trees keep their upstream licenses.

| Component | Local path | Upstream revision | License | Local treatment |
|---|---|---|---|---|
| Tau3 Bench | `tau2-bench/` | `sierra-research/tau2-bench` tag `v1.0.1`, commit `fc0055dc4e0a316c3f83133267fbd6faaa770992` | MIT | Vendored benchmark/runtime source; injected Airline DB isolation patch |
| veRL | `verl/` | `volcengine/verl` tag `v0.7.1`, commit `bec9ef74768dd201881cd4e54cd0385e87caae27` | Apache-2.0 | Vendored source with project patches; see inventory scope below |
| AReaL synthetic data | `data/raw/areal_tau2` | `inclusionAI/AReaL-tau2-data`, revision `86971dc03da6e7c1a7933295e05b84aab8215386` | See upstream dataset card | Frozen raw inputs and split manifests are versioned; see `data/README.md` for provenance and SHA256. Generated artifacts stay ignored |
| Layout reference | not vendored | `qiqihezh/agentic-grpo-longhorizon`, commit `2004fcbc747b9b282bc0a8ce0c006683c7c42751` | See upstream repository | Directory-layout reference only |

Machine-readable pins live in `env_info/versions.lock`.

## Airline environment patch

`tau2-bench/src/tau2/domains/airline/environment.py` deep-copies an explicitly
injected `FlightDB` for each environment. The official evaluator constructs both
predicted and gold replay environments with the same `env_kwargs`; retaining the
same mutable DB makes their final hashes spuriously equal. This patch preserves
the official evaluator, tool behavior and reward basis while isolating replay
state. Regression tests cover a missing required write and a correct write.

## veRL patch boundary

The original documented patch inventory is below; it is not a complete inventory of later Qwen3.5, reward-recording and export patches:

1. `verl/experimental/agent_loop/tool_agent_loop.py`
2. `verl/trainer/config/algorithm.py`
3. `verl/trainer/ppo/ray_trainer.py` — includes project telemetry and optional SwanLab training-trajectory logging
4. `verl/utils/config.py`
5. `verl/workers/actor/dp_actor.py`
6. `setup.py` — isolated Qwen3.5 dependency extra; retain legacy vLLM/NumPy bounds
7. `verl/models/transformers/monkey_patch.py` — native padded Qwen3.5 forward guard
8. `verl/utils/tokenizer.py` — explicit text-only Qwen3.5 processor bypass
9. `verl/workers/config/rollout.py` — opt-in sequential multi-call configuration
10. `verl/trainer/config/rollout/rollout.yaml` — Hydra execution-mode default

The project launcher enables `tool_execution_mode=sequential`: every parsed call
executes in list order, with one recorded assistant message for the entire batch.
SFT, RL and project evaluation use `airline_sequential_multicall_v1` from
`tau3_grpo/prompts.py`. This changes the single-call restriction in the agent
prompt; the pinned benchmark policy file, tools and evaluator stay unchanged.
Evaluation provenance explicitly records `benchmark_policy_modified=true`.
See `docs/sequential_multicall_20260911.md` for the execution and data contract.

Local changes are marked with `Tau3-GRPO local patch` and guarded by
`tests/test_patch_contract.py`. Algorithm semantics and
runtime responsibilities are described in
`docs/implementation_status.md`.

The root MIT license applies only to project-owned work. It does not relicense
the vendored source trees.

Qwen tokenizer/model revisions are recorded in
`configs/models/qwen35.json`; downloaded files are ignored.
The project-owned full-dialogue supervision helper was adapted from the sibling
`code_pytrio/src/tau3_pytrio/qwen35.py`, with no dependency on that cloud backend.


## 2026-09-18 adapter boundary update

The pinned revisions above were not upgraded. This batch changes project imports
in `verl/experimental/agent_loop/tool_agent_loop.py` and
`verl/trainer/ppo/ray_trainer.py` to the canonical
`tau3_grpo.integrations.verl.{gigpo,mt_gtpo}` adapters. Core NumPy estimators,
reward implementations and benchmark files keep their previous bytes. Legacy
adapter imports alias the same module objects; no parallel registry or statistics
state is introduced. CPU checks cover estimator coexistence, reward replay,
mask/filter boundaries and patch contracts; GPU update/resume remains unverified
for this refactor. Evidence: `docs/architecture/batch1_20260918.md`.

The machine-readable pinned upstream/runtime diff is now `env_info/vendor_patches.json`; rationale, test paths and GPU limitations are in `env_info/vendor_patch_notes.json`. Rebuild with `python -m tau3_grpo.integrations.vendor_inventory --archive-dir <pinned-archives>` and verify offline without this flag. The scope is every file in the veRL/tau2 runtime packages plus packaging inputs, excluding generated caches and upstream AGENTS instructions. Upstream docs, examples, datasets and test suites are outside this runtime inventory; it is not an assertion that the full upstream repository was vendored.
`patch_contract.py` checks known integration requirements; its string checks are
not a proof of all GPU behavior or an exhaustive inventory.

## 2026-09-20 engineering acceptance

The bounded three-algorithm interface checks, GRPO/GiGPO updates, GiGPO resume,
export/load and independent evaluation, and MT-GTPO reference_write v3 DF off/on
updates and replay have passed. See
[the final interface report](docs/architecture/interface_acceptance_20260919.md)
for exact evidence and limitations. Full active-text mapping covers dense TP1,
no quantization and no LoRA; vision is excluded. Earlier “unverified” statements
above describe their historical batches, not the final acceptance state. Pinned
upstream revisions remain unchanged.
