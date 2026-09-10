# Implementation status and 14-day mapping

Day labels are progress metadata only; no temporary `day1/` code paths exist.
Progress appears here, in source file headers and in commit messages.

| Milestone | Permanent implementation | Verification |
|---|---|---|
| D1 | `paths.py`, `tau2_bridge.py`, `data/schema.py`, `data/dataset.py` | `test_paths_and_layout.py`, `test_data_splits.py` |
| D2 | `data/manifest.py`, `data/leakage.py`, `data/sft.py`, `sft/dataset.py`, `env/adapter.py`, `env/verifier.py` | `test_data_splits.py`, `test_leakage_audit.py`, `test_sft_data.py`, `test_verifier.py` |
| D3 | `env/session.py`, `integration/interaction.py`, `integration/tools.py`, `integration/registry.py` | `test_session_isolation.py` |
| D4–D5 | `algo/dynamic_filtering.py`, ray_trainer DF patch, `configs/experiments/arms.yaml` | `test_dynamic_filtering.py`, `test_patch_contract.py` |
| D6–D7 | `algo/tau_gigpo.py`, `algo/verl_estimator.py`, `anchors/` | `test_tau_gigpo.py`, `test_anchors.py` |
| D8–D11 | `data/parquet_builder.py`, `experiment/manifest.py`, `cli/prepare_experiment.py`, `integration/trainer_telemetry.py`, `scripts/train/rl/run_base.sh` | `test_parquet_builder.py`, `test_experiment_manifest.py`, `test_trainer_telemetry.py`, `test_configs.py` |
| D12 | `anchors/encoder.py` three modes, `configs/experiments/arms.yaml` ablations | `test_anchors.py`, `test_configs.py` |
| D13 | `experiment/winner_lock.py`, `experiment/service_attestation.py`, `data/official.py`, `cli/evaluate.py` | `test_winner_guard.py`, `test_service_attestation.py`, `test_source_isolation.py` |
| D14 | `experiment/metrics.py`, `experiment/telemetry.py`, `cli/report.py` | `test_metrics.py`, `test_experiment_manifest.py` |

## veRL local patches

Eight veRL files contain nine logical patch areas. Each local change is marked
`Tau3-GRPO local patch` and verified by `tests/test_patch_contract.py` and
`python -m tau3_grpo.integrations.verify_patches`:

1. `verl/experimental/agent_loop/tool_agent_loop.py` — `tau3_anchor_hook` records an
   anchor id and token span per assistant generation and `None` at tool and user
   observation segments; `run()` publishes `anchor_ids` / `anchor_spans` through
   `extra_fields`, publishes the official terminal reward, and always releases the
   private session; `_postprocess` turns those fields into `non_tensor_batch` keys.
2. `verl/trainer/ppo/ray_trainer.py::compute_advantage` — registers `tau_gigpo`
   inside the Ray TaskRunner process and passes `non_tensor_batch` to it, alongside
   the existing GDPO precedent.
3. `verl/trainer/ppo/ray_trainer.py` — `tau3_dynamic_filter` applies the Dynamic
   Filtering response mask before `compute_advantage`; a no-op unless
   `algorithm.dynamic_filter.enable` is set. The same file performs zero-loss
   post-rollout padding from 128 to 144 for the agreed six policy ranks, excludes
   padding from DF/telemetry, and removes it before rollout logs and data metrics.
4. `verl/trainer/config/algorithm.py::AlgoConfig` — optional `dynamic_filter` and
   `gigpo` blocks, both defaulting to `None`.
5. `verl/utils/config.py` — validates the real rollout batch separately from the
   post-rollout padded policy batch used by six policy ranks.
6. `verl/workers/actor/dp_actor.py` — keeps FSDP ranks aligned when a padded
   micro-batch has an empty response mask by producing a graph-connected zero
   loss instead of skipping the collective.
7. `setup.py` — adds the isolated `qwen35` dependency extra and the directly
   imported `cachetools` dependency. Allows NumPy 2 for the new vLLM/OpenCV
   stack while keeping NumPy <2 in the legacy `vllm` extra.
8. `verl/models/transformers/monkey_patch.py` — requires native padded forward
   for dense Qwen3.5; rejects generic packing, sequence parallelism, fused
   kernels, prefix grouping and tiled MLP until separately implemented.
9. `verl/utils/tokenizer.py` — skips the vision processor only for dense
   Qwen3.5 when the launcher explicitly sets `TAU3_GRPO_TEXT_ONLY=1`.

The GAE and stock GRPO branches of `compute_advantage` are untouched, so E0
runs veRL's unmodified advantage path. E2 and E3 deliberately select the custom
`tau_gigpo` estimator, which lives in this package and registers by string
name, so no veRL enum changed.

## Frozen algorithm constants

- Tau-GiGPO: `A = A_episode + omega*A_step`, `omega=1`, `Fnorm=1`, `gamma=0.95`,
  `A_step = 0` exactly when no anchor group reaches size 2.
- Dynamic Filtering: fixed rollout, uid groups of 8, response mask only, no state
  carried across updates. `fixed_informative` exists but its cap is off by default.

## A800 verification status

### Environment and deterministic checks

- The complete remote suite passes with the live Tau3/veRL/Torch stack: 356 tests
  passed. CUDA behavior is validated only in the pinned A800 environment.
- The merged four-shard checkpoint passes CUDA FlashAttention 2 generation and
  vLLM OpenAI-compatible serving at 16K context.

### SFT and single-A800 diagnostics

- Qwen2.5-7B SFT completed with micro-batch 1, accumulation 8 and effective
  batch 8: 30/30 updates in 1060.467 seconds, train loss 0.599849 and final
  validation loss 0.582722. Micro-batch 2 OOMed on the first backward and is no
  longer the frozen default.
- One live AReaL Airline selection trajectory completed through the official
  orchestrator/verifier without infrastructure failure. Reward was 0 because
  the policy looped without tool calls; that is a policy-quality result.
- With a colocated simulator, the merged 7B policy reached
  `actor_optimizer.step()` and then OOMed while lazily allocating Adam state.
  This is a one-card topology limit, not a change to the formal six-policy-GPU
  plan.
- The diagnostic Qwen2.5-1.5B fallback completed one real rollout,
  old/reference log-probabilities, GRPO advantage, optimizer update,
  actor-to-vLLM synchronization and `training/global_step=1` in 27.712 seconds.
  Group size 1 and the three-turn cap produced zero reward/advantage/gradient,
  so this validates connectivity rather than learning signal.
- The successful single-card smoke log SHA-256 is
  `c1b3a19ba373ff2ae88fa0b053d0897f8b929e2520ab07d465c8551c3d7a9bfb`; the
  rollout SHA-256 is
  `38d2e7a908683bba293e2b4dc835f99aea7e7e0fe12f5b8b347d0d789ca6e419`.

### Eight-A800 distributed validation

The frozen distributed topology reserves GPUs 0–5 for policy training/rollout
and GPUs 6–7 for the independent user simulator. All four arms completed one
real optimizer update:

| Arm | Validation rollouts | Result |
|---|---:|---|
| E0 | 6 | vanilla GRPO, `global_step=1` |
| E1 | 12 | Dynamic Filtering + GRPO, `global_step=1` |
| E2 | 6 | Tau-GiGPO, average step-level group size 1.0, `global_step=1` |
| E3 | 12 | Dynamic Filtering + Tau-GiGPO, average step-level group size 2.0, `global_step=1` |

The runs jointly cover rollout JSON generation, isolated Tau3 environments,
tool calls, the external user simulator, official terminal verification,
old/reference log-probabilities, GRPO/GiGPO advantages, optimizer execution and
actor-to-vLLM weight synchronization.

Policy GPUs averaged roughly 38%–43% utilization during active phases, reached
88%–100% peak utilization and used about 13.4–14.1 GiB peak memory each. CPU
utilization averaged roughly 7%–14% and peaked near 27%. The two simulator GPUs
served 24/24 concurrent requests successfully, reached 83%/70% peak utilization
and used about 25.2 GiB each.

### Remaining experiment work

Infrastructure validation is complete. Still outstanding are formal 7B
40/60-update E0–E3 training, multiple seeds, parameter adjustment based on
bad-case evidence, internal selection, anchor ablations and the final 50-task
Tau3 evaluation. No algorithmic improvement should be claimed from the
one-update validation runs.

## Qwen2.5 / Qwen3.5 series support (2026-09-08)

The project supports the Qwen2.5 text Instruct and dense Qwen3.5 model families,
with configurable policy and simulator model sizes and separate training
environments. The Qwen2.5 A800 results above describe the legacy runtime. The new Qwen3.5
profile is implemented and locally verified, but has **not** passed A800
training or vLLM acceptance. Its pins are separate and
`QWEN35_GPU_VALIDATED=false` remains in `env_info/versions.lock`.

Project-owned model loading, native assistant masks, language-only SFT LoRA,
checkpoint merging, parser selection and dedicated launchers are documented
in [qwen35.md](qwen35.md). The two-card engineering profile starts with
Qwen3.5-0.8B policy on GPU 0 and Qwen3.5-4B simulator on GPU 1. Dense 9B uses
the same model-family path, a separate namespace and a larger weight-transfer
bucket; its GPU topology still requires memory validation.

The new CPU stack passes all 378 project tests, including real tiny-model
Qwen3.5 forward/backward, LoRA update/merge/reload, actual veRL actor log
probabilities and both native tool parsers. A separate legacy CPU subset
passes 327 tests with eight veRL-dependent skips. Linux dependency resolution,
patch contracts, lint and shell syntax also pass. The original frozen 45/5
SFT examples retain all tool schemas and complete dialogues; the Qwen3.5
profile increases its length cap to 24,576 to accommodate the measured
16,702-token maximum. See [qwen35_local_validation.md](qwen35_local_validation.md)
for reproducible commands, hashes and the remaining GPU gates.

The subsequent AutoDL no-GPU installation passes 375 project tests with three
tokenizer-asset skips. Qwen3.5-0.8B and 4B full checkpoints are downloaded and
their weight SHA-256s match the pinned official manifests. The Linux environment,
frozen SFT audit and launcher checks pass; CUDA/vLLM kernels still require the
two-GPU host mode. See [a800_qwen35_setup.md](a800_qwen35_setup.md) for paths,
versions and validation details.

The AutoDL deployment now keeps the independent Python interpreter and venv
alongside current `code` on file storage. Pinned model master copies persist
there; working models, runs, caches and temporary files use the node's data disk.
The former `code_agent` tree and all old experiment artifacts were deleted at
the user's request. This storage cleanup does not remove the source-level
Qwen2.5 compatibility path; its legacy environment can be recreated separately.
