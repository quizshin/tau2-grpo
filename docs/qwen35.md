# Qwen2.5 / Qwen3.5 series model support

The standard veRL project in **`code/` supports Qwen2.5 and Qwen3.5 series
models** for text-only Airline policy training and user simulation. Choose
policy and simulator sizes independently. Qwen2.5 support uses the text
Instruct model family; Qwen3.5 support currently covers dense checkpoints.

| Model family | Training environment | Entry points |
|---|---|---|
| Qwen2.5 series | `setup.sh a800`, legacy Torch/Transformers/vLLM pins | Existing `scripts/` launchers; see the [project README](../README.md) |
| Dense Qwen3.5 series (0.8B / 2B / 4B / 9B) | `setup.sh a800-qwen35`, separate new pins | `scripts/qwen35/`; detailed example below |

The current two-A800 example uses `Qwen/Qwen3.5-0.8B` for policy, reference
and rollout on GPU 0, and a fixed `Qwen/Qwen3.5-4B` simulator on GPU 1.
These are example defaults, not limits on model-family selection. Available
memory, parallelism and runtime validation must be checked for each size.
`code_agent/` and `code_pytrio/` are outside this adaptation.

The original Qwen2.5 configuration, legacy `vllm` extra and `setup.sh a800`
remain available. Use separate environments; do not install both `sft` and
`qwen35` extras into one environment (their Transformers constraints differ).

## Qwen3.5 model-family adaptation

- Load the complete `Qwen3_5ForConditionalGeneration` checkpoint via
  `AutoModelForImageTextToText`, freezing the unused visual tower.
- Select full language-parameter SFT or LoRA with `QWEN35_SFT_METHOD`.
  Full SFT trains language layers, embeddings and the LM head, using FP32
  master parameters/Adam states and BF16 autocast. The unused visual tower is frozen.
  LoRA targets language-layer linear modules, including Gated DeltaNet;
  rank 16, alpha 32, dropout .05; visual modules, embeddings and LM head are excluded.
  Both methods retain effective batch 8, five epochs / 30 updates.
- Construct assistant masks from one full native chat-template render. Check
  the exact token sequence against the unmodified template on every dialogue.
  Exclude observations and empty thinking scaffolding; supervise assistant
  content, native XML calls and EOS. Qwen2.5 retains its original mask path.
- Disable thinking for Qwen3.5 SFT, rollout and served evaluation. Use the
  existing `qwen3_coder` parser during rollout and in the vLLM evaluation API.
- Use native padded SDPA forward, SP=1, no fused kernels, no prefix grouping,
  no packed-sequence patches. Generic packing is explicitly rejected because
  Gated DeltaNet recurrent state needs independent sequence boundaries.
- Export either SFT method to a BF16 HF checkpoint. RL defaults to **full
  language-parameter training**, with FP32 actor parameters and BF16 FSDP
  computation. Explicit LoRA overrides use veRL's native adapter training and
  rollout synchronization. Reference is fixed to the SFT starting checkpoint.
  The Qwen3.5 profile preloads rollout safetensors. Keep this setting for LoRA:
  vLLM wraps GDN convolutions even when they are not adapter targets, and the
  current base-sync loader skips their unwrapped keys. Preloading initializes
  them correctly; LoRA sleep level 1 retains the frozen base weights.
  With frozen tied Qwen3.5 embeddings, the LoRA worker uses FSDP flattened
  parameters: Torch 2.11's `use_orig_params=True` otherwise fails on the second
  accumulated microbatch. The LoRA wrap policy keeps adapters separate from
  frozen base parameters; embedding tying and frozen weights are preserved.
- Engineering RL defaults are 2 groups × 4 trajectories, three updates,
  checkpoint every update. Formal 16 × 8 / 40–60 updates must be requested
  explicitly, with sufficient memory and runtime.

No algorithm/reward/data-split changes were made. E0–E3 still use standard
veRL's fixed-rollout filtering and Tau-GiGPO, not verl-agent's re-sampling path.
The original v2-2 SFT contract is LoRA. Full SFT is an explicitly requested
comparison variant, with its own learning rate and artifact namespace.

## Qwen3.5 environment

Candidate pins: Python 3.12, Torch 2.11.0, Transformers 5.5.1, vLLM 0.20.0,
PEFT .18.1, Accelerate 1.12.0, NumPy 2.2.6. Linux x86_64 dependency resolution
is checked separately from local macOS CPU execution. vLLM 0.20 needs OpenCV
4.13/NumPy 2; the legacy `vllm` extra retains NumPy <2.

**The default PyPI GPU dependency set uses CUDA 13.0.** The remote host needs
a CUDA-13-compatible NVIDIA driver (R580 or newer), not merely an old CUDA 12.8
image. `nvidia-smi` and a real CUDA allocation must be checked on arrival.
Do not upgrade an existing experiment environment in place. Alternative CUDA
wheels require a separately verified Torch/vLLM combination.

From the remote `code/` root:

```bash
bash setup.sh a800-qwen35
source .venv-a800-qwen35/bin/activate
python -m pip check

# Pinned 0.8B policy + 4B simulator weights, plus the original data-split tokenizer.
python -m tau3_grpo.models.download_qwen35 --size 0.8B --include-user --include-legacy-tokenizer
```

Models go under `models/`, which is ignored. For laptop
tokenizer checks use `--tokenizer-only`; those downloads cannot train a model.
Override `QWEN35_MODEL_PATH` only to use another complete local base snapshot.

## Frozen data and SFT

Reuse the existing 45 train / 5 validation JSONL files when present. If they
are absent, reproduce the original split with the original tokenizer first:

```bash
DOWNLOAD=1 REQUIRE_DB=1 \
SFT_MODEL_NAME_OR_PATH="$PWD/models/Qwen2.5-7B-Instruct" \
bash scripts/data/prepare.sh

python -m tau3_grpo.models.check_qwen35 \
  --model models/Qwen3.5-0.8B --audit-sft

bash scripts/train/sft/run_qwen35.sh
bash scripts/train/sft/merge_qwen35.sh
```

The Qwen2.5 assets here are tokenizer/config files only, used to reproduce
the frozen data selection. The running simulator is Qwen3.5-4B.

Do not reselect the 45/5 dialogues using a new tokenizer. The audit fails on
overlength examples; it never truncates, drops or resamples them. The Qwen3.5
SFT cap is **24,576**, instead of the old 16,384. With all original tool schema
fields retained, the frozen 0.8B train maximum is 16,473 and validation maximum
is 16,702. This cap change preserves the original complete dialogues and is
recorded as a model-profile difference, not hidden as a data change.

`check_qwen35 --gpu` verifies versions, CUDA visibility and worker imports.
It does not load policy weights or establish GPU training correctness.

## Short GPU validation

### Compare full SFT and LoRA

See [the running AutoDL comparison](sft_comparison_20260908.md) for exact settings,
GPU fixes and artifact locations.

The Qwen3.5 default SFT profile now selects `full`; `lora` remains available.
The two comparison configurations use identical data, seed 42, effective
batch 8, and 30 optimizer updates. Full SFT uses learning rate `2e-5`, while
LoRA uses `1e-4`. This compares two method-specific configurations, rather
than isolating trainable-parameter count with one shared learning rate.
Each evaluates the baseline, evaluates every epoch and exports its best
validation-loss checkpoint. `train_summary.json` and `trainer_state.json`
record validation loss, update count, trainable parameters, runtime and peak CUDA memory.

The pinned Transformers 5.5.1 / Accelerate 1.12.0 combination otherwise
divides accumulated gradients twice. The Qwen3.5 SFT Trainer leaves window
normalization to Transformers and sets Accelerate's divisor to one. Direct
batch and accumulated updates are tested for both methods, including a partial
last window. It also projects only supervised next-token positions through the
LM head while retaining the full dialogue in the transformer; native loss and
parameter gradients match in tied/untied full and LoRA tests. Both GPU runs use
`PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
Full SFT checkpoints and both methods' BF16 exports use native HF weight names
(`save_original_format=False`), because the pinned Transformers reverse mapping
is incompatible with Trainer's direct best-checkpoint restore. The final
validation loss must match the tracked best checkpoint before it is saved.

```bash
# Run concurrently in separate terminals, with no simulator occupying GPU 1.
# SwanLab is enabled by default; log in first, or pass --report-to none.
SFT_GPU=0 QWEN35_SFT_METHOD=full \
QWEN35_RUN_ROOT=/root/autodl-tmp/tau3/runs/sft_compare_20260908/full \
bash scripts/train/sft/run_qwen35.sh

SFT_GPU=1 QWEN35_SFT_METHOD=lora \
QWEN35_RUN_ROOT=/root/autodl-tmp/tau3/runs/sft_compare_20260908/lora \
bash scripts/train/sft/run_qwen35.sh
```

Pass the same method and run root to `scripts/train/sft/merge_qwen35.sh` for export.
For full SFT this converts the saved full checkpoint to BF16; for LoRA it
merges the adapter into the base. After both finish, use the same frozen
Qwen3.5-4B simulator and selection tasks to compare downstream performance.
The official 50-task final set must not select the SFT method.
SFT/RL now record metrics, selected dialogues and source artifacts through
[SwanLab](swanlab.md), including a command to upload completed SFT histories.

Terminal 1, same environment:

```bash
# Defaults to the pinned local Qwen3.5-4B snapshot on GPU 1, thinking disabled.
bash scripts/serve/simulator_qwen35.sh
```

Terminal 2:

```bash
# Prints the actual command; no data writes, weights or GPUs are allocated.
TAU3_DRY_RUN=1 bash scripts/train/rl/run_qwen35.sh e0 42

bash scripts/train/rl/run_qwen35.sh e0 42

# Explicit LoRA RL on the merged SFT checkpoint, in a separate result namespace.
# Include the Gated DeltaNet projections; exclude vision, embedding and LM head.
RESULTS_DIR="$PWD/results/qwen35_0.8b/lora_e0_seed42" \
LR=1e-6 bash scripts/train/rl/run_qwen35.sh e0 42 \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,gate_proj,up_proj,down_proj]' \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.load_format=safetensors

# Resume optimizer + actor and continue the same deterministic schedule.
TOTAL_UPDATES=4 bash scripts/train/rl/run_qwen35.sh e0 42 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$PWD/results/qwen35_0.8b/e0_seed42/global_step_3"

# Separate arms start from the same SFT checkpoint, not the E0 checkpoint.
bash scripts/train/rl/run_qwen35.sh e1 42
bash scripts/train/rl/run_qwen35.sh e2 42
bash scripts/train/rl/run_qwen35.sh e3 42
```

The per-run simulator config honors `TAU3_USER_MODEL`,
`TAU3_USER_SERVED_MODEL_NAME`, `TAU3_USER_BASE_URL` and the turn limits, leaving
shared YAML unchanged. The shared Qwen3.5 profile defaults to the local 4B
snapshot and API name `Qwen/Qwen3.5-4B` in serving, training and evaluation.
Both the simulator server and training requests disable thinking. If overriding
the simulator, apply the same model/name variables in every terminal.
Never start a second simulator on policy GPU 0.

Acceptance needs real trajectories/tools/user simulation, official terminal
verification, finite old/reference log probabilities, finite actor gradients,
optimizer execution, successful actor→vLLM synchronization, checkpoint merge,
and a successful resumed update. Record reward distributions separately:
all-identical reward groups give zero GRPO task-reward advantage. E1/E3 can
legitimately skip all-filtered updates; that does not demonstrate learning.

For a short TP=1 integration diagnosis, `VERL_QWEN35_WEIGHT_AUDIT_DIR` enables
worker-side comparison of GDN convolution tensors against the merged base
checkpoint, and hashes the actual GPU LoRA buffers after each complete sync.
It writes local JSON reports and fails if convolution weights differ. This is
opt-in because copying buffers to CPU adds overhead. `TAU3_GRPO_DEBUG_BATCH_DIR`
can also save the real pre-update batch for failure diagnosis. Such snapshots
must not replace fresh on-policy rollouts in training.

## Merge, selection and final

```bash
bash scripts/train/rl/merge_checkpoint.sh \
  results/qwen35_0.8b/e0_seed42/global_step_3/actor \
  results/qwen35_0.8b/e0_seed42/global_step_3/merged_hf

# Stop training before reusing GPU 0 for this server; keep the simulator running.
bash scripts/serve/policy_qwen35.sh \
  "$PWD/results/qwen35_0.8b/e0_seed42/global_step_3/merged_hf"

# Another terminal, same environment.
bash scripts/eval/run_qwen35.sh selection \
  "$PWD/results/qwen35_0.8b/e0_seed42/global_step_3/merged_hf"
```

These example checkpoints are engineering artifacts. After the actual formal
selection, use `freeze_winner --output-dir` pointing to
`results/qwen35_0.8b`, with the real selected checkpoint,
metric and measured score. The Qwen3.5 `run_eval.sh tau3-final` wrapper keeps
the existing winner-lock and service/PID/content attestation checks. Official
50-task results must not be used to choose or tune the policy.

## Moving to 9B

`QWEN35_SIZE=9B` changes the profile's model path and output namespace; download
the pinned 9B checkpoint first, then rerun SFT and RL from the 9B base. Do not
reuse 0.8B adapters, optimizer state or checkpoints. 0.8B/2B/4B tie input/output
embeddings; 9B does not, so both cases are covered by tiny-model local tests.
The profile also increases the weight-transfer bucket from 2,560 to 4,096 MiB
for 9B: each FP32 embedding/LM-head tensor is 3,880 MiB, and the sender does
not split individual tensors or cast them before transfer.

The one-policy-GPU default is not a 9B memory guarantee. Full-parameter 9B RL
may require more policy GPUs/offload and a separately hosted simulator. LoRA
RL would change the training method and is not silently substituted here.

## Local verification

Use a separate Python 3.12 environment with the new CPU Torch/Transformers/PEFT
versions, the project/benchmark/veRL packages and their dependencies. The full
tests include real tiny Qwen3.5 forward/backward and LoRA merge/reload for both
embedding modes. No full policy weights are downloaded for these tests.

```bash
python -m tau3_grpo.models.download_qwen35 --tokenizer-only --include-user --include-legacy-tokenizer
python -m tau3_grpo.models.download_qwen35 --tokenizer-only --size 9B
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q tests
python -m tau3_grpo.integrations.verify_patches --check-registration
```

Local regression tests pass (395 tests as of 2026-09-08). Both 0.8B SFT methods
have completed on A800, their best checkpoints and BF16 exports have been
validated, and standalone vLLM policy/simulator serving plus a full task smoke
test have passed. The injected Airline DB is copied for each environment to
isolate predicted and gold replay; selection trajectories are strictly rescored
with this fix before comparison. The selection comparison is recorded in
[the SFT comparison](sft_comparison_20260908.md).
FSDP on A800, vLLM sleep/wake/weight transfer, and the complete RL sequence
remain remote acceptance gates. See `qwen35_local_validation.md` for local evidence.
The subsequent Linux installation and complete model downloads are recorded in
[the AutoDL no-GPU setup record](a800_qwen35_setup.md).
