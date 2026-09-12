# Paratera RTX 5090 deployment

This profile targets the standard `code/` veRL implementation. All environment
installation and runtime checks run on the remote Linux host, never on the Mac.

## Storage and access

The shared mount was expanded to **150 GiB** on 2026-09-10. The user still limits
total shared filesystem use to **100 decimal GB**, and system filesystem use to
**25 decimal GB**. `run_guarded.py` measures both entire filesystems and terminates
only its own job at 95 GB / 23 GB, preserving headroom and partial outputs. This
is a monitored stop threshold, not a filesystem quota.

| Purpose | Remote path |
|---|---|
| Source | `/root/shared-nvme/tau3/code` |
| Models | `/root/shared-nvme/tau3/models` |
| Raw and prepared datasets | `/root/shared-nvme/tau3/data` |
| Adapters, merged models, checkpoints | `/root/shared-nvme/tau3/artifacts` |
| Run logs, rollouts, audits | `/root/shared-nvme/tau3/runs` |
| Environments | `/root/shared-nvme/tau3/envs` |
| Installation and validation records | `/root/shared-nvme/tau3/bootstrap` |
| Download and compilation caches | `/root/tau3-cache` on system disk |

SSH uses host `ssh.bj8.bz1.paratera.com`, port `2233`, and username
`root@ackcs-00gjhmys` (pass with `ssh -l`). The existing
`/Users/apple/.ssh/id_ed25519_a800.pub` was appended to the container's
`/root/.ssh/authorized_keys`; the private key remains on the Mac. Key login was
verified against the container's sshd through an authenticated localhost tunnel.
The public SSHPiper gateway still requests password authentication after partial
public-key success. Do not describe the public gateway as verified key-only access.

## Runtime

Observed host: Ubuntu 24.04, Python 3.12.3, driver 580.82.07, two RTX 5090 cards
with 32607 MiB each. The preinstalled CUDA compiler is 12.8.93. The fresh isolated
runtime targets Torch 2.11.0 / CUDA 13.0, vLLM 0.20.0, Transformers 5.5.1,
PEFT 0.18.1 and Accelerate 1.12.0. The simulator uses a separate Transformers
5.8.0 overlay sharing the large Torch/vLLM dependencies.

The Paratera installer also constrains datasets 4.8.5, fsspec 2026.2.0 and
PyArrow 25.0.1. An unconstrained resolution chose datasets 1.1.1, which calls
the removed PyArrow `PyExtensionType` API. The constrained environment passes
Trainer imports and loads the prepared 200/60-row train/selection parquet files.

```bash
source /root/shared-nvme/tau3/activate.sh
cd "$TAU3_ROOT/code"
python env_info/paratera/run_guarded.py -- bash env_info/paratera/install_qwen35.sh
python env_info/paratera/run_guarded.py -- bash env_info/paratera/validate_environment.sh
```

Installer downloads use a persistent system-disk uv cache with copy mode, so
installed environments remain independent of that cache. `qwen35-installed.txt`
and `qwen38-sim-installed.txt` record the actual resolved packages after success.
The two large Torch/vLLM wheels needed a byte-stream relay from official PyPI
through the Mac into `/root/tau3-cache/wheels/relay`; no wheel was written or
executed locally. Official hashes are saved in `bootstrap/tau3-wheels-manifest.json`.
Set `TAU3_WHEELHOUSE` to that remote directory to reuse those verified archives.
No replacement of the new NCCL library with the old platform Torch 2.7 library
is part of this profile.

The [platform academic proxy](https://ai.paratera.com/document/container/faq/env/network_turbo)
was configured in a remote mode-600 file outside source control. Its TCP endpoint
timed out from this instance during deployment, so it is not enabled globally.
The download fallback uses Aliyun PyPI and ModelScope. Every model file is checked
against pinned Hugging Face hashes; differing mirror metadata is obtained from
the pinned original revision and verified before transfer.
The simulator's third weight shard also used a Hugging Face byte-stream relay
through the Mac directly to remote storage, with no local weight file. Both
complete model directories have verified `tau3_source_revision.json` receipts.

Verified remotely on 2026-09-10: dependency checks, CUDA 13 BF16 forward/backward,
two-rank NCCL 2.28.9 all-reduce, 40 focused regression tests, and two-rank FSDP
LoRA gradient accumulation with empty local masks. See `bootstrap/validation-final.log`
and `bootstrap/fsdp-projection.log`. These checks alone do not establish full-model
memory fit or end-to-end rollout success. The 45/5 SFT token counts also match the
historical audit exactly; the longest dialogue is 16,702 tokens, without truncation.

## Models and validation stages

Start with **Qwen3.5-4B LoRA** and **Qwen3.8-27B-AWQ-INT4**. The manifests pin:

- Policy: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Simulator: `cyankiwi/Qwen3.8-27B-AWQ-INT4`, revision `63768c10df38c0395e12ef49edac1bd539eaeeea`.

`configs/hardware/2x5090.yaml` assigns GPU 0 to policy and GPU 1 to simulator.
The simulator passed with two concurrent sequences, 16K context, eager mode and
85% GPU memory budget, occupying about 27.2 GiB. These are the tested engineering
settings; higher concurrency and longer simulator context remain unverified.

```bash
python env_info/paratera/run_guarded.py -- python -m tau3_grpo.launch simulator \
  --config configs/simulator/qwen38_27b_paratera_5090.yaml

CUDA_VISIBLE_DEVICES=0 python env_info/paratera/run_guarded.py -- \
  python env_info/paratera/sft_memory_probe.py

python env_info/paratera/run_guarded.py -- python -m tau3_grpo.launch rl \
  --config configs/train/rl/qwen35_4b_lora_paratera_smoke.yaml --experiment e0
```

The RL smoke profile uses the **base model**, one update with 2 groups × 4
rollouts, and separate outputs. It is not an SFT-initialized E0 result. The SFT
memory probe discards its diagnostic updates. Formal SFT retains the frozen
45/5 complete dialogues, effective batch 8 and 30 optimizer updates in
`configs/train/sft/qwen35_4b_lora_paratera_5090.yaml`.

Measured full-model diagnostics on the two-card host:

| Check | Result |
|---|---|
| Simulator TP 1 | Full 27B INT4 model loaded; two concurrent API requests passed; warm requests took 0.34/0.40 s |
| SFT worst complete dialogue | 16,702 tokens / 4,517 labels; finite loss and actual LoRA update; 23.83 GiB allocated, 27.54 GiB reserved peak |
| RL actor diagnostic | Same dialogue padded to 24,576 tokens; old/ref scoring, synthetic-advantage loss and actual LoRA update passed |
| RL head chunking | Allocated peak remained 28.52 GiB; reserved peak fell from 30.22 to 29.38 GiB; loss unchanged, gradient norm 0.65065 vs 0.65075 |

Evidence is in `runs/paratera-preflight/{simulator-api,sft-memory,rl-memory}.json`.
`rl-memory-unchunked.json` preserves the earlier baseline. The SFT criteria
selected the same worst dialogue, so the probe performed one diagnostic update.
These updates are discarded; the RL memory probe uses synthetic advantages and
does not establish actual task reward or end-to-end GRPO success.

## Completed two-card RL smoke

`bootstrap/rl-smoke.exit` is **0**. The base-4B E0 profile completed one update,
checkpoint saving and the subsequent vLLM base/adapter weight synchronization.
Evidence: `runs/paratera_4b_lora_base_smoke/smoke-summary.json`,
`batch-summary.json`, `weight-audits/`, and `e0_seed42/rollouts/1.jsonl`.
The checkpoint is `artifacts/rl_4b_lora_base_smoke/global_step_1/`.

- Eight real rollouts across two tasks, all ended with `user_stop`; all scored
  1.0 on both DB and COMMUNICATE checks. No dummy rows or truncated responses.
- Native input batch was 8 × 24,576 tokens; response lengths were 2,123–7,807
  including observations, with 1,023–5,233 policy tokens per row.
- Measured actor allocated/reserved peaks were 29.20 / 29.38 GiB. Sampled total
  GPU use approached 31.4 GiB, including the sleeping inference process. The
  two-card profile therefore has little GPU memory headroom.
- One step took 1,538 s (25.6 min), excluding startup: generation 885 s,
  old/reference scoring 48 / 45 s, actor update 523 s, saving 23 s, sync 14 s.
- All within-group rewards were identical, so advantages, policy/KL losses and
  gradient norm were exactly zero. GPU LoRA fingerprints remained unchanged,
  as expected. This is a successful infrastructure test, **not evidence of
  learning or improved policy quality**. The separate synthetic-advantage
  full-model probe verifies nonzero gradients and actual parameter changes.
- All four base/adapter audits matched the original convolution weights.
  Changed-weight synchronization after a nonzero real GRPO update is not yet
  established by this all-equal-reward batch.

After saving this checkpoint, total filesystem usage was about 50.2 decimal GB
shared and 1.17 GB system. At that stage, formal SFT and export were still pending.
Nonzero real-task GRPO learning and eight-card training remain unverified.
Test GPU services are stopped after validation; use the launch commands above
to start them again. The verified environments, models and outputs are retained.

## SFT export; RL deferred to the eight-card host

The user requested **SFT only**, with the adapter and merged model retained in
shared storage. The previous waiting RL continuation was terminated before it
started any simulator or RL process. `continue_sft_rl.py` now defaults to waiting
for the 30-step SFT job, checking its completion record, and exporting the selected
adapter. It requires explicit `--run-rl` to start subsequent RL; that option is
not used for this run. Stage logs and `sft-export-status.json` are in `bootstrap/`.
The detached continuation is protected by `run_guarded.py`. Training keeps the
original native implementation, complete 45/5 dialogues and 30 optimizer steps.

Shared outputs:

- `artifacts/sft4b_lora_5090/adapter/`: selected LoRA adapter, tokenizer,
  training summary, trainer state and retained epoch checkpoints.
- `artifacts/sft4b_lora_5090/sft_merged_seed42/`: merged BF16 Hugging Face model,
  tokenizer and SFT provenance, for later eight-card RL initialization.

The unused three-step E0 profile
`qwen35_4b_lora_paratera_sft_validation.yaml` remains available for future manual
validation. It uses four concurrent policy sequences at 40% vLLM memory and
2 groups × 4 rollouts per update. No post-SFT RL run is scheduled on this host.

### Completed SFT on 2026-09-10

SFT and export both exited **0**. The run completed 5 epochs / 30 optimizer
updates on all 45 training and 5 validation dialogues, effective batch 8, seed 42.
Validation loss went from **0.5583231** before training to **0.3553452** at the
selected `checkpoint-30`; mean training loss was 0.3316057. Training took
3,854 seconds. This is a held-out language-model loss result, not an RL reward
evaluation. No simulator or RL was started by the post-SFT continuation.

The merged BF16 model and tokenizer occupy about 9.10 decimal GB. The adapter
directory, including two retained resumable checkpoints, occupies about 0.97 GB.
Total shared/system filesystem usage after export is about **60.83 / 1.30 GB**.
Both GPUs are idle after training. Models and logs remain on shared storage;
only small summaries and hash manifests are copied back to the Mac.

`verify_sft_export.py` checks that the final adapter equals the selected
checkpoint, LoRA B tensors are nonzero, all merged tensors are finite, every
LoRA matrix matches base plus its update within BF16 precision, the complete
model reloads without missing/unexpected weights, and tokenizer/template agree.
It writes `export_verification.json`, a copy of `sft_config.yaml`, source/data
hashes in `sft_provenance.json`, and `SHA256SUMS` in both export directories.
After copying a model directory to another host, run `sha256sum -c SHA256SUMS`
inside that directory. Eight-card runtime validation is still required later.
The final audit exited **0**: all 496 adapter tensors matched `checkpoint-30`,
all 248 LoRA B matrices were nonzero, and all 248 merged update matrices matched
the independently computed BF16 result exactly (maximum absolute error 0).
All 723 merged tensors were finite; a complete reload reported zero missing,
unexpected or mismatched keys, and the tokenizer/chat template matched.

An optional **fla-core 0.5.2** overlay was tested on the idle GPU in
`envs/fla-core-0.5.2`; it is **not enabled in the training environment**. Although
small kernel/model comparisons passed and the warm full-model diagnostic took
18.66 s instead of about 77 s, the complete 24K LoRA-gradient comparison failed.
With identical initial adapter SHA256 fingerprints, gradient relative L2 error
was 0.573 and cosine similarity 0.836. It is therefore rejected for this run;
neither SFT nor RL is switched to FLA. See
`runs/fla-gradient-audit-seeded/comparison.json` and `fla_probe.py` for evidence.
The native SFT training entrypoint was retained unchanged; no checkpoint resume
or kernel switch was performed.

For 32 GB cards, `VERL_QWEN35_LOSS_ONLY_LOGITS=1` opts into projecting only response
columns used by the policy mask. Full padded inputs, positions and native Gated
DeltaNet/attention processing remain unchanged. Unused output columns are zero;
an empty filtered microbatch still executes a graph-connected zero-loss backward.
The flag rejects non-Qwen3.5, packing/fused/sequence-parallel and multimodal paths.
Numerical tests compare masked log probabilities, entropy and LoRA gradients
against the full native projection, including empty masks and tied embeddings.
`VERL_QWEN35_HEAD_CHUNK_SIZE=128` additionally checkpoints the frozen linear
head and token-statistic reductions in chunks inside the native model forward.
Set it to 0 to use the unchunked projection. The transformer input and model
parameters are unchanged. CPU/GPU numerical comparisons and a two-rank FSDP
accumulation/update probe pass with chunks of 2, including empty local masks.

`configs/hardware/8x5090_candidate.yaml` reserves six policy GPUs (rollout TP 2)
and two simulator GPUs (TP 2). It is a candidate only: the current host has two
cards. Six-rank FSDP, eight-card topology, 9B memory and full training throughput
require validation on the later host. Do not infer those results from A800 logs.
The corresponding simulator and RL `*_8x5090_candidate.yaml` profiles keep
16 × 8 real rollouts, a PPO mini-group count of 6 and padding divisor 48. Their
single update is a capacity check, not the formal 40/60-update experiment.
