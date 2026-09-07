# τ² → τ³ Tau-GiGPO long-horizon agent RL

Trains a Qwen2.5-7B tool agent on AReaL's 1,148 synthetic τ²-style Airline tasks
with veRL, picks a winner on a frozen 60-task internal split, then evaluates that
single frozen winner on the 50 official Airline tasks from τ³-bench v1.0.1.

Official τ³ tasks are never used for training or model selection. That is enforced
in code, not by convention: `data.official.assert_trainable_source` rejects the
τ³ source on every training path, and `experiment.winner_lock` refuses to run the
final set without a frozen lock naming the exact checkpoint.
The lock hashes checkpoint bytes, and the evaluation process also requires the
PID/content attestation written by the vLLM launch script.

## Layout

This project is one of four sibling directories under the `code/` root. Every
relative path in configs, scripts and tests resolves against that root
(`tau3_grpo.paths`).

```text
code/
├── tau3-grpo-longhorizon/   this project
├── env_info/                version locks + A800 notes
├── tau2-bench/              Sierra tau2-bench v1.0.1 (fc0055d)
└── verl/                    veRL v0.7.1 (bec9ef7) + minimal local patches
```

## Runtime composition

```text
AReaL JSONL + per-record FlightDB
        │
        ▼
schema → tau2 Task adapter ──► per-trajectory Environment + official verifier
        │                                  ▲
        ▼                                  │
veRL parquet ──► patched ToolAgentLoop ──► Airline BaseTool + BaseInteraction
        │                                        (real tau2 UserSimulator)
        ├── E0 vanilla GRPO
        ├── E1 fixed-rollout Dynamic Filtering
        ├── E2 Tau-GiGPO structured anchors
        └── E3 Dynamic Filtering + Tau-GiGPO
```

Each rollout owns its own `FlightDB` and `Environment`, so the eight rollouts of a
uid group cannot observe each other's writes.

## Data preparation (CPU)

```bash
python -m tau3_grpo.cli.prepare_data --seed 42 --download --require-db
python -m tau3_grpo.cli.build_parquet --seed 42 --split train
# Reconstruct 45 train + 5 validation complete SFT dialogues. This audits them
# against the 60-task selection split and official final intents, rejects
# tokenizer/tool-rendered sequences over 16K, and caps each near-duplicate
# semantic intent at one dialogue.
python -m tau3_grpo.cli.prepare_sft --seed 42 \
  --model-name-or-path /path/to/Qwen2.5-7B-Instruct
# Or prepare the frozen manifests, parquet and SFT data together:
DOWNLOAD=1 REQUIRE_DB=1 scripts/prepare_all.sh
```

The builder checks the real counts (1,982 total, 1,148 Airline) and writes exactly
200 train / 60 selection / 888 reserve. Ordering is a stable hash rank, not an RNG
shuffle, so the split is identical across Python and numpy versions. It never
backfills or resamples.

## Training (GPU only)

```bash
python -m tau3_grpo.cli.verify_patches --check-registration
# SFT uses one A800, Python 3.12 / Torch 2.8 / CUDA 12.8;
# 45 dialogues x 5 epochs,
# micro-batch 1 x accumulation 8, effective batch 8, 30 optimizer steps.
scripts/run_sft.sh
scripts/merge_sft.sh

# Start the user simulator before a smoke or formal training run. The frozen
# 6+2 topology uses the default 7B simulator on GPUs 6,7.
scripts/serve_user_simulator.sh

# Alternatively, one-A800 diagnostic terminal 1: a small simulator on GPU 0,
# advertised under the interaction config's frozen API model name.
TAU3_USER_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
TAU3_USER_SERVED_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
TAU3_USER_TP=1 TAU3_USER_CUDA_DEVICES=0 \
TAU3_USER_GPU_MEMORY_UTILIZATION=0.12 TAU3_USER_MAX_MODEL_LEN=8192 \
scripts/serve_user_simulator.sh

# One-A800 diagnostic, terminal 2: after confirming the merged 7B policy OOMs at
# micro-batch 1 during Adam-state allocation when it shares the card with the
# simulator, use a local 1.5B/3B snapshot to verify the complete optimizer path.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct \
scripts/smoke/run_one_trajectory.sh

# Every formal E0-E3 arm defaults to the merged SFT checkpoint above.
scripts/run_train.sh e3 42
```

## Validation status

The frozen 7B SFT run was validated on one A800. Micro-batch 2 OOMed on the
first backward; micro-batch 1 with accumulation 8 completed 30/30 updates in
1060.467 seconds (train loss 0.599849, final validation loss 0.582722). The
merged checkpoint loads with both Transformers/FlashAttention 2 and vLLM. A
one-task live Airline selection rollout completed official verification without
an infrastructure error; its reward was 0 because the policy entered a dialogue
loop without making the required tool calls, which is a model-quality result.

The complete one-A800 veRL smoke was also exercised. The merged 7B policy passed
trajectory generation, official terminal reward, old/reference log-probabilities
and GRPO advantage computation, then OOMed in `Adam.step()` while lazily creating
optimizer state with micro-batch 1 and a colocated 1.5B simulator. This satisfies
the documented small-model fallback condition for the diagnostic only. A 1.54B
policy then completed `global_step=1`, including the optimizer update and
actor-to-vLLM weight resynchronization, in 27.712 seconds. Its actor peak was
17.269 GiB allocated / 19.342 GiB reserved; the external simulator used 10.892
GiB. Reward, advantage and gradient were zero because the smoke intentionally
uses group size 1 and the trajectory hit the three-turn cap. Formal E0-E3 remains
the merged 7B SFT policy on six policy GPUs plus two simulator GPUs.

The full 8×A800 topology was then validated with six policy GPUs and two
user-simulator GPUs. E0–E3 each completed one real optimizer update:

| Arm | Validation rollouts | Additional path exercised |
|---|---:|---|
| E0 | 6 | vanilla GRPO |
| E1 | 12 | fixed-rollout Dynamic Filtering |
| E2 | 6 | Tau-GiGPO; average step-level group size 1.0 |
| E3 | 12 | Dynamic Filtering + Tau-GiGPO; average step-level group size 2.0 |

All four runs reached `training/global_step=1` and covered official verification,
old/reference log-probabilities, advantage computation, optimizer execution and
actor-to-vLLM synchronization. Across policy GPUs, active-phase utilization
averaged roughly 38%–43%, peak utilization reached 88%–100%, and peak memory was
about 13.4–14.1 GiB per GPU. Both simulator GPUs served all 24 concurrent
requests successfully.

These one-update runs validate the distributed training path only. Formal
7B 40/60-update experiments, multiple seeds, internal selection, ablations and
the final 50-task Tau3 evaluation are still outstanding; no benchmark
improvement is claimed from smoke or validation results.

`run_train.sh` first expands the frozen 200-task pool into an exact, non-shuffled
per-update schedule under the run's `results/` directory. It then calls
`python -m tau3_grpo.cli.train`; that wrapper registers the
custom estimator before entering veRL, and the Ray TaskRunner also performs an
idempotent local registration. vLLM is the project's only rollout backend. The
script verifies the local veRL patches
and regenerates live tool schemas before allocating GPUs. Every formal run also
writes full rollout records under `results/<arm>_seed<seed>/rollouts/` by
default. Set `ROLLOUT_DATA_DIR` only when an alternate artifact location is
required. These records are the task-level evidence used for bad-case analysis;
the whole `results/` tree is Git-ignored.

The 6-policy-GPU topology cannot directly split 128 rows. The trainer therefore
pads only post-rollout compute from 128 to 144 with 16 explicitly marked dummy
rows. Dummy response masks/rewards/anchors are zeroed and the rows are removed
before trajectory logging and scalar metrics. The real experiment remains
exactly 16 uid groups x 8 rollouts; no environment call is added or dropped.

## Frozen final evaluation

```bash
scripts/merge_checkpoint.sh \
  results/e3_seed42/global_step_40/actor \
  results/e3_seed42/global_step_40/merged_hf

# Freeze the exact merged bytes that vLLM will serve, after winner selection.
python -m tau3_grpo.cli.freeze_winner --experiment e3 \
  --checkpoint results/e3_seed42/global_step_40/merged_hf \
  --metric selection_solve_rate --score 0.42 --task-count 60 --seeds 42 43

scripts/serve_policy_eval.sh results/e3_seed42/global_step_40/merged_hf

TAU3_POLICY_BASE_URL=http://127.0.0.1:8000/v1 \
TAU3_USER_BASE_URL=http://127.0.0.1:8100/v1 \
scripts/run_eval.sh tau3-final results/e3_seed42/global_step_40/merged_hf
```

The final run is refused unless a valid lock exists, names and hashes that exact
checkpoint, the live vLLM PID attests the same checkpoint hash/model/URL, and the
task set has exactly 50 entries. The script then runs real tau2
orchestrators against the two served endpoints and stores full trajectories,
errors and a summary under `results/evaluation/`. Internal selection uses the
same runner but reconstructs each AReaL task with its record-specific FlightDB.

## Reporting

```bash
python -m tau3_grpo.cli.report \
  --arm e0 results/e0_seed42/telemetry.jsonl \
  --arm e3 results/e3_seed42/telemetry.jsonl \
  --baseline e0 --treatment e3 --per-task results/per_task.json
```

pass@k uses the unbiased combinatorial estimator; the bootstrap is paired over
tasks and seeded, so a report regenerates byte-identically.

## Frozen algorithm constants

- **Tau-GiGPO**: `A = A_episode + omega*A_step`, `omega=1`, `Fnorm=1`, `gamma=0.95`.
  `A_step` is exactly 0 when no anchor group reaches size 2, so the estimator
  degrades to episode-level GRPO instead of inventing step credit.
- **Dynamic Filtering**: fixed rollout, uid groups of 8, response mask only, no
  state carried across updates. Reports candidate/effective groups, `d_bar` and
  `1/(1-d_bar)`.

Two upstream semantics worth knowing before reading `d_bar`: the tau2 ALL reward is
the **product** of the components in the task's `reward_basis`, and a run that
terminates for any reason other than `agent_stop`/`user_stop` scores 0.0 *without
being evaluated*. Telemetry therefore keeps `failure_category` separate from the
reward so "finished and wrong" is distinguishable from "never finished".

## Tests

```bash
pytest -q                    # CPU: data, anchors, DF, Tau-GiGPO, guards, patches
pytest -q -m tau3            # + sibling tau2-bench install
pytest -q -m verl            # + sibling veRL install (A800)
```

Tests needing GPU, a served policy or an absent dependency skip explicitly rather
than passing vacuously. See `docs/implementation_status.md` for the milestone
mapping and what still requires the A800 host.
