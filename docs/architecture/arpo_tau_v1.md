# ARPO τ adaptation v1

Status: cpu_verified (2026-09-27); integrated into local publication checkout on 2026-09-28.
The old GRPO run subsequently completed and its server was shut down; this publication
does not deploy ARPO into that server checkout. Historical development boundaries below
retain the original run/source identity.
No GPU authorization or GPU validation.

## Sources and scope

- Paper: Agentic Reinforced Policy Optimization, arXiv:2507.19849v1,
  https://arxiv.org/html/2507.19849v1 (sections 3.1–3.2).
- Official repository: https://github.com/RUC-NLPIR/ARPO
- Reference revision: `4d19a746b631b93cb382bdaa20fdeff272ed7361`.
- Reference implementation: `ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py`.
- Reproduce adaptive branching and soft attribution in the existing τ airline
  environment, not the paper's search/math benchmark scores. Keep τ terminal
  reward, model, tokenizer and existing training lifecycle. Do not import AEPO.

## Explicit protocol: arpo_tau_v1

| Item | Reference | τ v1 decision |
| --- | --- | --- |
| Entropy | Paper full-vocabulary entropy; code sums first 20 token top-10 logprob contributions divided by log(V) | Explicit `topk_partial` proxy, first 20 tokens, top 10 (including sampled token if returned by vLLM); no renormalization; missing values fail closed |
| Branch probability | Code clamps U − weight × entropy_delta then compares with base probability | Bernoulli with clipped base + weight × delta; endpoints mathematically defined; deterministic per-group RNG |
| Branch timing | Paper monitors after tool feedback; published code measures generation before executing the next tool and forks after execution | Measure next generation after a completed tool batch; fork from its pre-generation state, retaining the source continuation; no tool side effects repeated |
| Baseline | Initial trajectory entropy; code does not copy baseline into new branch entries | Inherit root baseline across descendants |
| Budget | N initial roots, at most M final trajectories; fill unused slots with roots | M = configured group size; N = 4 of M = 8 by default, one added child per eligible decision; deterministic FIFO allocation within a group |
| Credit | Paper default soft (GRPO loss), hard optional | Soft only v1; retain shared prefix in each leaf; no deduplication or invented process rewards; hard is a later ablation |
| Loss | Token PPO/GRPO clipped loss with KL | Explicit sequence-mean/token-mean reduction, matching paper length weighting; preserve old-policy logprobs separately from KL reference |
| Environment | Tools in the public repo primarily search/code | Snapshot mutable database, simulator object/state, messages, counters and registry state at quiescent turn boundaries; new request ID and branch sampling seed |
| Evaluation | Independent full trajectories | Adaptive branching disabled during evaluation |

This is a documented adaptation, not byte-for-byte equivalence with the public
repository. No claim that hard and soft gradients are exactly equivalent under
clipping or differing trajectory lengths. Branching changes correlations and
compute usage even when leaf counts match GRPO.

## Integration and acceptance

Algorithms and protocol validation belong in `algorithms/arpo.py`; grouped
rollout, replay and batch adaptation in `integrations/verl/arpo*.py`. Existing
ToolAgentLoop handles all actual generation/tool/user transitions. Opt-in vendor
hooks only; existing grpo/tau_gigpo/mt_gtpo behavior must remain unchanged.

Record group/root/parent/node IDs, prefix token count/hash, policy step, entropy
measurement and branch decisions, exact tokens/masks/logprobs and terminal reward.
Reject partial task groups, inconsistent prompt identities, unsupported reward
filtering, missing entropy, incompatible resume and unknown settings. Keep task
groups on one worker and preserve trainer row order. Clean all child sessions on
success, exceptions and cancellation.

CPU acceptance: numerical entropy/probability/advantage fixtures; state and RNG
isolation; budget/fallback; shared-prefix identity; multi-tool completion;
termination/truncation; cancellation cleanup; configuration and estimator
coexistence; real pinned τ environment and mocked model transport. GPU sampling,
weight updates, save/load and performance remain separately gated.

## Deployment boundary

Base local/remote HEAD: `b9c981a992da0d238448b2ec3b885cf470f9dea9`.
At development start remote GRPO runner PID 184501 and teacher PID 184448 use the
canonical checkout. Development uses temporary Git worktree branch
`codex/arpo-tau-v1`. No deployment into active checkout. Capture current run/source
identities before deployment; the activity snapshot is not a permanent process
assertion. Existing unrelated uncommitted configuration/document edits are not
part of this work.


## Implementation notes

- Root and branch seeds derive from data seed, global step, original group ordinal,
  leaf ordinal and turn; random UUIDs and worker placement do not affect sampling.
  The external simulator service's internal RNG state cannot be snapshotted;
  inherited local simulator state plus explicit continuation seed is recorded.
- The pinned τ airline constructor copies its input DB. Snapshots use the live
  tool-owned DB, rebuild bound tool methods, and verify DB hash and policy.
- Groups run concurrently; within each group roots/branches run in deterministic
  FIFO order. GPU throughput is unmeasured and may need subsequent scheduling
  optimization without changing allocation semantics.
- Missing entropy is an error. A zero-token budget termination without a generation
  is accepted. A short generation uses its actual observed window length.
- Branch budgets count final leaves; unused slots become fresh roots. Branches
  inherit remaining turn/context limits. Every sibling gets a distinct session.
- Soft attribution deliberately retains all shared prefix copies and uses
  sequence-mean/token-mean loss. Existing GRPO/GiGPO/MT-GTPO defaults stay unchanged.
- Final Hydra is validated; resume checks bind ARPO settings, group size, seed,
  sampling and loss aggregation. Existing controller owns checkpoints/evaluation.


## Comparison boundaries

The current GRPO profiles retain their existing loss reduction and simulator seed
policy. A controlled ARPO-vs-GRPO experiment must explicitly match those choices
(or report them as additional factors), as well as initialization, reward,
task schedule and evaluation. Equal leaf counts do not imply equal compute.
No claim of superiority, token savings, or matched historical E0–E3 performance
is made by this CPU integration.


## CPU verification receipt (2026-09-27)

Runtime implementation: `533b55d340fb0bad2f49a6858c98800b09e78d24`.

- Local core: 502 passed.
- Local relevant integration first run: 262 passed, 9 failed. Two regressions
  (legacy minimal manager config and old error wording) were fixed; remaining
  failures were missing tokenizer asset link / subprocess Python PATH in the
  temporary test environment. Relevant final retest: 31 passed, no skips.
- Remote pinned qwen35 environment, CUDA hidden: first run 142 passed, 5 failed
  (old error wording and subprocess PATH). Final retest: 54 passed, no skips,
  240.25 seconds. Runs overlap; counts must not be added.
- Verified native τ tools and DB, real tokenizer, scripted policy/user transports,
  nested branch budgets, exact prefix/mask/logprob inheritance, cancellation,
  truncation, complete-group routing/original order, soft arithmetic and offline
  replay, real Hydra composition, three existing estimators, vendor inventory.
- vLLM generation method body was executed with a CPU engine double; this does
  not validate a live vLLM GPU engine or memory requirements.
- Pinned upstream vendor inventory rebuilt and verified. No new Ruff diagnostics
  in 16 changed Python files relative to b9c981a. Full repository lint still fails
  on existing unregistered debt; its baseline was not expanded.
- Receipts: local canonical `results/maintenance/arpo-tau-v1-20260927/` and
  remote `/root/autodl-fs/tau3-core/maintenance/arpo-tau-v1-20260927/`.
- GPU rollout/update/weight sync/checkpoint recovery and performance are pending
  separate authorization. Temporary source worktrees are removed after receipts
  and commit identities are verified; the feature branch preserves all code.
