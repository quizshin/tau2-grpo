# ARPO τ adaptation v1

Status: implementation in progress; no GPU authorization or GPU validation.

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
