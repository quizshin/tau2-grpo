# Implementation status and 14-day mapping

Day labels are progress metadata only; no temporary `day1/` code paths exist.

| Milestone | Permanent implementation | Verification |
|---|---|---|
| D1 | source locks, four-root layout, AReaL schema, τ²/τ³ runtime factory | install preflight + single session |
| D2 | deterministic 200/60 manifests, DB isolation, verifier bridge | unit/integration tests |
| D3 | simulator client, 8-way smoke runner, throughput recorder | smoke scripts |
| D4–D5 | E0/E1 configs and fixed-rollout Dynamic Filtering | advantage/mask tests |
| D6–D7 | E2/E3 Tau-GiGPO, structured/similarity anchors | recurrence/collision tests |
| D8–D11 | screening/extension/seed orchestration and checkpoint locks | manifest-driven launch scripts |
| D12 | structured-vs-DB-hash ablation | E2 ablation config |
| D13 | τ³ final and SGLang parity | frozen-winner guard |
| D14 | pass@k/bootstrap/telemetry report generation | deterministic report tests |

The repository contains implementations and entrypoints for every row. Actual
GPU results remain absent until the code is synchronized to the A800 host.

