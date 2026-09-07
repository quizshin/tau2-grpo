# τ² training and τ³ final boundary

`tau2-bench v1.0.1` (commit fc0055d) is the official τ³ code release despite
keeping `tau2` as its Python import and CLI name. Both data paths share that one
runtime; only the task source differs.

## Two sources, one runtime

| | training / internal selection | τ³ official final |
|---|---|---|
| source | `inclusionAI/AReaL-tau2-data` rev 86971dc | tau2-bench Airline `base` split |
| `DataSource` | `areal_tau2_airline` | `tau3_official_airline` |
| loader | `data.dataset` → `env.adapter.adapt_record` | `data.official.load_official_airline_tasks` |
| count | 1,148 Airline → 200 / 60 / 888 | exactly 50 |
| DB | per-record snapshot from `db_path` | Sierra's stock Airline DB |
| may reach an optimizer | yes | **never** |

AReaL is τ²-style synthetic data. Converting its schema into a tau2 `Task` does
not make it τ³ official, and nothing in this repository labels it as such.

## How the boundary is enforced

Not by convention or by separate config defaults alone:

- `data.official.assert_trainable_source` raises `SourceIsolationError` for the
  τ³ source. It is called by `Tau3AirlineInteraction.start_interaction` (so a τ³
  task cannot even begin a training rollout) and by
  `data.parquet_builder.build_rows` (so it cannot enter a training parquet).
- `assert_trainable_entries` additionally rejects any entry whose split is
  `tau3_final`.
- `experiment.winner_lock.assert_final_run_allowed` refuses the final run unless a
  valid, unedited lock exists, it names and content-hashes the exact merged
  checkpoint being evaluated, and the task count is exactly 50.
- `experiment.service_attestation` binds the vLLM exec PID, served model name,
  endpoint URL and checkpoint content hash. A stale process, renamed endpoint or
  changed checkpoint is rejected before real evaluation rollouts begin.

The practical consequence: there is no code path from a τ³ result back to a model
choice, because running τ³ at all requires a lock that already fixed the winner.

## Verifier

`env.verifier` builds an official `SimulationRun` from the session's recorded
messages and calls `evaluate_simulation(..., EvaluationType.ALL)`. Scoring is
entirely Sierra's; we add only the failure categorisation that upstream's single
0.0 reward cannot express.

Two upstream behaviours shape the surrounding code:

1. The ALL reward is the **product** of the components in the task's
   `reward_basis`, so one zero component zeroes the trajectory.
2. A run whose `termination_reason` is not `agent_stop` or `user_stop` returns
   0.0 **without being evaluated**.

`classify_failure` therefore keeps `turn_limit`, `tool_errors`, `agent_error`,
`infrastructure` and `unfinished` distinct from `wrong_outcome`. Without that,
Dynamic Filtering's `d_bar` would be uninterpretable: a batch full of turn-limit
timeouts and a batch full of genuinely wrong answers both look like all-zero
groups.

## Per-trajectory isolation

`airline.get_environment(db=...)` accepts an injected `FlightDB`, so each
`TrajectorySession` loads its own snapshot from the record's `db_path` and builds
its own `Environment`. Nothing is cached or shared, which is what lets eight
rollouts of one uid mutate the same task independently. `resolve_db_path` rejects
any `db_path` that escapes the dataset root.
