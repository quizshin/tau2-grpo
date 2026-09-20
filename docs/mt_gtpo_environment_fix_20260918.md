# MT-GTPO environment matching fix

## Scope

`paper_env_v2` is an independent reward version. `paper_v1` remains replayable and is not silently changed. The new version mirrors the pinned Airline execution boundary before comparing tool arguments:

- nested `FlightInfo` / `Passenger` / `Payment` values are converted with the official models;
- top-level IDs, empty values, scalar types, and list order remain strict;
- execution errors do not consume a gold action or duplicate history;
- changed repeated-read observations are `read_only`, not duplicates, and do not renew gold;
- unavailable or truncated observations cannot certify a duplicate.

Dynamic filtering is unchanged and remains an optional configuration switch. This repair changes reward semantics only; it does not start training or claim an effectiveness improvement.

## Why the versions differ

The historical recursive scorer (`paper_v1`) sorts lists of dictionaries. The execution scorer intentionally does not. The pinned Airline tool assigns the submitted passenger list directly to the reservation, and the official database hash sorts dictionary keys but not list elements. Therefore passenger order is part of the executed database state.

The remote proof used two isolated copies of the same real database and the same `update_reservation_passengers` call, changing only the passenger order:

| case | resulting DB hash |
| --- | --- |
| Chloe, Daniel, Ethan | `905b93c30505d68a04699924c8177f2e97589e0ece38d081e2b3bc84b912990f` |
| Chloe, Ethan, Daniel | `a2a757979f37e6ba69e1687589e334c1e68ee24f5e3641219659299bc4b8834c` |

Both started from `224cede469c9ab8a718204d1edc9e763b53b80c7c80bc888978bfc2eddde13f4`. Consequently, the 77 historical `gold -> soft` migrations caused by passenger-list reordering are expected under the current benchmark, not a reason to sort the list in `paper_env_v2`.

This does not mean every changed call was the only cause of a failed trajectory. It only establishes that sorting would misrepresent the official environment. A permutation-invariant benchmark would require changing the evaluator and reward protocol together.

## Validation

- Remote final regression suite: 180 passed.
- Remote lint, veRL patch contract, and diff checks: passed.
- `paper_env_v2` formal dry-run: passed with the existing 20-step / dynamic-filter / checkpoint / selection settings.
- Historical buffer replay: 1,280 trajectories and 12,591 turns replayed successfully.
- Version migration: 291 soft calls became gold, 77 gold calls became soft, 27 changed-read duplicates became `read_only`; advantage decomposition error was below `8e-15`.
- New focused regression: passenger-list reordering remains execution-significant.

The independent IRC still rejects a frozen recipe because soft and duplicate category directions/support are not calibrated. That is a separate reward-calibration issue; this environment repair must not be hidden by dynamic filtering or by changing the benchmark's database semantics.

