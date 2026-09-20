# Contributing

Thanks for collaborating on Tau3-GRPO. This repository vendors the exact
Tau3 Bench and veRL revisions used by the experiments, so changes to project
code and changes to upstream runtime code must remain easy to distinguish.

## Branch workflow

- `main` is the current GitHub default and integration branch. Local `master`
  and `dev` refs are historical; do not assume corresponding remote branches exist.
- Use focused feature branches and pull requests for collaborative changes.
- For an explicitly requested repository synchronization, verify the actual
  upstream and publish a fast-forward update; never force-push shared history.
- Keep runtime assets out of commits and preserve existing worktree changes.

## Local CPU setup

Python 3.12 is required.

```bash
bash setup.sh cpu-test
source .venv-cpu/bin/activate
python -m pip install ruff==0.16.6
python scripts/maintenance/check_lint.py
python scripts/maintenance/check_cpu.py --suite core
python -m tau3_grpo.integrations.vendor_inventory
```

GPU, vLLM, FlashAttention and full veRL integration checks run only on the
pinned A800 environment. Do not treat a laptop-only result as an A800 runtime
validation.

## Repository boundaries

- Put project-owned code in `tau3_grpo/`; keep `scripts/` as thin entry points
  and parameters in `configs/`. Follow [the development standard](docs/architecture/development_standard.md).
- Keep generated data/parquet, SFT artifacts, checkpoints, model weights, rollout
  outputs and logs out of Git. The reviewed frozen inputs already listed in
  `data/SHA256SUMS.json` are an explicit exception, documented in `data/README.md`.
- `tau2-bench/` is the unmodified Sierra Tau3 Bench v1.0.1 snapshot.
- Changes under `verl/` must be minimal, marked with
  `Tau3-GRPO local patch`, documented in
  `env_info/vendor_patch_notes.json` and `env_info/vendor_patches.json`, and
  covered by the relevant tests. `tests/test_patch_contract.py` checks known
  integration contracts; the vendor inventory verifies the runtime file set.
- Keep `THIRD_PARTY_SOURCES.md` and `env_info/versions.lock` synchronized
  whenever an upstream revision or local patch boundary changes.
- Do not use official Tau3 final tasks for training or model selection.

## Pull request checklist

- Explain the algorithm or runtime behavior being changed.
- Add or update deterministic tests.
- Run the CPU checks above.
- If GPU behavior changes, attach the A800 command, environment versions and
  relevant metrics/log hashes.
- Confirm that no unreviewed dataset, checkpoint, credential or generated result
  is included. Historical runtime evidence stays outside the source publication.
- Update documentation when configs, CLI flags, patch boundaries or experiment
  invariants change.
- Describe smoke and one-update runs as infrastructure validation; do not claim
  algorithmic improvement without the frozen multi-update, multi-seed and
  selection protocol.
