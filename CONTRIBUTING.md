# Contributing

Thanks for collaborating on Tau3-GRPO. This repository vendors the exact
Tau3 Bench and veRL revisions used by the experiments, so changes to project
code and changes to upstream runtime code must remain easy to distinguish.

## Branch workflow

- `master` is the stable integration branch.
- `dev` is the shared development branch.
- Create a focused feature branch from `dev`.
- Open a pull request back to `dev`; merge `dev` into `master` only after
  the agreed experiment gate passes.
- Do not commit directly to `master`.

## Local CPU setup

Python 3.12 is required.

```bash
bash setup.sh cpu-test
source .venv-cpu/bin/activate
ruff check tau3_grpo tests
pytest -q \
  tests/test_anchors.py \
  tests/test_dynamic_filtering.py \
  tests/test_tau_gigpo.py \
  tests/test_metrics.py \
  tests/test_configs.py \
  tests/test_experiment_manifest.py \
  tests/test_leakage_audit.py \
  tests/test_path_traversal.py \
  tests/test_patch_contract.py
```

GPU, vLLM, FlashAttention and full veRL integration checks run only on the
pinned A800 environment. Do not treat a laptop-only result as an A800 runtime
validation.

## Repository boundaries

- Put project-owned code in ``.
- Keep AReaL raw data, generated manifests/parquet, SFT artifacts, checkpoints,
  model weights, rollout outputs and logs out of Git.
- `tau2-bench/` is the unmodified Sierra Tau3 Bench v1.0.1 snapshot.
- Changes under `verl/` must be minimal, marked with
  `Tau3-GRPO local patch`, documented in
  `docs/implementation_status.md`, and covered by
  `tests/test_patch_contract.py`.
- Keep `THIRD_PARTY_SOURCES.md` and `env_info/versions.lock` synchronized
  whenever an upstream revision or local patch boundary changes.
- Do not use official Tau3 final tasks for training or model selection.

## Pull request checklist

- Explain the algorithm or runtime behavior being changed.
- Add or update deterministic tests.
- Run the CPU checks above.
- If GPU behavior changes, attach the A800 command, environment versions and
  relevant metrics/log hashes.
- Confirm that no dataset, checkpoint, credential or generated result is
  included.
- Update documentation when configs, CLI flags, patch boundaries or experiment
  invariants change.
- Describe smoke and one-update runs as infrastructure validation; do not claim
  algorithmic improvement without the frozen multi-update, multi-seed and
  selection protocol.
