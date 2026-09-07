# Tau2 → Tau3 GiGPO long-horizon agent RL

This project trains a Qwen2.5-7B tool agent with AReaL's 1,148 synthetic
τ²-style Airline tasks and veRL, selects a winner on a frozen 60-task internal
split, then evaluates Base/SFT/E0/winner on the 50 official Airline tasks from
τ³-bench v1.0.1. Official τ³ tasks are never used for training or selection.

## Runtime composition

```text
AReaL JSONL + per-task DB
        │
        ▼
manifest/schema adapter ──► τ³/`tau2` Environment + official verifier
        │                                  ▲
        ▼                                  │
veRL parquet ──► veRL ToolAgentLoop ──► Airline tools + user simulator
        │
        ├── E0 vanilla GRPO
        ├── E1 fixed-rollout Dynamic Filtering
        ├── E2 Tau-GiGPO structured anchors
        └── E3 Dynamic Filtering + Tau-GiGPO
```

## Data preparation

```bash
python scripts/data/download_areal.py --output-dir data/raw/areal_tau2
python scripts/data/build_manifests.py \
  --rl-jsonl data/raw/areal_tau2/tau2_rl_train.jsonl \
  --output-dir experiments/manifests --seed 20260828
python scripts/data/build_grpo_parquet.py \
  --manifest experiments/manifests/train.jsonl \
  --output data/processed/train.parquet
```

The split builder strictly checks the real counts: 1,982 total records and
1,148 Airline records, then writes exactly 200 train and 60 internal-selection
records plus hashes. It does not silently backfill or resample.

## Training

```bash
bash scripts/train/grpo/run_experiment.sh e0_vanilla seed42
bash scripts/train/grpo/run_experiment.sh e1_dynamic_filter seed42
bash scripts/train/grpo/run_experiment.sh e2_tau_gigpo seed42
bash scripts/train/grpo/run_experiment.sh e3_joint seed42
```

The scripts call `python -m verl.trainer.main_ppo`; vLLM is the primary rollout
backend and SGLang is configured only for parity smoke tests. The local veRL
copy carries small, auditable patches for Dynamic Filtering masks and passing
anchor metadata into the registered `tau_gigpo` advantage estimator.

## Frozen final evaluation

```bash
python scripts/evaluation/freeze_winner.py --selection-results ...
python scripts/evaluation/run_tau3_final.py --winner-lock experiments/winner.lock.json
```

`run_tau3_final.py` refuses to run unless the winner lock contains the selection
manifest hash and no τ³ result already influenced the winner decision.

## Tests

```bash
pytest -q tests/unit
pytest -q -m tau3 tests/integration   # Python 3.12 + sibling tau2-bench
pytest -q -m verl tests/integration   # A800/veRL environment
```

See `docs/implementation_status.md` for the 14-day milestone mapping.

