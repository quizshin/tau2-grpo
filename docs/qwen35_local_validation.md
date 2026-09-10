# Qwen3.5 local validation — 2026-09-08

Scope: `code/`, branch `feat/qwen35-local-support`. No remote server was used.
No pretrained policy weights were downloaded or trained. The model tests use
tiny randomly initialized Qwen3.5 and Qwen2.5 models; native tokenizer/config
assets are downloaded at the revisions in `configs/models/qwen35.json`.

## Environments and results

The isolated `.venv-qwen35-local` uses macOS arm64, Python 3.12.14, Torch
2.11.0, Transformers 5.5.1, PEFT 0.18.1, Accelerate 1.12.0, NumPy 2.2.6,
SciPy 1.14.1, TensorDict 0.10.0 and LiteLLM 1.82.6. vLLM is not installed on
this Mac. The editable project, benchmark and veRL packages are installed.

| Check | Result |
|---|---|
| Full project suite, new CPU stack | **378 passed**, 2 deprecation warnings, 20.86 s |
| Legacy CPU subset, existing sibling environment | **327 passed, 8 skipped**, 1 deprecation warning |
| Project Ruff check | Passed |
| All project launchers and both setup entrypoints | `bash -n`: 19 passed |
| Local veRL patch contracts and `tau_gigpo` registration | Passed |
| New local environment `pip check` | No broken requirements |
| Linux x86_64 / Python 3.12 dependency resolution | Passed under `qwen35-constraints.txt` |

The legacy subset used Torch 2.8 / Transformers 4.57.6, without veRL installed.
It excluded `test_qwen35.py` and `test_rollout_json_serialization.py`; eight
other integration checks skipped for missing veRL. This is a compatibility
regression check, not a rerun of the previously validated CUDA 12.8 /
Transformers 4.56.1 deployment.

The full new-stack suite covers:

- Native 0.8B, 4B and 9B assistant token masks: exact native token equality,
  XML calls and EOS labeled, user/tool observations and empty thinking
  scaffolding excluded. The Qwen2.5 JSON template retains its original path.
- Actual veRL `qwen3_coder` and `hermes` parsers, including string ID `0012`.
- Real Qwen3.5 full backward through both Gated DeltaNet and full attention;
  SFT LoRA optimizer update, merge, save and reload with tied and untied
  embeddings. The unused vision parameters remain frozen.
- Independent padded batch rows under native Gated DeltaNet; actual veRL
  actor log probabilities compared with native logits, followed by backward.
- Explicit rejection of unsupported packed forward, and text-only processor
  handling, including an isolated tool-observation template render.
- Real launcher arguments composed through Hydra for 0.8B and 9B; the FP32
  embedding fits its selected weight-transfer bucket. The evaluation wrapper
  preserves the existing `tau3-final` guard and model/result namespace.
- Tiny Qwen2.5 causal-LM loading and backward, plus existing algorithm, data,
  integration and evaluation guard regressions.

2B has a revision-pinned download/profile entry but its tokenizer was not
downloaded separately in this local run. No model size has GPU acceptance yet.

The simulator default was subsequently set to **Qwen3.5-4B** on GPU 1,
independently of the policy size. The full suite was rerun: **378 passed**,
2 deprecation warnings, 17.14 s (`code/.cache/qwen35-simulator-tests.log`).
An offline launcher check intercepted the server command and verified the
pinned local 4B snapshot, the matching training/evaluation API name, TP=1,
GPU 1 and disabled thinking. Download requests were intercepted to verify
that `--include-user` fetches 4B weights and `--include-legacy-tokenizer`
fetches only the original data-split tokenizer/config assets. No live server
or model download was started by these checks.

## Frozen SFT audit

The existing files in `../code_pytrio/data/sft/` were read without modifying
that project. The audit uses the complete original `code/` tool schemas and
the pinned Qwen3.5-0.8B tokenizer, with thinking disabled. No task was dropped,
resampled or truncated.

| Split | Dialogues | Total tokens | Assistant label tokens | Longest dialogue |
|---|---:|---:|---:|---:|
| Train | 45 | 488,834 | 97,413 | 16,473 |
| Validation | 5 | 62,000 | 15,492 | 16,702 |

Input SHA-256:

- Train: `a930dbffab46ebb5ce2e9df453ec7c6ecc69752dbd8de9a1f3f1969f974cfe55`
- Validation: `38f55df325f2bfd6a9aa7f16cd627329b292056c269d43ea8cbb40207b16e22e`

The Qwen3.5 SFT cap is 24,576 because these complete dialogues exceed the
old 16,384-token cap. The original Qwen2.5 config and data selection are
unchanged. Audit the actual remote input files before training and compare
these hashes when reusing this frozen split.

## Reproduction

From `code/`, with the isolated local environment already prepared and pinned
tokenizer assets downloaded as described in [qwen35.md](qwen35.md):

```bash
source .venv-qwen35-local/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q tests
ruff check tau3_grpo tests
python -m tau3_grpo.integrations.verify_patches --check-registration
python -m pip check

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m tau3_grpo.models.check_qwen35 \
  --model models/Qwen3.5-0.8B --audit-sft \
  --train ../code_pytrio/data/sft/airline_sft_train_seed42.jsonl \
  --validation ../code_pytrio/data/sft/airline_sft_validation_seed42.jsonl
```

The Linux resolution input lists `-e ./verl[qwen35]`, `-e ./tau2-bench` and
`-e .[dev,data,stats,qwen35]`, one per line:

```bash
uv pip compile .cache/qwen35-linux.in \
  --constraint env_info/qwen35-constraints.txt \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --output-file .cache/qwen35-linux-lock.txt --quiet
```

This checks package compatibility; it is not a Linux installation or execution
test. The resulting default PyPI dependency set uses CUDA 13.0 and needs a
compatible NVIDIA driver (R580 or newer). The new environment installer is
`bash setup.sh a800-qwen35`; the legacy `a800` mode remains separate.

Local ignored evidence files and SHA-256:

| File under `code/.cache/` | SHA-256 |
|---|---|
| `qwen35-final-tests.log` | `32fcc828cf642ad36d5ff0eeafac923be9032d9ab116be1479f09fd04b10d8a8` |
| `qwen25-legacy-tests.log` | `aa0c87e9659a37ae23cf14798870ccb19cdaec99a48a139314918072fcef0eed` |
| `qwen35-linux-lock.txt` | `653a72b1bc7408d057fb70f6aad9d28f75dd9c0ff384375bc89ef243e0bff7f5` |
| `qwen35-sft-audit.json` | `f6b70af43c58c5396199ed1b8b38baebd3287eb339d3a3391649217cb1d5d781` |

## Still required on the remote A800 host

During the subsequent AutoDL installation, the `stats` extra was added to the
Qwen3.5 installer so the constrained SciPy version is actually installed for
reporting. The Linux lock hash above reflects this addition. A constraints file
alone does not install an otherwise optional package.

Follow [qwen35.md](qwen35.md) for the exact commands. Validate real CUDA
allocation, full-weight SFT and merge, vLLM simulator/policy serving, complete
tool trajectories and official terminal verification, finite actor/reference
log probabilities and gradients, a real optimizer update, actor-to-vLLM
weight synchronization, checkpoint merge and a resumed update. Exercise all
E0–E3 arms, then selection and the final evaluation guard at the proper stage.

FSDP, CUDA kernels, vLLM sleep/wake and weight synchronization are unverified
locally. The three-update engineering profile does not establish training
quality or replace the formal experiment schedule. Full-parameter 9B RL may
need more policy GPUs/offload and a separately hosted simulator.
