#!/usr/bin/env bash
# Run on the server after install_qwen35.sh has completed successfully.
set -euo pipefail
source /root/shared-nvme/tau3/code/env_info/paratera/activate.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TAU3_TRACKING_BACKEND=none TRAINER_LOGGERS='[console]'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TAU3_ROOT}/code"
python -m pip check
python -c 'from datasets import Dataset; from transformers import Trainer; from tau3_grpo.training.sft.trainer import Qwen35SFTTrainer; print("SFT Trainer and Dataset imports passed")'
timeout 180 torchrun --standalone --nproc-per-node=2 \
  env_info/paratera/runtime_probe.py
python -m tau3_grpo.integrations.verify_patches
python -m pytest -q tests/test_launcher.py tests/test_logprob_memory.py \
  tests/test_qwen35_fsdp.py tests/test_qwen35_loss_projection.py
python -m tau3_grpo.data.prepare --seed 42 --require-db
python -m tau3_grpo.data.build_parquet --seed 42 --split train
python -m tau3_grpo.data.build_parquet --seed 42 --split selection
python -m tau3_grpo.launch simulator \
  --config configs/simulator/qwen38_27b_paratera_5090.yaml --dry-run \
  > "${TAU3_ROOT}/bootstrap/simulator-resolved.json"
python -m tau3_grpo.launch sft \
  --config configs/train/sft/qwen35_4b_lora_paratera_5090.yaml --dry-run \
  > "${TAU3_ROOT}/bootstrap/sft-resolved.json"
# Overlay adds only Transformers 5.8; shares the pinned Torch/vLLM libraries.
bash env_info/setup_qwen38_simulator.sh
"${TAU3_UV}" pip freeze --python "${TAU3_SIM_PYTHON}" \
  > "${TAU3_ROOT}/bootstrap/qwen38-sim-installed.txt"
