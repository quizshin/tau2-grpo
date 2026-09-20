#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/tau3-5xa800-20260911
cd "$ROOT/code"
export PYTHONPATH="$PWD"
mkdir -p "$ROOT/model_store" /root/tau3-models-5xa800
/root/miniconda3/bin/python -u -m tau3_grpo.models.download_simulator --manifest env_info/paratera/qwen35-4b-download.json --output "$ROOT/model_store/Qwen3.5-4B" --workers 3
/root/miniconda3/bin/python -u -m tau3_grpo.models.download_simulator --manifest configs/simulator/qwen38_download.json --output /root/tau3-models-5xa800/Qwen3.8-27B-AWQ-INT4 --workers 3
ln -s /root/tau3-models-5xa800/Qwen3.8-27B-AWQ-INT4 "$ROOT/model_store/Qwen3.8-27B-AWQ-INT4"
