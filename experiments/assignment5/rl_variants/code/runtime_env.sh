#!/usr/bin/env bash
export PATH="/root/ocr-rl-venv/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MAX_JOBS=8
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_SERVER_DEV_MODE=1 PYTHONUNBUFFERED=1
export WANDB_CONSOLE=off WANDB_SILENT=true
export PYTHONPATH="/root/autodl-tmp/cs336-a5-20261029/code:/root/autodl-tmp/cs336-a5-20261029/deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/root/autodl-tmp/cs336-a5-20261029/cache/huggingface
export VLLM_CACHE_ROOT=/root/autodl-tmp/cs336-a5-20261029/cache/vllm
export TRITON_CACHE_DIR=/root/autodl-tmp/cs336-a5-20261029/cache/triton
export TMPDIR=/root/autodl-tmp/cs336-a5-20261029/cache/tmp
TASK_NVCC=$(python -c 'from pathlib import Path; print(next(Path("/root/ocr-rl-venv/lib/python3.12/site-packages/nvidia").glob("**/bin/nvcc")))')
export CUDA_HOME="$(dirname "$(dirname "$TASK_NVCC")")"
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR"
