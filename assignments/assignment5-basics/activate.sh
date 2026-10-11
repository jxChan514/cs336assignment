#!/usr/bin/env bash
# Source from bash. Paths can be configured without editing tracked files.
_cs336_a5_project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_cs336_a5_env="${CS336_A5_ENV:-${VIRTUAL_ENV:-$_cs336_a5_project/.venv}}"
if [[ ! -f "$_cs336_a5_env/bin/activate" ]]; then
  echo "A5 environment missing: run uv sync, or set CS336_A5_ENV to an existing venv." >&2
  unset _cs336_a5_project _cs336_a5_env
  return 1
fi
source "$_cs336_a5_env/bin/activate" || return 1
export CS336_A5_MODEL_DIR="${CS336_A5_MODEL_DIR:-$_cs336_a5_project/models/OLMo-2-0425-1B}"
export CS336_A5_DATA_DIR="${CS336_A5_DATA_DIR:-$_cs336_a5_project/data/gsm8k}"
export HF_HOME="${HF_HOME:-$_cs336_a5_project/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$_cs336_a5_project/.cache/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-$_cs336_a5_project/outputs/wandb}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$WANDB_DIR"
cd "$_cs336_a5_project" || return 1
unset _cs336_a5_project _cs336_a5_env
