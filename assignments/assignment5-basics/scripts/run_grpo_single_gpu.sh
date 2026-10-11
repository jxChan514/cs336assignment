#!/usr/bin/env bash
# Alternate vLLM generation and full-parameter training on one 48GB GPU.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/activate.sh"
profile="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
exec python "$project_dir/scripts/train_grpo.py" \
  --profile "$profile" --execution-mode single_gpu --sampling-gpus 0 \
  --vllm-memory-utilization 0.35 --microbatch-size 1 \
  --wandb-group standard-on-policy-single-48gb "$@"
