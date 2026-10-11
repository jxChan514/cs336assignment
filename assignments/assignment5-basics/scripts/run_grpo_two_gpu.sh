#!/usr/bin/env bash
# One local GPU trains, one local GPU generates. No DDP or remote SSH needed.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/activate.sh"
profile="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
exec python "$project_dir/scripts/train_grpo.py" \
  --profile "$profile" --execution-mode separate_gpus --sampling-gpus 1 \
  --microbatch-size 1 --wandb-group standard-on-policy-two-local-gpus "$@"
