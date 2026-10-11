#!/usr/bin/env bash
# Two training GPUs + two independent vLLM sampling GPUs. Overrides follow profile.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/activate.sh"
profile="${1:-pilot}"
if [[ $# -gt 0 ]]; then shift; fi
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
# Only preload a custom NCCL library when the caller explicitly configures one.
if [[ -n "${VLLM_NCCL_SO_PATH:-}" ]]; then
  export LD_PRELOAD="${VLLM_NCCL_SO_PATH}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  --master-addr=127.0.0.1 --master-port="${CS336_GRPO_MASTER_PORT:-29564}" \
  "$project_dir/scripts/train_grpo.py" --profile "$profile" --sampling-gpus 2 3 "$@"
