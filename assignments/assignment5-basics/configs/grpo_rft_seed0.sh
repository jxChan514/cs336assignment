#!/usr/bin/env bash
# Portable configuration: GPU 0 trains, GPU 1 generates on the same machine.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
method="${1:-grpo}"
if [[ $# -gt 0 ]]; then shift; fi
case "$method" in grpo|rft) ;; *) echo "Usage: $0 {grpo|rft} [train_grpo.py options]" >&2; exit 2 ;; esac
exec bash "$project_dir/scripts/run_grpo_two_gpu.sh" standard \
  --method "$method" --prompt r1_zero --seed 0 --steps 200 \
  --train-examples 6400 --val-examples 1024 \
  --rollout-batch-size 256 --group-size 8 --microbatch-size 8 \
  --learning-rate 1e-5 --max-grad-norm 1 --attention-implementation sdpa \
  --temperature 1 --max-tokens 512 --eval-every 10 \
  --samples-every 40 --sample-count 8 --request-batch-size 32 \
  --vllm-memory-utilization 0.35 --vllm-max-model-len 2048 \
  --wandb-project cs336-a5-grpo --wandb-group on-policy-seed0 \
  --output-dir "$project_dir/outputs/${method}-seed0" "$@"
