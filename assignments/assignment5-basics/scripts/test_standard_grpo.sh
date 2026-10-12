#!/usr/bin/env bash
# Run completed on-policy components and variants with snapshot-compatible HF deps.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/activate.sh"
test_deps="${CS336_A5_TEST_DEPS:-}"
if [[ -n "$test_deps" && ! -f "$test_deps/transformers/__init__.py" ]]; then
    echo "CS336_A5_TEST_DEPS must contain snapshot-compatible Transformers." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${test_deps:+$test_deps:}$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"
exec python -m pytest \
  tests/test_grpo.py tests/test_rollout_rewards_smoke.py tests/test_grpo_train_step_sanity.py tests/test_grpo_distributed.py tests/test_sampled_trajectories.py \
  -k 'not off_policy' \
    -q --tb=short "$@"
