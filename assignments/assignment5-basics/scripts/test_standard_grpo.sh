#!/usr/bin/env bash
# Run the completed standard on-policy components with snapshot-compatible HF deps.
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
  tests/test_grpo.py tests/test_rollout_rewards_smoke.py tests/test_grpo_train_step_sanity.py tests/test_grpo_distributed.py \
  -k 'test_tokenize_prompt_and_output or test_get_response_log_probs or test_compute_rollout_rewards or test_compute_group_normalized_rewards or test_compute_policy_gradient_loss_on_policy or test_aggregate_loss_across_microbatch_sequence or test_grpo_train_step_standard_on_policy or real_grader or test_train_step or test_ddp' \
    -q --tb=short "$@"
