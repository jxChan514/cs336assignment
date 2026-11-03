#!/usr/bin/env bash
set -euo pipefail
snapshot_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test_python="${A5_PYTHON:-python3}"
test_python="$(command -v "$test_python")"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
(
  cd "$snapshot_root/rl_variants/code"
  "$test_python" -m unittest test_objectives -v
  "$test_python" test_scoring.py
)
(
  cd "$snapshot_root/dapo/code"
  "$test_python" -m unittest test_dapo -v
)
(
  cd "$snapshot_root/gkd_opd/code"
  "$test_python" -m unittest test_gkd -v
)
