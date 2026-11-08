#!/usr/bin/env bash
set -euo pipefail

uv run python -m experiments.benchmark_runner parity \
  --pilot-replicates 5 \
  --replicates 5 \
  --n-agents 100 \
  --world-size 100 \
  --max-steps 100 \
  --bootstrap-resamples 199 \
  --permutations 99
