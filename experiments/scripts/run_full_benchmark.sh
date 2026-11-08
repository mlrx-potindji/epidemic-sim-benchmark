#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-results/full_benchmark_${RUN_ID}}"
PARITY_PILOT_REPLICATES="${PARITY_PILOT_REPLICATES:-100}"
PARITY_REPLICATES="${PARITY_REPLICATES:-250}"
RUNTIME_REPEATS="${RUNTIME_REPEATS:-30}"
RUNTIME_WARMUPS="${RUNTIME_WARMUPS:-3}"
MAX_STEPS="${MAX_STEPS:-50}"
PARITY_STEPS="${PARITY_STEPS:-300}"
WORLD_SIZE="${WORLD_SIZE:-200}"
FIRST_SEED="${FIRST_SEED:-1000}"
RUNTIME_SEED="${RUNTIME_SEED:-28022026}"
read -r -a POPULATION_SIZES <<< "${POPULATION_SIZES:-300 1000 2000}"
SCENARIOS=(hazard_similarity hazard_quarantine)

if [[ -e "$OUTPUT_DIR" ]]; then
    printf 'Output directory already exists: %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'git_revision=%s\n' "$(git rev-parse HEAD)"
    printf 'git_worktree_dirty=%s\n' "$(if [[ -n "$(git status --porcelain)" ]]; then printf true; else printf false; fi)"
    printf 'parity_pilot_replicates=%s\n' "$PARITY_PILOT_REPLICATES"
    printf 'parity_confirmatory_replicates=%s\n' "$PARITY_REPLICATES"
    printf 'runtime_repeats=%s\n' "$RUNTIME_REPEATS"
    printf 'runtime_warmups=%s\n' "$RUNTIME_WARMUPS"
    printf 'parity_steps=%s\n' "$PARITY_STEPS"
    printf 'runtime_steps=%s\n' "$MAX_STEPS"
    printf 'world_size=%s\n' "$WORLD_SIZE"
    printf 'population_sizes=%s\n' "${POPULATION_SIZES[*]}"
    printf 'parity_scenarios=%s\n' "${SCENARIOS[*]}"
} > "$OUTPUT_DIR/run_manifest.txt"

printf 'Saving benchmark outputs to %s\n' "$OUTPUT_DIR"

uv run pytest -q 2>&1 | tee "$OUTPUT_DIR/tests.log"

for scenario in "${SCENARIOS[@]}"; do
    uv run python -m experiments.benchmark_runner parity \
        --scenario "$scenario" \
        --pilot-replicates "$PARITY_PILOT_REPLICATES" \
        --replicates "$PARITY_REPLICATES" \
        --n-agents 300 \
        --world-size "$WORLD_SIZE" \
        --max-steps "$PARITY_STEPS" \
        --first-seed "$FIRST_SEED" \
        --alpha 0.05 \
        --mean-margin-sd 0.8 \
        --major-outbreak-margin 0.10 \
        --scale-ratio-bounds 0.5 2.0 \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/parity_${scenario}.log"
done

for n_agents in "${POPULATION_SIZES[@]}"; do
    uv run python -m experiments.profiling.benchmark_models \
        --scenario hazard_similarity \
        --n-agents "$n_agents" \
        --world-size "$WORLD_SIZE" \
        --steps "$MAX_STEPS" \
        --repeats "$RUNTIME_REPEATS" \
        --warmups "$RUNTIME_WARMUPS" \
        --seed "$RUNTIME_SEED" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/runtime_n${n_agents}.log"
done

uv run python -m experiments.plot_benchmark_results \
    --runtime "$OUTPUT_DIR"/runtime_*.json \
    --parity "$OUTPUT_DIR"/epi_parity_*.json \
    --output-dir "$OUTPUT_DIR"

printf 'Benchmark complete. Results and logs: %s\n' "$OUTPUT_DIR"