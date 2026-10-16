# Benchmark Protocol

Run these commands from the repository root with `uv`. The protocol separates
epidemiological parity from runtime: first establish that both implementations
represent the intended model, then compare their runtime on fixed workloads.

## 1. Prepare the environment

Install the locked dependencies and run the structural tests:

```bash
uv sync
uv run pytest tests/test_structural_parity.py
```

Use an otherwise idle machine. Record the commit, OS, CPU, Python version, and
dependency versions with the results. Do not compare timings collected on
different hardware or under different system load as though they were a single
experiment.

## 2. Check epidemiological parity

Run independent replicates for each mechanism you intend to benchmark. For
example:

```bash
uv run python -m experiments.benchmark_runner parity --scenario hazard_similarity --replicates 100 --n-agents 300
uv run python -m experiments.benchmark_runner parity --scenario hazard_quarantine --replicates 100 --n-agents 300
```

The parity runner uses the same parameters but independent random streams; a
matching seed does not pair or synchronize random draws. Review all three
estimands (attack rate, peak prevalence, and time to peak), and inspect the
saved raw replicate values. Set `--familywise-scenarios` to the total number of
scenarios in the planned manuscript analysis on every run. The default of one
applies Bonferroni correction across three hypothesis tests and the variance
ratio interval check for each of the three estimands at family-wise alpha 0.01.
For two scenarios, the run-level alpha is allocated across all 24 screening
criteria.
“No test rejected” is a screen, not evidence of equivalence.

For an equivalence claim, set domain-justified margins before the experiment
and supply all three mean-difference margins and bounds for the Mesa/ECS
variance ratio. For example, the following margins are illustrative only and
must be replaced with values justified for the scientific use case:

```bash
uv run python -m experiments.benchmark_runner parity \
  --scenario hazard_similarity --replicates 500 --n-agents 300 --alpha 0.05 \
  --familywise-scenarios 2 \
  --attack-rate-margin 0.02 --peak-prevalence-margin 0.02 \
  --time-to-peak-margin 2 --variance-ratio-bounds 0.8 1.25
```

This performs a Welch confidence-interval equivalence check for each mean and
a bootstrap confidence-interval check for each variance ratio. The family-wise
alpha is divided across all planned scenario/estimand/equivalence components;
use the same `--familywise-scenarios` setting for every scenario. Estimates, intervals,
per-test alpha, and the chosen margins are saved. Choose replicate counts to
provide adequate precision for those margins using a pilot and prospective
power/precision analysis; 100 is a starting run, not a universal sample-size
guarantee. Constant-valued endpoints are reported as inconclusive for variance
parity rather than being accepted as a match.

## 3. Measure runtime

The runtime command records model construction and simulation separately,
without per-system timing instrumentation. Each repeat uses fresh models and
counterbalanced ECS-first/Mesa-first order; warm-ups are excluded from recorded
measurements. Spatial trajectory collection is disabled for both
implementations, while their normal epidemic time-series collection remains on.
Reports include raw measurements, medians and IQRs, a paired median
ECS/Mesa simulation-time ratio with a bootstrap 95% interval, the full resolved
model configuration, package versions, CPU model/availability, OS, git revision,
and whether the worktree was dirty.

Run a small scaling sweep with one scenario held constant:

```bash
uv run python -m experiments.profiling.benchmark_models --scenario hazard_similarity --n-agents 300 --steps 50 --repeats 10 --warmups 2
uv run python -m experiments.profiling.benchmark_models --scenario hazard_similarity --n-agents 1000 --steps 50 --repeats 10 --warmups 2
uv run python -m experiments.profiling.benchmark_models --scenario hazard_similarity --n-agents 2000 --steps 50 --repeats 10 --warmups 2
```

Use the same scenario, world size, and step count across population sizes.
Repeat the sweep for other scenarios only when those mechanisms are relevant.
The similarity-network setup currently requires at least 51 agents.

Use at least 30 measured repeats and 3 warm-ups for a primary benchmark; these
are defaults. Repeat the benchmark command in separate invocations to check
day-to-day stability. Report the median and IQR across measured runs for
simulation time and construction time separately, plus the paired speed ratio
and its interval. Keep per-system timings from
`experiments.benchmark_runner profile` diagnostic only; instrumentation changes
the measured execution. For publication-quality timing, increase repeats and
run independent benchmark processes on an idle, consistently configured host.
For resource-use claims beyond wall-clock time, add a separately validated
peak-memory measurement; the current runner does not measure memory.

## 4. Plot and retain results

Each parity and runtime command writes raw JSON under `results/`. Generate the
comparison charts after running the desired scenarios and sizes:

```bash
uv run python -m experiments.plot_benchmark_results \
  --runtime results/runtime_*.json \
  --parity results/epi_parity_*.json
```

The plotter pools timestamped files only when their workload, model parameters,
software, and execution environment match. It writes
`results/runtime_comparison.png` (repeat-level construction and simulation
timing distributions) and `results/epidemiological_parity.png` (replicate-level
outcome distributions). Inspect raw JSON and numerical intervals as well as
plots; the figures do not replace uncertainty estimates or statistical
decisions. Generated JSON and PNG files are ignored by git so local experiment
outputs do not accidentally become source changes.
