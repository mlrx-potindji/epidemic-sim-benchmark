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

The comparison uses a separate pilot and confirmatory block. The pilot fixes
mean-equivalence margins at `0.8 × pooled pilot SD`; pilot outcomes are not used
for confirmatory inference. The confirmatory defaults are 250 fresh replicates
per implementation. For example:

```bash
uv run python -m experiments.benchmark_runner parity --scenario hazard_similarity \
  --pilot-replicates 100 --replicates 250 --n-agents 300
uv run python -m experiments.benchmark_runner parity --scenario hazard_quarantine \
  --pilot-replicates 100 --replicates 250 --n-agents 300
```

The model streams remain independent; equal seed values do not synchronize
draws. The primary mean estimands are attack rate, peak prevalence, and time to
peak. Their TOST margins are fixed from the pilot as 0.8 times each pooled
within-implementation SD. The major-outbreak probability uses a difference
margin of ±0.10, with a major outbreak defined as at least five times the seed
count ever infected. Thresholds of three and ten are reported as sensitivity
analyses. If each arm has at least 30 major outbreaks, the three mean estimands
are also analyzed conditional on a major outbreak.

Dispersion equivalence is assessed for both the variance ratio and IQR ratio,
with each 90% bootstrap interval required to fall inside [0.5, 2]. The default
uses 9,999 BCa bootstrap resamples, falling back to percentile intervals when
BCa is undefined. Welch and Newcombe 90% intervals are used for means and
major-outbreak probabilities. Each TOST uses alpha 0.05; the global
equivalence claim is a conjunction of component claims, so no multiplicity
adjustment is applied to equivalence tests. Difference-detection screens
(KS, Wasserstein-1, and Brown-Forsythe permutation tests) are adjusted by Holm
within each scenario and are descriptive only: a non-significant result is
not evidence of equivalence.

The margins are a provisional, lenient statistical choice—not a claim that
those absolute differences are scientifically negligible. Review the
pilot-derived absolute tolerances and confirmatory precision before making
substantive claims. Options can be changed explicitly:

```bash
uv run python -m experiments.benchmark_runner parity \
  --scenario hazard_similarity --pilot-replicates 100 --replicates 250 \
  --n-agents 300 --alpha 0.05 --mean-margin-sd 0.8 \
  --major-outbreak-margin 0.10 --scale-ratio-bounds 0.5 2.0
```

Pilot and confirmatory seeds use disjoint blocks. The JSON report retains both
sets of raw outcomes, pilot SDs, fixed margins, intervals, test adjustments,
resolved configuration, and environment metadata. Constant-valued scales are
reported as inconclusive. The default replicate count is a starting design,
not a power guarantee; inspect the resulting absolute margins and intervals.

### Frozen transition semantics for phase-level parity

The maintained executable oracle is [`experiments/parity/oracle.py`](parity/oracle.py);
run its focused checks with `uv run pytest tests/test_oracle.py`. It is kept
with the parity analysis and does not import either implementation.
The maintained keyed RNG is [`experiments/parity/keyed_rng.py`](parity/keyed_rng.py).
Parity tests can pass it to `SIREpidemicModel.step(phase_rng=...,
phase_step=...)`, `MesaSIREpidemicModel.step_with_phase_rng(...)`, or directly
to ECS processor `process(phase_rng=..., step=...)`. Mesa exposes the named
helper because its framework replaces public `step()` with its scheduler.
Without injection, both implementations continue to use their existing
model-owned NumPy generators. ECS adapters map entity IDs to the shared
snapshot IDs with `KeyedRNG(id_map=...)`; Mesa uses each agent's stable snapshot
index. The keyed purposes for movement, infection, viral load, duration,
recovery, and quarantine entry are kept stable in that module. Verify the seam
with `uv run pytest tests/test_keyed_rng.py`.

The production adapters in [`experiments/parity/adapters.py`](parity/adapters.py)
load explicit oracle snapshots into one Mesa or ECS instance, execute a single
phase, and return the resulting state or hazard arrays. Movement, hazards,
infection resolution, disease progression, and quarantine are each compared
independently with the oracle. Boundary cases cover inclusive spatial radius,
strict infection/recovery/quarantine-entry comparisons, infection-age and
recovery-duration thresholds, quarantine expiry, and immediate quarantine
entry ticks. Failures report the phase, state/hazard field, and affected agent
IDs. Each adapter is a context manager so its test model is cleaned up after
use; tests intentionally compare phase outputs rather than full trajectories,
which can magnify harmless floating-point differences.

[`experiments/parity/loop_adapter.py`](parity/loop_adapter.py) is a separately
written scalar implementation used only to validate the oracle/test harness.
`uv run pytest tests/test_parity_mutations.py` checks the clean loop against
the oracle, runs 17 catalogued mutations, and deliberately alters the real
Mesa/ECS spatial-hazard calculation to verify that the adapters expose a
specific omitted-immunity defect. This mutation score is evidence that tests
detect the planted faults, not evidence that the frozen equations are
scientifically correct.

Rewiring is validated separately by `uv run pytest tests/test_rewiring_parity.py`.
The structural/raw-weight check uses adapter seed `1901` and rewire seed `7319`,
with non-default similarity scales (`alpha=3.5`, `tau=11`) and checks edges
against those active values. Degree-distribution tests use 1,800 agents,
`mean_degree=6`, `dispersion=0.6`, and rewire seeds `8141` and `8142`; the
sample mean tolerance is four theoretical standard errors and the binned
negative-binomial chi-square test requires `p > 0.001` (bins 0 through 11 and
12 or more). Weighted inclusion probabilities use 2,000 independently seeded
rewires (`12000` through `13999`), fixed degree two, and the oracle's exact
successive-sampling probabilities; each empirical inclusion rate must be
within five binomial standard errors plus `0.005`. Failures include the seed
and measured/expected values.

The following choices were checked against the manuscript equations and the
current ECS and Mesa update paths. They are the parity oracle's required
semantics; changing one is a model change that must be reflected in both
implementations and the manuscript.

1. **New infection age:** infection resolution assigns age zero, then disease
   progression runs later in that same update and increments the counter. A
   newly infected agent therefore ends that update at age one; initially seeded
   agents are also initialized at age one.
2. **Recovery/death threshold:** increment infection age before testing
   `age >= recovery_duration`; make the recovery/death decision once on the
   first update reaching the threshold. Recovery occurs only when an independent
   draw is strictly less than the sampled recovery probability (`u < rho`).
3. **Quarantine stage and expiry:** after disease progression, first attempt
   quarantine entry for infectious agents that are not currently quarantined,
   then increment the duration counter for agents in quarantine and release
   when it reaches 14. A newly quarantined agent receives its first duration
   tick in its entry update; an infectious agent released by expiry cannot
   re-enter until the next update. Recovery releases quarantine immediately
   and restores mobility; death sets mobility to zero and the quarantine flag
   remains until its duration expires.
4. **Rewiring candidates and degree cap:** rewiring excludes dead agents as
   both sources and candidates. Thus a source's requested out-degree is capped
   by the number of other living agents (`alive_count - 1`), not population
   size minus one. This cap applies to rewiring; initial-network construction
   is a separate operation.
5. **Recovery duration conversion:** use `max(1, int(Z))`, with `Z` sampled
   from the specified normal distribution. Python's integer conversion
   truncates toward zero; it does not round to nearest.

The phase tests explicitly check these choices: same-update infection ageing,
the inclusive recovery-duration threshold and strict recovery comparison,
quarantine entry/expiry ordering and mobility changes, dead-agent rewiring
exclusion/degree caps, and truncation of fractional recovery durations.
When a test reports a first divergence, it identifies the tested phase, field,
and agent IDs; that localizes a mismatch but does not decide which equation is
scientifically justified. Recheck the relevant numbered choice and manuscript
equation before changing implementation or oracle. An oracle/implementation
agreement can still reproduce a mistaken or underspecified equation.

The quarantine order above matches both production update paths but differs
from decision D3 in the archived scripts under `revisiting benchmark`, which
release expired quarantine before attempting entry and therefore permit
same-update re-entry. The archived oracle must not be used as evidence for the
maintained D3 choice without being corrected.

The `Phase parity` GitHub Actions workflow runs keyed-randomness, oracle,
phase-adapter, mutation-sensitivity, and rewiring tests on pushes and pull
requests. Keep these checks ahead of any interpretation of outcome-level
comparisons.

## 3. Measure runtime

The runtime command records model construction and simulation separately,
without per-system timing instrumentation. Protocol v3 disables spatial-history
collection in both models and advances exactly the requested number of updates,
even if one implementation reaches extinction first. After extinction, each
model continues movement, network rewiring, and time-series collection for the
remaining timed steps; this is a fixed-workload benchmark, not a time-to-end
comparison. Each repeat uses fresh models and counterbalanced ECS-first/Mesa-
first order; warm-ups are excluded from recorded measurements. Normal epidemic
time-series collection remains on in both implementations.
Reports tagged protocol version 3 are the corrected runtime comparison. Older
version-1 and version-2 timing reports predate equal spatial-history settings
and fixed-step execution; the plotter labels them separately and does not pool
them with v3. Reports include raw measurements, medians and IQRs, a paired median
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

## 5. Run the complete benchmark

Run the full test suite, the two planned parity scenarios, the protocol-v3
runtime sweep at 300/1,000/2,000 agents, and plot generation with one command:

```bash
bash experiments/scripts/run_full_benchmark.sh
```

By default this runs `uv run pytest -q`, 100 pilot and 250 confirmatory
replicates per parity scenario,
30 measured timing repeats with 3 warm-ups at each population size, and saves
everything under a unique `results/full_benchmark_<UTC timestamp>/` directory.
That directory contains raw JSON reports, separate command logs, a run manifest,
and both figures. The benchmark can take several minutes, particularly at the
largest population size.

The settings can be overridden through environment variables. For example, a
short-workload smoke of the orchestration workflow can be run with:

```bash
OUTPUT_DIR=/tmp/benchmark-smoke \
PARITY_PILOT_REPLICATES=2 PARITY_REPLICATES=2 \
RUNTIME_REPEATS=2 RUNTIME_WARMUPS=0 \
MAX_STEPS=3 PARITY_STEPS=3 WORLD_SIZE=50 POPULATION_SIZES="60" \
bash experiments/scripts/run_full_benchmark.sh
```

This still runs the full automated test suite; the reduced replicate and step
counts are for checking orchestration only and must not be used as scientific
results. `OUTPUT_DIR` must not already exist. Other supported overrides are
`PARITY_PILOT_REPLICATES`, `PARITY_REPLICATES`, `RUNTIME_REPEATS`,
`RUNTIME_WARMUPS`, `MAX_STEPS`,
`PARITY_STEPS`, `WORLD_SIZE`, `FIRST_SEED`, `RUNTIME_SEED`, and
`POPULATION_SIZES`.
