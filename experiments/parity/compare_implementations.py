"""Compare epidemiological outcomes from ECS (Esper) and Mesa over independent replicates.

    uv run python -m experiments.parity.compare_implementations --replicates 100 --n-agents 300

Writes a JSON report under ``results/`` and prints a summary table.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from dataclasses import asdict, replace
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.parity.statistical_tests import ESTIMANDS, compare_estimand, summarise
from experiments.metadata import collect_environment_metadata
from epidemic_sim.config import BenchmarkConfig, SimulationConfig
from epidemic_sim.disease_model import DiseaseModelConfig
from epidemic_sim.ecs.simulation import SIREpidemicModel
from epidemic_sim.mesa import MesaSIREpidemicModel

SCENARIOS = {
    "hazard_similarity": dict(spatial_new=True, network_new=True,
                              space_attribute_similarity=True, dispersion=0.65),
    "hazard_random": dict(spatial_new=True, network_new=True, space_attribute_similarity=False),
    "sequential_random": dict(spatial_new=False, network_new=False, space_attribute_similarity=False),
    "hazard_quarantine": dict(spatial_new=True, network_new=True, enable_quarantine=True,
                              space_attribute_similarity=True, dispersion=0.65),
}


def alpha_budgets(familywise_alpha: float, planned_scenarios: int) -> tuple[float, float]:
    if not 0 < familywise_alpha < 0.5:
        raise ValueError("familywise alpha must be between 0 and 0.5")
    if planned_scenarios < 1:
        raise ValueError("planned_scenarios must be positive")
    screening = familywise_alpha / (planned_scenarios * len(ESTIMANDS) * 4)
    equivalence = familywise_alpha / (planned_scenarios * len(ESTIMANDS) * 2)
    return screening, equivalence


def replicate(
    config: BenchmarkConfig,
    first_seed: int = 1000,
) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {
        framework: [] for framework in config.frameworks or ()
    }
    for r in range(config.n_replicates):
        sim_config = replace(config.simulation, seed=first_seed + r)
        with contextlib.redirect_stdout(io.StringIO()):
            if "ecs" in out:
                ecs = SIREpidemicModel(
                    simulation_config=sim_config,
                    disease_model_config=config.disease_model,
                )
                ecs.run(config.simulation.max_steps)
                ecs_result = summarise(ecs.time_series_data, config.simulation.n_agents)
                ecs_result["seed"] = sim_config.seed
                out["ecs"].append(ecs_result)
                ecs.clean_up()
            if "mesa" in out:
                mesa_config = replace(sim_config, seed=first_seed + 100_000 + r, collect_spatial=False)
                model = MesaSIREpidemicModel(
                    simulation_config=mesa_config,
                    disease_model_config=config.disease_model,
                )
                model.run(config.simulation.max_steps)
                mesa_result = summarise(model.time_series_data, config.simulation.n_agents)
                mesa_result["seed"] = mesa_config.seed
                out["mesa"].append(mesa_result)
    return out


def build_benchmark_config(args: argparse.Namespace, scenario: dict) -> BenchmarkConfig:
    n_agents = args.n_agents
    return BenchmarkConfig(
        simulation=SimulationConfig(
            n_agents=n_agents,
            world_size=args.world_size,
            initial_infected=max(3, n_agents // 40),
            max_steps=args.max_steps,
        ),
        disease_model=DiseaseModelConfig(
            beta_spatial=0.3,
            beta_network=0.25,
            **scenario,
        ),
        n_replicates=args.replicates,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="hazard_similarity")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=300)
    parser.add_argument("--world-size", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="family-wise alpha across all planned scenarios, estimands, and tests")
    parser.add_argument("--familywise-scenarios", type=int, default=1,
                        help="number of scenarios included in this manuscript-level analysis")
    parser.add_argument("--attack-rate-margin", type=float)
    parser.add_argument("--peak-prevalence-margin", type=float)
    parser.add_argument("--time-to-peak-margin", type=float)
    parser.add_argument("--variance-ratio-bounds", type=float, nargs=2, metavar=("LOWER", "UPPER"))
    args = parser.parse_args(argv)
    margins = {
        "attack_rate": args.attack_rate_margin,
        "peak_prevalence": args.peak_prevalence_margin,
        "time_to_peak": args.time_to_peak_margin,
    }
    provided_margins = [margin is not None for margin in margins.values()]
    try:
        screening_alpha, equivalence_alpha = alpha_budgets(args.alpha, args.familywise_scenarios)
    except ValueError as error:
        parser.error(str(error))
    if any(provided_margins) and (
        not all(provided_margins) or args.variance_ratio_bounds is None
    ):
        parser.error(
            "formal equivalence requires all three estimand margins and --variance-ratio-bounds"
        )
    if any(margin is not None and margin <= 0 for margin in margins.values()):
        parser.error("all estimand margins must be positive")
    if args.variance_ratio_bounds is not None and not (
        0 < args.variance_ratio_bounds[0] < 1 < args.variance_ratio_bounds[1]
    ):
        parser.error("variance-ratio bounds must be positive and bracket 1")

    start = time.perf_counter()
    config = build_benchmark_config(args, SCENARIOS[args.scenario])
    runs = replicate(config, args.first_seed)
    rng = np.random.default_rng(0)
    report = {
        estimand: compare_estimand(
            [row[estimand] for row in runs["ecs"]],
            [row[estimand] for row in runs["mesa"]],
            rng,
            alpha=screening_alpha,
            mean_equivalence_margin=margins[estimand],
            variance_ratio_bounds=tuple(args.variance_ratio_bounds)
            if args.variance_ratio_bounds is not None else None,
            equivalence_alpha=equivalence_alpha,
        )
        for estimand in ESTIMANDS
    }
    print(f"{args.scenario}: {config.n_replicates} replicates, N={config.simulation.n_agents}, "
          f"{time.perf_counter() - start:.0f}s")
    print(
        f"Family-wise alpha={args.alpha:g}; Bonferroni screening threshold "
        f"per test={screening_alpha:.4g} across "
        f"{args.familywise_scenarios * len(ESTIMANDS) * 4} planned criteria."
    )
    if all(provided_margins):
        print(
            f"Equivalence alpha per component={equivalence_alpha:.4g}; "
            "margins are researcher-specified."
        )
    print(f"{'estimand':16s} {'ECS mean (sd)':>16s} {'Mesa mean (sd)':>16s} "
          f"{'Welch p':>8s} {'KS p':>7s} {'BF p':>7s} {'var ratio [CI]':>24s} "
          f"{'screen':>8s} {'equiv.':>9s}")
    for m, r in report.items():
        def show_p(value: float | None) -> str:
            return f"{value:.3f}" if value is not None else "NA"

        if r["variance_ratio"] is None or r["variance_ratio_low"] is None:
            variance_summary = "not estimable"
        else:
            variance_summary = (
                f"{r['variance_ratio']:.2g} "
                f"[{r['variance_ratio_low']:.2g}, {r['variance_ratio_high']:.2g}]"
            )
        equivalent = r["equivalent"]
        equivalence_status = "not run" if equivalent is None else ("PASS" if equivalent else "FAIL")
        print(f"{m:16s} {r['mean_ecs']:8.3f} ({r['sd_ecs']:.3f}) {r['mean_mesa']:8.3f} ({r['sd_mesa']:.3f}) "
              f"{show_p(r['p_mean_welch']):>8s} {show_p(r['p_ks']):>7s} "
              f"{show_p(r['p_variance_brown_forsythe']):>7s} {variance_summary:>24s} "
              f"{'NO REJECT' if r['parity'] else 'REJECT':>8s} {equivalence_status:>9s}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (output_dir / f"epi_parity_{args.scenario}_{run_id}.json").write_text(
        json.dumps({"kind": "epidemiological_parity", "args": vars(args),
                    "resolved_config": {
                        "simulation": asdict(config.simulation),
                        "disease_model": asdict(config.disease_model),
                    },
                    "analysis": {
                        "familywise_alpha": args.alpha,
                        "familywise_scenarios": args.familywise_scenarios,
                        "screening_alpha_per_test": screening_alpha,
                        "screening_criteria_per_estimand": 4,
                        "estimands": list(ESTIMANDS),
                        "screening_correction": "Bonferroni over all estimand-test pairs",
                        "equivalence_alpha_per_component": equivalence_alpha,
                        "equivalence_margins": margins,
                        "variance_ratio_bounds": args.variance_ratio_bounds,
                        "equivalence_enabled": all(provided_margins),
                        "environment": collect_environment_metadata(),
                    },
                    "report": report, "runs": runs}, default=float, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
