"""Run pilot and confirmatory ECS-versus-Mesa epidemiological comparisons.

    uv run python -m experiments.parity.compare_implementations --scenario hazard_similarity

The pilot fixes standardized mean margins; only fresh confirmatory replicates
are used for equivalence conclusions.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from experiments.metadata import collect_environment_metadata
from experiments.parity.statistical_tests import ESTIMANDS, analyze_outcomes, pooled_sd, summarise
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


def replicate(
    config: BenchmarkConfig,
    first_seed: int = 1000,
) -> dict[str, list[dict[str, float]]]:
    """Run independent implementation replicates from a contiguous seed block."""
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
                mesa_config = replace(sim_config, seed=first_seed + 100_000 + r,
                                      collect_spatial=False)
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
        output_dir=getattr(args, "output_dir", "results"),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="hazard_similarity")
    parser.add_argument("--pilot-replicates", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=250,
                        help="fresh confirmatory replicates per implementation")
    parser.add_argument("--n-agents", type=int, default=300)
    parser.add_argument("--world-size", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="one-sided TOST alpha; also sets two-sided confidence levels")
    parser.add_argument("--mean-margin-sd", type=float, default=0.8,
                        help="mean equivalence margin as a multiple of pooled pilot SD")
    parser.add_argument("--major-outbreak-margin", type=float, default=0.10)
    parser.add_argument("--scale-ratio-bounds", type=float, nargs=2, default=(0.5, 2.0),
                        metavar=("LOWER", "UPPER"))
    parser.add_argument("--bootstrap-resamples", type=int, default=9999)
    parser.add_argument("--permutations", type=int, default=4999)
    args = parser.parse_args(argv)
    if args.pilot_replicates < 2 or args.replicates < 2:
        parser.error("pilot and confirmatory runs need at least two replicates per arm")
    if args.pilot_replicates > 100_000 or args.replicates > 100_000:
        parser.error("each seed block supports at most 100,000 replicates")
    if not 0 < args.alpha < 0.5:
        parser.error("--alpha must be between 0 and 0.5")
    if args.mean_margin_sd <= 0 or args.major_outbreak_margin <= 0:
        parser.error("equivalence margins must be positive")
    if not 0 < args.scale_ratio_bounds[0] < 1 < args.scale_ratio_bounds[1]:
        parser.error("--scale-ratio-bounds must be positive and bracket 1")
    if args.bootstrap_resamples < 1 or args.permutations < 1:
        parser.error("bootstrap resamples and permutations must be positive")

    start = time.perf_counter()
    pilot_args = argparse.Namespace(**{
        **vars(args),
        "replicates": args.pilot_replicates,
    })
    pilot_config = build_benchmark_config(pilot_args, SCENARIOS[args.scenario])
    confirmatory_config = build_benchmark_config(args, SCENARIOS[args.scenario])
    scenario_seed = args.first_seed + sorted(SCENARIOS).index(args.scenario) * 400_000
    pilot_runs = replicate(pilot_config, scenario_seed)
    confirmatory_runs = replicate(confirmatory_config, scenario_seed + 200_000)
    pilot_sd = {
        estimand: pooled_sd(
            [row[estimand] for row in pilot_runs["ecs"]],
            [row[estimand] for row in pilot_runs["mesa"]],
        )
        for estimand in ESTIMANDS
    }
    report = analyze_outcomes(
        confirmatory_runs["ecs"],
        confirmatory_runs["mesa"],
        pilot_sd,
        args.n_agents,
        max(3, args.n_agents // 40),
        alpha=args.alpha,
        margin_sd=args.mean_margin_sd,
        outbreak_margin=args.major_outbreak_margin,
        scale_bounds=tuple(args.scale_ratio_bounds),
        n_boot=args.bootstrap_resamples,
        n_permutations=args.permutations,
    )
    print(f"{args.scenario}: pilot={pilot_config.n_replicates}, "
          f"confirmatory={confirmatory_config.n_replicates}, N={confirmatory_config.simulation.n_agents}, "
          f"{time.perf_counter() - start:.0f}s")
    print(f"TOST alpha={args.alpha:g}; mean margins={args.mean_margin_sd:g} x pooled pilot SD; "
          f"scale-ratio bounds={tuple(args.scale_ratio_bounds)}.")
    print(f"{'estimand':18s} {'ECS mean (sd)':>20s} {'Mesa mean (sd)':>20s} "
          f"{'mean CI':>23s} {'margin':>10s} {'mean eq.':>10s}")
    for key in ESTIMANDS:
        item = report["estimands"][key]
        mean = item["mean_equivalence"]
        print(f"{key:18s} {item['ecs']['mean']:8.3f} ({item['ecs']['sd']:.3f}) "
              f"{item['mesa']['mean']:8.3f} ({item['mesa']['sd']:.3f}) "
              f"[{mean['ci'][0]:.4g}, {mean['ci'][1]:.4g}] "
              f"{mean['margin']:.4g} {mean['equivalent']!s:>10s}")
        for kind, scale in item["scale_equivalence"].items():
            if scale["ci"] is None:
                interval = "undecidable"
            else:
                interval = f"{scale['ratio']:.4g} [{scale['ci'][0]:.4g}, {scale['ci'][1]:.4g}]"
            print(f"  {kind:18s} {interval}; method={scale['method']}; "
                  f"equivalent={scale['equivalent']}")
    major = report["major_outbreak"]
    print(f"Major-outbreak probability difference: {major['difference']:.4g} "
          f"(CI {major['ci']}, margin ±{args.major_outbreak_margin:g}); "
          f"equivalent={major['equivalent']}")
    for k, sensitivity in major["sensitivity"].items():
        print(f"  outbreak threshold k={k}: difference={sensitivity['difference']:.4g}, "
              f"CI={sensitivity['ci']}")
    if report["conditional_major"] is None:
        print("Conditional major-outbreak means: not analyzed (fewer than 30 major outbreaks in an arm).")
    else:
        print("Conditional major-outbreak mean equivalence:")
        for key, result in report["conditional_major"].items():
            print(f"  {key:16s} difference={result['difference']:.4g}, CI={result['ci']}, "
                  f"margin={result['margin']:.4g}, equivalent={result['equivalent']}")
    print(f"All primary equivalence criteria passed: "
          f"{report['summary']['all_primary_equivalent']}")
    print("Descriptive difference screens (Holm-adjusted within scenario):")
    for row in report["difference_screens"]:
        print(f"  {row['estimand']:16s} {row['test']:14s} "
              f"p={row['p_value']:.4g}, Holm p={row['holm_p_value']:.4g}")

    output_dir = Path(confirmatory_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_args = vars(args).copy()
    report_args["output_dir"] = str(args.output_dir)
    (output_dir / f"epi_parity_{args.scenario}_{run_id}.json").write_text(
        json.dumps({
            "kind": "epidemiological_parity",
            "args": report_args,
            "resolved_config": {
                "simulation": asdict(confirmatory_config.simulation),
                "disease_model": asdict(confirmatory_config.disease_model),
            },
            "analysis": {
                "protocol_version": 4,
                "alpha": args.alpha,
                "estimands": list(ESTIMANDS),
                "mean_margin_sd": args.mean_margin_sd,
                "major_outbreak_margin": args.major_outbreak_margin,
                "scale_ratio_bounds": args.scale_ratio_bounds,
                "bootstrap_resamples": args.bootstrap_resamples,
                "permutations": args.permutations,
                "pilot_replicates_per_arm": args.pilot_replicates,
                "confirmatory_replicates_per_arm": args.replicates,
                "scenario_seed_block_start": scenario_seed,
                "pilot_sd": pilot_sd,
                "difference_screen_correction": "Holm within scenario across 9 screens",
                "conditional_major_threshold_per_arm": 30,
                "environment": collect_environment_metadata(),
            },
            "report": report,
            "pilot_runs": pilot_runs,
            "runs": confirmatory_runs,
        }, default=float, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
