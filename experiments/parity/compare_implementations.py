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
from dataclasses import replace
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from experiments.parity.statistical_tests import ESTIMANDS, compare_estimand, summarise
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
                out["ecs"].append(summarise(ecs.time_series_data, config.simulation.n_agents))
                ecs.clean_up()
            if "mesa" in out:
                mesa_config = replace(sim_config, seed=first_seed + 100_000 + r, collect_spatial=False)
                model = MesaSIREpidemicModel(
                    simulation_config=mesa_config,
                    disease_model_config=config.disease_model,
                )
                model.run(config.simulation.max_steps)
                out["mesa"].append(summarise(model.time_series_data, config.simulation.n_agents))
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
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = build_benchmark_config(args, SCENARIOS[args.scenario])
    runs = replicate(config, args.first_seed)
    rng = np.random.default_rng(0)
    report = {m: compare_estimand([r[m] for r in runs["ecs"]], [r[m] for r in runs["mesa"]], rng)
              for m in ESTIMANDS}
    print(f"{args.scenario}: {config.n_replicates} replicates, N={config.simulation.n_agents}, "
          f"{time.perf_counter() - start:.0f}s")
    print(f"{'estimand':16s} {'ECS mean (sd)':>16s} {'Mesa mean (sd)':>16s} "
          f"{'Welch p':>8s} {'KS p':>7s} {'BF p':>7s} {'var ratio [95% CI]':>24s}  parity")
    for m, r in report.items():
        print(f"{m:16s} {r['mean_ecs']:8.3f} ({r['sd_ecs']:.3f}) {r['mean_mesa']:8.3f} ({r['sd_mesa']:.3f}) "
              f"{r['p_mean_welch']:8.3f} {r['p_ks']:7.3f} {r['p_variance_brown_forsythe']:7.3f} "
              f"{r['variance_ratio']:8.2f} [{r['variance_ratio_low']:.2f}, {r['variance_ratio_high']:.2f}]"
              f"  {'OK' if r['parity'] else 'DIFFERENT'}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"epi_parity_{args.scenario}.json").write_text(
        json.dumps({"args": vars(args), "report": report, "runs": runs}, default=float, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
