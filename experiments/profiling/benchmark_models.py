"""Repeatable, uninstrumented ECS-versus-Mesa runtime benchmark.

Example:
    uv run python -m experiments.profiling.benchmark_models --n-agents 1000 --steps 50
"""
import argparse
import contextlib
import io
import json
import random
import sys
import time
from argparse import Namespace
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import esper
import numpy as np

from experiments.parity.compare_implementations import SCENARIOS, build_benchmark_config
from experiments.metadata import collect_environment_metadata
from epidemic_sim.ecs.simulation import SIREpidemicModel
from epidemic_sim.mesa import MesaSIREpidemicModel


def run_once(framework: str, simulation_config, disease_model_config, steps: int) -> tuple[float, float]:
    """Return construction and simulation wall time for one fresh model."""
    constructor = SIREpidemicModel if framework == "ecs" else MesaSIREpidemicModel
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model = constructor(
            simulation_config=simulation_config,
            disease_model_config=disease_model_config,
        )
    construction_seconds = time.perf_counter() - start

    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.run(steps)
    simulation_seconds = time.perf_counter() - start

    if framework == "ecs":
        model.clean_up()
        esper.switch_world("default")
    return construction_seconds, simulation_seconds


def summarize_measurements(measurements: list[dict], seed: int) -> dict:
    framework_summary = {}
    for framework in ("ecs", "mesa"):
        values = np.asarray([
            row["simulation_seconds"]
            for row in measurements
            if row["framework"] == framework
        ])
        framework_summary[framework] = {
            "median_seconds": float(np.median(values)),
            "q1_seconds": float(np.percentile(values, 25)),
            "q3_seconds": float(np.percentile(values, 75)),
        }

    pairs = {}
    for row in measurements:
        pair_id = (row.get("run_id"), row["repeat"])
        pairs.setdefault(pair_id, {})[row["framework"]] = row["simulation_seconds"]
    speedups = np.asarray([
        pair["ecs"] / pair["mesa"]
        for pair in pairs.values()
        if "ecs" in pair and "mesa" in pair and pair["mesa"] > 0
    ])
    if len(speedups) == 0:
        paired_speedup = None
    else:
        rng = np.random.default_rng(seed)
        bootstrap_medians = np.asarray([
            np.median(rng.choice(speedups, size=len(speedups), replace=True))
            for _ in range(10_000)
        ])
        paired_speedup = {
            "definition": "ECS simulation time / Mesa simulation time; >1 favors Mesa",
            "n_pairs": int(len(speedups)),
            "median": float(np.median(speedups)),
            "ci_level": 0.95,
            "ci_low": float(np.percentile(bootstrap_medians, 2.5)),
            "ci_high": float(np.percentile(bootstrap_medians, 97.5)),
        }
    return {"simulation_time_seconds": framework_summary, "paired_speedup": paired_speedup}


def balanced_framework_orders(repeats: int, rng: random.Random) -> list[list[str]]:
    orders = (["ecs", "mesa"] for _ in range(repeats // 2))
    counterbalanced = list(orders) + [["mesa", "ecs"] for _ in range(repeats // 2)]
    if repeats % 2:
        counterbalanced.append(rng.choice((["ecs", "mesa"], ["mesa", "ecs"])))
    rng.shuffle(counterbalanced)
    return counterbalanced


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="hazard_similarity")
    parser.add_argument("--n-agents", type=int, default=1000)
    parser.add_argument("--world-size", type=int, default=200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=28022026)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    if args.n_agents < 1 or args.steps < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("n-agents, steps, and repeats must be positive; warmups cannot be negative")
    if args.n_agents < 51 and SCENARIOS[args.scenario].get("space_attribute_similarity", False):
        parser.error("similarity-network scenarios require at least 51 agents")

    benchmark = build_benchmark_config(
        Namespace(
            n_agents=args.n_agents,
            world_size=args.world_size,
            max_steps=args.steps,
            replicates=args.repeats,
        ),
        SCENARIOS[args.scenario],
    )
    simulation_config = replace(benchmark.simulation, collect_spatial=False)
    order_rng = random.Random(args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    for warmup in range(args.warmups):
        order = ["ecs", "mesa"]
        order_rng.shuffle(order)
        for framework in order:
            run_once(
                framework,
                replace(simulation_config, seed=args.seed - warmup - 1),
                benchmark.disease_model,
                args.steps,
            )

    measurements = []
    for repeat, order in enumerate(balanced_framework_orders(args.repeats, order_rng)):
        seed = args.seed + repeat
        for position, framework in enumerate(order):
            construction_seconds, simulation_seconds = run_once(
                framework,
                replace(simulation_config, seed=seed),
                benchmark.disease_model,
                args.steps,
            )
            measurements.append({
                "run_id": run_id,
                "framework": framework,
                "repeat": repeat,
                "seed": seed,
                "order_position": position,
                "construction_seconds": construction_seconds,
                "simulation_seconds": simulation_seconds,
                "total_seconds": construction_seconds + simulation_seconds,
            })

    summary = summarize_measurements(measurements, args.seed)
    output = {
        "kind": "runtime_benchmark",
        "config": {
            "scenario": args.scenario,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "order_seed": args.seed,
            "simulation": asdict(simulation_config),
            "disease_model": asdict(benchmark.disease_model),
        },
        "environment": collect_environment_metadata(),
        "summary": summary,
        "measurements": measurements,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"runtime_{args.scenario}_n{args.n_agents}_s{args.steps}_{run_id}.json"
    )
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    for framework in ("ecs", "mesa"):
        stats = summary["simulation_time_seconds"][framework]
        print(
            f"{framework.upper():4s} simulation median [IQR]: "
            f"{stats['median_seconds']:.4f} s [{stats['q1_seconds']:.4f}, {stats['q3_seconds']:.4f}]"
        )
    speedup = summary["paired_speedup"]
    if speedup is not None:
        print(
            f"ECS/Mesa paired median speed ratio: {speedup['median']:.2f} "
            f"[95% bootstrap CI {speedup['ci_low']:.2f}, {speedup['ci_high']:.2f}]"
        )
    print(f"Wrote raw measurements: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])