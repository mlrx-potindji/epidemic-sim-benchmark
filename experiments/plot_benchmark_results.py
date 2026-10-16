"""Create comparison plots from runtime benchmark and parity JSON reports.

Example:
    uv run python -m experiments.plot_benchmark_results \
        --runtime results/runtime_*.json --parity results/epi_parity_*.json
"""
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FRAMEWORKS = ("ecs", "mesa")
COLORS = {"ecs": "#167d8d", "mesa": "#d17a22"}
ESTIMANDS = ("attack_rate", "peak_prevalence", "time_to_peak")


def load_reports(paths: list[Path], kind: str) -> list[tuple[Path, dict[str, Any]]]:
    reports = []
    for path in paths:
        report = json.loads(path.read_text())
        is_parity_report = kind == "epidemiological_parity" and all(
            key in report for key in ("args", "report", "runs")
        )
        if report.get("kind") != kind and not is_parity_report:
            raise ValueError(f"{path} is not a {kind} report")
        reports.append((path, report))
    return reports


def grouped_boxplot(axis, datasets, centers, width: float = 0.28) -> None:
    for framework, offset in (("ecs", -width / 1.7), ("mesa", width / 1.7)):
        values = datasets[framework]
        positions = [center + offset for center in centers]
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#20252b", "linewidth": 1.4},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(COLORS[framework])
            box.set_alpha(0.72)


def group_runtime_reports(reports: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Any]] = {}
    for path, report in reports:
        config = report["config"]
        simulation = config.get("simulation", config)
        environment = report.get("environment", {})
        key = (
            config["scenario"],
            simulation["n_agents"],
            simulation["world_size"],
            config.get("steps", simulation.get("max_steps")),
            simulation.get("collect_spatial", config.get("collect_spatial")),
            json.dumps(config.get("disease_model", {}), sort_keys=True),
            environment.get("platform"),
            environment.get("python"),
            environment.get("cpu_model"),
            environment.get("logical_cpu_count", environment.get("cpu_count")),
            environment.get("available_cpu_count"),
            json.dumps(environment.get("packages", {}), sort_keys=True),
            config.get("benchmark_protocol_version", "legacy"),
        )
        group = groups.setdefault(key, {"key": key, "measurements": [], "files": []})
        group["measurements"].extend(report["measurements"])
        group["files"].append(path)
    return [
        groups[key]
        for key in sorted(
            groups,
            key=lambda item: (item[0], item[1], item[2], item[3], str(item[4]), *map(str, item[5:])),
        )
    ]


def group_parity_reports(reports: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Any]] = {}
    def report_priority(item: tuple[Path, dict[str, Any]]) -> tuple[bool, float, str]:
        path, report = item
        has_seeds = any(
            "seed" in outcome
            for framework_runs in report.get("runs", {}).values()
            for outcome in framework_runs
        )
        modified = path.stat().st_mtime if path.exists() else 0.0
        return has_seeds, modified, path.name

    for path, report in sorted(reports, key=report_priority, reverse=True):
        args = report["args"]
        key = (
            args["scenario"],
            args["n_agents"],
            args.get("world_size"),
            args.get("max_steps"),
        )
        group = groups.setdefault(
            key,
            {
                "key": key,
                "runs": {framework: [] for framework in FRAMEWORKS},
                "seen_seeds": {framework: set() for framework in FRAMEWORKS},
                "files": [],
            },
        )
        for framework in FRAMEWORKS:
            seed_offset = 100_000 if framework == "mesa" else 0
            for index, outcome in enumerate(report["runs"][framework]):
                seed = outcome.get("seed", args.get("first_seed", 1000) + seed_offset + index)
                if seed not in group["seen_seeds"][framework]:
                    group["runs"][framework].append(outcome)
                    group["seen_seeds"][framework].add(seed)
        group["files"].append(path)
    return [groups[key] for key in sorted(groups, key=lambda item: item[:4] + item[4:])]


def plot_runtime(reports: list[tuple[Path, dict[str, Any]]], output: Path) -> None:
    groups = group_runtime_reports(reports)
    scenario_names = {
        "hazard_similarity": "sim",
        "hazard_quarantine": "quar",
        "hazard_random": "rand",
        "sequential_random": "seq",
    }
    labels = []
    for group in groups:
        scenario = scenario_names.get(group["key"][0], group["key"][0])
        repeats = sum(row["framework"] == "ecs" for row in group["measurements"])
        version_label = group["key"][-1]
        protocol = "legacy" if version_label == "legacy" else f"v{version_label}"
        labels.append(
            f"{scenario} {protocol}\nN{group['key'][1]}, {group['key'][3]}s\n{repeats} reps"
        )
    centers = list(range(1, len(groups) + 1))
    figure, axes = plt.subplots(1, 2, figsize=(max(10, 2.7 * len(groups)), 5.5))
    for axis, field, title in (
        (axes[0], "simulation_seconds", "Simulation time per run (seconds)"),
        (axes[1], "construction_seconds", "Model construction time (seconds)"),
    ):
        datasets = {
            framework: [
                [row[field] for row in group["measurements"] if row["framework"] == framework]
                for group in groups
            ]
            for framework in FRAMEWORKS
        }
        grouped_boxplot(axis, datasets, centers)
        axis.set_title(title)
        axis.set_ylabel("Seconds (log scale)")
        axis.set_yscale("log")
        axis.set_xticks(centers, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.legend(
        handles=[Patch(facecolor=COLORS[name], label=name.upper()) for name in FRAMEWORKS],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92),
    )
    figure.suptitle("ECS and Mesa runtime comparison", y=0.99, fontsize=14)
    figure.subplots_adjust(top=0.78, bottom=0.31, wspace=0.25)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_parity(reports: list[tuple[Path, dict[str, Any]]], output: Path) -> None:
    groups = group_parity_reports(reports)
    labels = [
        f"{group['key'][0]}\nN={group['key'][1]}, n={len(group['runs']['ecs'])}"
        for group in groups
    ]
    centers = list(range(1, len(groups) + 1))
    figure, axes = plt.subplots(1, 3, figsize=(max(12, 3.4 * len(groups)), 5.5))
    for axis, estimand in zip(axes, ESTIMANDS):
        datasets = {
            framework: [
                [row[estimand] for row in group["runs"][framework]]
                for group in groups
            ]
            for framework in FRAMEWORKS
        }
        grouped_boxplot(axis, datasets, centers)
        axis.set_title(estimand.replace("_", " ").title())
        axis.set_xticks(centers, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.legend(
        handles=[Patch(facecolor=COLORS[name], label=name.upper()) for name in FRAMEWORKS],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92),
    )
    figure.suptitle("Epidemiological outcome distributions", y=0.99, fontsize=14)
    figure.subplots_adjust(top=0.78, bottom=0.30, wspace=0.25)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", nargs="+", type=Path, default=[])
    parser.add_argument("--parity", nargs="+", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not args.runtime and not args.parity:
        parser.error("provide at least one --runtime or --parity JSON file")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.runtime:
        plot_runtime(
            load_reports(args.runtime, "runtime_benchmark"),
            args.output_dir / "runtime_comparison.png",
        )
        print(f"Wrote {args.output_dir / 'runtime_comparison.png'}")
    if args.parity:
        plot_parity(
            load_reports(args.parity, "epidemiological_parity"),
            args.output_dir / "epidemiological_parity.png",
        )
        print(f"Wrote {args.output_dir / 'epidemiological_parity.png'}")


if __name__ == "__main__":
    main()