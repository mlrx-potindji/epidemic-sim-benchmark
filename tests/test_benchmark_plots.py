from pathlib import Path

from experiments.plot_benchmark_results import (
    group_parity_reports,
    group_runtime_reports,
    plot_parity,
    plot_runtime,
)


def test_runtime_reports_pool_by_workload_and_sort_population_numerically():
    base_config = {
        "scenario": "hazard_similarity",
        "world_size": 200,
        "steps": 50,
        "collect_spatial": False,
    }
    environment = {"platform": "linux", "processor": "x86_64"}
    reports = []
    for filename, n_agents, seconds in (
        ("n1000-a.json", 1000, 1.0),
        ("n300-a.json", 300, 0.3),
        ("n300-b.json", 300, 0.4),
    ):
        reports.append((
            Path(filename),
            {
                "config": {**base_config, "n_agents": n_agents},
                "environment": environment,
                "measurements": [
                    {"framework": framework, "simulation_seconds": seconds}
                    for framework in ("ecs", "mesa")
                ],
            },
        ))

    groups = group_runtime_reports(reports)

    assert [group["key"][1] for group in groups] == [300, 1000]
    assert len(groups[0]["files"]) == 2
    assert len(groups[0]["measurements"]) == 4


def test_runtime_report_accepts_resolved_nested_config():
    report = {
        "config": {
            "scenario": "hazard_similarity",
            "simulation": {"n_agents": 300, "world_size": 200, "max_steps": 50,
                           "collect_spatial": False},
            "disease_model": {"beta_spatial": 0.3},
        },
        "environment": {"platform": "linux", "cpu_model": "test CPU", "packages": {"mesa": "3.x"}},
        "measurements": [],
    }

    groups = group_runtime_reports([(Path("resolved.json"), report)])

    assert len(groups) == 1
    assert groups[0]["key"][1:4] == (300, 200, 50)


def test_runtime_reports_from_different_protocol_versions_are_not_pooled():
    reports = []
    for version in (None, 2):
        config = {
            "scenario": "hazard_similarity",
            "n_agents": 300,
            "world_size": 200,
            "steps": 50,
            "collect_spatial": False,
        }
        if version is not None:
            config["benchmark_protocol_version"] = version
        reports.append((
            Path(f"protocol-{version}.json"),
            {
                "config": config,
                "environment": {"platform": "linux"},
                "measurements": [],
            },
        ))

    assert len(group_runtime_reports(reports)) == 2


def test_parity_reports_pool_replicates_and_both_plots_render(tmp_path):
    reports = []
    for filename, scenario, offset, first_seed in (
        ("similarity-a.json", "hazard_similarity", 0.0, 500),
        ("similarity-b.json", "hazard_similarity", 0.1, 501),
        ("similarity-duplicate.json", "hazard_similarity", 0.5, 500),
        ("quarantine.json", "hazard_quarantine", 0.2, 500),
    ):
        report = {
            "args": {"scenario": scenario, "n_agents": 300, "world_size": 200,
                     "max_steps": 300, "first_seed": first_seed},
            "runs": {
                framework: [
                    {
                        "attack_rate": 0.3 + offset,
                        "peak_prevalence": 0.2 + offset,
                        "time_to_peak": 10 + offset,
                    }
                ]
                for framework in ("ecs", "mesa")
            },
        }
        reports.append((Path(filename), report))

    groups = group_parity_reports(reports)
    assert [group["key"][0] for group in groups] == ["hazard_quarantine", "hazard_similarity"]
    assert len(groups[1]["runs"]["ecs"]) == 2

    runtime_reports = [
        (
            Path("runtime.json"),
            {
                "config": {
                    "scenario": "hazard_similarity",
                    "n_agents": 300,
                    "world_size": 200,
                    "steps": 50,
                    "collect_spatial": False,
                },
                "environment": {"platform": "linux", "processor": "x86_64"},
                "measurements": [
                    {"framework": framework, "simulation_seconds": 0.5, "construction_seconds": 0.1}
                    for framework in ("ecs", "mesa")
                ],
            },
        )
    ]
    plot_runtime(runtime_reports, tmp_path / "runtime.png")
    plot_parity(reports, tmp_path / "parity.png")
    assert (tmp_path / "runtime.png").stat().st_size > 0
    assert (tmp_path / "parity.png").stat().st_size > 0