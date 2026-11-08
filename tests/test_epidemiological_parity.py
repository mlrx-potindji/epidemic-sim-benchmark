"""Tests for the pilot-based epidemiological equivalence protocol."""
from argparse import Namespace

import numpy as np
import pytest

from experiments.parity.compare_implementations import build_benchmark_config
from experiments.parity.statistical_tests import (
    ESTIMANDS,
    analyze_outcomes,
    newcombe_difference_ci,
    pooled_sd,
    scale_equivalence,
    summarise,
    welch_tost,
)


SCENARIO = dict(spatial_new=True, network_new=True,
                space_attribute_similarity=True, dispersion=0.65)


def test_benchmark_runner_uses_shared_benchmark_config():
    config = build_benchmark_config(
        Namespace(n_agents=200, world_size=150, max_steps=300, replicates=12),
        SCENARIO,
    )
    assert config.n_replicates == 12
    assert config.simulation.n_agents == 200
    assert config.disease_model.spatial_new
    assert config.frameworks == ["ecs", "mesa"]


def test_summarise_gives_the_three_estimands():
    series = {"susceptible": [8, 5, 4, 4], "infected": [2, 4, 5, 3], "recovered": [0, 1, 1, 3],
              "death": [0, 0, 0, 0]}
    assert summarise(series, 10) == {
        "attack_rate": 0.6,
        "peak_prevalence": 0.5,
        "time_to_peak": 2.0,
    }


def test_pooled_pilot_sd_and_welch_tost():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 1.1, 2.1, 3.1])
    assert pooled_sd(x, y) == pytest.approx(np.sqrt(5 / 3))
    result = welch_tost(x, y, margin=2.0)
    assert result["difference"] == pytest.approx(0.1)
    assert result["ci"][0] < result["ci"][1]
    assert result["equivalent"]


def test_newcombe_interval_and_major_outbreak_margin_scale():
    difference, ci = newcombe_difference_ci(60, 100, 62, 100)
    assert difference == pytest.approx(0.02)
    assert ci[0] < difference < ci[1]


def test_scale_equivalence_uses_bootstrap_ci_and_reports_method():
    rng = np.random.default_rng(42)
    x = rng.normal(size=300)
    y = rng.normal(0.01, 1.02, size=300)
    result = scale_equivalence(
        x, y, (0.5, 2.0), "variance", n_resamples=199, seed=7
    )
    assert result["ratio"] > 0
    assert result["ci"][0] < result["ci"][1]
    assert result["method"] in {"BCa", "percentile"}
    assert result["equivalent"] is True


def test_constant_scale_is_inconclusive():
    result = scale_equivalence([4.0] * 20, [4.0] * 20, (0.5, 2.0), "variance")
    assert result == {
        "ratio": None,
        "ci": None,
        "method": None,
        "equivalent": None,
    }


def test_analysis_reports_all_prespecified_estimands_and_holm_screens():
    rng = np.random.default_rng(123)
    ecs, mesa = [], []
    for _ in range(120):
        x = {
            "attack_rate": float(rng.normal(0.55, 0.08)),
            "peak_prevalence": float(rng.normal(0.25, 0.04)),
            "time_to_peak": float(round(rng.normal(16, 3))),
        }
        ecs.append(x)
        mesa.append({
            "attack_rate": x["attack_rate"] + float(rng.normal(0, 0.003)),
            "peak_prevalence": x["peak_prevalence"] + float(rng.normal(0, 0.002)),
            "time_to_peak": x["time_to_peak"] + float(rng.choice([-1, 0, 1])),
        })
    result = analyze_outcomes(
        ecs,
        mesa,
        {"attack_rate": 0.08, "peak_prevalence": 0.04, "time_to_peak": 3.0},
        n_agents=300,
        n_seed=7,
        n_boot=99,
        n_permutations=49,
        seed=5,
    )

    assert tuple(result["estimands"]) == ESTIMANDS
    assert all(
        result["estimands"][name]["mean_equivalence"]["margin"] == pytest.approx(
            0.8 * {"attack_rate": 0.08, "peak_prevalence": 0.04,
                   "time_to_peak": 3.0}[name]
        )
        for name in ESTIMANDS
    )
    assert len(result["difference_screens"]) == 9
    assert all("holm_p_value" in row for row in result["difference_screens"])
    assert set(result["major_outbreak"]["sensitivity"]) == {"3", "10"}
