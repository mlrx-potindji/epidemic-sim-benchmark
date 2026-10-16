"""Epidemiological parity: distribution of the epidemic indicators of ECS vs Mesa.

Estimands, tests and acceptance rule are documented in
``experiments.parity.statistical_tests``.
Both implementations are run on independent seeds; the means, the whole distributions
and the variances of the three estimands must agree.
"""
from argparse import Namespace

import numpy as np
import pytest

from experiments.parity.compare_implementations import alpha_budgets, build_benchmark_config, replicate
from experiments.parity.statistical_tests import ESTIMANDS, compare_estimand, summarise

# Regression guard run on 4 scenarios x 3 estimands x 3 tests = 36 comparisons: the error level of
# each is lowered to 0.001 (family-wise false-alarm rate about 3 %). The report script
# compare_implementations.py uses alpha = 0.01 with many more replicates.
ALPHA = 0.001

SCENARIOS = {
    "hazard_similarity": dict(spatial_new=True, network_new=True,
                              space_attribute_similarity=True, dispersion=0.65),
    "hazard_random": dict(spatial_new=True, network_new=True, space_attribute_similarity=False),
    "sequential_random": dict(spatial_new=False, network_new=False, space_attribute_similarity=False),
    "hazard_quarantine": dict(spatial_new=True, network_new=True, enable_quarantine=True,
                              space_attribute_similarity=True, dispersion=0.65),
}


def run_replicates(scenario: dict, replicates: int, n_agents: int = 200, world_size: int = 150):
    config = build_benchmark_config(
        Namespace(
            n_agents=n_agents,
            world_size=world_size,
            max_steps=300,
            replicates=replicates,
        ),
        scenario,
    )
    return replicate(config, first_seed=500)


def test_benchmark_runner_uses_shared_benchmark_config():
    config = build_benchmark_config(
        Namespace(n_agents=200, world_size=150, max_steps=300, replicates=12),
        SCENARIOS["hazard_similarity"],
    )
    assert config.n_replicates == 12
    assert config.simulation.n_agents == 200
    assert config.disease_model.spatial_new
    assert config.frameworks == ["ecs", "mesa"]


def test_familywise_alpha_is_allocated_across_planned_scenarios():
    screening, equivalence = alpha_budgets(0.05, planned_scenarios=4)

    assert screening == pytest.approx(0.05 / (4 * 3 * 4))
    assert equivalence == pytest.approx(0.05 / (4 * 3 * 2))


def test_summarise_gives_the_three_estimands():
    series = {"susceptible": [8, 5, 4, 4], "infected": [2, 4, 5, 3], "recovered": [0, 1, 1, 3],
              "death": [0, 0, 0, 0]}
    assert summarise(series, 10) == {"attack_rate": 0.6, "peak_prevalence": 0.5, "time_to_peak": 2.0}


def test_comparison_detects_a_shift_and_a_variance_change():
    rng = np.random.default_rng(0)
    same = compare_estimand(rng.normal(0, 1, 200), rng.normal(0, 1, 200), rng)
    assert same["parity"]
    shifted = compare_estimand(rng.normal(0, 1, 200), rng.normal(1, 1, 200), rng)
    assert not shifted["parity"] and shifted["p_mean_welch"] < 0.01
    wider = compare_estimand(rng.normal(0, 1, 200), rng.normal(0, 2, 200), rng)
    assert not wider["parity"] and wider["variance_ratio_low"] > 1


def test_equivalence_requires_mean_and_variance_intervals_within_margins():
    rng = np.random.default_rng(31)
    ecs = rng.normal(0.4, 0.08, 400)
    mesa = ecs + rng.normal(0.005, 0.002, 400)
    result = compare_estimand(
        ecs,
        mesa,
        np.random.default_rng(32),
        alpha=0.025,
        mean_equivalence_margin=0.03,
        variance_ratio_bounds=(0.8, 1.25),
    )
    assert result["mean_equivalent"]
    assert result["variance_equivalent"]
    assert result["equivalent"]
    assert result["mean_equivalence_ci_low"] < result["mean_equivalence_ci_high"]


def test_constant_estimands_are_reported_as_inconclusive():
    result = compare_estimand(
        [4.0] * 20,
        [4.0] * 20,
        np.random.default_rng(33),
        mean_equivalence_margin=1.0,
        variance_ratio_bounds=(0.8, 1.25),
    )

    assert result["variance_ratio"] is None
    assert result["variance_equivalent"] is None
    assert result["equivalent"] is None
    assert result["parity"] is False


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_epidemic_indicators_have_the_same_mean_and_variance(name):
    runs = run_replicates(SCENARIOS[name], replicates=60)
    rng = np.random.default_rng(0)
    for estimand in ESTIMANDS:
        result = compare_estimand([r[estimand] for r in runs["ecs"]],
                                  [r[estimand] for r in runs["mesa"]], rng, alpha=ALPHA)
        assert result["parity"], f"{name}/{estimand}: {result}"
