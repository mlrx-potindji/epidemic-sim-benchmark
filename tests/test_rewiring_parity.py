"""Statistical and structural checks for production network rewiring."""
import numpy as np
import pytest
from scipy.stats import chisquare, nbinom

from experiments.parity.adapters import ECSAdapter, MesaAdapter
from experiments.parity.oracle import (
    D,
    I,
    S,
    Edges,
    Params,
    State,
    check_rewire_invariants,
    exact_inclusion_probs,
    similarity_weight,
)


ADAPTERS = [MesaAdapter, ECSAdapter]


def make_rewire_state(n=7, dead_ids=()):
    index = np.arange(n)
    health = np.full(n, S, dtype=np.int8)
    health[0] = I
    health[list(dead_ids)] = D
    mobility = np.full(n, 2.0)
    mobility[health == D] = 0.0
    positions = np.column_stack(((index % 60) * 1.5, (index // 60) * 1.5))
    return State(
        pos=positions,
        age=(index * 7 + 18) % 75,
        mobility=mobility,
        base_mobility=mobility.copy(),
        immunity=np.zeros(n),
        health=health,
        vload=np.where(health == I, 800.0, 0.0),
        inf_age=np.zeros(n, dtype=np.int64),
        rec_dur=np.full(n, 12, dtype=np.int64),
        quarantined=np.zeros(n, dtype=bool),
        q_start=np.zeros(n, dtype=np.int64),
    )


def empty_edges():
    return Edges(
        src=np.empty(0, dtype=np.int64),
        dst=np.empty(0, dtype=np.int64),
        w=np.empty(0, dtype=float),
    )


def make_params(**overrides):
    values = {
        "world_size": 100.0,
        "mean_degree": 4.0,
        "rewiring_dispersion": 0.7,
        "similarity_length": 3.5,
        "age_similarity_scale": 11.0,
    }
    values.update(overrides)
    return Params(**values)


@pytest.mark.parametrize("adapter_type", ADAPTERS)
def test_rewire_invariants_raw_weights_and_active_similarity_parameters(adapter_type):
    state = make_rewire_state(dead_ids=(6,))
    params = make_params()
    with adapter_type(state, empty_edges(), params, seed=1901) as adapter:
        result = adapter.rewire(seed=7319)

    violations, expected_weights = check_rewire_invariants(
        state, result.edges, result.requested_degrees, params
    )
    assert violations == []
    assert result.realized_degrees == pytest.approx(
        np.bincount(result.edges.src, minlength=state.n)
    )
    assert np.all(result.realized_degrees <= result.requested_degrees)
    assert np.all(result.realized_degrees[state.health == D] == 0)
    assert result.edges.w == pytest.approx(expected_weights, rel=1e-12, abs=1e-12)
    assert np.all(result.edges.w < 1.0)
    assert np.all(result.edges.dst != 6)


@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize(("seed", "n_agents"), [(8141, 1800), (8142, 1800)])
def test_rewire_degree_distribution_matches_configured_negative_binomial(
    adapter_type, seed, n_agents
):
    state = make_rewire_state(n_agents)
    params = make_params(mean_degree=6.0, rewiring_dispersion=0.6)
    probability = params.rewiring_dispersion / (
        params.rewiring_dispersion + params.mean_degree
    )

    with adapter_type(state, empty_edges(), params, seed=seed - 1) as adapter:
        result = adapter.rewire(seed=seed)

    degrees = result.realized_degrees
    assert np.all(degrees <= result.requested_degrees)
    variance = params.mean_degree + params.mean_degree**2 / params.rewiring_dispersion
    mean_tolerance = 4 * np.sqrt(variance / n_agents)
    assert abs(float(degrees.mean()) - params.mean_degree) <= mean_tolerance

    bins = np.arange(12)
    observed = np.bincount(np.minimum(degrees, 12), minlength=13)
    expected_probabilities = np.concatenate(
        (
            nbinom.pmf(bins, params.rewiring_dispersion, probability),
            [nbinom.sf(11, params.rewiring_dispersion, probability)],
        )
    )
    expected = expected_probabilities * n_agents
    statistic, p_value = chisquare(observed, expected)
    assert p_value > 0.001, (
        f"seed={seed}, chi-square={statistic:.3f}, p={p_value:.4g}; "
        "bins are 0..11 and >=12"
    )


@pytest.mark.parametrize("adapter_type", ADAPTERS)
def test_weighted_sampling_inclusion_matches_successive_sampling_oracle(adapter_type):
    state = make_rewire_state(n=6)
    params = make_params(
        mean_degree=3.0,
        rewiring_dispersion=0.8,
        similarity_length=2.3,
        age_similarity_scale=7.0,
    )
    candidates = np.arange(1, state.n)
    weights = similarity_weight(
        state.pos[0],
        state.pos[candidates],
        state.age[0],
        state.age[candidates],
        params,
    )
    expected = exact_inclusion_probs(weights, 2)
    trials = 2000
    inclusion_counts = np.zeros(len(candidates), dtype=np.int64)

    with adapter_type(state, empty_edges(), params, seed=4221) as adapter:
        for trial in range(trials):
            result = adapter.rewire(seed=12000 + trial, degree_override=2)
            selected = result.edges.dst[result.edges.src == 0]
            assert selected.size == 2
            inclusion_counts[np.isin(candidates, selected)] += 1

    observed = inclusion_counts / trials
    tolerance = 5 * np.sqrt(expected * (1 - expected) / trials) + 0.005
    assert np.all(np.abs(observed - expected) <= tolerance), (
        f"adapter={adapter_type.__name__}, seed range=12000..{12000 + trials - 1}, "
        f"trials={trials}, tolerance={tolerance.tolist()}, "
        f"observed={observed.tolist()}, expected={expected.tolist()}"
    )
