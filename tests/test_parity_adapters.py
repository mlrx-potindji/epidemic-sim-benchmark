"""Production phase adapters are checked against the independent oracle."""
import numpy as np
import pytest

from experiments.parity.adapters import ECSAdapter, MesaAdapter
from experiments.parity.keyed_rng import KeyedRNG
from experiments.parity.oracle import (
    I,
    S,
    Edges,
    Params,
    State,
    movement,
    network_hazard,
    spatial_hazard,
)


@pytest.fixture
def snapshot():
    return State(
        pos=np.asarray([[1.0, 1.0], [4.0, 1.0], [20.0, 20.0], [2.0, 1.0]]),
        age=np.asarray([20, 25, 30, 40]),
        mobility=np.asarray([4.0, 2.0, 3.0, 0.0]),
        base_mobility=np.asarray([4.0, 2.0, 3.0, 0.0]),
        immunity=np.asarray([0.0, 0.2, 0.1, 0.0]),
        health=np.asarray([I, S, S, I], dtype=np.int8),
        vload=np.asarray([900.0, 0.0, 0.0, 800.0]),
        inf_age=np.asarray([3, 0, 0, 2]),
        rec_dur=np.asarray([12, 12, 12, 12]),
        quarantined=np.asarray([False, False, False, True]),
        q_start=np.asarray([0, 0, 0, 2]),
    )


@pytest.fixture
def edges():
    return Edges(
        src=np.asarray([0, 3, 3]),
        dst=np.asarray([1, 1, 2]),
        w=np.asarray([0.5, 0.75, 0.4]),
    )


@pytest.fixture
def params():
    return Params(
        world_size=100.0,
        beta_spatial=0.1,
        beta_network=0.2,
    )


def assert_states_equal(actual, expected):
    for field in expected.__dataclass_fields__:
        assert getattr(actual, field) == pytest.approx(getattr(expected, field))


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
def test_implementation_movement_matches_oracle(adapter_type, snapshot, edges, params):
    rng = KeyedRNG(101)
    expected = movement(snapshot, params, rng, step=7)

    with adapter_type(snapshot, edges, params, seed=13) as adapter:
        actual = adapter.movement(rng, step=7)

    assert_states_equal(actual, expected)


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
def test_implementation_hazards_match_oracle(adapter_type, snapshot, edges, params):
    expected_spatial = spatial_hazard(snapshot, params)
    expected_network = network_hazard(snapshot, edges, params)

    with adapter_type(snapshot, edges, params, seed=13) as adapter:
        actual = adapter.hazards()

    assert actual.spatial == pytest.approx(expected_spatial)
    assert actual.network == pytest.approx(expected_network)
