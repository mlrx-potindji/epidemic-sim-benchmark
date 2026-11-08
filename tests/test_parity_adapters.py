"""Production phase adapters are checked against the independent oracle."""
import esper
import numpy as np
import pytest

from experiments.parity.adapters import ECSAdapter, MesaAdapter
from experiments.parity.keyed_rng import KeyedRNG
from epidemic_sim.ecs.components import Quarantined
from experiments.parity.oracle import (
    D,
    I,
    R,
    S,
    Edges,
    Params,
    State,
    movement,
    network_hazard,
    progress_disease,
    quarantine_stage,
    resolve_infection,
    spatial_hazard,
)


@pytest.fixture
def snapshot():
    return State(
        pos=np.asarray([[1.0, 1.0], [5.0, 1.0], [20.0, 20.0], [2.0, 1.0]]),
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


def assert_states_equal(phase, actual, expected):
    for field in expected.__dataclass_fields__:
        actual_values = np.asarray(getattr(actual, field))
        expected_values = np.asarray(getattr(expected, field))
        equal = np.isclose(actual_values, expected_values, rtol=1e-12, atol=1e-12)
        if equal.ndim > 1:
            equal = equal.reshape((equal.shape[0], -1)).all(axis=1)
        if not np.all(equal):
            ids = np.flatnonzero(~equal).tolist()
            pytest.fail(
                f"first divergent phase: {phase}; field {field}; agent IDs {ids}; "
                f"actual={actual_values[~equal]!r}; expected={expected_values[~equal]!r}"
            )


def assert_phase_array_equal(phase, field, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    equal = np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
    ids = np.flatnonzero(~equal).tolist()
    if ids:
        pytest.fail(
            f"first divergent phase: {phase}; field {field}; agent IDs {ids}; "
            f"actual={actual[~equal]!r}; expected={expected[~equal]!r}"
        )


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
def test_implementation_movement_matches_oracle(adapter_type, snapshot, edges, params):
    rng = KeyedRNG(101)
    expected = movement(snapshot, params, rng, step=7)

    with adapter_type(snapshot, edges, params, seed=13) as adapter:
        actual = adapter.movement(rng, step=7)

    assert_states_equal("movement", actual, expected)


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
def test_implementation_hazards_match_oracle(adapter_type, snapshot, edges, params):
    expected_spatial = spatial_hazard(snapshot, params)
    expected_network = network_hazard(snapshot, edges, params)

    with adapter_type(snapshot, edges, params, seed=13) as adapter:
        actual = adapter.hazards()

    assert_phase_array_equal("hazards", "spatial", actual.spatial, expected_spatial)
    assert_phase_array_equal("hazards", "network", actual.network, expected_network)


class ScriptedRNG:
    def __init__(self, **values):
        self.values = values

    def uniform(self, purpose, step, ids):
        return np.full(np.asarray(ids).shape, self.values.get(purpose, 0.0))

    def normal(self, purpose, step, ids):
        return np.full(np.asarray(ids).shape, self.values.get(purpose, 0.0))


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
@pytest.mark.parametrize(
    ("infection_draw", "expected_health"),
    [(1.0, [I, S, S, I]), (0.0, [I, I, I, I])],
)
def test_infection_resolution_matches_oracle(
    adapter_type, snapshot, edges, params, infection_draw, expected_health
):
    spatial = np.asarray([0.0, 0.2, 0.5, 0.0])
    network = np.zeros(snapshot.n)
    rng = ScriptedRNG(
        infect=infection_draw,
        vload=0.4,
        dur=-3.0,
    )
    expected, expected_probability = resolve_infection(
        snapshot, spatial, network, params, rng, step=7
    )

    with adapter_type(snapshot, edges, params, seed=13) as adapter:
        actual, actual_probability = adapter.resolve_infections(
            spatial, network, rng, step=7
        )

    assert_states_equal("infection resolution", actual, expected)
    assert_phase_array_equal(
        "infection resolution", "probability", actual_probability, expected_probability
    )
    assert actual.health.tolist() == expected_health
    if infection_draw == 0.0:
        assert actual.rec_dur[1:3].tolist() == [1, 1]


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
def test_infection_resolution_uses_strict_probability_boundary(
    adapter_type, snapshot, edges, params
):
    spatial = np.asarray([0.0, 0.2, 0.0, 0.0])
    network = np.zeros(snapshot.n)
    probability = 1.0 - np.exp(-spatial[1] * params.dt)
    rng = ScriptedRNG(infect=probability, vload=0.5, dur=0.0)
    expected, expected_probability = resolve_infection(
        snapshot, spatial, network, params, rng, step=7
    )

    with adapter_type(snapshot, edges, params, seed=13) as adapter:
        actual, actual_probability = adapter.resolve_infections(
            spatial, network, rng, step=7
        )

    assert_states_equal("infection resolution", actual, expected)
    assert_phase_array_equal(
        "infection resolution", "probability", actual_probability, expected_probability
    )
    assert actual.health[1] == S


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
def test_infection_duration_uses_truncation_not_rounding(
    adapter_type, snapshot, edges, params
):
    spatial = np.zeros(snapshot.n)
    spatial[1:3] = 100.0
    network = np.zeros(snapshot.n)
    rng = ScriptedRNG(infect=0.0, vload=0.5, dur=0.15)
    expected, _ = resolve_infection(snapshot, spatial, network, params, rng, step=7)

    with adapter_type(snapshot, edges, params, seed=13) as adapter:
        actual, _ = adapter.resolve_infections(spatial, network, rng, step=7)

    assert_states_equal("infection resolution", actual, expected)
    assert actual.rec_dur[1:3].tolist() == [12, 12]


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
@pytest.mark.parametrize(
    ("recovery_draw", "expected_health"),
    [(0.96, R), (0.97, D)],
)
def test_recovery_threshold_and_strict_boundary_match_oracle(
    adapter_type, snapshot, edges, params, recovery_draw, expected_health
):
    state = snapshot.copy()
    state.quarantined[:] = False
    state.q_start[:] = 0
    state.quarantined[0] = True
    state.q_start[0] = 4
    state.base_mobility[0] = 4.0
    state.mobility[0] = 0.4
    state.inf_age[0] = 11
    state.rec_dur[0] = 12
    state.inf_age[3] = 2
    state.rec_dur[3] = 12
    rng = ScriptedRNG(rho=0.0, recover=recovery_draw)
    expected = progress_disease(state, params, rng, step=8)

    with adapter_type(state, edges, params, seed=13) as adapter:
        actual = adapter.progress_disease(rng, step=8)

    assert_states_equal("disease progression", actual, expected)
    assert actual.health[0] == expected_health
    assert actual.inf_age.tolist() == expected.inf_age.tolist()


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
@pytest.mark.parametrize(
    ("quarantine_start", "expected_quarantined"),
    [(1, False), (2, True)],
)
def test_quarantine_entry_compliance_boundary_and_expiry_match_oracle(
    adapter_type, snapshot, edges, params, quarantine_start, expected_quarantined
):
    state = snapshot.copy()
    state.mobility[3] = 1.0
    state.base_mobility[3] = 10.0
    state.q_start[3] = quarantine_start
    rng = ScriptedRNG(quar_enter=params.quarantine_compliance)
    expected = quarantine_stage(state, params, rng, step=14)

    with adapter_type(state, edges, params, seed=13) as adapter:
        actual = adapter.quarantine(rng, step=14)

    assert_states_equal("quarantine", actual, expected)
    assert not actual.quarantined[0]  # Equality does not pass the strict entry test.
    assert bool(actual.quarantined[3]) == expected_quarantined
    if expected_quarantined:
        assert actual.mobility[3] == pytest.approx(state.mobility[3])
    else:
        assert actual.mobility[3] == pytest.approx(state.base_mobility[3])


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
def test_quarantine_entry_reduces_mobility_and_ticks_immediately(
    adapter_type, snapshot, edges, params
):
    state = snapshot.copy()
    state.quarantined[:] = False
    state.q_start[:] = 0
    rng = ScriptedRNG(quar_enter=0.0)
    expected = quarantine_stage(state, params, rng, step=5)

    with adapter_type(state, edges, params, seed=13) as adapter:
        actual = adapter.quarantine(rng, step=5)
        if adapter_type is MesaAdapter:
            ticks = adapter.model.space.active_agents[0].days_in_quarantine
        else:
            esper.switch_world(adapter.model.world_name)
            quarantine = esper.component_for_entity(
                adapter.entity_ids[0], Quarantined
            )
            ticks = quarantine.days_in_quarantine

    assert_states_equal("quarantine", actual, expected)
    assert actual.quarantined.tolist() == [True, False, False, True]
    assert actual.q_start[[0, 3]].tolist() == [5, 5]
    assert actual.mobility[0] == pytest.approx(
        state.mobility[0] * (1 - params.quarantine_compliance * params.quarantine_effect)
    )
    assert ticks == 1
