"""Unit tests of the Mesa implementation alone (no ECS involved)."""
import contextlib
import io
import subprocess
import sys

import numpy as np
import pytest

from epidemic_sim.config import SimulationConfig
from epidemic_sim.disease_model import DiseaseModelConfig, HealthState
from epidemic_sim.mesa import MesaSIREpidemicModel
from epidemic_sim.mesa.model import health_counts

HAZARD = dict(spatial_new=True, network_new=True)


def run_quietly(model: MesaSIREpidemicModel, steps: int) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        model.run(steps)


def test_model_is_reproducible():
    kwargs = dict(seed=11, n_agents=150, world_size=150, initial_infected=5,
                  space_attribute_similarity=True, dispersion=0.65, **HAZARD)
    runs = []
    for _ in range(2):
        model = MesaSIREpidemicModel(**kwargs)
        run_quietly(model, 40)
        runs.append(model.time_series_data)
    assert runs[0] == runs[1]


def test_simulation_draws_use_only_the_numpy_generator():
    model = MesaSIREpidemicModel(seed=11, n_agents=30, initial_infected=2, **HAZARD)
    python_rng_state = model.random.getstate()
    model.run_for(3)
    assert model.random.getstate() == python_rng_state


def test_different_seeds_give_different_runs():
    kwargs = dict(n_agents=150, world_size=150, initial_infected=5, **HAZARD)
    a = MesaSIREpidemicModel(seed=1, **kwargs)
    b = MesaSIREpidemicModel(seed=2, **kwargs)
    assert not np.array_equal(a.space.agent_positions, b.space.agent_positions)


def test_population_is_conserved():
    model = MesaSIREpidemicModel(seed=3, n_agents=200, world_size=150, initial_infected=6,
                                 enable_quarantine=True, **HAZARD)
    run_quietly(model, 60)
    series = model.time_series_data
    for i in range(len(series["time"])):
        assert (series["susceptible"][i] + series["infected"][i]
                + series["recovered"][i] + series["death"][i]) == 200


def test_positions_are_held_by_the_mesa_space_and_stay_in_the_world():
    model = MesaSIREpidemicModel(seed=4, n_agents=100, world_size=80, initial_infected=3, **HAZARD)
    run_quietly(model, 30)
    positions = model.space.agent_positions
    assert positions.shape == (100, 2)
    assert positions.min() >= 0 and positions.max() <= 80
    person = model.space.active_agents[7]
    assert np.array_equal(person.position, positions[7]) and person.index == 7


def test_initial_health_counts_and_proportions():
    model = MesaSIREpidemicModel(seed=5, n_agents=200, initial_infected=8, **HAZARD)
    states = [a.state for a in model.agents]
    assert states.count(HealthState.INFECTED) == 8 and states.count(HealthState.SUSCEPTIBLE) == 192
    mixed = MesaSIREpidemicModel(
        seed=5, n_agents=200, **HAZARD,
        initial_proportions={HealthState.SUSCEPTIBLE: 0.8, HealthState.INFECTED: 0.1,
                             HealthState.RECOVERED: 0.05, HealthState.DEAD: 0.05})
    states = [a.state for a in mixed.agents]
    assert (states.count(HealthState.SUSCEPTIBLE), states.count(HealthState.INFECTED),
            states.count(HealthState.RECOVERED), states.count(HealthState.DEAD)) == (160, 20, 10, 10)
    assert all(a.mobility == 0 for a in mixed.agents if a.state is HealthState.DEAD)
    with pytest.raises(ValueError):
        health_counts(100, 0, {HealthState.SUSCEPTIBLE: 0.5, HealthState.INFECTED: 0.2})


def test_days_infected_counter_is_incremented_by_the_model_step():
    model = MesaSIREpidemicModel(seed=6, n_agents=30, world_size=500, initial_infected=1,
                                 beta_spatial=0.0, beta_network=0.0, **HAZARD)
    person = next(a for a in model.agents if a.state is HealthState.INFECTED)
    person.recovery_time = 100
    start = person.days_infected
    for _ in range(3):
        model.step()
    assert person.days_infected == start + 3
    fresh = next(a for a in model.agents if a.state is HealthState.SUSCEPTIBLE)
    fresh.infect(800.0, 10)
    assert fresh.days_infected == 0  # infect() takes no counter argument


def test_quarantine_reduces_mobility_and_restores_it():
    model = MesaSIREpidemicModel(seed=5, n_agents=20, world_size=50, initial_infected=1, **HAZARD)
    person = next(a for a in model.agents if a.state is HealthState.INFECTED)
    original = person.mobility
    person.try_quarantine(compliance=1.0)  # certain to enter quarantine
    assert person.quarantined
    assert person.mobility == pytest.approx(original * (1 - 0.9))
    for _ in range(14):
        person.tick_quarantine()
    assert not person.quarantined
    assert person.mobility == pytest.approx(original)


def test_run_for_advances_the_model_and_run_stops_when_nobody_is_infected():
    model = MesaSIREpidemicModel(seed=8, n_agents=50, world_size=100, initial_infected=2, **HAZARD)
    model.run_for(5)
    assert model.steps == 5 and len(model.time_series_data["time"]) == 5
    run_quietly(model, 1000)
    assert not model.running and model.time_series_data["infected"][-1] == 0


def test_outputs_have_the_ecs_format():
    model = MesaSIREpidemicModel(seed=9, n_agents=40, world_size=60, initial_infected=2, **HAZARD)
    model.run_for(3)
    assert set(model.time_series_data) == {"time", "susceptible", "infected", "recovered", "death"}
    assert model.time_series_data["time"] == [0, 1, 2]
    spatial = model.spatial_location_series_data
    assert set(spatial) == {"susceptible", "infected", "recovered", "death"}
    assert sum(len(v) for v in spatial.values()) == 3 * 40
    step, x, y = spatial["susceptible"][0]
    assert isinstance(step, int) and 0 <= x <= 60 and 0 <= y <= 60


@pytest.mark.parametrize("bad", [
    dict(n_agents=0), dict(initial_infected=999), dict(world_size=0), dict(dt=0),
    dict(beta_spatial=1.5), dict(tau=0), dict(dispersion=0), dict(alpha=-1),
])
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        MesaSIREpidemicModel(seed=1, **bad)


def test_seed_must_be_integer():
    with pytest.raises(TypeError):
        MesaSIREpidemicModel(seed=1.5)


def test_mesa_module_does_not_import_esper():
    """A Mesa-only run must not load Esper (fair memory measurement in the benchmark)."""
    code = "import sys, epidemic_sim.mesa; assert 'esper' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_choose_contacts_is_the_successive_weighted_sample():
    """Person.choose_contacts (Mesa space distances) draws k contacts without
    replacement, one after the other, proportionally to the similarity weights:
    inclusion frequencies match exact enumeration."""
    from itertools import permutations

    model = MesaSIREpidemicModel(n_agents=6, initial_infected=1, world_size=20, seed=3)
    people = list(model.agents)
    ages = np.array([p.age for p in model.space.agents])
    alive = np.ones(len(ages), dtype=bool)
    me = people[0]
    k, alpha, tau = 2, 8.0, 15.0
    dist, _ = model.space.calculate_distances(me.position)
    w = np.exp(-dist / alpha - np.abs(ages - me.age) / tau)
    w[me.index] = 0.0
    expected = np.zeros(len(w))
    for perm in permutations([j for j in range(len(w)) if w[j] > 0], k):
        p, rest = 1.0, w.sum()
        for j in perm:
            p *= w[j] / rest
            rest -= w[j]
        for j in perm:
            expected[j] += p
    counts = np.zeros(len(w))
    n_rep = 20000
    for _ in range(n_rep):
        contacts, _ = me.choose_contacts(k, ages, alive, alpha, tau)
        for c in contacts:
            counts[c.index] += 1
    assert np.allclose(counts / n_rep, expected, atol=0.015)


def test_model_accepts_shared_configs():
    simulation = SimulationConfig(seed=42, n_agents=30, world_size=80, initial_infected=2)
    disease = DiseaseModelConfig(
        beta_spatial=0.0,
        beta_network=0.0,
        spatial_new=True,
        network_new=True,
    )
    model = MesaSIREpidemicModel(
        simulation_config=simulation,
        disease_model_config=disease,
    )
    assert model.seed == 42
    assert model.n_agents == 30
    assert model.world_size == 80
    assert model.spatial_new and model.network_new
