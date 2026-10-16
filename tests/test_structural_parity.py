"""Structural parity between the ECS (Esper) and the Mesa implementations.

Both models are built with the same parameters and must expose the same population,
the same initial infections, the same law of the initial contact network and the same
output format. This is NOT the epidemiological parity (see test_epidemiological_parity.py):
the two models use different random draws, so nothing is compared draw by draw.
"""
import contextlib
import io
from dataclasses import dataclass

import esper
import numpy as np
import pytest
from scipy.stats import ks_2samp

from epidemic_sim.ecs.components import ContactNetwork, Demographics, Infected, Location
from epidemic_sim.ecs.simulation import SIREpidemicModel
from epidemic_sim.config import SimulationConfig
from epidemic_sim.disease_model import DiseaseModelConfig, HealthState
from epidemic_sim.mesa import MesaSIREpidemicModel

PARAMS = dict(n_agents=600, world_size=300, initial_infected=12, spatial_new=True,
              network_new=True, space_attribute_similarity=True, dispersion=0.65)


def build_ecs(**kwargs) -> SIREpidemicModel:
    with contextlib.redirect_stdout(io.StringIO()):
        return SIREpidemicModel(**kwargs)


@dataclass
class Snapshot:
    """Population and contact-network summary of a model."""

    x: np.ndarray
    y: np.ndarray
    age: np.ndarray
    mobility: np.ndarray
    infected: int
    degree: np.ndarray        # contacts per agent (0 when the agent has none)
    edge_distance: np.ndarray
    edge_age_gap: np.ndarray
    edge_strength: np.ndarray


def ecs_snapshot(model: SIREpidemicModel) -> Snapshot:
    esper.switch_world(model.world_name)
    where, years, rows = {}, {}, []
    for entity, (location, demographics) in esper.get_components(Location, Demographics):
        where[entity], years[entity] = (location.x, location.y), demographics.age
        rows.append((location.x, location.y, demographics.age, demographics.mobility))
    degree = {entity: 0 for entity in where}
    distance, gap, strength = [], [], []
    for entity, (network,) in esper.get_components(ContactNetwork):
        degree[entity] = len(network.contacts)
        for other, weight in zip(network.contacts, network.contact_strength):
            distance.append(np.hypot(where[entity][0] - where[other][0], where[entity][1] - where[other][1]))
            gap.append(abs(years[entity] - years[other]))
            strength.append(weight)
    x, y, age, mobility = (np.array(column, dtype=float) for column in zip(*rows))
    return Snapshot(x, y, age, mobility, len(esper.get_component(Infected)),
                    np.array(list(degree.values())), np.array(distance), np.array(gap), np.array(strength))


def mesa_snapshot(model: MesaSIREpidemicModel) -> Snapshot:
    people = model.space.active_agents
    distance, gap, strength = [], [], []
    for person in people:
        for other in model.contact_graph.successors(person):
            weight = model.contact_graph[person][other]["weight"]
            distance.append(np.hypot(person.x - other.x, person.y - other.y))
            gap.append(abs(person.age - other.age))
            strength.append(weight)
    return Snapshot(model.space.agent_positions[:, 0].copy(), model.space.agent_positions[:, 1].copy(),
                    np.array([p.age for p in people], dtype=float),
                    np.array([p.mobility for p in people]),
                    sum(p.state is HealthState.INFECTED for p in people),
                    np.array([model.contact_graph.out_degree(p) for p in people]),
                    np.array(distance), np.array(gap), np.array(strength))


@pytest.fixture(scope="module")
def snapshots() -> tuple[Snapshot, Snapshot]:
    return ecs_snapshot(build_ecs(seed=21, **PARAMS)), mesa_snapshot(MesaSIREpidemicModel(seed=21, **PARAMS))


def test_same_population_size_and_initial_infections(snapshots):
    ecs, mesa = snapshots
    assert len(ecs.x) == len(mesa.x) == PARAMS["n_agents"]
    assert ecs.infected == mesa.infected == PARAMS["initial_infected"]


@pytest.mark.parametrize("attribute", ["x", "y", "age", "mobility"])
def test_same_law_of_the_population_attributes(snapshots, attribute):
    ecs, mesa = snapshots
    assert ks_2samp(getattr(ecs, attribute), getattr(mesa, attribute)).pvalue > 0.001


def test_same_law_of_the_contact_network(snapshots):
    ecs, mesa = snapshots
    # number of contacts: negative binomial with mean ``average_contacts`` = 10 (3 standard errors)
    for snapshot in (ecs, mesa):
        assert abs(snapshot.degree.mean() - 10) < 1.6
    # homophily signature of the edges: mean distance, mean age gap, mean tie strength
    for name in ("edge_distance", "edge_age_gap", "edge_strength"):
        a, b = getattr(ecs, name).mean(), getattr(mesa, name).mean()
        assert abs(a - b) / a < 0.08, f"{name}: ECS {a:.3f} vs Mesa {b:.3f}"


def test_same_output_format_and_conservation():
    kwargs = dict(n_agents=120, world_size=100, initial_infected=4, spatial_new=True, network_new=True)
    ecs = build_ecs(seed=2, **kwargs)
    mesa = MesaSIREpidemicModel(seed=2, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        ecs.run(15)
        mesa.run(15)
    for model in (ecs, mesa):
        series = model.time_series_data
        assert set(series) == {"time", "susceptible", "infected", "recovered", "death"}
        for i in range(len(series["time"])):
            assert (series["susceptible"][i] + series["infected"][i]
                    + series["recovered"][i] + series["death"][i]) == 120
    ecs.clean_up()


def test_ecs_respects_spatial_collection_setting():
    for collect_spatial in (False, True):
        model = build_ecs(
            simulation_config=SimulationConfig(
                seed=12,
                n_agents=60,
                world_size=100,
                initial_infected=3,
                max_steps=2,
                collect_spatial=collect_spatial,
            ),
            disease_model_config=DiseaseModelConfig(),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.run(2)

        assert len(model.time_series_data["time"]) == 2
        spatial_points = sum(map(len, model.spatial_location_series_data.values()))
        assert spatial_points == (120 if collect_spatial else 0)
        model.clean_up()


def test_mesa_contact_graph_is_directed_and_weighted():
    model = MesaSIREpidemicModel(seed=21, **PARAMS)
    assert model.contact_graph.is_directed()
    assert model.contact_graph.number_of_nodes() == PARAMS["n_agents"]
    assert all("weight" in data for _, _, data in model.contact_graph.edges(data=True))
    assert all(source is not target for source, target in model.contact_graph.edges)


def test_ecs_accepts_shared_configs():
    model = SIREpidemicModel(
        simulation_config=SimulationConfig(seed=21, n_agents=20, world_size=80),
        disease_model_config=DiseaseModelConfig(spatial_new=True, network_new=True),
    )
    assert model.n_agents == 20
    assert model.world_size == 80
    assert model.seed == 21
    assert isinstance(model.rng.numpy, np.random.Generator)
    assert not hasattr(model.rng, "python")
    model.clean_up()
