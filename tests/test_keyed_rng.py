"""Tests for keyed phase randomness and its test-only model seam."""
import esper
import numpy as np
import pytest

from experiments.parity.keyed_rng import KeyedRNG, ScriptedRNG
from epidemic_sim.ecs.components import Demographics, Location
from epidemic_sim.ecs.simulation import SIREpidemicModel
from epidemic_sim.ecs.systems.transmission import MovementSystem
from epidemic_sim.mesa import MesaSIREpidemicModel


def test_keyed_draws_are_reproducible_and_independent_of_id_order():
    ids = np.array([3, 8, 2, 5])
    rng = KeyedRNG(42)
    reordered = KeyedRNG(42)

    draws = rng.uniform("infect", 17, ids)
    reversed_draws = reordered.uniform("infect", 17, ids[::-1])

    assert draws == pytest.approx(reversed_draws[::-1])
    assert rng.normal("dur", 17, ids) == pytest.approx(
        KeyedRNG(42).normal("dur", 17, ids)
    )
    assert np.all((draws > 0) & (draws < 1))


def test_keyed_ids_can_be_mapped_to_shared_snapshot_ids():
    ids = np.array([101, 202, 303])
    mapped = KeyedRNG(9, id_map={101: 0, 202: 1, 303: 2})

    assert mapped.uniform("move_x", 4, ids) == pytest.approx(
        KeyedRNG(9).uniform("move_x", 4, np.arange(3))
    )


def test_mesa_and_ecs_movement_use_same_keyed_draws_in_different_orders():
    n_agents = 8
    kwargs = {
        "seed": 12,
        "n_agents": n_agents,
        "world_size": 100,
        "initial_infected": 1,
        "space_attribute_similarity": False,
    }
    ecs = SIREpidemicModel(**kwargs)
    mesa = MesaSIREpidemicModel(**kwargs)
    ids_by_index = {entity_id: index for index, entity_id in enumerate(ecs.entity_iDs)}
    keyed_ecs = KeyedRNG(77, id_map=ids_by_index)
    keyed_mesa = KeyedRNG(77)
    positions = np.column_stack((np.arange(n_agents) * 5 + 10, np.arange(n_agents) * 3 + 15))

    try:
        esper.switch_world(ecs.world_name)
        for index, entity_id in enumerate(ecs.entity_iDs):
            location = esper.component_for_entity(entity_id, Location)
            demographics = esper.component_for_entity(entity_id, Demographics)
            location.x, location.y = positions[index]
            demographics.mobility = 4.0
        people = mesa.space.active_agents
        for index, person in enumerate(people):
            person.position = positions[index].astype(float)
            person.mobility = 4.0

        mesa_people = list(reversed(people))
        for person in mesa_people:
            person.move(phase_rng=keyed_mesa, step=6)
        esper.switch_world(ecs.world_name)
        esper.get_processor(MovementSystem).process(phase_rng=keyed_ecs, step=6)

        mesa_positions = np.asarray([person.position for person in people])
        ecs_positions = np.asarray([
            (
                esper.component_for_entity(entity_id, Location).x,
                esper.component_for_entity(entity_id, Location).y,
            )
            for entity_id in ecs.entity_iDs
        ])
        assert ecs_positions == pytest.approx(mesa_positions)
    finally:
        ecs.clean_up()


def test_scripted_rng_overrides_are_broadcast_per_agent():
    rng = ScriptedRNG(5, {"infect": 0.25})

    assert rng.uniform("infect", 1, np.array([7, 2])) == pytest.approx([0.25, 0.25])


def test_model_steps_accept_keyed_rng_without_replacing_production_rng():
    kwargs = {
        "seed": 31,
        "n_agents": 12,
        "world_size": 100,
        "initial_infected": 2,
        "spatial_new": True,
        "network_new": True,
        "enable_quarantine": True,
    }
    ecs = SIREpidemicModel(**kwargs)
    mesa = MesaSIREpidemicModel(**kwargs)
    keyed_ecs = KeyedRNG(
        19,
        id_map={entity_id: index for index, entity_id in enumerate(ecs.entity_iDs)},
    )
    try:
        ecs.step(phase_rng=keyed_ecs, phase_step=1)
        mesa.step_with_phase_rng(KeyedRNG(19), phase_step=1)

        assert ecs.step_count == 1
        assert mesa.steps == 1
    finally:
        ecs.clean_up()
