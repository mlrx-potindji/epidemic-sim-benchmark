"""Harness sensitivity checks using an independent loop and planted defects."""
import numpy as np
import pytest

import esper
from experiments.parity.adapters import ECSAdapter, MesaAdapter
from experiments.parity.keyed_rng import KeyedRNG
from experiments.parity.loop_adapter import LoopAdapter
from experiments.parity.oracle import (
    D,
    I,
    S,
    Edges,
    Params,
    State,
    check_rewire_invariants,
    exact_inclusion_probs,
    movement,
    network_hazard,
    progress_disease,
    quarantine_stage,
    resolve_infection,
    spatial_hazard,
)
from epidemic_sim.ecs.components import InfectionHazard
from epidemic_sim.ecs.systems.transmission import SpatialTransmissionSystemNew


class ScriptedRNG:
    def __init__(self, **values):
        self.values = values

    def uniform(self, purpose, step, ids):
        return np.full(np.asarray(ids).shape, self.values.get(purpose, 0.0))

    def normal(self, purpose, step, ids):
        return np.full(np.asarray(ids).shape, self.values.get(purpose, 0.0))


def make_state():
    health = np.asarray([I, S, S, I, D, S], dtype=np.int8)
    return State(
        pos=np.asarray(
            [[0, 0], [4, 0], [1, 0], [1, 0], [0, 0], [8, 8]],
            dtype=float,
        ),
        age=np.asarray([30, 30, 50, 20, 40, 65]),
        mobility=np.asarray([4, 3, 2, 3, 0, 2], dtype=float),
        base_mobility=np.asarray([4, 3, 2, 3, 4, 2], dtype=float),
        immunity=np.asarray([0, 0.5, 0.2, 0, 0, 0], dtype=float),
        health=health,
        vload=np.asarray([800, 0, 0, 900, 0, 0], dtype=float),
        inf_age=np.asarray([2, 0, 0, 2, 0, 0]),
        rec_dur=np.full(6, 12, dtype=np.int64),
        quarantined=np.asarray([False, False, False, True, True, False]),
        q_start=np.asarray([0, 0, 0, 2, 1, 0]),
    )


def make_params():
    return Params(
        world_size=10,
        transmission_radius=4,
        beta_spatial=0.3,
        beta_network=0.25,
        similarity_length=2.0,
        age_similarity_scale=8.0,
        mean_degree=3.0,
        rewiring_dispersion=0.7,
    )


def differences(expected: State, actual: State) -> list[tuple[str, list[int]]]:
    result = []
    for field in expected.__dataclass_fields__:
        want = np.asarray(getattr(expected, field))
        got = np.asarray(getattr(actual, field))
        equal = np.isclose(want, got, rtol=1e-12, atol=1e-12)
        if equal.ndim > 1:
            equal = equal.reshape((equal.shape[0], -1)).all(axis=1)
        ids = np.flatnonzero(~equal).tolist()
        if ids:
            result.append((field, ids))
    return result


def assert_detects(mutation, phase, expected, actual):
    mismatches = differences(expected, actual)
    assert mismatches, f"mutation {mutation!r} escaped {phase} comparison"
    field, agent_ids = mismatches[0]
    assert agent_ids
    return phase, field, agent_ids


@pytest.mark.parametrize("adapter_type", [MesaAdapter, ECSAdapter])
def test_real_model_adapter_detects_seeded_immunity_omission_mutation(
    adapter_type, monkeypatch
):
    state = make_state()
    params = make_params()
    edges = Edges(
        src=np.empty(0, dtype=np.int64),
        dst=np.empty(0, dtype=np.int64),
        w=np.empty(0, dtype=float),
    )
    expected = spatial_hazard(state, params)

    with adapter_type(state, edges, params, seed=1907) as adapter:
        if adapter_type is MesaAdapter:
            original = adapter.model._spatial_transmission_hazard

            def omit_immunity():
                original()
                target = adapter.model.space.active_agents[1]
                target.hazard += (
                    params.beta_spatial
                    * state.vload[0]
                    / 1000.0
                    * state.immunity[1]
                )

            monkeypatch.setattr(adapter.model, "_spatial_transmission_hazard", omit_immunity)
        else:
            original = SpatialTransmissionSystemNew.process

            def omit_immunity(system):
                original(system)
                esper.switch_world(adapter.model.world_name)
                target_id = adapter.entity_ids[1]
                hazard = esper.component_for_entity(target_id, InfectionHazard)
                hazard.hazard += (
                    params.beta_spatial
                    * state.vload[0]
                    / 1000.0
                    * state.immunity[1]
                )

            monkeypatch.setattr(SpatialTransmissionSystemNew, "process", omit_immunity)

        actual = adapter.hazards().spatial
        mismatch_ids = np.flatnonzero(
            ~np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
        ).tolist()
        assert mismatch_ids == [1], (
            "first divergent phase: spatial hazard; field spatial; "
            f"agent IDs {mismatch_ids}"
        )


def test_defect_free_independent_loop_matches_oracle_for_core_phases():
    state = make_state()
    params = make_params()
    edges = Edges(
        src=np.asarray([0, 3, 0]),
        dst=np.asarray([1, 2, 2]),
        w=np.asarray([0.8, 0.6, 0.3]),
    )
    rng = KeyedRNG(5101)
    loop = LoopAdapter()

    expected_movement = movement(state, params, rng, step=20)
    actual_movement = loop.movement(state, params, rng, step=20)
    assert not differences(expected_movement, actual_movement)

    expected_spatial = spatial_hazard(state, params)
    expected_network = network_hazard(state, edges, params)
    actual_spatial, actual_network = loop.hazards(state, edges, params)
    np.testing.assert_allclose(actual_spatial, expected_spatial, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_network, expected_network, rtol=1e-12, atol=1e-12)

    hazards = np.asarray([0.0, 0.25, 0.4, 0.0, 0.0, 0.0])
    expected_resolved, _ = resolve_infection(
        state, hazards, np.zeros(state.n), params, rng, step=20
    )
    actual_resolved, _ = loop.resolve(
        state, hazards, np.zeros(state.n), params, rng, step=20
    )
    assert not differences(expected_resolved, actual_resolved)

    progressed_expected = progress_disease(
        state, params, ScriptedRNG(), step=20
    )
    progressed_actual = loop.progress(state, params, ScriptedRNG(), step=20)
    assert not differences(progressed_expected, progressed_actual)

    quarantined_expected = quarantine_stage(
        state, params, ScriptedRNG(), step=20
    )
    quarantined_actual = loop.quarantine(state, params, ScriptedRNG(), step=20)
    assert not differences(quarantined_expected, quarantined_actual)


@pytest.mark.parametrize(
    ("mutation", "phase"),
    [
        ("strict_radius", "hazards"),
        ("spatial_quarantine_leak", "hazards"),
        ("ignore_immunity_spatial", "hazards"),
        ("q_theta_only", "hazards"),
        ("round_duration", "resolution"),
        ("newborn_age_one", "resolution"),
        ("recover_leq", "progression"),
        ("no_release_on_recovery", "progression"),
        ("dead_mobility_restored", "quarantine"),
        ("enter_prob_one", "quarantine"),
        ("movement_wrap", "movement"),
        ("quarantine_before_progression", "update"),
        ("rewire_uniform", "rewiring"),
        ("store_normalized_weight", "rewiring"),
        ("rewire_ignores_age", "rewiring"),
        ("rewire_includes_dead", "rewiring"),
        ("degree_wrong_p", "degree distribution"),
    ],
)
def test_independent_loop_mutation_catalog(mutation, phase):
    state = make_state()
    params = make_params()
    edges = Edges(
        src=np.asarray([0, 3]),
        dst=np.asarray([1, 2]),
        w=np.asarray([0.8, 0.6]),
    )
    loop = LoopAdapter({mutation})
    step = 14
    rng = ScriptedRNG(
        move_x=0.0,
        move_y=0.5,
        infect=0.0,
        vload=0.4,
        dur=0.15,
        rho=0.0,
        recover=0.97,
        quar_enter=1.0,
    )

    if mutation == "strict_radius":
        expected = spatial_hazard(state, params)
        actual = loop.hazards(state, edges, params)[0]
        assert not np.allclose(expected, actual)
    elif mutation in {
        "spatial_quarantine_leak",
        "ignore_immunity_spatial",
        "q_theta_only",
    }:
        oracle_spatial = spatial_hazard(state, params)
        oracle_network = network_hazard(state, edges, params)
        actual_spatial, actual_network = loop.hazards(state, edges, params)
        if mutation == "q_theta_only":
            assert not np.allclose(actual_network, oracle_network)
        else:
            assert not np.allclose(actual_spatial, oracle_spatial)
    elif mutation in {"round_duration", "newborn_age_one"}:
        sparse = state.copy()
        sparse.health[:] = S
        sparse.health[0] = I
        hazards = np.zeros(state.n)
        hazards[1] = 100.0
        expected, _ = resolve_infection(
            sparse, hazards, hazards, params, rng, step
        )
        actual, _ = loop.resolve(sparse, hazards, hazards, params, rng, step)
        assert_detects(mutation, phase, expected, actual)
    elif mutation in {"recover_leq", "no_release_on_recovery"}:
        due = state.copy()
        due.health[0] = I
        due.inf_age[0] = 11
        due.rec_dur[0] = 12
        due.quarantined[0] = True
        due.mobility[0] = 0.4
        test_rng = (
            ScriptedRNG(rho=0, recover=0.0)
            if mutation == "no_release_on_recovery"
            else ScriptedRNG(rho=0, recover=0.97)
        )
        expected = progress_disease(due, params, test_rng, step)
        actual = loop.progress(due, params, test_rng, step)
        assert_detects(mutation, phase, expected, actual)
    elif mutation in {"dead_mobility_restored", "enter_prob_one"}:
        test_rng = ScriptedRNG(quar_enter=1.0)
        expected = quarantine_stage(state, params, test_rng, step)
        actual = loop.quarantine(state, params, test_rng, step)
        assert_detects(mutation, phase, expected, actual)
    elif mutation == "movement_wrap":
        moved = state.copy()
        moved.pos[1] = [0.0, 5.0]
        expected = movement(moved, params, rng, step)
        actual = loop.movement(moved, params, rng, step)
        assert_detects(mutation, phase, expected, actual)
    elif mutation == "quarantine_before_progression":
        due = state.copy()
        due.health[0] = I
        due.inf_age[0] = 11
        due.rec_dur[0] = 12
        due.quarantined[0] = False
        due_edges = Edges(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )
        test_rng = ScriptedRNG(
            move_x=0.5, move_y=0.5, rho=0.0, recover=1.0, quar_enter=0.0
        )
        expected = LoopAdapter().full_update(
            due, due_edges, params, test_rng, step, True
        )
        actual = loop.full_update(due, due_edges, params, test_rng, step, True)
        assert_detects(mutation, phase, expected, actual)
    elif mutation in {
        "rewire_uniform",
        "store_normalized_weight",
        "rewire_ignores_age",
        "rewire_includes_dead",
    }:
        rewire_state = state.copy()
        rewire_state.health[:] = S
        rewire_state.health[0] = I
        requested = np.zeros(state.n, dtype=np.int64)
        requested[0] = 1
        if mutation == "rewire_uniform":
            rewire_state.health[:] = S
            rewire_state.health[0] = I
            weights = np.exp(
                -np.linalg.norm(rewire_state.pos[1:] - rewire_state.pos[0], axis=1)
                / params.similarity_length
                - np.abs(rewire_state.age[1:] - rewire_state.age[0])
                / params.age_similarity_scale
            )
            target_probs = exact_inclusion_probs(weights, 1)
            counts = np.zeros(len(weights), dtype=float)
            np_rng = np.random.default_rng(8551)
            for _ in range(2500):
                drawn = loop.rewire(
                    rewire_state, params, np_rng, requested, sources=[0]
                )
                counts[drawn.dst[0] - 1] += 1
            observed = counts / 2500
            assert np.max(np.abs(observed - target_probs)) > 0.12
        else:
            if mutation == "rewire_includes_dead":
                rewire_state.health[:] = D
                rewire_state.health[:2] = [I, S]
            rewire = loop.rewire(
                rewire_state,
                params,
                np.random.default_rng(4409),
                requested,
                sources=[0],
            )
            violations, expected_weights = check_rewire_invariants(
                rewire_state, rewire, requested, params
            )
            if mutation == "store_normalized_weight":
                assert not np.allclose(rewire.w, expected_weights)
            elif mutation == "rewire_ignores_age":
                assert not np.allclose(rewire.w, expected_weights)
            else:
                assert "dead agent used as source or candidate" in violations
    elif mutation == "degree_wrong_p":
        degrees = loop.degree_sample(
            params, size=30000, rng=np.random.default_rng(8811)
        )
        assert abs(float(degrees.mean()) - params.mean_degree) > 2.0
