"""Focused tests for the executable phase-level reference model."""
import numpy as np
import pytest

from experiments.parity.oracle import (
    D,
    I,
    R,
    S,
    Edges,
    Params,
    State,
    check_rewire_invariants,
    exact_inclusion_probs,
    full_update,
    quarantine_stage,
    spatial_hazard,
    progress_disease,
)


class FixedRNG:
    def __init__(self, values=None):
        self.values = values or {}

    def uniform(self, purpose, step, ids):
        return np.broadcast_to(self.values.get(purpose, 0.0), np.asarray(ids).shape).copy()

    def normal(self, purpose, step, ids):
        return np.broadcast_to(self.values.get(purpose, 0.0), np.asarray(ids).shape).copy()


def make_state(
    health,
    *,
    positions=None,
    mobility=None,
    base_mobility=None,
    immunity=None,
    infected_age=None,
    recovery_duration=None,
    quarantined=None,
    quarantine_start=None,
):
    n = len(health)
    return State(
        pos=np.asarray(positions if positions is not None else np.zeros((n, 2)), dtype=float),
        age=np.full(n, 30, dtype=np.int64),
        mobility=np.asarray(mobility if mobility is not None else np.full(n, 10.0), dtype=float),
        base_mobility=np.asarray(
            base_mobility if base_mobility is not None else np.full(n, 10.0), dtype=float
        ),
        immunity=np.asarray(immunity if immunity is not None else np.zeros(n), dtype=float),
        health=np.asarray(health, dtype=np.int8),
        vload=np.where(np.asarray(health) == I, 1000.0, 0.0),
        inf_age=np.asarray(infected_age if infected_age is not None else np.zeros(n), dtype=np.int64),
        rec_dur=np.asarray(
            recovery_duration if recovery_duration is not None else np.full(n, 12),
            dtype=np.int64,
        ),
        quarantined=np.asarray(
            quarantined if quarantined is not None else np.zeros(n), dtype=bool
        ),
        q_start=np.asarray(
            quarantine_start if quarantine_start is not None else np.zeros(n), dtype=np.int64
        ),
    )


def test_spatial_hazard_includes_radius_boundary_and_excludes_quarantined_source():
    state = make_state(
        [I, S, S, I],
        positions=[[0, 0], [4, 0], [4.000001, 0], [1, 0]],
        quarantined=[False, False, False, True],
        immunity=[0, 0.2, 0, 0],
    )

    hazard = spatial_hazard(state, Params())

    assert hazard == pytest.approx([0.0, 0.24, 0.0, 0.0])


def test_recovery_threshold_and_strict_uniform_comparison():
    state = make_state([I], infected_age=[11], recovery_duration=[12])
    rng = FixedRNG({"recover": 0.97, "rho": 0.0})

    result = progress_disease(state, Params(), rng, step=10)

    assert result.inf_age.tolist() == [12]
    assert result.health.tolist() == [D]
    assert result.mobility.tolist() == [0.0]


def test_recovery_releases_quarantine_and_restores_mobility():
    state = make_state(
        [I],
        mobility=[2.0],
        base_mobility=[10.0],
        infected_age=[11],
        recovery_duration=[12],
        quarantined=[True],
        quarantine_start=[1],
    )
    rng = FixedRNG({"recover": 0.0, "rho": 0.0})

    result = progress_disease(state, Params(), rng, step=10)

    assert result.health.tolist() == [R]
    assert result.quarantined.tolist() == [False]
    assert result.mobility.tolist() == [10.0]


def test_new_infection_progresses_to_age_one_in_same_update():
    state = make_state([I, S], positions=[[0, 0], [1, 0]])
    empty_edges = Edges(
        src=np.empty(0, dtype=np.int64),
        dst=np.empty(0, dtype=np.int64),
        w=np.empty(0, dtype=float),
    )
    rng = FixedRNG({"move_x": 0.5, "move_y": 0.5, "infect": 0.0, "vload": 0.5})

    result = full_update(state, empty_edges, Params(), rng, step=1, quarantine_enabled=False)

    assert result.health.tolist() == [I, I]
    assert result.inf_age.tolist() == [1, 1]


def test_quarantine_entry_ticks_immediately_and_expiry_cannot_reenter_same_update():
    params = Params(quarantine_duration=2)
    rng = FixedRNG({"quar_enter": 0.0})
    state = make_state([I])

    entered = quarantine_stage(state, params, rng, step=4)
    expired = quarantine_stage(entered, params, rng, step=5)
    reentered = quarantine_stage(expired, params, rng, step=6)

    assert entered.quarantined.tolist() == [True]
    assert entered.mobility.tolist() == pytest.approx([2.8])
    assert expired.quarantined.tolist() == [False]
    assert expired.mobility.tolist() == pytest.approx([10.0])
    assert reentered.quarantined.tolist() == [True]


def test_rewiring_invariants_use_passed_similarity_parameters():
    params = Params(similarity_length=2.0, age_similarity_scale=10.0)
    state = make_state([S, S, D], positions=[[0, 0], [2, 0], [3, 0]], immunity=[0, 0, 0])
    edges = Edges(
        src=np.array([0], dtype=np.int64),
        dst=np.array([1], dtype=np.int64),
        w=np.array([np.exp(-1.0)]),
    )

    violations, expected = check_rewire_invariants(
        state, edges, np.array([1, 1, 0]), params
    )

    assert violations == []
    assert expected.tolist() == pytest.approx([np.exp(-1.0)])
    assert exact_inclusion_probs(np.array([1.0, 2.0, 3.0]), 1) == pytest.approx(
        [1 / 6, 2 / 6, 3 / 6]
    )
