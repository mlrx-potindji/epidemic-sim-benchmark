"""Executable reference specification for phase-level epidemic parity tests.

This intentionally straightforward NumPy implementation is a correctness
oracle, not a production simulator. Model phases receive random values through
the ``RandomVariates`` protocol so tests can supply deterministic, keyed draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

S, I, R, D = 0, 1, 2, 3


class RandomVariates(Protocol):
    """Random draws addressed by purpose, update, and agent identifiers."""

    def uniform(self, purpose: str, step: int, ids: NDArray) -> NDArray: ...

    def normal(self, purpose: str, step: int, ids: NDArray) -> NDArray: ...


@dataclass(frozen=True)
class Params:
    world_size: float = 200.0
    transmission_radius: float = 4.0
    beta_spatial: float = 0.30
    beta_network: float = 0.25
    quarantine_compliance: float = 0.8
    quarantine_effect: float = 0.9
    quarantine_duration: int = 14
    similarity_length: float = 40.0
    age_similarity_scale: float = 25.0
    dt: float = 1.0
    recovery_mean: float = 12.0
    recovery_sd: float = 4.0
    recovery_probability_low: float = 0.97
    recovery_probability_high: float = 0.995
    viral_load_low: float = 500.0
    viral_load_high: float = 1000.0
    recovered_immunity: float = 0.9
    mean_degree: float = 10.0


@dataclass
class State:
    """One explicit model snapshot; each array has one entry per agent."""

    pos: NDArray
    age: NDArray
    mobility: NDArray
    base_mobility: NDArray
    immunity: NDArray
    health: NDArray
    vload: NDArray
    inf_age: NDArray
    rec_dur: NDArray
    quarantined: NDArray
    q_start: NDArray

    @property
    def n(self) -> int:
        return len(self.health)

    @property
    def ids(self) -> NDArray[np.int64]:
        return np.arange(self.n)

    def copy(self) -> State:
        return State(**{name: value.copy() for name, value in self.__dict__.items()})


@dataclass
class Edges:
    """Directed contacts with raw, unnormalized similarity strengths."""

    src: NDArray
    dst: NDArray
    w: NDArray


def movement(state: State, params: Params, rng: RandomVariates, step: int) -> State:
    """Apply coordinate-wise random-walk movement and clip to the world."""
    out = state.copy()
    ux = rng.uniform("move_x", step, state.ids)
    uy = rng.uniform("move_y", step, state.ids)
    displacement = np.stack(
        [
            (2 * ux - 1) * state.mobility,
            (2 * uy - 1) * state.mobility,
        ],
        axis=1,
    )
    displacement[state.health == D] = 0.0
    out.pos = np.clip(state.pos + displacement, 0.0, params.world_size)
    return out


def similarity_weight(pos_a, pos_b, age_a, age_b, params: Params):
    distance = np.sqrt(((pos_a - pos_b) ** 2).sum(axis=-1))
    return np.exp(
        -distance / params.similarity_length
        - np.abs(age_a - age_b) / params.age_similarity_scale
    )


def spatial_hazard(state: State, params: Params) -> NDArray[np.float64]:
    """Accumulate spatial hazard from non-quarantined infectious sources."""
    delta = state.pos[:, None, :] - state.pos[None, :, :]
    distance = np.sqrt((delta**2).sum(axis=-1))
    sources = (state.health == I) & ~state.quarantined
    targets = state.health == S
    connected = sources[:, None] & (distance <= params.transmission_radius) & targets[None, :]
    source_hazard = params.beta_spatial * state.vload / 1000.0
    hazard = (connected * source_hazard[:, None]).sum(axis=0) * (1.0 - state.immunity)
    return np.where(targets, hazard, 0.0)


def network_hazard(state: State, edges: Edges, params: Params) -> NDArray[np.float64]:
    """Accumulate outgoing-edge hazard, including quarantine attenuation."""
    source_modifier = (
        1.0
        - state.quarantined[edges.src]
        * params.quarantine_compliance
        * params.quarantine_effect
    )
    active = (state.health[edges.src] == I) & (state.health[edges.dst] == S)
    contribution = (
        params.beta_network
        * source_modifier
        * edges.w
        * (1.0 - state.immunity[edges.dst])
        * state.vload[edges.src]
        / 1000.0
    )
    return np.bincount(edges.dst[active], weights=contribution[active], minlength=state.n)


def resolve_infection(
    state: State,
    spatial: NDArray,
    network: NDArray,
    params: Params,
    rng: RandomVariates,
    step: int,
) -> tuple[State, NDArray[np.float64]]:
    """Resolve the combined hazard once per susceptible agent."""
    probability = 1.0 - np.exp(-(spatial + network) * params.dt)
    newly_infected = (state.health == S) & (rng.uniform("infect", step, state.ids) < probability)
    out = state.copy()
    viral_load = params.viral_load_low + (
        params.viral_load_high - params.viral_load_low
    ) * rng.uniform("vload", step, state.ids)
    duration_sample = params.recovery_mean + params.recovery_sd * rng.normal(
        "dur", step, state.ids
    )
    duration = np.maximum(1, np.trunc(duration_sample).astype(np.int64))
    out.health[newly_infected] = I
    out.vload[newly_infected] = viral_load[newly_infected]
    out.inf_age[newly_infected] = 0
    out.rec_dur[newly_infected] = duration[newly_infected]
    return out, probability


def progress_disease(
    state: State, params: Params, rng: RandomVariates, step: int
) -> State:
    """Age infections, then resolve recovery or death at the duration threshold."""
    out = state.copy()
    infected = out.health == I
    out.inf_age[infected] += 1
    due = infected & (out.inf_age >= out.rec_dur)
    rho = params.recovery_probability_low + (
        params.recovery_probability_high - params.recovery_probability_low
    ) * rng.uniform("rho", step, state.ids)
    recovered = due & (rng.uniform("recover", step, state.ids) < rho)
    dead = due & ~recovered
    out.health[recovered] = R
    out.immunity[recovered] = params.recovered_immunity
    out.health[dead] = D
    out.mobility[dead] = 0.0
    return out


def quarantine_stage(
    state: State, params: Params, rng: RandomVariates, step: int
) -> State:
    """Attempt entry, tick quarantines, and release at the duration threshold.

    Entry precedes the duration tick, matching both production update paths.
    Therefore a new entrant receives its first tick immediately, while an
    agent released from an existing quarantine cannot re-enter this update.
    """
    out = state.copy()
    can_enter = (out.health == I) & ~out.quarantined
    entering = can_enter & (
        rng.uniform("quar_enter", step, state.ids) < params.quarantine_compliance
    )
    out.mobility[entering] *= 1.0 - params.quarantine_compliance * params.quarantine_effect
    out.quarantined[entering] = True
    out.q_start[entering] = step

    # q_start marks the update of entry; that update itself counts as tick one.
    expired = out.quarantined & (
        (step - out.q_start + 1) >= params.quarantine_duration
    )
    recovered = out.quarantined & (out.health == R)
    release = expired | recovered
    restore_mobility = release & (out.health != D)
    out.mobility[restore_mobility] = out.base_mobility[restore_mobility]
    out.quarantined[release] = False
    return out


def full_update(
    state: State,
    edges: Edges,
    params: Params,
    rng: RandomVariates,
    step: int,
    quarantine_enabled: bool,
) -> State:
    """Movement, hazards, resolution, progression, then optional quarantine.

    Scheduled network rewiring occurs after movement and before hazard
    accumulation in the production models, so callers provide that update's
    edge snapshot.
    """
    moved = movement(state, params, rng, step)
    spatial = spatial_hazard(moved, params)
    network = network_hazard(moved, edges, params)
    resolved, _ = resolve_infection(moved, spatial, network, params, rng, step)
    progressed = progress_disease(resolved, params, rng, step)
    if quarantine_enabled:
        return quarantine_stage(progressed, params, rng, step)
    return progressed


def exact_inclusion_probs(weights: NDArray, k: int) -> NDArray[np.float64]:
    """Exact inclusion probabilities for successive weighted sampling without replacement."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be a one-dimensional array of finite positive values")
    if not 0 <= k <= len(weights):
        raise ValueError("k must be between zero and the number of weights")
    inclusion = np.zeros(len(weights), dtype=float)
    if k == 0:
        return inclusion
    for sequence in permutations(range(len(weights)), k):
        probability = 1.0
        remaining = float(weights.sum())
        for index in sequence:
            probability *= weights[index] / remaining
            remaining -= weights[index]
        inclusion[list(sequence)] += probability
    return inclusion


def check_rewire_invariants(
    state: State,
    edges: Edges,
    out_degree_caps: NDArray,
    params: Params,
) -> tuple[list[str], NDArray[np.float64]]:
    """Check graph structure and return expected raw edge weights."""
    violations = []
    src = np.asarray(edges.src)
    dst = np.asarray(edges.dst)
    weights = np.asarray(edges.w)
    caps = np.asarray(out_degree_caps)
    if src.ndim != 1 or dst.ndim != 1 or weights.ndim != 1:
        return ["edge arrays must be one-dimensional"], np.empty(0, dtype=float)
    if not (len(src) == len(dst) == len(weights)):
        return ["edge arrays have different lengths"], np.empty(0, dtype=float)
    if caps.shape != (state.n,):
        return ["out-degree caps must have one value per agent"], np.empty(0, dtype=float)
    if np.any(src < 0) or np.any(src >= state.n) or np.any(dst < 0) or np.any(dst >= state.n):
        return ["edge endpoint outside population"], np.empty(0, dtype=float)
    if np.any(src == dst):
        violations.append("self-loop present")
    alive = state.health != D
    if np.any(~alive[src]) or np.any(~alive[dst]):
        violations.append("dead agent used as source or candidate")
    pairs = src.astype(np.int64) * state.n + dst.astype(np.int64)
    if len(np.unique(pairs)) != len(pairs):
        violations.append("duplicate contacts (sampling not without replacement)")
    degree = np.bincount(src.astype(np.int64), minlength=state.n)
    if np.any(degree > int(alive.sum()) - 1):
        violations.append("out-degree exceeds #alive - 1")
    if np.any(degree > caps):
        violations.append("out-degree exceeds requested K_i")
    expected = similarity_weight(
        state.pos[src], state.pos[dst], state.age[src], state.age[dst], params
    )
    return violations, expected
