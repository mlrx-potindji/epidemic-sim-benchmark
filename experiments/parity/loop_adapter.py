"""Independent scalar phase implementation used to validate the parity harness."""
from __future__ import annotations

import math

import numpy as np

from .oracle import D, I, R, S, Edges, Params, State


class LoopAdapter:
    """Simple per-agent reference with optional known defects for mutation checks."""

    MUTATIONS = frozenset(
        {
            "strict_radius",
            "spatial_quarantine_leak",
            "ignore_immunity_spatial",
            "q_theta_only",
            "round_duration",
            "newborn_age_one",
            "recover_leq",
            "no_release_on_recovery",
            "dead_mobility_restored",
            "enter_prob_one",
            "movement_wrap",
            "quarantine_before_progression",
            "rewire_uniform",
            "store_normalized_weight",
            "rewire_ignores_age",
            "rewire_includes_dead",
            "degree_wrong_p",
        }
    )

    def __init__(self, mutations=()):
        unknown = set(mutations) - self.MUTATIONS
        if unknown:
            raise ValueError(f"unknown mutations: {sorted(unknown)}")
        self.mutations = frozenset(mutations)

    def movement(self, state: State, params: Params, rng, step: int) -> State:
        result = state.copy()
        for agent in range(state.n):
            if state.health[agent] == D:
                continue
            for axis, purpose in enumerate(("move_x", "move_y")):
                displacement = (
                    2 * rng.uniform(purpose, step, np.asarray([agent]))[0] - 1
                ) * state.mobility[agent]
                position = state.pos[agent, axis] + displacement
                if "movement_wrap" in self.mutations:
                    position %= params.world_size
                else:
                    position = min(max(position, 0.0), params.world_size)
                result.pos[agent, axis] = position
        return result

    def hazards(self, state: State, edges: Edges, params: Params):
        spatial = np.zeros(state.n)
        network = np.zeros(state.n)
        for target in range(state.n):
            if state.health[target] != S:
                continue
            for source in range(state.n):
                if state.health[source] != I:
                    continue
                if (
                    state.quarantined[source]
                    and "spatial_quarantine_leak" not in self.mutations
                ):
                    continue
                distance = math.dist(state.pos[source], state.pos[target])
                inside = (
                    distance < params.transmission_radius
                    if "strict_radius" in self.mutations
                    else distance <= params.transmission_radius
                )
                if inside:
                    spatial[target] += (
                        params.beta_spatial * state.vload[source] / 1000.0
                    )
            if "ignore_immunity_spatial" not in self.mutations:
                spatial[target] *= 1.0 - state.immunity[target]

        for source, target, weight in zip(edges.src, edges.dst, edges.w):
            if state.health[source] != I or state.health[target] != S:
                continue
            attenuation = (
                params.quarantine_effect
                if "q_theta_only" in self.mutations
                else params.quarantine_compliance * params.quarantine_effect
            )
            modifier = 1.0 - (
                attenuation if state.quarantined[source] else 0.0
            )
            network[target] += (
                params.beta_network
                * modifier
                * weight
                * (1.0 - state.immunity[target])
                * state.vload[source]
                / 1000.0
            )
        return spatial, network

    def resolve(self, state: State, spatial, network, params: Params, rng, step: int):
        out = state.copy()
        probability = 1.0 - np.exp(-(spatial + network) * params.dt)
        for agent in range(state.n):
            if state.health[agent] != S:
                continue
            if rng.uniform("infect", step, np.asarray([agent]))[0] >= probability[agent]:
                continue
            out.health[agent] = I
            out.vload[agent] = params.viral_load_low + (
                params.viral_load_high - params.viral_load_low
            ) * rng.uniform("vload", step, np.asarray([agent]))[0]
            duration_sample = params.recovery_mean + params.recovery_sd * rng.normal(
                "dur", step, np.asarray([agent])
            )[0]
            duration = (
                round(duration_sample)
                if "round_duration" in self.mutations
                else int(duration_sample)
            )
            out.rec_dur[agent] = max(1, duration)
            out.inf_age[agent] = (
                1 if "newborn_age_one" in self.mutations else 0
            )
        return out, probability

    def progress(self, state: State, params: Params, rng, step: int) -> State:
        out = state.copy()
        for agent in range(state.n):
            if out.health[agent] != I:
                continue
            out.inf_age[agent] += 1
            if out.inf_age[agent] < out.rec_dur[agent]:
                continue
            rho = params.recovery_probability_low + (
                params.recovery_probability_high - params.recovery_probability_low
            ) * rng.uniform("rho", step, np.asarray([agent]))[0]
            recover_draw = rng.uniform("recover", step, np.asarray([agent]))[0]
            recovered = (
                recover_draw <= rho
                if "recover_leq" in self.mutations
                else recover_draw < rho
            )
            if recovered:
                out.health[agent] = R
                out.immunity[agent] = params.recovered_immunity
                if "no_release_on_recovery" not in self.mutations:
                    out.mobility[agent] = out.base_mobility[agent]
                    out.quarantined[agent] = False
            else:
                out.health[agent] = D
                out.mobility[agent] = 0.0
        return out

    def quarantine(self, state: State, params: Params, rng, step: int) -> State:
        out = state.copy()
        for agent in range(state.n):
            if out.health[agent] == I and not out.quarantined[agent]:
                draw = rng.uniform("quar_enter", step, np.asarray([agent]))[0]
                if draw < params.quarantine_compliance or "enter_prob_one" in self.mutations:
                    out.mobility[agent] *= (
                        1.0
                        - params.quarantine_compliance * params.quarantine_effect
                    )
                    out.quarantined[agent] = True
                    out.q_start[agent] = step
        for agent in range(state.n):
            if not out.quarantined[agent]:
                continue
            expired = step - out.q_start[agent] + 1 >= params.quarantine_duration
            recovered = (
                out.health[agent] == R
                and "no_release_on_recovery" not in self.mutations
            )
            if expired or recovered:
                if (
                    out.health[agent] != D
                    or "dead_mobility_restored" in self.mutations
                ):
                    out.mobility[agent] = out.base_mobility[agent]
                out.quarantined[agent] = False
        return out

    def full_update(
        self,
        state: State,
        edges: Edges,
        params: Params,
        rng,
        step: int,
        quarantine_enabled: bool,
    ) -> State:
        moved = self.movement(state, params, rng, step)
        spatial, network = self.hazards(moved, edges, params)
        resolved, _ = self.resolve(moved, spatial, network, params, rng, step)
        if quarantine_enabled and "quarantine_before_progression" in self.mutations:
            return self.progress(
                self.quarantine(resolved, params, rng, step), params, rng, step
            )
        progressed = self.progress(resolved, params, rng, step)
        return (
            self.quarantine(progressed, params, rng, step)
            if quarantine_enabled
            else progressed
        )

    def degree_sample(self, params: Params, size: int, rng) -> np.ndarray:
        dispersion = params.rewiring_dispersion
        probability = dispersion / (dispersion + params.mean_degree)
        if "degree_wrong_p" in self.mutations:
            probability = params.mean_degree / (dispersion + params.mean_degree)
        return rng.negative_binomial(dispersion, probability, size=size)

    def rewire(
        self,
        state: State,
        params: Params,
        rng,
        requested_degrees: np.ndarray,
        sources=None,
    ) -> Edges:
        alive = np.flatnonzero(state.health != D)
        pool = (
            np.arange(state.n)
            if "rewire_includes_dead" in self.mutations
            else alive
        )
        selected_sources = alive if sources is None else np.asarray(sources)
        srcs, dsts, weights = [], [], []
        for source in selected_sources:
            candidates = pool[pool != source]
            candidate_ages = (
                np.zeros(len(candidates))
                if "rewire_ignores_age" in self.mutations
                else state.age[candidates]
            )
            source_age = (
                0 if "rewire_ignores_age" in self.mutations else state.age[source]
            )
            distance = np.linalg.norm(state.pos[candidates] - state.pos[source], axis=1)
            candidate_weights = np.exp(
                -distance / params.similarity_length
                - np.abs(candidate_ages - source_age) / params.age_similarity_scale
            )
            count = min(int(requested_degrees[source]), len(alive) - 1)
            if count <= 0 or len(candidates) == 0:
                continue
            probabilities = (
                None
                if "rewire_uniform" in self.mutations
                else candidate_weights / candidate_weights.sum()
            )
            selected = rng.choice(len(candidates), size=count, replace=False, p=probabilities)
            stored_weights = candidate_weights[selected]
            if "store_normalized_weight" in self.mutations:
                stored_weights = stored_weights / candidate_weights.sum()
            srcs.extend([source] * count)
            dsts.extend(candidates[selected].tolist())
            weights.extend(stored_weights.tolist())
        return Edges(
            src=np.asarray(srcs, dtype=np.int64),
            dst=np.asarray(dsts, dtype=np.int64),
            w=np.asarray(weights, dtype=float),
        )
