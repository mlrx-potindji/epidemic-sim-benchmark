"""Adapters that expose production model phases to parity tests."""
from __future__ import annotations

from dataclasses import dataclass

import esper
import numpy as np

from .oracle import D, I, R, S, Edges, Params, State
from epidemic_sim.disease_model import HealthState, QUARANTINE_EFFECT
from epidemic_sim.ecs.components import (
    ContactNetwork,
    Dead,
    Demographics,
    InfectionHazard,
    Infected,
    Location,
    Quarantined,
    Recovered,
    Susceptible,
)
from epidemic_sim.ecs.simulation import SIREpidemicModel
from epidemic_sim.ecs.systems.transmission import (
    MovementSystem,
    NetworkTransmissionSystemNew,
    SpatialTransmissionSystemNew,
)
from epidemic_sim.mesa import MesaSIREpidemicModel
from epidemic_sim.randomness import PhaseRandomVariates, RandomStreams


@dataclass(frozen=True)
class PhaseHazards:
    spatial: np.ndarray
    network: np.ndarray


class _MappedRandomVariates:
    def __init__(self, rng: PhaseRandomVariates, id_map: dict[int, int]) -> None:
        self.rng = rng
        self.id_map = id_map

    def _map_ids(self, ids: np.ndarray) -> np.ndarray:
        values = np.asarray(ids)
        return np.asarray(
            [self.id_map[int(agent_id)] for agent_id in values.flat],
            dtype=np.int64,
        ).reshape(values.shape)

    def uniform(self, purpose: str, step: int, ids: np.ndarray) -> np.ndarray:
        return self.rng.uniform(purpose, step, self._map_ids(ids))

    def normal(self, purpose: str, step: int, ids: np.ndarray) -> np.ndarray:
        return self.rng.normal(purpose, step, self._map_ids(ids))


def _validate_supported_params(params: Params) -> None:
    if params.quarantine_effect != QUARANTINE_EFFECT:
        raise ValueError(
            "production implementations use the fixed QUARANTINE_EFFECT "
            f"({QUARANTINE_EFFECT}), got {params.quarantine_effect}"
        )


def _health_to_mesa(health: int) -> HealthState:
    return {
        S: HealthState.SUSCEPTIBLE,
        I: HealthState.INFECTED,
        R: HealthState.RECOVERED,
        D: HealthState.DEAD,
    }[int(health)]


def _health_from_mesa(health: HealthState) -> int:
    return {
        HealthState.SUSCEPTIBLE: S,
        HealthState.INFECTED: I,
        HealthState.RECOVERED: R,
        HealthState.DEAD: D,
    }[health]


class MesaAdapter:
    """Load snapshots into Mesa, execute one production phase, and read it back."""

    def __init__(
        self,
        state: State,
        edges: Edges,
        params: Params,
        seed: int = 0,
    ) -> None:
        _validate_supported_params(params)
        self.params = params
        self.model = MesaSIREpidemicModel(
            seed=seed,
            n_agents=state.n,
            world_size=params.world_size,
            initial_infected=min(1, state.n),
            beta_spatial=params.beta_spatial,
            beta_network=params.beta_network,
            transmission_radius=params.transmission_radius,
            spatial_new=True,
            network_new=True,
            enable_quarantine=True,
            dt=params.dt,
            collect_spatial=False,
        )
        self.load(state, edges)

    def __enter__(self) -> MesaAdapter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.model.contact_graph.clear()
        for person in list(self.model.space.active_agents):
            person.remove()

    def load(self, state: State, edges: Edges) -> None:
        people = list(self.model.space.active_agents)
        if len(people) != state.n:
            raise ValueError("snapshot population does not match Mesa model")
        for index, person in enumerate(people):
            person.position = state.pos[index].astype(float, copy=True)
            person.age = int(state.age[index])
            person.mobility = float(state.mobility[index])
            person.original_mobility = float(state.base_mobility[index])
            person.immunity = float(state.immunity[index])
            person.state = _health_to_mesa(state.health[index])
            person.viral_load = float(state.vload[index])
            person.days_infected = int(state.inf_age[index])
            person.recovery_time = int(state.rec_dur[index])
            person.infectious = state.health[index] == I
            person.quarantined = bool(state.quarantined[index])
            person.quarantine_compliance = self.params.quarantine_compliance
            person.quarantine_duration = self.params.quarantine_duration
            person.days_in_quarantine = 0
            person._parity_q_start = int(state.q_start[index])
            person.hazard = 0.0

        self.model.contact_graph.clear_edges()
        self.model.contact_graph.add_weighted_edges_from(
            (
                people[int(src)],
                people[int(dst)],
                float(weight),
            )
            for src, dst, weight in zip(edges.src, edges.dst, edges.w)
        )

    def read_state(self) -> State:
        people = list(self.model.space.active_agents)
        return State(
            pos=np.asarray([person.position for person in people], dtype=float),
            age=np.asarray([person.age for person in people], dtype=np.int64),
            mobility=np.asarray([person.mobility for person in people], dtype=float),
            base_mobility=np.asarray(
                [person.original_mobility for person in people], dtype=float
            ),
            immunity=np.asarray([person.immunity for person in people], dtype=float),
            health=np.asarray(
                [_health_from_mesa(person.state) for person in people],
                dtype=np.int8,
            ),
            vload=np.asarray([person.viral_load for person in people], dtype=float),
            inf_age=np.asarray([person.days_infected for person in people], dtype=np.int64),
            rec_dur=np.asarray([person.recovery_time for person in people], dtype=np.int64),
            quarantined=np.asarray(
                [person.quarantined for person in people], dtype=bool
            ),
            q_start=np.asarray(
                [person._parity_q_start for person in people], dtype=np.int64
            ),
        )

    def movement(self, rng: PhaseRandomVariates, step: int) -> State:
        self.model._move(rng, step)
        return self.read_state()

    def hazards(self) -> PhaseHazards:
        people = list(self.model.space.active_agents)
        for person in people:
            person.hazard = 0.0
        self.model._spatial_transmission_hazard()
        spatial = np.asarray([person.hazard for person in people], dtype=float)
        for person in people:
            person.hazard = 0.0
        self.model._network_transmission_hazard()
        network = np.asarray([person.hazard for person in people], dtype=float)
        return PhaseHazards(spatial=spatial, network=network)


class ECSAdapter:
    """Load snapshots into ECS, execute one production phase, and read it back."""

    _HEALTH_COMPONENTS = (Susceptible, Infected, Recovered, Dead)

    def __init__(
        self,
        state: State,
        edges: Edges,
        params: Params,
        seed: int = 0,
    ) -> None:
        _validate_supported_params(params)
        self.params = params
        self.model = SIREpidemicModel(
            seed=seed,
            n_agents=state.n,
            world_size=params.world_size,
            initial_infected=min(1, state.n),
            beta_spatial=params.beta_spatial,
            beta_network=params.beta_network,
            transmission_radius=params.transmission_radius,
            spatial_new=True,
            network_new=True,
            enable_quarantine=True,
            dt=params.dt,
        )
        self.entity_ids = list(self.model.entity_iDs)
        self.id_map = {entity_id: index for index, entity_id in enumerate(self.entity_ids)}
        self.load(state, edges)

    def __enter__(self) -> ECSAdapter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.model.clean_up()

    def load(self, state: State, edges: Edges) -> None:
        if len(self.entity_ids) != state.n:
            raise ValueError("snapshot population does not match ECS model")
        self._base_mobility = state.base_mobility.copy()
        self._immunity = state.immunity.copy()
        self._vload = state.vload.copy()
        self._infection_age = state.inf_age.copy()
        self._recovery_duration = state.rec_dur.copy()
        esper.switch_world(self.model.world_name)
        for index, entity in enumerate(self.entity_ids):
            for component in (*self._HEALTH_COMPONENTS, Quarantined, InfectionHazard):
                if esper.has_component(entity, component):
                    esper.remove_component(entity, component)
            esper.add_component(
                entity,
                Location(float(state.pos[index, 0]), float(state.pos[index, 1])),
            )
            demographics = esper.component_for_entity(entity, Demographics)
            demographics.age = int(state.age[index])
            demographics.mobility = float(state.mobility[index])
            if state.health[index] == S:
                esper.add_component(entity, Susceptible(float(state.immunity[index])))
            elif state.health[index] == I:
                esper.add_component(
                    entity,
                    Infected(
                        viral_load=float(state.vload[index]),
                        days_infected=int(state.inf_age[index]),
                        infectious=True,
                        recovery_time=int(state.rec_dur[index]),
                    ),
                )
            elif state.health[index] == R:
                esper.add_component(entity, Recovered(float(state.immunity[index])))
            else:
                esper.add_component(entity, Dead())
                demographics.mobility = float(state.mobility[index])
            if state.quarantined[index]:
                esper.add_component(
                    entity,
                    Quarantined(
                        start_day=int(state.q_start[index]),
                        compliance_level=self.params.quarantine_compliance,
                        duration=self.params.quarantine_duration,
                        original_mobility=float(state.base_mobility[index]),
                    ),
                )

        contacts: dict[int, tuple[list[int], list[float]]] = {
            entity: ([], []) for entity in self.entity_ids
        }
        for src, dst, weight in zip(edges.src, edges.dst, edges.w):
            contacts[self.entity_ids[int(src)]][0].append(self.entity_ids[int(dst)])
            contacts[self.entity_ids[int(src)]][1].append(float(weight))
        for entity in self.entity_ids:
            targets, strengths = contacts[entity]
            esper.add_component(
                entity, ContactNetwork(contacts=targets, contact_strength=strengths)
            )

    def read_state(self) -> State:
        esper.switch_world(self.model.world_name)
        positions = []
        ages = []
        mobilities = []
        base_mobilities = []
        immunities = []
        health = []
        viral_loads = []
        infection_ages = []
        recovery_durations = []
        quarantined = []
        quarantine_starts = []
        for index, entity in enumerate(self.entity_ids):
            location = esper.component_for_entity(entity, Location)
            demographics = esper.component_for_entity(entity, Demographics)
            positions.append((location.x, location.y))
            ages.append(demographics.age)
            mobilities.append(demographics.mobility)
            if esper.has_component(entity, Infected):
                infected = esper.component_for_entity(entity, Infected)
                health.append(I)
                viral_loads.append(infected.viral_load)
                infection_ages.append(infected.days_infected)
                recovery_durations.append(infected.recovery_time)
            elif esper.has_component(entity, Recovered):
                health.append(R)
                viral_loads.append(self._vload[index])
                infection_ages.append(self._infection_age[index])
                recovery_durations.append(self._recovery_duration[index])
            elif esper.has_component(entity, Dead):
                health.append(D)
                viral_loads.append(self._vload[index])
                infection_ages.append(self._infection_age[index])
                recovery_durations.append(self._recovery_duration[index])
            else:
                health.append(S)
                viral_loads.append(self._vload[index])
                infection_ages.append(self._infection_age[index])
                recovery_durations.append(self._recovery_duration[index])
            if esper.has_component(entity, Susceptible):
                immunities.append(esper.component_for_entity(entity, Susceptible).immunity)
            elif esper.has_component(entity, Recovered):
                immunities.append(esper.component_for_entity(entity, Recovered).immunity)
            else:
                immunities.append(self._immunity[index])
            if esper.has_component(entity, Quarantined):
                quarantine = esper.component_for_entity(entity, Quarantined)
                quarantined.append(True)
                quarantine_starts.append(quarantine.start_day)
                base_mobilities.append(quarantine.original_mobility)
            else:
                quarantined.append(False)
                quarantine_starts.append(0)
                base_mobilities.append(self._base_mobility[index])
        return State(
            pos=np.asarray(positions, dtype=float),
            age=np.asarray(ages, dtype=np.int64),
            mobility=np.asarray(mobilities, dtype=float),
            base_mobility=np.asarray(base_mobilities, dtype=float),
            immunity=np.asarray(immunities, dtype=float),
            health=np.asarray(health, dtype=np.int8),
            vload=np.asarray(viral_loads, dtype=float),
            inf_age=np.asarray(infection_ages, dtype=np.int64),
            rec_dur=np.asarray(recovery_durations, dtype=np.int64),
            quarantined=np.asarray(quarantined, dtype=bool),
            q_start=np.asarray(quarantine_starts, dtype=np.int64),
        )

    def movement(self, rng: PhaseRandomVariates, step: int) -> State:
        esper.switch_world(self.model.world_name)
        MovementSystem(
            self.params.world_size,
            self.params.world_size,
            self.model.rng,
        ).process(
            phase_rng=_MappedRandomVariates(rng, self.id_map),
            step=step,
        )
        return self.read_state()

    def hazards(self) -> PhaseHazards:
        esper.switch_world(self.model.world_name)
        for entity in self.entity_ids:
            if esper.has_component(entity, InfectionHazard):
                esper.remove_component(entity, InfectionHazard)
        SpatialTransmissionSystemNew(
            self.params.transmission_radius,
            self.params.beta_spatial,
            self.model.rng,
        ).process()
        spatial = self._read_hazards()
        for entity in self.entity_ids:
            if esper.has_component(entity, InfectionHazard):
                esper.remove_component(entity, InfectionHazard)
        NetworkTransmissionSystemNew(
            self.params.beta_network,
            self.model.rng,
        ).process()
        network = self._read_hazards()
        return PhaseHazards(spatial=spatial, network=network)

    def _read_hazards(self) -> np.ndarray:
        return np.asarray(
            [
                esper.component_for_entity(entity, InfectionHazard).hazard
                if esper.has_component(entity, InfectionHazard)
                else 0.0
                for entity in self.entity_ids
            ],
            dtype=float,
        )
