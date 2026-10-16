"""Mesa (object-oriented) implementation of the spatial epidemic model.

Benchmark counterpart of ``epidemic_sim.ecs.simulation.SIREpidemicModel``: same
parameters, same mechanisms, same order of the systems inside one step, same outputs.
It is written with Mesa's own tools only, the way a Mesa user would write it:

* agents are created with ``Person.create_agents`` and activated through ``AgentSet``
  (``select`` / ``get`` / ``groupby`` / ``do``);
* positions live in the Mesa continuous space; contacts within the transmission radius are
  found with ``Person.get_neighbors_in_radius`` and distances with ``space.calculate_distances``;
* stochastic model draws use ``model.rng`` (NumPy);
* ``DataCollector`` records the series; ``Model.run_until`` advances the model and
  ``Model.steps`` is the step counter.

Order of one step (highest ECS priority first):

    1. movement                       (ECS priority 100)
    2. contact-network rewiring       (95, similarity network only, every ``rewire_every`` steps)
    3. spatial transmission           (90)
    4. network transmission           (80)
    5. infection resolution           (70, when a "new" hazard-based system is used)
    6. disease progression            (60)
    7. quarantine                     (60, optional, after progression)
    8. data collection

Parity with the ECS is epidemiological (distribution of the epidemic indicators over
replicates), not draw by draw: the two frameworks do not consume random numbers in the
same order.

Behaviours copied as-is from the ECS code, that the author may want to review:
    * the rewiring uses a fixed dispersion ``k = 0.5`` while the initial network uses the
      ``dispersion`` parameter;
    * contacts are directed (i chooses j; j does not necessarily choose i).
"""

import math
from typing import Mapping, Optional

import mesa
import networkx as nx
import numpy as np
from mesa.datacollection import DataCollector
from mesa.experimental.continuous_space import ContinuousSpace

from .agents import Person
from ..config import SimulationConfig
from ..disease_model import (
    AGE_RANGE,
    IMMUNITY_AFTER_RECOVERY,
    IMMUNITY_RANGE,
    INITIAL_DAYS_INFECTED,
    INITIAL_VIRAL_LOAD_RANGE,
    MOBILITY_RANGE,
    NEW_VIRAL_LOAD_RANGE,
    QUARANTINE_COMPLIANCE,
    RANDOM_NETWORK_STRENGTH_RANGE,
    RECOVERY_TIME_MEAN,
    RECOVERY_TIME_SD,
    REWIRING_DISPERSION,
    DiseaseModelConfig,
    HealthState,
)
from ..validation import validate_all

STATE_SERIES_NAMES = {state: state.value for state in HealthState}


def health_counts(
    n_agents: int,
    initial_infected: int,
    proportions: Optional[Mapping[HealthState, float]] = None,
) -> dict[HealthState, int]:
    if proportions is None:
        return {
            HealthState.SUSCEPTIBLE: n_agents - initial_infected,
            HealthState.INFECTED: initial_infected,
            HealthState.RECOVERED: 0,
            HealthState.DEAD: 0,
        }
    if any(p < 0 for p in proportions.values()) or abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("initial proportions must be non-negative and sum to 1")
    counts = {state: int(round(proportions.get(state, 0.0) * n_agents)) for state in HealthState}
    others = sum(count for state, count in counts.items() if state is not HealthState.SUSCEPTIBLE)
    if others > n_agents:
        raise ValueError("initial proportions give more people than n_agents")
    counts[HealthState.SUSCEPTIBLE] = n_agents - others
    return counts


class MesaSIREpidemicModel(mesa.Model):
    def __init__(self, seed: int | None = None, tau: float = 25, alpha: Optional[float] = None,
                 n_agents: int = 500, world_size: int = 100, initial_infected: int = 5,
                 average_contacts: int = 10, beta_spatial: float = 0.10,
                 beta_network: float = 0.20, enable_quarantine: bool = False,
                 transmission_radius: float = 4.0, spatial_new: Optional[bool] = False,
                 network_new: Optional[bool] = False,
                 space_attribute_similarity: Optional[bool] = False,
                 dt: float = 1.0, dispersion: float = 0.7, rewire_every: int = 18,
                 collect_spatial: bool = True,
                 initial_proportions: Optional[Mapping[HealthState, float]] = None,
                 simulation_config: SimulationConfig | None = None,
                 disease_model_config: DiseaseModelConfig | None = None):
        """Initialize from shared configs or legacy keyword parameters.

        A supplied config takes precedence over legacy parameters in its group.
        """
        simulation_config = simulation_config or SimulationConfig(
            seed=seed,
            n_agents=n_agents,
            world_size=world_size,
            initial_infected=initial_infected,
            average_contacts=average_contacts,
            collect_spatial=collect_spatial,
        )
        disease_model_config = disease_model_config or DiseaseModelConfig(
            beta_spatial=beta_spatial,
            beta_network=beta_network,
            transmission_radius=transmission_radius,
            dt=dt,
            alpha=alpha,
            tau=tau,
            dispersion=dispersion,
            spatial_new=bool(spatial_new),
            network_new=bool(network_new),
            space_attribute_similarity=bool(space_attribute_similarity),
            enable_quarantine=enable_quarantine,
        )
        validate_all(simulation_config, disease_model_config)
        self.simulation_config = simulation_config
        self.disease_model_config = disease_model_config
        seed = simulation_config.seed
        n_agents = simulation_config.n_agents
        world_size = simulation_config.world_size
        initial_infected = simulation_config.initial_infected
        average_contacts = simulation_config.average_contacts
        collect_spatial = simulation_config.collect_spatial
        beta_spatial = disease_model_config.beta_spatial
        beta_network = disease_model_config.beta_network
        enable_quarantine = disease_model_config.enable_quarantine
        transmission_radius = disease_model_config.transmission_radius
        dt = disease_model_config.dt
        alpha = disease_model_config.alpha
        tau = disease_model_config.tau
        dispersion = disease_model_config.dispersion
        spatial_new = disease_model_config.spatial_new
        network_new = disease_model_config.network_new
        space_attribute_similarity = disease_model_config.space_attribute_similarity
        super().__init__(rng=seed)

        self.n_agents: int = n_agents
        self.world_size: int = world_size
        self.initial_infected: int = initial_infected
        self.average_contacts: int = average_contacts
        self.beta_spatial: float = beta_spatial
        self.beta_network: float = beta_network
        self.enable_quarantine: bool = enable_quarantine
        self.seed: int | None = seed
        self.transmission_radius: float = transmission_radius
        self.dt: float = dt
        self.alpha: float = alpha if alpha is not None else 0.20 * world_size
        self.tau: float = tau
        self.dispersion: float = dispersion
        self.rewire_every: int = rewire_every
        self.collect_spatial: bool = collect_spatial
        self.spatial_new: bool = bool(spatial_new)
        self.network_new: bool = bool(network_new)
        self._uses_similarity_network: bool = bool(space_attribute_similarity)
        self.seed_locations: list[tuple[float, float]] = []

        self.space: ContinuousSpace = ContinuousSpace(
            [[0, world_size], [0, world_size]], torus=False,
            random=self.random, n_agents=n_agents)
        self._populate(health_counts(n_agents, initial_infected, initial_proportions))
        self.contact_graph: nx.DiGraph = nx.DiGraph()
        self.contact_graph.add_nodes_from(self.space.active_agents)
        if self._uses_similarity_network:
            self._build_similarity_network(self.dispersion, alive_only=False)
        else:
            self._create_social_network()

        self._counts: dict[HealthState, int] = {}
        model_reporters = {"time": lambda m: m.steps - 1}
        for state, name in STATE_SERIES_NAMES.items():
            model_reporters[name] = (lambda m, s=state: m._counts.get(s, 0))
        # attribute names (not lambdas): faster in the DataCollector for large populations
        agent_reporters = ({"state": "state", "x": "x", "y": "y"} if collect_spatial else None)
        self.datacollector = DataCollector(model_reporters=model_reporters,
                                           agent_reporters=agent_reporters)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _draw_recovery_time(self) -> int:
        return max(1, int(self.rng.normal(RECOVERY_TIME_MEAN, RECOVERY_TIME_SD)))

    def _draw_new_infection(self) -> tuple[float, int]:
        """(viral_load, recovery_time) of a new infection."""
        return self.rng.uniform(*NEW_VIRAL_LOAD_RANGE), self._draw_recovery_time()

    def _populate(self, counts: dict[HealthState, int]) -> None:
        n: int = self.n_agents
        people = Person.create_agents(
            self, n, self.space,
            position=self.rng.uniform(0, self.world_size, (n, 2)),
            age=self.rng.integers(*AGE_RANGE, n).tolist(),
            mobility=self.rng.uniform(*MOBILITY_RANGE, n).tolist(),
            immunity=self.rng.uniform(*IMMUNITY_RANGE, n).tolist()).to_list()
        order: np.ndarray = self.rng.permutation(n)
        cursor = 0
        for state in (HealthState.INFECTED, HealthState.RECOVERED, HealthState.DEAD):
            for i in order[cursor:cursor + counts[state]]:
                person: Person = people[i]
                if state is HealthState.INFECTED:
                    person.infect(self.rng.uniform(*INITIAL_VIRAL_LOAD_RANGE),
                                  self._draw_recovery_time())
                    person.days_infected = INITIAL_DAYS_INFECTED  # seeding day, as in the ECS
                    self.seed_locations.append((person.x, person.y))
                elif state is HealthState.RECOVERED:
                    person.state = HealthState.RECOVERED
                    person.immunity = IMMUNITY_AFTER_RECOVERY
                else:
                    person.state = HealthState.DEAD
                    person.mobility = 0.0
            cursor += counts[state]

    def _create_social_network(self) -> None:
        """Random network: Poisson number of contacts, strengths U(0.1, 1)."""
        people: list[Person] = self.space.active_agents
        n: int = len(people)
        for i, person in enumerate(people):
            k: int = min(max(0, int(self.rng.poisson(self.average_contacts))), n - 1)
            if k > 0:
                chosen: np.ndarray = self.rng.choice(n - 1, size=k, replace=False)
                chosen += chosen >= i  # skip the agent itself
                strengths = self.rng.uniform(*RANDOM_NETWORK_STRENGTH_RANGE, k)
                self.contact_graph.add_weighted_edges_from(
                    (person, people[j], float(weight))
                    for j, weight in zip(chosen, strengths)
                )

    def _build_similarity_network(self, dispersion: float, alive_only: bool) -> None:
        """Spatial + age homophily network: every (living) agent draws a negative-binomial
        number of contacts (``Person.choose_contacts``)."""
        people: list[Person] = self.space.active_agents
        ages: np.ndarray = np.asarray(self.space.agents.get("age"), dtype=float)
        alive: np.ndarray = (np.asarray([p.state is not HealthState.DEAD for p in people])
                             if alive_only else np.ones(len(people), dtype=bool))
        wanted: np.ndarray = self.rng.negative_binomial(
            dispersion, dispersion / (dispersion + self.average_contacts), len(people))
        for person, k, is_alive in zip(people, wanted, alive):
            if is_alive:
                if k > 0:
                    contacts, strengths = person.choose_contacts(
                        int(k), ages, alive, self.alpha, self.tau
                    )
                    self.contact_graph.add_weighted_edges_from(
                        (person, contact, strength)
                        for contact, strength in zip(contacts, strengths)
                    )

    # ------------------------------------------------------------------
    # Systems
    # ------------------------------------------------------------------
    def _move(self) -> None:
        for person in self.rng.permutation(list(self.agents)):
            person.move()

    def _rewire_network(self) -> None:
        """Dynamic rewiring every ``rewire_every`` steps (ECS ``NetworkRewiringystem``)."""
        if self.steps % self.rewire_every == 0:
            self.contact_graph.clear_edges()
            self._build_similarity_network(REWIRING_DISPERSION, alive_only=True)

    def _spatial_transmission_hazard(self) -> None:
        """Hazard-based spatial transmission (ECS ``SpatialTransmissionSystemNew``): every
        infectious source adds its hazard to the susceptibles within the radius."""
        for source in self.agents.select(
                lambda a: a.state is HealthState.INFECTED and a.infectious and not a.quarantined):
            base: float = self.beta_spatial * source.viral_load / 1000
            neighbors, _ = source.get_neighbors_in_radius(radius=self.transmission_radius)
            for target in neighbors:
                if target.state is HealthState.SUSCEPTIBLE:
                    target.hazard += base * (1 - target.immunity)

    def _spatial_transmission_immediate(self) -> None:
        """Sequential spatial transmission (ECS ``SpatialTransmissionSystem``). The neighbours
        are filtered on their state when each source acts, so a person infected by an earlier
        source of the same pass is not drawn again."""
        for source in self.agents.select(
                lambda a: a.state is HealthState.INFECTED and a.infectious and not a.quarantined):
            base: float = self.beta_spatial * source.viral_load / 1000
            neighbors, _ = source.get_neighbors_in_radius(radius=self.transmission_radius)
            for target in [t for t in neighbors if t.state is HealthState.SUSCEPTIBLE]:
                if self.rng.random() < base * (1 - target.immunity):
                    target.infect(*self._draw_new_infection())

    def _network_transmission_hazard(self) -> None:
        """Hazard-based network transmission (ECS ``NetworkTransmissionSystemNew``)."""
        for source in self.agents.select(lambda a: a.state is HealthState.INFECTED and a.infectious):
            adjusted: float = self.beta_network * source.transmission_modifier()
            for contact in self.contact_graph.successors(source):
                if contact.state is HealthState.SUSCEPTIBLE:
                    strength = self.contact_graph[source][contact]["weight"]
                    hazard: float = adjusted * strength * (1 - contact.immunity) * source.viral_load / 1000
                    if hazard > 0:
                        contact.hazard += hazard

    def _network_transmission_immediate(self) -> None:
        """Sequential network transmission (ECS ``NetworkTransmissionSystem``). The state of a
        contact is read when the draw is made: it may have been infected earlier in the pass."""
        for source in self.agents.select(lambda a: a.state is HealthState.INFECTED and a.infectious
                                         and self.contact_graph.out_degree(a) > 0):
            adjusted: float = self.beta_network * source.transmission_modifier()
            for contact in self.contact_graph.successors(source):
                strength = self.contact_graph[source][contact]["weight"]
                if contact.state is HealthState.SUSCEPTIBLE:
                    probability: float = adjusted * strength * (1 - contact.immunity) * source.viral_load / 1000
                    if self.rng.random() < probability:
                        contact.infect(*self._draw_new_infection())

    def _resolve_infections(self) -> None:
        """Hazard -> probability p = 1 - exp(-hazard * dt) (ECS ``InfectionResolutionSystem``)."""
        for person in self.agents.select(lambda a: a.state is HealthState.SUSCEPTIBLE and a.hazard > 0):
            probability: float = 1 - math.exp(-person.hazard * self.dt)
            if self.rng.random() < probability:
                person.infect(*self._draw_new_infection())
            person.hazard = 0.0

    # ------------------------------------------------------------------
    # One step / run
    # ------------------------------------------------------------------
    def step(self) -> None:
        if not self.running:  # nobody left to infect: the remaining steps do nothing
            return
        self._move()
        if self._uses_similarity_network:
            self._rewire_network()
        if self.spatial_new:
            self._spatial_transmission_hazard()
        else:
            self._spatial_transmission_immediate()
        if self.network_new:
            self._network_transmission_hazard()
        else:
            self._network_transmission_immediate()
        if self.spatial_new or self.network_new:
            self._resolve_infections()
        self.agents.select(lambda a: a.state is HealthState.INFECTED).do("progress_disease")
        if self.enable_quarantine:
            self.agents.select(lambda a: a.state is HealthState.INFECTED and a.infectious
                               and not a.quarantined).do("try_quarantine", QUARANTINE_COMPLIANCE)
            self.agents.select(lambda a: a.quarantined).do("tick_quarantine")

        self._counts = self.agents.groupby("state").count()
        self.datacollector.collect(self)
        if self._counts.get(HealthState.INFECTED, 0) == 0:
            self.running = False  # same stopping rule as the ECS

    def run(self, max_steps: int) -> None:
        """Advance until step ``max_steps`` (``Model.run_until``); the model stops by itself
        when nobody is infected any more."""
        self.run_until(max_steps)
        if not self.running:
            print(f"Simulation ended at step {self.datacollector.model_vars['time'][-1] + 1}"
                  " - no more infected entities.")

    # ------------------------------------------------------------------
    # Outputs (same structure as the ECS model)
    # ------------------------------------------------------------------
    @property
    def time_series_data(self) -> dict[str, list]:
        """Counts of each health state at every step (dict of lists, ECS format)."""
        return self.datacollector.get_model_vars_dataframe().to_dict("list")

    @property
    def spatial_location_series_data(self) -> dict[str, list[tuple[int, float, float]]]:
        """(step, x, y) of every agent, by health state (ECS format)."""
        frame = self.datacollector.get_agent_vars_dataframe().reset_index()
        series: dict[str, list[tuple[int, float, float]]] = {n: [] for n in STATE_SERIES_NAMES.values()}
        for step, state, x, y in zip(frame["Step"], frame["state"], frame["x"], frame["y"]):
            series[STATE_SERIES_NAMES[state]].append((int(step) - 1, float(x), float(y)))
        return series
