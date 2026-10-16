"""Object-oriented agent of the Mesa implementation.

One ``Person`` object holds the state of one individual (demographics, health state,
infection, quarantine, contacts) and owns its behaviour (movement, disease progression,
quarantine, choice of contacts). The position is held by the Mesa continuous space and
neighbours are found with Mesa's own ``get_neighbors_in_radius``.

The numerical rules are the ones of the ECS systems, so both implementations share
the same model semantics.
"""

from typing import TYPE_CHECKING

import numpy as np
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent

from ..disease_model import (
    IMMUNITY_AFTER_RECOVERY,
    QUARANTINE_DURATION,
    QUARANTINE_EFFECT,
    RECOVERY_PROBABILITY_RANGE,
    HealthState,
)

if TYPE_CHECKING:
    from .model import MesaSIREpidemicModel


class Person(ContinuousSpaceAgent):
    """One individual of the population."""

    model: "MesaSIREpidemicModel"

    def __init__(self, model: "MesaSIREpidemicModel", space: ContinuousSpace,
                 position: np.ndarray, age: int, mobility: float, immunity: float):
        super().__init__(space, model)
        self.position = position
        self.index: int = len(space.active_agents) - 1  # row of the space position array

        self.age: int = age
        self.mobility: float = mobility
        self.immunity: float = immunity

        self.state: HealthState = HealthState.SUSCEPTIBLE

        # Infection (counter incremented at every step spent infected, see progress_disease)
        self.viral_load: float = 0.0
        self.days_infected: int = 0
        self.infectious: bool = False
        self.recovery_time: int = 0
        self.hazard: float = 0.0  # accumulated infection hazard of the current step

        # Quarantine
        self.quarantined: bool = False
        self.quarantine_compliance: float = 0.0
        self.quarantine_duration: int = QUARANTINE_DURATION
        self.days_in_quarantine: int = 0
        self.original_mobility: float = 0.0

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    # ------------------------------------------------------------------
    # State changes
    # ------------------------------------------------------------------
    def infect(self, viral_load: float, recovery_time: int) -> None:
        """Susceptible -> Infected. The day counter starts at 0 and is incremented by
        ``progress_disease`` at every step spent in the Infected state."""
        self.state = HealthState.INFECTED
        self.viral_load = viral_load
        self.days_infected = 0
        self.infectious = True
        self.recovery_time = recovery_time

    def transmission_modifier(self) -> float:
        """Reduction of the network transmission caused by quarantine."""
        if self.quarantined and self.quarantine_compliance > 0:
            return 1.0 - self.quarantine_compliance * QUARANTINE_EFFECT
        return 1.0

    # ------------------------------------------------------------------
    # Behaviours called by the model through ``AgentSet.do(...)``
    # ------------------------------------------------------------------
    def move(self) -> None:
        """Random walk bounded to the square world (the dead have mobility 0)."""
        if self.mobility > 0:
            step: np.ndarray = self.model.rng.uniform(-self.mobility, self.mobility, 2)
            self.position = np.clip(self.position + step, 0, self.space.dimensions[:, 1])

    def choose_contacts(self, wanted: int, ages: np.ndarray, alive: np.ndarray,
                        alpha: float, tau: float) -> tuple[list["Person"], list[float]]:
        """Draw ``wanted`` contacts without replacement, one after the other, with
        probability proportional to w_ij = exp(-d_ij/alpha) * exp(-|age_i - age_j|/tau)
        over the living others. The tie strength is the raw similarity w_ij.
        Distances come from the Mesa space (``calculate_distances``)."""
        distances: np.ndarray
        people: list["Person"]
        distances, people = self.space.calculate_distances(self.position)
        weights: np.ndarray = np.exp(-distances / alpha - np.abs(ages - self.age) / tau)
        weights[~alive] = 0.0
        weights[self.index] = 0.0
        total: float = weights.sum()
        candidates: int = int(np.count_nonzero(weights))
        if total == 0 or candidates == 0:
            return [], []
        chosen: np.ndarray = self.model.rng.choice(
            len(weights), size=min(wanted, candidates), replace=False, p=weights / total)
        return [people[j] for j in chosen], weights[chosen].tolist()

    def progress_disease(self) -> None:
        """Infected: count one more day, then recover or die (ECS ``DiseaseProgressionSystem``)."""
        if self.state is not HealthState.INFECTED:
            return
        self.days_infected += 1
        if self.days_infected >= self.recovery_time:
            random = self.model.random
            recovers: bool = random.random() < random.uniform(*RECOVERY_PROBABILITY_RANGE)
            if recovers:
                self.state = HealthState.RECOVERED
                self.immunity = IMMUNITY_AFTER_RECOVERY
                if self.quarantined:  # a recovered agent leaves quarantine at once
                    self.mobility = self.original_mobility
                    self.quarantined = False
            else:
                self.state = HealthState.DEAD
                self.mobility = 0.0

    def try_quarantine(self, compliance: float) -> None:
        """Infectious agents enter quarantine with probability ``compliance``."""
        if self.model.random.random() < compliance:
            self.quarantined = True
            self.quarantine_compliance = compliance
            self.quarantine_duration = QUARANTINE_DURATION
            self.days_in_quarantine = 0
            self.original_mobility = self.mobility
            self.mobility *= (1 - compliance * QUARANTINE_EFFECT)

    def tick_quarantine(self) -> None:
        """One more day in quarantine; release when the duration is over."""
        self.days_in_quarantine += 1
        if self.days_in_quarantine >= self.quarantine_duration:
            if self.state is not HealthState.DEAD:
                self.mobility = self.original_mobility
            self.quarantined = False
