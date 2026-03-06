from dataclasses import dataclass, field
from typing import List

@dataclass
class Location:
    """spatial location component"""
    x: float
    y: float

@dataclass
class Susceptible:
    """Susceptible entity (agent)"""
    immunity: float = 0.0

@dataclass
class Infected:
    """Infected entity (agent)"""
    viral_load: float
    days_infected: int = 0
    infectious: bool = True # can be used to model incubation period or isolation effects 
                            # if set to False and then back to True after some days
    recovery_time: int = 0  # days until recovery (or death) --- can be used to model disease progression and quarantine duration

@dataclass
class Recovered:
    """Recovered entity (agent)"""
    immunity: float 

@dataclass
class Dead:
    """Dead entity (agent)"""
    reason: str = "Disease"
    day_of_death: int = 0

@dataclass
class Demographics:
    """Basic entity (agent) characteristics"""
    age: int
    mobility: float

@dataclass
class ContactNetwork:
    """Who contacts whom on network"""
    contacts: List[int] = field(default_factory=list)  # entities iD
    contact_strength: List[float] = field(default_factory=list)  # how strong is the contact (0-1)

@dataclass
class Quarantined:
    """Quarantine switch"""
    start_day: int 
    compliance_level: float
    duration: int = 14
    days_in_quarantine: int = 0
    original_mobility: float = 0.0  # Store original mobility to restore after quarantine

@dataclass
class InfectionHazard:
    """
    Transient component that is created or updated by hazard calculation system 
    and consumed by infection system to determine if susceptible agents become infected.

    Multiple hazard systems write additively into this component so that InfectionResolutionSystem 
    sees a single FOI and combine it once: p = 1 - exp(-lambda * dt)
    """
    hazard: float = 0.0