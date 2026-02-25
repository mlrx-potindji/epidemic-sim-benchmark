from dataclasses import dataclass
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
    infectious: bool = True

@dataclass
class Recovered:
    """Recovered entity (agent)"""
    immunity: float 

@dataclass
class Demographics:
    """Basic entity (agent) characteristics"""
    age: int
    mobility: float

@dataclass
class ContactNetwork:
    """Who contacts whom on network"""
    contacts: List[int] # entities iD
    contact_strenght: List[float] # 

@dataclass
class Quarantined:
    """Quarantine switch"""
    start_day: int 
    compliance_level: float
    duration: int = 14
    days_in_quarantine: int = 0
    