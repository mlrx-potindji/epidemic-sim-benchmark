import esper
import random
from .components import *

class MovementSystem(esper.Processor):
    """Entities move based on their mobility"""

    def __init__(self, world_width: float, world_height: float):
        super().__init__()
        # The world dimensions are needed to ensure entities stay within bounds
        self.width = world_width
        self.height = world_height
        

    def process(self):

        # list[tuple[int, tuple[_C, _C2]]] == list[tuple[entity_iD, (Location, Demographics)]]
        for entity, (location, demographics) in esper.get_components(Location, Demographics):
            # Random walk based on mobility
            mobility = demographics.mobility
            dx = random.uniform(-mobility, mobility)
            dy = random.uniform(-mobility, mobility)

            # Update location
            location.x = max(0, min(self.width, location.x + dx))
            location.y = max(0, min(self.height, location.y + dy))

class SpatialTransmissionSystem(esper.Processor):
    """Infectious entities can transmit to nearby susceptible entities"""

    def __init__(self, transmission_radius: float, base_transmission_prob: float):
        super().__init__()
        self.transmission_radius = transmission_radius
        self.base_prob = base_transmission_prob

    def process(self):
        # Get all infected and susceptible entities
        infected = list(esper.get_components(Infected, Location))
        susceptible = list(esper.get_components(Susceptible, Location))

        # list[tuple[int, tuple[_C, _C2]]] == list[tuple[inf_entity_iD, (Infected, Location)]]
        for inf_entity, (inf_status, inf_loc) in infected:
            if not inf_status.infectious or esper.has_component(inf_entity, Quarantined):
                # If the infected entity is not currently infectious or is quarantined, skip transmission attempts
                continue  # Skip if not currently infectious or if quarantined
                
            # list[tuple[int, tuple[_C, _C2]]] == list[tuple[sus_entity_iD, (Susceptible, Location)]]
            for sus_entity, (sus_status, sus_loc) in susceptible:
                # Calculate distance
                dx = inf_loc.x - sus_loc.x
                dy = inf_loc.y - sus_loc.y
                distance = (dx**2 + dy**2)**0.5

                # Check if within transmission radius
                if distance <= self.transmission_radius:
                    prob = (self.base_prob * inf_status.viral_load / 1000) * (1 - sus_status.immunity) 

                    # Attempt transmission
                    if random.random() < prob:
                        if esper.has_component(sus_entity, Susceptible):
                        # Infect the susceptible entity
                            esper.remove_component(sus_entity, Susceptible)
                            esper.add_component(sus_entity, Infected(
                                viral_load=random.uniform(500, 1000), 
                                days_infected=0,
                                infectious=True
                            ))

class NetworkTransmissionSystem(esper.Processor):
    """Infectious entities can transmit to connected susceptible entities"""

    def __init__(self, base_transmission_prob: float):
        super().__init__()
        self.base_prob = base_transmission_prob

    def process(self):
        # Get all infected and susceptible entities
        infected = list(esper.get_components(Infected, ContactNetwork))

        for inf_entity, (inf_status, contact_net) in infected:
            if not inf_status.infectious or esper.has_component(inf_entity, Quarantined):
                # If the infected entity is not currently infectious skip transmission attempts
                continue  

            transmission_modifier = 1.0

            if esper.has_component(inf_entity, Quarantined):
                quarantine = esper.component_for_entity(inf_entity, Quarantined)
                if quarantine.compliance_level > 0:
                    transmission_modifier *= (1 - quarantine.compliance_level * 0.9)  # Reduce transmission probability based on compliance level, up to 90% reduction
                
            adjusted_prob = self.base_prob * transmission_modifier

            for contact_iD, strength in zip(contact_net.contacts, contact_net.contact_strength):
                if esper.has_component(contact_iD, Susceptible):
                    sus_status = esper.component_for_entity(contact_iD, Susceptible)
                    prob = adjusted_prob * strength * (1 - sus_status.immunity) * inf_status.viral_load / 1000

                    if random.random() < prob:
                        # Infect the susceptible entity
                        esper.remove_component(contact_iD, Susceptible)
                        esper.add_component(contact_iD, Infected(
                            viral_load=random.uniform(500, 1000), 
                            days_infected=0,
                            infectious=True
                        ))

class DiseaseProgressionSystem(esper.Processor):
    """Infected entities progress through disease stages"""

    def __init__(self, recovery_time: float = 12): # Purposely set to 12 (shorter that quarantine duration)
        super().__init__()
        self.recovery_time = recovery_time

    def process(self):
        for entity, infected in esper.get_component(Infected):
            # Increase days infected
            infected.days_infected += 1

            # Simple progression logic: after recovery_time days, recover or die
            if infected.days_infected >= self.recovery_time:
                if random.random() < 0.98:  # 98% chance to recover
                    esper.remove_component(entity, Infected)
                    esper.add_component(entity, Recovered(immunity=0.9))  # Recovered with some immunity
                else:
                    esper.remove_component(entity, Infected)
                    esper.add_component(entity, Dead(reason = "Disease",
                                                     day_of_death = infected.days_infected))  # Died from disease

class QuarantineSystem(esper.Processor):
    """Reduce mobility and transmission for quarantined entities"""

    def __init__(self, quarantine_compliance: float):
        super().__init__()
        self.compliance = quarantine_compliance

    def process(self):
        for entity, (infected, demographics) in esper.get_components(Infected, Demographics):
            if infected.infectious and not esper.has_component(entity, Quarantined):
                # Quarantine infectious entities with some compliance level
                if random.random() < self.compliance:
                    esper.add_component(entity, Quarantined(
                        start_day=infected.days_infected,
                        compliance_level=self.compliance,
                        duration=14
                    ))

        # Update quarantine status
        for entity, quarantine in esper.get_component(Quarantined):
            quarantine.days_in_quarantine += 1

            # End quarantine after duration
            if quarantine.days_in_quarantine >= quarantine.duration:
                esper.remove_component(entity, Quarantined)     
        
        # Reduce mobility for quarantined entities
        for entity, (quarantine, demographics) in esper.get_components(Quarantined, Demographics):
            demographics.mobility *= (1 - quarantine.compliance_level * 0.9)  # Reduce mobility based on compliance level, up to 90% reduction      

        # Reduce transmission for quarantined entities in the SpatialTransmissionSystem and NetworkTransmissionSystem
        for entity, quarantine in esper.get_component(Quarantined):
            if quarantine.compliance_level > 0:
                # This will be handled in the respective transmission systems by checking for the Quarantined component
                pass
        









       
