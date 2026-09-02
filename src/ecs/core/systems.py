import math
import esper
import random
import numpy as np
from .components import *
from .randomness import RandomStreams

# --------------------------------------------------
# MovementSystem
# --------------------------------------------------

class MovementSystem(esper.Processor):
    """Entities move based on their mobility"""

    def __init__(self, world_width: float, world_height: float, rng: RandomStreams):
        super().__init__()
        # The world dimensions are needed to ensure entities stay within bounds
        self.width = world_width
        self.height = world_height
        self.rng = rng
        
    def process(self):

        # list[tuple[int, tuple[_C, _C2]]] == list[tuple[entity_iD, (Location, Demographics)]]
        for entity, (location, demographics) in esper.get_components(Location, Demographics):
            # Random walk based on mobility
            mobility = demographics.mobility
            dx = self.rng.python.uniform(-mobility, mobility)
            dy = self.rng.python.uniform(-mobility, mobility)

            # Update location
            location.x = max(0, min(self.width, location.x + dx))
            location.y = max(0, min(self.height, location.y + dy))

# --------------------------------------------------
# SpatialTransmissionSystem
# --------------------------------------------------

class SpatialTransmissionSystem(esper.Processor):
    """Infectious entities can transmit to nearby susceptible entities"""

    def __init__(self, transmission_radius: float, base_transmission_prob: float, rng: RandomStreams):
        super().__init__()
        self.transmission_radius = transmission_radius
        self.base_prob = base_transmission_prob
        self.rng = rng

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
                    if self.rng.python.random() < prob:
                        if esper.has_component(sus_entity, Susceptible):
                        # Infect the susceptible entity
                            esper.remove_component(sus_entity, Susceptible)
                            esper.add_component(sus_entity, Infected(
                                viral_load=self.rng.python.uniform(500, 1000),
                                days_infected=0,
                                infectious=True,
                                recovery_time = max(1, int(self.rng.python.normalvariate(12, 4))) # Recovery time drawn from normal distribution with mean 12 days and std dev 4 days, minimum of 1 day
                            ))

class SpatialTransmissionSystemNew(esper.Processor):
    def __init__(self, transmission_radius: float, base_transmission_prob: float, rng: RandomStreams):
        super().__init__()
        self.transmission_radius = transmission_radius
        self.base_prob = base_transmission_prob
        self.rng = rng

    def process(self):
        infected_entities = [
            (eid, inf, loc)
            for eid, (inf, loc) in esper.get_components(Infected, Location)
            if inf.infectious and not esper.has_component(eid, Quarantined)
        ]

        if not infected_entities:
            return  # No infectious entities, skip processing
        
        for sus_entity, (sus, sus_loc) in esper.get_components(Susceptible, Location):
            accumulated_prob = 0.0

            for inf_entity, inf, inf_loc in infected_entities:
                dx = inf_loc.x - sus_loc.x
                dy = inf_loc.y - sus_loc.y
                distance = math.sqrt(dx**2 + dy**2)

                if distance <= self.transmission_radius:
                    prob = (self.base_prob * inf.viral_load / 1000) * (1 - sus.immunity)
                    accumulated_prob += prob

            if accumulated_prob > 0:
                self._add_hazard(sus_entity, accumulated_prob)
   
    @staticmethod
    def _add_hazard(entity_id: int, hazard: float):
        if esper.has_component(entity_id, InfectionHazard):
            esper.component_for_entity(entity_id, InfectionHazard).hazard += hazard
        else:
            esper.add_component(entity_id, InfectionHazard(hazard=hazard))

# --------------------------------------------------
# NetworkTransmissionSystem
# --------------------------------------------------

class NetworkTransmissionSystem(esper.Processor):
    """Infectious entities can transmit to connected susceptible entities"""

    def __init__(self, base_transmission_prob: float, rng: RandomStreams):
        super().__init__()
        self.base_prob = base_transmission_prob
        self.rng = rng

    def process(self):
        # Get all infected and susceptible entities
        infected = list(esper.get_components(Infected, ContactNetwork))

        for inf_entity, (inf_status, contact_net) in infected:
            if not inf_status.infectious: #or esper.has_component(inf_entity, Quarantined):
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

                    if self.rng.python.random() < prob:
                        # Infect the susceptible entity
                        esper.remove_component(contact_iD, Susceptible)
                        esper.add_component(contact_iD, Infected(
                                viral_load=self.rng.python.uniform(500, 1000),
                            days_infected=0,
                            infectious=True,
                                recovery_time = max(1, int(self.rng.python.normalvariate(12, 4))) # Recovery time drawn from normal distribution with mean 12 days and std dev 4 days, minimum of 1 day
                        ))

class NetworkTransmissionSystemNew(esper.Processor):
    def __init__(self, base_transmission_prob: float):
        super().__init__()
        self.base_prob = base_transmission_prob
    
    def process(self):

        for inf_entity, (inf_status, contact_net) in esper.get_components(Infected, ContactNetwork):
            if not inf_status.infectious:
                continue  # Skip if not currently infectious

            # for quarantine, same logic as before, it reduces but does not eliminate transmission
            transmission_modifier = 1.0

            if esper.has_component(inf_entity, Quarantined):
                quarantine = esper.component_for_entity(inf_entity, Quarantined)
                if quarantine.compliance_level > 0:
                    transmission_modifier *= (1 - quarantine.compliance_level * 0.9)

            adjusted_prob = self.base_prob * transmission_modifier

            for contact_iD, strength in zip(contact_net.contacts, contact_net.contact_strength):
                if not esper.has_component(contact_iD, Susceptible):
                    continue  # Skip if contact is not susceptible
                sus_entity = esper.component_for_entity(contact_iD, Susceptible)
                hazard_network = adjusted_prob * strength * (1 - sus_entity.immunity) * inf_status.viral_load / 1000

                if hazard_network > 0:
                    self._add_hazard(contact_iD, hazard_network)

    @staticmethod
    def _add_hazard(entity_id: int, hazard: float):
        if esper.has_component(entity_id, InfectionHazard):
            esper.component_for_entity(entity_id, InfectionHazard).hazard += hazard
        else:
            esper.add_component(entity_id, InfectionHazard(hazard=hazard))

class NetworkRewiringystem(esper.Processor):
    def __init__(self, entity_iDs: List, average_contacts: int,  
                 tau: float, alpha: float, diagnostic: bool = False, rewire_every: int = 18,
                 rng: RandomStreams | None = None):
        super().__init__()

        self.entity_iDs = entity_iDs
        self.average_contacts = average_contacts
        self.tau = tau
        self.alpha = alpha
        self.rewire_every = rewire_every
        self._step = 0
        self.diagnostic = diagnostic
        self.rng = rng or RandomStreams.from_seed(None)

    def process(self):

        self._step += 1
        if self._step % self.rewire_every != 0:
            return
        
        for entity in self.entity_iDs:
            if esper.has_component(entity, Dead):
                continue
            # remove older contact
            if esper.has_component(entity, ContactNetwork):
                esper.remove_component(entity, ContactNetwork)

            #num_contacts = max(0, int(np.random.poisson(self.average_contacts))) 
            k = 0.5  # dispersion parameter 
            num_contacts = max(0, int(self.rng.numpy.negative_binomial(k, k / (k + self.average_contacts))))

            if num_contacts == 0:
                continue

            # Fetch current position and age for this entity
            if not esper.has_component(entity, Location) or \
               not esper.has_component(entity, Demographics):
                continue

            position_i = esper.component_for_entity(entity, Location)
            age_i = esper.component_for_entity(entity, Demographics).age

            candidates = []
            weights = []

            for other in self.entity_iDs:
                if other == entity:
                    continue
                if esper.has_component(other, Dead):
                    continue  # exclude dead agents as contacts
                if not esper.has_component(other, Location) or \
                   not esper.has_component(other, Demographics):
                    continue

                position_j = esper.component_for_entity(other, Location)
                age_j = esper.component_for_entity(other, Demographics).age

                # Current positions — not frozen initial ones
                distance_ij = math.sqrt(
                    (position_i.x - position_j.x) ** 2 +
                    (position_i.y - position_j.y) ** 2
                )
                age_diff_ij = abs(age_i - age_j)

                similarity_ij = (math.exp(-distance_ij / self.alpha) *
                                 math.exp(-age_diff_ij / self.tau))

                candidates.append(other)
                weights.append(similarity_ij)

            total_weight = sum(weights)
            if total_weight == 0:
                continue

            probability = [w / total_weight for w in weights]
            contacts = list(self.rng.numpy.choice(
                candidates,
                size=min(num_contacts, len(candidates)),
                replace=False,
                p=probability
            ))

            # Raw similarity as tie strength — not normalised by total_weight
            strengths = [weights[candidates.index(c)] for c in contacts]

            esper.add_component(entity, ContactNetwork(
                contacts=contacts,
                contact_strength=strengths
            ))

        # ── DIAGNOSTIC ────────────────────────────────────────────────────────
        if self.diagnostic:
            all_weights = []
            sample = [e for e in self.entity_iDs
                      if not esper.has_component(e, Dead)][:50]

            for entity in sample:
                if not esper.has_component(entity, Location) or \
                   not esper.has_component(entity, Demographics):
                    continue
                position_i = esper.component_for_entity(entity, Location)
                age_i = esper.component_for_entity(entity, Demographics).age
                for other in self.entity_iDs:
                    if other == entity or esper.has_component(other, Dead):
                        continue
                    if not esper.has_component(other, Location) or \
                       not esper.has_component(other, Demographics):
                        continue
                    position_j = esper.component_for_entity(other, Location)
                    age_j = esper.component_for_entity(other, Demographics).age
                    d = math.sqrt((position_i.x - position_j.x)**2 +
                                  (position_i.y - position_j.y)**2)
                    a = abs(age_i - age_j)
                    all_weights.append(math.exp(-d / self.alpha) *
                                       math.exp(-a / self.tau))

            all_weights_arr = np.array(all_weights)
            weights_per_agent = all_weights_arr.reshape(len(sample), -1)
            effective_candidates = np.mean(np.sum(weights_per_agent > 0.01, axis=1))
            n_alive = sum(1 for e in self.entity_iDs
                          if not esper.has_component(e, Dead))

            print(f"[Rewire diagnostic] step={self._step} | alpha={self.alpha:.1f}, tau={self.tau}")
            print(f"  live agents:               {n_alive} / {len(self.entity_iDs)}")
            print(f"  mean weight:               {np.mean(all_weights_arr):.4f}")
            print(f"  median weight:             {np.median(all_weights_arr):.4f}")
            print(f"  % weights < 0.01:          {np.mean(all_weights_arr < 0.01)*100:.1f}%")
            print(f"  mean effective candidates: {effective_candidates:.1f} / {n_alive - 1}")

# --------------------------------------------------
# InfectionResolutionSystem
# --------------------------------------------------

class InfectionResolutionSystem(esper.Processor):
    def __init__(self, dt: float = 1.0, rng: RandomStreams | None = None):
        super().__init__()
        self.dt = dt
        self.rng = rng or RandomStreams.from_seed(None)

    def process(self):
        to_resolve = list(esper.get_components(Susceptible, InfectionHazard))

        for entity, (susceptible, hazard) in to_resolve:
            prob = 1 - math.exp(-hazard.hazard * self.dt)  # Convert hazard to probability

            if self.rng.python.random() < prob:
                esper.remove_component(entity, Susceptible)
                esper.add_component(entity, Infected(
                    viral_load = self.rng.python.uniform(500, 1000),
                    days_infected = 0,
                    infectious = True,
                    recovery_time = max(1, int(self.rng.python.normalvariate(12, 4))) #
                ))

            # Remove the hazard component after processing
            if esper.has_component(entity, InfectionHazard):
                esper.remove_component(entity, InfectionHazard)

# --------------------------------------------------
# DiseaseProgressionSystem
# --------------------------------------------------

class DiseaseProgressionSystem(esper.Processor):
    """Infected entities progress through disease stages"""

    def __init__(self, rng: RandomStreams | None = None): # Purposely set to 12 (shorter that quarantine duration)
        super().__init__()
        self.rng = rng or RandomStreams.from_seed(None)

    def process(self):
        for entity, infected in esper.get_component(Infected):
            # Increase days infected
            infected.days_infected += 1

            # Simple progression logic: after recovery_time days, recover or die

                    # Possible changes and improvements
            # With repspect to changes to quanrantine logic on recovery and death:
            #   - consider we should consider including death probability here also
            #   and draw from that probability to determine if the entity dies during the infectious period
            #   which would then also impact quarantine end logic in the QuarantineSystem.
            if infected.days_infected >= infected.recovery_time:
                if self.rng.python.random() < self.rng.python.uniform(0.97, 0.995):
                    esper.remove_component(entity, Infected)
                    esper.add_component(entity, Recovered(immunity=0.9))  # Recovered with some immunity
                else:
                    esper.remove_component(entity, Infected)
                    esper.add_component(entity, Dead(reason = "Disease",
                                                     day_of_death = infected.days_infected))  # Died from disease
                    if esper.has_component(entity, Demographics):
                        esper.component_for_entity(entity, Demographics).mobility = 0.0  # Dead entities do not move
            
            # End quarantine if entity recovers or dies during quarantine
            # This logic assumes that quarantine does not end immediately upon recovery or death, 
            # but rather that the entity is removed from quarantine at the next scheduled check
                if esper.has_component(entity, Quarantined) and not esper.has_component(entity, Dead) and not esper.has_component(entity, Infected):
                    quarantine = esper.component_for_entity(entity, Quarantined)
                    if esper.has_component(entity, Demographics):
                        demo = esper.component_for_entity(entity, Demographics)
                        demo.mobility = quarantine.original_mobility
                    esper.remove_component(entity, Quarantined)   

# --------------------------------------------------
# QuarantineSystem
# --------------------------------------------------

class QuarantineSystem(esper.Processor):
    """Reduce mobility and transmission for quarantined entities"""

    def __init__(self, quarantine_compliance: float, rng: RandomStreams | None = None):
        super().__init__()
        self.compliance = quarantine_compliance
        self.rng = rng or RandomStreams.from_seed(None)

    def process(self):
        for entity, (infected, demographics) in esper.get_components(Infected, Demographics):
            if infected.infectious and not esper.has_component(entity, Quarantined):
                # Quarantine infectious entities with some compliance level
                if self.rng.python.random() < self.compliance:
                    esper.add_component(entity, Quarantined(
                        start_day = infected.days_infected,
                        compliance_level = self.compliance,
                        duration = 14,
                        original_mobility = demographics.mobility
                    ))
                    demographics.mobility *= (1 - self.compliance * 0.9)  # Reduce mobility based on compliance level, up to 90% reduction

        # Update quarantine status
        for entity, quarantine in esper.get_component(Quarantined):
            quarantine.days_in_quarantine += 1

            # End quarantine after duration

                    # Possible changes and improvements
            # With respect to recovery and death
            #   - Allow for early release if recovered
            #   - allow for death during quarantine (would need to add death probability from being infectious) and 
            #     change quarantine end logic to also check for death.
            if quarantine.days_in_quarantine >= quarantine.duration:
                if esper.has_component(entity, Demographics) and not esper.has_component(entity, Dead):
                    demographics = esper.component_for_entity(entity, Demographics)
                    demographics.mobility = quarantine.original_mobility  # Restore original mobility
                esper.remove_component(entity, Quarantined)     
        
        # Reduce mobility for quarantined entities
        #for entity, (quarantine, demographics) in esper.get_components(Quarantined, Demographics):
        #    demographics.mobility *= (1 - quarantine.compliance_level * 0.9)  # Reduce mobility based on compliance level, up to 90% reduction      

        # Reduce transmission for quarantined entities in the SpatialTransmissionSystem and NetworkTransmissionSystem
        for entity, quarantine in esper.get_component(Quarantined):
            if quarantine.compliance_level > 0:
                # This will be handled in the respective transmission systems by checking for the Quarantined component
                pass
        









       
