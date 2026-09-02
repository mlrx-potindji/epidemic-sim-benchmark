from collections import defaultdict
from .systems.transmission import *
from typing import List, Optional
import numpy as np
import esper
import math
import uuid
from ..randomness import RandomStreams

# --------------------------------------------------
# Entity class
# --------------------------------------------------
class Entity:
    """
    Represents an individual in the simulation with a unique ID.
    This class is responsible for creating entities and managing their IDs.

    Parameters:
    - n_agents: The number of agents (entities) to create in the simulation.
    - _id_entities: A list to store the unique IDs of the created entities.

    Methods:
    - populate(): Creates n_agents of entities and stores their IDs in _id_entities.
    - get_iDs(): Returns a copy of the list of entity IDs.

    Returns:
    - A list of unique entity IDs created in the simulation.
    """
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self._id_entities = [] 

    def populate(self):
        for _ in range(self.n_agents):
            entity = esper.create_entity()
            self._id_entities.append(entity)

    def get_iDs(self):
        return self._id_entities.copy()

# --------------------------------------------------
# Model class
# --------------------------------------------------

class SIREpidemicModel:
    def __init__(self, seed: int, tau: float = 25, alpha = None, n_agents: int = 500, 
                 world_size: int = 100, initial_infected: int = 5, average_contacts: int = 10, 
                 beta_spatial: float = 0.10, beta_network: float = 0.20,
                 enable_quarantine: bool = False, transmission_radius: float = 4.0, 
                 world_name: Optional[str] = None, spatial_new: Optional[bool] = False,
                 network_new: Optional[bool] = False, space_attribute_similarity: Optional[bool] = False,  
                 dt: float = 1.0, dispersion: float = 0.7):

    # Initialize model parameters
        self._validate_parameters(seed, tau, alpha, n_agents, world_size,
                                  initial_infected, average_contacts,
                                  beta_spatial, beta_network,
                                  transmission_radius, dt, dispersion)
        self.n_agents = n_agents
        self.world_size = world_size
        self.initial_infected = initial_infected
        self.average_contacts = average_contacts
        self.beta_spatial = beta_spatial
        self.beta_network = beta_network
        #self.recovery_time = recovery_time
        self.enable_quarantine = enable_quarantine
        self.seed = seed
        self.transmission_radius = transmission_radius
        self.dt = dt

        self.alpha = alpha if alpha is not None else 0.20 * world_size
        self.tau = tau
        self.dispersion = dispersion
        self._uses_similarity_network = bool(space_attribute_similarity)
        
        self.world_name = world_name or f"world_{uuid.uuid4().hex}"
        esper.switch_world(self.world_name)

        self.rng = RandomStreams.from_seed(seed)

        self.step_count = 0

        self.entities = Entity(n_agents) # store full Entity object to access both population and IDs
        self.entities.populate() # create the population of entities
        self.entity_iDs = self.entities.get_iDs() # get the list of entity IDs for component assignment

        self._population_components() # assign components to each entity in the population

        self.seed_locations: List[tuple[float, float]] = []

        self._initial_infection() # Infect initial entities at the start of the simulation

        if space_attribute_similarity:
            self._space_attribute_similarity_network(alpha = self.alpha,
                                                      tau = self.tau,
                                                      dispersion = dispersion,
                                                      diagnostic = True) # Create contact network based on spatial and attribute similarity
        else:
            self._create_social_network() # Create contact network based on Poisson distribution of average contacts

        self._register_systems(spatial_new, network_new, enable_quarantine)
        
        self.time_series_data: defaultdict[str, List[int]] = defaultdict(list) # Initialize time series data storage

        self.spatial_location_series_data: defaultdict[str, List[tuple[int, float, float]]] = defaultdict(list) # Initialize spatial location series data storage

    @staticmethod
    def _validate_parameters(seed, tau, alpha, n_agents, world_size,
                             initial_infected, average_contacts, beta_spatial,
                             beta_network, transmission_radius, dt, dispersion):
        if seed is not None and not isinstance(seed, int):
            raise TypeError("seed must be an integer or None")
        if n_agents < 1 or initial_infected < 0 or initial_infected > n_agents:
            raise ValueError("n_agents must be positive and initial_infected must be in [0, n_agents]")
        if world_size <= 0 or average_contacts < 0 or transmission_radius < 0 or dt <= 0:
            raise ValueError("world_size and dt must be positive; contacts and radius cannot be negative")
        if tau <= 0 or dispersion <= 0:
            raise ValueError("tau and dispersion must be positive")
        if alpha is not None and alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= beta_spatial <= 1 or not 0 <= beta_network <= 1:
            raise ValueError("transmission probabilities must be between 0 and 1")

    def _register_systems(self, spatial_new: bool, network_new: bool,
                          enable_quarantine: bool):
        """Install all processors for this model's world in execution order."""
        esper.add_processor(MovementSystem(self.world_size, self.world_size, self.rng), priority=100)
        spatial_system = SpatialTransmissionSystemNew if spatial_new else SpatialTransmissionSystem
        network_system = NetworkTransmissionSystemNew if network_new else NetworkTransmissionSystem
        esper.add_processor(spatial_system(self.transmission_radius, self.beta_spatial, self.rng), priority=90)
        esper.add_processor(network_system(self.beta_network, self.rng), priority=80)
        if spatial_new or network_new:
            esper.add_processor(InfectionResolutionSystem(self.dt, self.rng), priority=70)
        esper.add_processor(DiseaseProgressionSystem(self.rng), priority=60)
        if enable_quarantine:
            esper.add_processor(QuarantineSystem(0.8, self.rng), priority=60)
        if hasattr(self, "_uses_similarity_network") and self._uses_similarity_network:
            esper.add_processor(NetworkRewiringystem(
                self.entity_iDs, self.average_contacts, self.tau, self.alpha,
                diagnostic=True, rng=self.rng), priority=95)

    # --------------------------------------------------
    # Population initialization method
    # --------------------------------------------------

    def _population_components(self):
        """
        Assigns components to each entity in the population, 
        including Location, Demographics, Susceptible, and ContactNetwork 
        components.

        Parameters:
        - entity_iDs: List of entity IDs to assign components to.

        Returns:
        - None (components are added directly to entities in the world)
        """

        for entity in self.entity_iDs:
            esper.add_component(entity, Location(
                x = self.rng.numpy.uniform(0, self.world_size),
                y = self.rng.numpy.uniform(0, self.world_size)
            ))

            esper.add_component(entity, Demographics(
                age = self.rng.numpy.integers(0, 75),
                mobility = self.rng.numpy.uniform(0.5, 20.0)
            ))

            esper.add_component(entity, Susceptible(
                immunity = self.rng.numpy.uniform(0, 0.05)
            ))

    # --------------------------------------------------
    # Contact network creation methods
    # --------------------------------------------------

    def _create_social_network(self):
        """
        Creates a contact network for each entity based on a Poisson distribution of average contacts.
        Each contact is assigned a random strength between 0.1 and 1.0.
        
        Parameters:
        - entity_iDs: List of entity IDs to create contact networks for.
        - average_contacts: The average number of contacts each entity has, used as the lambda parameter

        Returns:
        - None (contact networks are added as components to each entity). 

        Results: 
        - Add component ContactNetwork to each entity with a list of contact IDs and corresponding strengths
        """

        for entity in self.entity_iDs:
            num_contacts = max(0, int(self.rng.numpy.poisson(self.average_contacts)))

            if num_contacts > 0:
                possible_contacts = [x for x in self.entity_iDs if x != entity] 
                contacts = self.rng.python.sample(possible_contacts, num_contacts)

                strengths = [self.rng.python.uniform(0.1, 1.0) for _ in range(num_contacts)]

                esper.add_component(entity, ContactNetwork(contacts = contacts, 
                                                           contact_strength = strengths))
    
    # --------------------------------------------------
    # Contact network creation methods based on spatial and attribute similarity
    # --------------------------------------------------

    def _space_attribute_similarity_network(self, alpha: float, tau: float, 
                                            dispersion: float, diagnostic: Optional[bool]):
        """
        Constructs a weighted contact network where tie formation is jointly governed
        by spatial proximity and age-based attribute similarity (homophily).

        Network topology
        ----------------
        This is a spatially-embedded homophily network. Each agent forms contacts
        preferentially with others who are both geographically close and
        demographically similar in age. The resulting structure exhibits:

        - High local clustering: agents predominantly connect within tight
          spatial and demographic neighbourhoods
        - Low long-range bridging: few connections span large distances or age gaps,
          producing lattice-like rather than small-world topology
        - Spatially structured epidemic diffusion: the pathogen spreads as a
          wave front through local clusters rather than jumping globally,
          producing slower epidemic growth and lower final attack rates compared
          to random contact networks

        Tie probability
        ---------------
        For each pair (i, j), a similarity weight is computed as:

            w_ij = exp(-d_ij / alpha) * exp(-|age_i - age_j| / tau)

        where d_ij is the Euclidean distance between agents i and j.
        Both terms are in (0, 1], so w_ij = 1.0 only if agents are co-located
        and identical in age, decaying toward 0 with distance and age difference.

        This weight serves two roles:
        - Sampling probability: normalised across all candidates to select contacts
        - Tie strength: used directly as contact_strength in ContactNetwork,
          reflecting how strongly two agents interact

        Parameters
        ----------
        alpha : float
            Distance sensitivity. Controls the spatial decay rate — the distance
            at which spatial similarity drops to exp(-1) ≈ 0.37. Should be scaled
            relative to world size; recommended range is 0.05 to 0.20 * world_size.
            Values too small relative to world_size produce degenerate networks
            where most agents have no effective candidates (verified via diagnostic).

        tau : float
            Attribute similarity sensitivity. Controls the age decay rate — the
            age difference at which demographic similarity drops to exp(-1) ≈ 0.37.
            Should be calibrated against the age distribution of the population;
            for ages drawn from uniform(0, 75), the mean pairwise age difference
            is ~25 years, making tau=25 a neutral baseline.

        Epidemiological implications
        ----------------------------
        - Slower epidemic growth relative to random networks due to local saturation
        - Higher run-to-run variance: outbreak size is sensitive to the spatial
          position of initially infected agents
        - Isolated susceptible pockets may survive the epidemic if no bridge
          contacts connect them to infected clusters, suppressing the final
          attack rate
        - To restore global reachability while preserving homophily structure,
          a Watts-Strogatz rewiring step can be applied post-construction,
          randomly redirecting a small fraction of edges (e.g. 5%) to random agents

        Notes
        -----
        The number of contacts per agent is drawn from Poisson(average_contacts).
        If total_weight across all candidates is zero for a given agent (can occur
        with extreme parameter values), that agent receives no ContactNetwork
        component and is effectively isolated from network transmission.
        Network transmission can still occur via SpatialTransmissionSystem.
        """

        alpha = alpha if alpha is not None else 0.10 * self.world_size
        tau = tau if tau is not None else 25
        dispersion = dispersion

        for entity in self.entity_iDs:
            #num_contacts = max(0, int(np.random.poisson(self.average_contacts)))
            
            # dispersion parameter
            num_contacts = max(0, int(self.rng.numpy.negative_binomial(dispersion, dispersion / (dispersion + self.average_contacts))))
            if num_contacts == 0:
                continue

            position_i = esper.component_for_entity(entity, Location)
            demographics_i = esper.component_for_entity(entity, Demographics).age

            candidates = []
            weights = []

            others = [ent for ent in self.entity_iDs if ent != entity]
            for other in others:
                position_j = esper.component_for_entity(other, Location)
                demographics_j = esper.component_for_entity(other, Demographics).age

                distance_ij = math.sqrt((position_i.x - position_j.x) ** 2 + (position_i.y - position_j.y) ** 2)
                attribute_simlarity_ij = abs(demographics_i - demographics_j)

                similarity_ij = math.exp(-distance_ij / self.alpha) * math.exp(-attribute_simlarity_ij / self.tau)

                candidates.append(other)
                weights.append(similarity_ij)

            total_weight = sum(weights)
            if total_weight == 0:
                continue
            probability = [weight / total_weight for weight in weights]
            contacts = list(self.rng.numpy.choice(candidates,
                                             size = min(num_contacts, len(candidates)), 
                                             replace = False, 
                                             p = probability))
            
            #strengths = [(weights[candidates.index(contact)] / total_weight) for contact in contacts]
            strengths = [weights[candidates.index(c)] for c in contacts]
            esper.add_component(entity, ContactNetwork(contacts = contacts,
                                                           contact_strength = strengths))
        # ── DIAGNOSTIC ────────────────────────────────────────────────────────
        if diagnostic: 
            all_weights = []
            for entity in self.entity_iDs[:50]:
                position_i = esper.component_for_entity(entity, Location)
                age_i = esper.component_for_entity(entity, Demographics).age
                for other in self.entity_iDs:
                    if other == entity:
                        continue
                    position_j = esper.component_for_entity(other, Location)
                    age_j = esper.component_for_entity(other, Demographics).age
                    d = math.sqrt((position_i.x - position_j.x)**2 + (position_i.y - position_j.y)**2)
                    a = abs(age_i - age_j)
                    all_weights.append(math.exp(-d / alpha) * math.exp(-a / tau))
            print(f"[Network diagnostic] alpha={alpha}, tau={tau}")
            print(f"  mean weight:         {np.mean(all_weights):.4f}")
            print(f"  median weight:       {np.median(all_weights):.4f}")
            print(f"  % weights < 0.01:    {np.mean(np.array(all_weights) < 0.01)*100:.1f}%")

            print(f"  % weights < 0.01:    {np.mean(np.array(all_weights) < 0.01)*100:.1f}%")

            # Add this:
            weights_per_agent = np.array(all_weights).reshape(50, -1)  # 50 agents × 999 candidates
            effective_contacts = np.mean(np.sum(weights_per_agent > 0.01, axis=1))
            print(f"  mean effective candidates per agent (w > 0.01): {effective_contacts:.1f} / {len(self.entity_iDs)-1}")
            

    def _degree_constrained_similarity_network(self):
        pass

    def _multidim_homophily_network(self):
        pass

    # --------------------------------------------------
    # Initial infection method
    # --------------------------------------------------

    def _initial_infection(self):
        """
        Infects a specified number of entities at the start of the simulation by changing their status from Susceptible to Infected.

        Parameters:
        - initial_infected: The number of entities to initially infect.

        Returns:
        - None (components are modified directly on entities in the world).
        """
        
        entities_to_infect = self.rng.python.sample(self.entity_iDs, min(self.initial_infected, len(self.entity_iDs)))
        

        for entity in entities_to_infect:
            
            if esper.has_component(entity, Susceptible):
                esper.remove_component(entity, Susceptible)
                esper.add_component(entity, Infected(
                    viral_load = self.rng.numpy.uniform(700, 1000),
                    days_infected = 1,
                    infectious = True,
                    recovery_time = max(1, int(self.rng.python.normalvariate(12, 4)))))
                
                if esper.has_component(entity, Location):
                    loc = esper.component_for_entity(entity, Location)
                    self.seed_locations.append((loc.x, loc.y))

    # --------------------------------------------------
    # Simulation step method
    # --------------------------------------------------
      
    def step(self):
        """
        Advances the simulation by one step, processing all systems in the world.

        Parameters:
        - None

        Returns:
        - None (systems are processed directly on the world).
        """
        
        esper.switch_world(self.world_name)
        esper.process()

        self._collect_data()  # Collect data after processing systems
        self.get_spatial_data()  # Collect spatial data after processing systems
        self.step_count += 1 # Increment the step count after processing systems and collecting data

    # --------------------------------------------------
    # Data collection methods
    # --------------------------------------------------

    def _collect_data(self):
        """
        Collects data on the current state of the simulation, such as counts of Susceptible, Infected, and Recovered entities.

        Parameters:
        - None

        Returns:
        - A dictionary containing counts of each health status in the population.
        """

        n_susceptible = len(esper.get_component(Susceptible))
        n_infected = len(esper.get_component(Infected))
        n_recovered = len(esper.get_component(Recovered))
        n_death = len(esper.get_component(Dead))

        self.time_series_data["time"].append(self.step_count)
        self.time_series_data["susceptible"].append(n_susceptible)
        self.time_series_data["infected"].append(n_infected)
        self.time_series_data["recovered"].append(n_recovered)
        self.time_series_data["death"].append(n_death)

        return self.time_series_data

    # --------------------------------------------------
    # model run method
    # --------------------------------------------------

    def run(self, max_steps: int):  
        """
        Runs the simulation for a specified number of steps, collecting data at each step.

        Parameters:
        - max_steps: The maximum number of steps to run the simulation.

        Returns:
        - None (simulation runs and collects data internally).
        """

        esper.switch_world(self.world_name)  # Ensure we're operating in the correct world

        for _ in range(max_steps):
            self.step()

            n_infected = len(esper.get_component(Infected))
            if n_infected == 0:
                print(f"Simulation ended at step {self.step_count} - no more infected entities.")
                break
    
    # --------------------------------------------------
    # Spatial data collection method
    # --------------------------------------------------

    def get_spatial_data(self):
        """
        Retrieves the spatial data (locations) of all entities in the simulation.

        Parameters:
        - None

        Returns:
        - A list of tuples containing the (x, y) coordinates of each entity.
        """

        esper.switch_world(self.world_name)  # Ensure we're operating in the correct world

        for entity, (sus, location) in esper.get_components(Susceptible, Location):
            self.spatial_location_series_data["susceptible"].append((self.step_count, location.x, location.y))

        for entity, (inf, location) in esper.get_components(Infected, Location):
            self.spatial_location_series_data["infected"].append((self.step_count, location.x, location.y))

        for entity, (rec, location) in esper.get_components(Recovered, Location):
            self.spatial_location_series_data["recovered"].append((self.step_count, location.x, location.y))

        for entity, (death, location) in esper.get_components(Dead, Location):
            self.spatial_location_series_data["death"].append((self.step_count, location.x, location.y))

        return self.spatial_location_series_data
    
    # --------------------------------------------------
    # Model clean up method
    # --------------------------------------------------

    def clean_up(self):
        """
        Cleans up the world by removing all entities and their components, resetting the simulation state.

        Parameters:
        - None

        Returns:
        - None (world is cleaned up directly).
        """

        esper.switch_world("default")  # Ensure we're operating in the correct world

        if self.world_name != "default":
            esper.delete_world(self.world_name)  # Delete the custom world to clean up all entities and components




    