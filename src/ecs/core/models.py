from collections import defaultdict
from .systems import *
import random
from typing import List
import numpy as np
import esper
import copy

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

class SIREpidemicModel:
    def __init__(self, seed: int, n_agents: int = 500, world_size: int = 100, 
                 initial_infected: int = 5, average_contacts: int = 10, 
                 beta_spatial: float = 0.05, beta_network: float = 0.1,
                 recovery_time: int = 12, enable_quarantine: bool = False, 
                 transmission_radius: float = 4.0, world_name: str = "default_world"):

    # Initialize model parameters
        self.n_agents = n_agents
        self.world_size = world_size
        self.initial_infected = initial_infected
        self.average_contacts = average_contacts
        self.beta_spatial = beta_spatial
        self.beta_network = beta_network
        self.recovery_time = recovery_time
        self.enable_quarantine = enable_quarantine
        self.seed = seed
        self.transmission_radius = transmission_radius
        
        # Switch to world if not provided
        self.world_name = world_name
        esper.switch_world(world_name)

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()

        self.step_count = 0

        # Register systems
        esper.add_processor(MovementSystem(world_height = world_size, world_width = world_size))

        esper.add_processor(SpatialTransmissionSystem(transmission_radius = transmission_radius, 
                                                      base_transmission_prob = beta_spatial))
        
        esper.add_processor(NetworkTransmissionSystem(base_transmission_prob = beta_network))

        if enable_quarantine:
            # Example quarantine compliance level, can be adjusted as needed
            esper.add_processor(QuarantineSystem(quarantine_compliance = 0.8)) 

        esper.add_processor(DiseaseProgressionSystem(recovery_time = recovery_time))

        self.entities = Entity(n_agents) # store full Entity object to access both population and IDs
        self.entities.populate() # create the population of entities
        self.entity_iDs = self.entities.get_iDs() # get the list of entity IDs for component assignment

        self._population_components() # assign components to each entity in the population

        self._initial_infection() # Infect initial entities at the start of the simulation
        
        self.time_series_data: defaultdict[str, List[int]] = defaultdict(list) # Initialize time series data storage

        self.spatial_location_series_data: defaultdict[str, List[tuple[float, float]]] = defaultdict(list) # Initialize spatial location series data storage

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
                x = np.random.uniform(0, self.world_size),
                y = np.random.uniform(0, self.world_size)
            ))

            esper.add_component(entity, Demographics(
                age = np.random.randint(0, 75),
                mobility = np.random.uniform(0.5, 2.0)
            ))

            esper.add_component(entity, Susceptible(
                immunity = np.random.uniform(0, 0.05)
            ))

        self._create_social_network()

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
            num_contacts = max(0, int(np.random.poisson(self.average_contacts)))  

            if num_contacts > 0:
                possible_contacts = [x for x in self.entity_iDs if x != entity] 
                contacts = random.sample(possible_contacts, num_contacts)

                strengths = [random.uniform(0.1, 1.0) for _ in range(num_contacts)]

                esper.add_component(entity, ContactNetwork(contacts = contacts, 
                                                           contact_strength = strengths))
    
    def _initial_infection(self):
        """
        Infects a specified number of entities at the start of the simulation by changing their status from Susceptible to Infected.

        Parameters:
        - initial_infected: The number of entities to initially infect.

        Returns:
        - None (components are modified directly on entities in the world).
        """
        
        entities_to_infect = random.sample(self.entity_iDs, min(self.initial_infected, len(self.entity_iDs)))

        for entity in entities_to_infect:
            if esper.has_component(entity, Susceptible):
                esper.remove_component(entity, Susceptible)
                esper.add_component(entity, Infected(
                    viral_load = np.random.uniform(700, 1000),
                    days_infected = 1,
                    infectious = True))
                
    def step(self):
        """
        Advances the simulation by one step, processing all systems in the world.

        Parameters:
        - None

        Returns:
        - None (systems are processed directly on the world).
        """
        
        esper.process()

        self._collect_data()  # Collect data after processing systems
        self.step_count += 1 # Increment the step count after processing systems and collecting data

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
            self.spatial_location_series_data["susceptible"].append((location.x, location.y))

        for entiy, (inf, location) in esper.get_components(Infected, Location):
            self.spatial_location_series_data["infected"].append((location.x, location.y))

        for entity, (rec, location) in esper.get_components(Recovered, Location):
            self.spatial_location_series_data["recovered"].append((location.x, location.y))

        for entity, (death, location) in esper.get_components(Dead, Location):
            self.spatial_location_series_data["death"].append((location.x, location.y))

        return self.spatial_location_series_data
    
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




    