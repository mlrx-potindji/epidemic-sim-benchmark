from collections import defaultdict
from typing import List
import esper

class Entity:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self._id_entities = [] 

    def populate(self):
        for _ in range(self.n_agents):
            entity = esper.create_entity()
            self._id_entities.append(entity)

# x = Entity(n_agents=10)
# x.populate()
# print(x._id_entities)

class SIREpidemicModel:
    pass