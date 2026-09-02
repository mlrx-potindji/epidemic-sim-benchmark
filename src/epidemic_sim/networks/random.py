"""Random contact-network utilities."""

from epidemic_sim.ecs.simulation import SIREpidemicModel


def create_random_network(model: SIREpidemicModel) -> None:
    """Populate a model with its configured random contact network."""
    model._create_social_network()
