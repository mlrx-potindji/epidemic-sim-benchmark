"""Spatial and demographic homophily network utilities."""

from epidemic_sim.ecs.simulation import SIREpidemicModel


def create_homophily_network(model: SIREpidemicModel) -> None:
    """Populate a model with its configured similarity network."""
    model._space_attribute_similarity_network(
        alpha=model.alpha,
        tau=model.tau,
        dispersion=model.dispersion,
        diagnostic=False,
    )
