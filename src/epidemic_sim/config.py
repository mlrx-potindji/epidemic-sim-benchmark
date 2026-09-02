from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Shared high-level configuration for ECS and Mesa implementations."""

    n_agents: int = 500
    world_size: int = 100
    initial_infected: int = 5
    average_contacts: int = 10
    beta_spatial: float = 0.10
    beta_network: float = 0.20
    transmission_radius: float = 4.0
    dt: float = 1.0
    seed: int | None = None
