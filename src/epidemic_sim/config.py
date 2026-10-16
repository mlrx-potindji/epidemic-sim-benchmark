"""Simulation configuration: model sizing, random seeds, and I/O."""

from dataclasses import dataclass

from .disease_model import DiseaseModelConfig


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation setup independent of the disease model."""

    n_agents: int = 500
    world_size: int = 100
    initial_infected: int = 5
    average_contacts: int = 10
    seed: int | None = None
    max_steps: int = 400
    collect_spatial: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for comparing simulation frameworks."""

    simulation: SimulationConfig
    disease_model: DiseaseModelConfig
    frameworks: list[str] | None = None
    n_replicates: int = 1
    output_dir: str = "results"

    def __post_init__(self) -> None:
        if self.frameworks is None:
            object.__setattr__(self, "frameworks", ["ecs", "mesa"])
        if not self.frameworks or any(
            framework not in {"ecs", "mesa"} for framework in self.frameworks
        ):
            raise ValueError("frameworks must contain 'ecs', 'mesa', or both")
        if self.n_replicates < 1:
            raise ValueError(f"n_replicates must be >= 1, got {self.n_replicates}")
