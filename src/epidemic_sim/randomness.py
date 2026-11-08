from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PhaseRandomVariates(Protocol):
    """Random values addressed by phase, update, and stable agent identifier."""

    def uniform(self, purpose: str, step: int, ids: np.ndarray) -> np.ndarray: ...

    def normal(self, purpose: str, step: int, ids: np.ndarray) -> np.ndarray: ...


@dataclass
class RandomStreams:
    """NumPy random generator owned by one simulation."""

    numpy: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int | None) -> "RandomStreams":
        return cls(numpy=np.random.default_rng(seed))
