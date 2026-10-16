from dataclasses import dataclass

import numpy as np


@dataclass
class RandomStreams:
    """NumPy random generator owned by one simulation."""

    numpy: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int | None) -> "RandomStreams":
        return cls(numpy=np.random.default_rng(seed))
