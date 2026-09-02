from dataclasses import dataclass
import random

import numpy as np


@dataclass
class RandomStreams:
    """Independent random streams owned by one simulation."""

    numpy: np.random.Generator
    python: random.Random

    @classmethod
    def from_seed(cls, seed: int | None) -> "RandomStreams":
        return cls(
            numpy=np.random.default_rng(seed),
            python=random.Random(seed),
        )
