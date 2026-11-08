"""Counter-based random variates for deterministic cross-implementation tests."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)

# Purpose IDs are part of the test protocol; only add new values.
PURPOSES = {
    "move_x": 1,
    "move_y": 2,
    "infect": 3,
    "vload": 4,
    "dur": 5,
    "rho": 6,
    "recover": 7,
    "quar_enter": 8,
    "rewire_key": 9,
    "init": 10,
}


def _splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        mixed = values + _GOLDEN
        mixed = (mixed ^ (mixed >> np.uint64(30))) * _M1
        mixed = (mixed ^ (mixed >> np.uint64(27))) * _M2
        return mixed ^ (mixed >> np.uint64(31))


class KeyedRNG:
    """Pure random variates keyed by ``(seed, purpose, step, agent_id)``."""

    def __init__(self, seed: int, id_map: dict[int, int] | None = None):
        self.seed = int(seed)
        self.id_map = id_map or {}

    def _bits(self, purpose: str, step: int, ids) -> np.ndarray:
        if purpose not in PURPOSES:
            raise ValueError(f"unknown keyed random purpose: {purpose}")
        raw_ids = np.asarray(ids)
        agent_ids = np.asarray(
            [self.id_map.get(int(agent_id), int(agent_id)) for agent_id in raw_ids.flat],
            dtype=np.uint64,
        ).reshape(raw_ids.shape)
        value = _splitmix64(np.full(agent_ids.shape, self.seed, dtype=np.uint64))
        value = _splitmix64(value ^ np.uint64(PURPOSES[purpose]))
        value = _splitmix64(value ^ np.uint64(step))
        return _splitmix64(value ^ agent_ids)

    def uniform(self, purpose: str, step: int, ids) -> np.ndarray:
        """Return values in the open interval (0, 1) with 53-bit resolution."""
        bits = self._bits(purpose, step, ids) >> np.uint64(11)
        return (bits.astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)

    def normal(self, purpose: str, step: int, ids) -> np.ndarray:
        """Return standard-normal values through the inverse normal CDF."""
        return ndtri(self.uniform(purpose, step, ids))


class ScriptedRNG(KeyedRNG):
    """Keyed variates with purpose overrides for exact boundary conditions."""

    def __init__(
        self,
        seed: int,
        overrides: dict | None = None,
        id_map: dict[int, int] | None = None,
    ):
        super().__init__(seed, id_map)
        self.overrides = overrides or {}

    def uniform(self, purpose: str, step: int, ids) -> np.ndarray:
        if purpose in self.overrides:
            return np.broadcast_to(
                np.asarray(self.overrides[purpose], dtype=np.float64),
                np.asarray(ids).shape,
            ).copy()
        return super().uniform(purpose, step, ids)

    def normal(self, purpose: str, step: int, ids) -> np.ndarray:
        if purpose in self.overrides:
            return np.broadcast_to(
                np.asarray(self.overrides[purpose], dtype=np.float64),
                np.asarray(ids).shape,
            ).copy()
        return super().normal(purpose, step, ids)
