"""A child's genome: the architecture the Creator designs.

Float genes are kept rounded to 6 decimals so normalization round-trips exactly
and run output stays byte-stable.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np

GENE_BOUNDS: dict[str, tuple[float, float]] = {
    "perception_radius": (1, 5),
    "memory_capacity": (0, 50),
    "learning_rate": (0.01, 1.0),
    "exploration": (0.0, 1.0),
    "exploration_decay": (0.9, 1.0),
    "planning_horizon": (1, 12),
    "risk_aversion": (0.0, 3.0),
    "surprise_threshold": (0.1, 5.0),
    "curiosity_bonus": (0.0, 2.0),
}
GENE_NAMES: tuple[str, ...] = tuple(GENE_BOUNDS)
INT_GENES = {"perception_radius", "memory_capacity", "planning_horizon"}
SURPRISE_OFF = 5.0  # a threshold at the upper bound disables adaptation resets

_DEFAULTS = {
    "perception_radius": 2, "memory_capacity": 10, "learning_rate": 0.3, "exploration": 0.3,
    "exploration_decay": 0.97, "planning_horizon": 6, "risk_aversion": 1.0,
    "surprise_threshold": 1.5, "curiosity_bonus": 0.5,
}


@dataclass(frozen=True)
class Genome:
    perception_radius: int
    memory_capacity: int
    learning_rate: float
    exploration: float
    exploration_decay: float
    planning_horizon: int
    risk_aversion: float
    surprise_threshold: float
    curiosity_bonus: float

    @classmethod
    def default(cls) -> "Genome":
        return cls(**_DEFAULTS)

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Genome":
        v = rng.random(len(GENE_NAMES))
        return cls.from_vector(v)

    def clamped(self) -> "Genome":
        out = {}
        for name in GENE_NAMES:
            lo, hi = GENE_BOUNDS[name]
            x = float(getattr(self, name))
            if not math.isfinite(x):
                x = float(_DEFAULTS[name])
            x = min(max(x, lo), hi)
            out[name] = int(round(x)) if name in INT_GENES else round(x, 6)
        return Genome(**out)

    def to_vector(self) -> np.ndarray:
        return np.array(
            [(getattr(self, n) - GENE_BOUNDS[n][0]) / (GENE_BOUNDS[n][1] - GENE_BOUNDS[n][0]) for n in GENE_NAMES],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, v) -> "Genome":
        v = np.asarray(v, dtype=np.float64)
        vals = {n: GENE_BOUNDS[n][0] + float(v[i]) * (GENE_BOUNDS[n][1] - GENE_BOUNDS[n][0])
                for i, n in enumerate(GENE_NAMES)}
        return cls(**vals).clamped()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        names = {f.name for f in fields(cls)}
        return cls(**{k: d[k] for k in names}).clamped()
