"""Null baseline: copies good parents exactly. Any 'improvement' it shows is selection luck, not design."""
from __future__ import annotations

import numpy as np

from .base import ChildSpec, GenerationReport, top_half
from .random_mutation import RandomMutationCreator


class CloneCreator:
    name = "clone"

    def seed_population(self, n: int, rng: np.random.Generator) -> list[ChildSpec]:
        return RandomMutationCreator().seed_population(n, rng)

    def next_generation(self, report: GenerationReport, slots: int, rng: np.random.Generator) -> list[ChildSpec]:
        parents = top_half(report.population)
        return [ChildSpec(genome=p.genome_obj(), parent_id=p.id, inherit_beliefs=False,
                          curriculum=tuple(p.curriculum),
                          reason=f"Clone of {p.name} (null baseline: no design changes)")
                for p in (parents[i % len(parents)] for i in range(slots))]
