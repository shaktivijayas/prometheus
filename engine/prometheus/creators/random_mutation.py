"""Control arm: no intelligence. Random mutations of good parents, random lessons."""
from __future__ import annotations

import numpy as np

from ..benchmark import CURRICULUM_PRESETS, LESSONS
from ..genome import Genome
from .base import ChildSpec, GenerationReport, curriculum_text, top_half

MUTATION_SIGMA = 0.1


def _random_curriculum(rng: np.random.Generator) -> tuple[str, ...]:
    return tuple(CURRICULUM_PRESETS[int(i)] for i in rng.integers(len(CURRICULUM_PRESETS), size=LESSONS))


class RandomMutationCreator:
    name = "random"

    def seed_population(self, n: int, rng: np.random.Generator) -> list[ChildSpec]:
        return [ChildSpec(Genome.random(rng), None, False, _random_curriculum(rng), "Random initial design")
                for _ in range(n)]

    def next_generation(self, report: GenerationReport, slots: int, rng: np.random.Generator) -> list[ChildSpec]:
        parents = top_half(report.population)
        specs = []
        for _ in range(slots):
            parent = parents[int(rng.integers(len(parents)))]
            v = parent.genome_obj().to_vector() + rng.normal(0.0, MUTATION_SIGMA, size=len(parent.genome))
            curriculum = _random_curriculum(rng)
            specs.append(ChildSpec(
                genome=Genome.from_vector(np.clip(v, 0.0, 1.0)),
                parent_id=parent.id,
                inherit_beliefs=bool(rng.random() < 0.5),
                curriculum=curriculum,
                reason=f"Random mutation of {parent.name}; curriculum {curriculum_text(curriculum)}",
            ))
        return specs
