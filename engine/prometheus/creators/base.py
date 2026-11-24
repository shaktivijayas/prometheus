"""What every Creator produces (ChildSpec), what it sees (GenerationReport), and its interface."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..benchmark import CURRICULUM_PRESETS, DEFAULT_CURRICULUM, LESSONS
from ..genome import GENE_NAMES, Genome
from ..jsonutil import jsonable

if TYPE_CHECKING:
    from ..registry import EntityRecord


@dataclass
class ChildSpec:
    """A Creator's design for one new child: its nature (genome), inheritance and teaching plan."""
    genome: Genome
    parent_id: str | None
    inherit_beliefs: bool
    curriculum: tuple[str, ...]
    reason: str
    specialization: str = ""
    predicted_change: float | None = None
    decision: dict = field(default_factory=dict)  # raw decision details for analysis (Creator-specific)

    def validated(self) -> "ChildSpec":
        """Clamp the genome into bounds; replace a malformed curriculum with the default one."""
        curriculum = tuple(self.curriculum)
        if len(curriculum) != LESSONS or any(p not in CURRICULUM_PRESETS for p in curriculum):
            curriculum = DEFAULT_CURRICULUM
        predicted = self.predicted_change
        if predicted is not None and not math.isfinite(float(predicted)):
            predicted = None
        return ChildSpec(self.genome.clamped(), self.parent_id, bool(self.inherit_beliefs), curriculum,
                         str(self.reason), str(self.specialization),
                         None if predicted is None else float(predicted), jsonable(dict(self.decision)))


@dataclass
class GenerationReport:
    generation: int
    population: list["EntityRecord"]
    history: list[dict]


class Creator(Protocol):
    name: str

    def seed_population(self, n: int, rng: np.random.Generator) -> list[ChildSpec]: ...

    def next_generation(self, report: GenerationReport, slots: int, rng: np.random.Generator) -> list[ChildSpec]: ...


def ranked(population: list["EntityRecord"]) -> list["EntityRecord"]:
    """Best first; ties broken by id so selection is deterministic."""
    return sorted(population, key=lambda r: (-r.fitness, r.id))


def top_half(population: list["EntityRecord"]) -> list["EntityRecord"]:
    ordered = ranked(population)
    return ordered[: max(1, (len(ordered) + 1) // 2)]


def describe_edits(before: Genome, after: Genome) -> str:
    parts = []
    for name in GENE_NAMES:
        a, b = getattr(before, name), getattr(after, name)
        if a != b:
            delta = b - a
            parts.append(f"{name} {delta:+d}" if isinstance(delta, int) else f"{name} {delta:+.2f}")
    return ", ".join(parts) if parts else "no gene changes"


def curriculum_text(curriculum) -> str:
    return " > ".join(curriculum)
