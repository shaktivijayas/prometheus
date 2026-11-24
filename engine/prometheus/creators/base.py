"""What every Creator produces (ChildSpec), what it sees (GenerationReport), and its interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..genome import Genome

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


@dataclass
class GenerationReport:
    generation: int
    population: list["EntityRecord"]
    history: list[dict]


class Creator(Protocol):
    name: str

    def seed_population(self, n: int, rng: np.random.Generator) -> list[ChildSpec]: ...

    def next_generation(self, report: GenerationReport, slots: int, rng: np.random.Generator) -> list[ChildSpec]: ...
