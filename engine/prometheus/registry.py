"""Every child's identity, lineage and history, plus the ecosystem's event timeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

import numpy as np

from .genome import Genome

if TYPE_CHECKING:
    from .creators.base import ChildSpec

# gene -> specialization; surprise_threshold counts when it goes *below* default (more adaptive)
_SPECIALIZATION_GENES = {
    "perception_radius": "SCOUT",
    "risk_aversion": "SENTINEL",
    "surprise_threshold": "ADAPTER",
    "curiosity_bonus": "EXPLORER",
    "exploration": "EXPLORER",
    "planning_horizon": "NAVIGATOR",
    "learning_rate": "LEARNER",
}
_GENERALIST_THRESHOLD = 0.1


def specialization_for(genome: Genome) -> str:
    names = list(Genome.__dataclass_fields__)
    v, d = genome.to_vector(), Genome.default().to_vector()
    best, best_dev = "GENERALIST", _GENERALIST_THRESHOLD
    for gene, label in _SPECIALIZATION_GENES.items():
        i = names.index(gene)
        dev = (d[i] - v[i]) if gene == "surprise_threshold" else (v[i] - d[i])
        if dev > best_dev:
            best, best_dev = label, dev
    return best


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class EntityRecord:
    id: str
    name: str
    parent_id: str | None
    generation: int
    created_at: str
    specialization: str
    reason: str
    genome: dict
    curriculum: list[str]
    inherit_beliefs: bool
    predicted_change: float | None
    status: str = "active"
    metrics: dict | None = None
    beliefs: dict | None = None
    actual_change: float | None = None
    retired_at_generation: int | None = None
    decision: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fitness(self) -> float:
        return float(self.metrics["fitness"]) if self.metrics else 0.0

    def genome_obj(self) -> Genome:
        return Genome.from_dict(self.genome)


class Registry:
    def __init__(self, clock: Callable[[], str] | None = None):
        self._clock = clock or _utc_now
        self._records: dict[str, EntityRecord] = {}
        self._prefix_counts: dict[str, int] = {}

    def create(self, spec: "ChildSpec", generation: int) -> EntityRecord:
        entity_id = f"E{len(self._records) + 1:04d}"
        label = spec.specialization or specialization_for(spec.genome)
        self._prefix_counts[label] = self._prefix_counts.get(label, 0) + 1
        record = EntityRecord(
            id=entity_id,
            name=f"{label}-{self._prefix_counts[label]:03d}",
            parent_id=spec.parent_id,
            generation=generation,
            created_at=self._clock(),
            specialization=label,
            reason=spec.reason,
            genome=spec.genome.to_dict(),
            curriculum=list(spec.curriculum),
            inherit_beliefs=bool(spec.inherit_beliefs),
            predicted_change=None if spec.predicted_change is None else round(float(spec.predicted_change), 6),
            decision=dict(spec.decision) if spec.decision else None,
        )
        self._records[entity_id] = record
        return record

    def get(self, entity_id: str) -> EntityRecord:
        return self._records[entity_id]

    def active(self) -> list[EntityRecord]:
        return [r for r in self._records.values() if r.status == "active"]

    def retire(self, entity_id: str, generation: int) -> None:
        rec = self._records[entity_id]
        rec.status = "retired"
        rec.retired_at_generation = generation

    def all(self) -> list[EntityRecord]:
        return list(self._records.values())


class EventLog:
    def __init__(self):
        self.events: list[dict] = []

    def emit(self, type: str, generation: int, entity_id: str | None = None, **data) -> dict:
        event = {"seq": len(self.events) + 1, "type": type, "generation": generation,
                 "entity_id": entity_id, "data": _jsonable(data)}
        self.events.append(event)
        return event


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x
