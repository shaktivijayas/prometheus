"""Hand-designed baseline: fixed if-then rules a human wrote. The trained Creator must beat this."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..benchmark import DEFAULT_CURRICULUM
from ..genome import Genome
from ..registry import EntityRecord
from .base import ChildSpec, GenerationReport, curriculum_text, describe_edits, top_half

SEED_SIGMA = 0.05
NUDGE_SIGMA = 0.03


def _pct(x: float) -> int:
    return int(round(100 * x))


def _diagnose(parent: EntityRecord, g: Genome, rng: np.random.Generator) -> tuple[Genome, tuple[str, ...], str]:
    m = parent.metrics or {}
    deaths = m.get("death_causes", {})
    if deaths.get("hazard", 0.0) > 0.3:
        return (replace(g, risk_aversion=g.risk_aversion + 0.5, perception_radius=g.perception_radius + 1),
                ("easy", "hard", "normal", "shift"),
                f"{parent.name} died to hazards in {_pct(deaths['hazard'])}% of worlds")
    if m.get("adaptation_ratio", 1.0) < 0.6:
        return (replace(g, surprise_threshold=g.surprise_threshold - 0.5, learning_rate=g.learning_rate + 0.1),
                ("normal", "shift", "shift", "shift"),
                f"{parent.name} failed to adapt after rule shifts (ratio {m['adaptation_ratio']:.2f})")
    if deaths.get("starvation", 0.0) > 0.3:
        radius = g.perception_radius - 1 if g.perception_radius > 2 else g.perception_radius
        return (replace(g, exploration=g.exploration + 0.1, curiosity_bonus=g.curiosity_bonus + 0.3,
                        perception_radius=radius),
                ("easy", "normal", "normal", "shift"),
                f"{parent.name} starved in {_pct(deaths['starvation'])}% of worlds")
    if m.get("learning_speed", 0.0) > 0.5:
        return (replace(g, learning_rate=g.learning_rate + 0.15, memory_capacity=g.memory_capacity + 5),
                ("easy", "easy", "normal", "shift"),
                f"{parent.name} learned slowly (speed {m['learning_speed']:.2f})")
    v = replace(g, planning_horizon=g.planning_horizon + 1).clamped().to_vector()
    nudged = Genome.from_vector(np.clip(v + rng.normal(0.0, NUDGE_SIGMA, size=len(v)), 0.0, 1.0))
    return nudged, DEFAULT_CURRICULUM, f"{parent.name} has no dominant weakness; refining planning"


class HeuristicCreator:
    name = "heuristic"

    def seed_population(self, n: int, rng: np.random.Generator) -> list[ChildSpec]:
        base = Genome.default().to_vector()
        return [ChildSpec(Genome.from_vector(np.clip(base + rng.normal(0.0, SEED_SIGMA, size=len(base)), 0, 1)),
                          None, False, DEFAULT_CURRICULUM, "Hand-designed default with small variation")
                for _ in range(n)]

    def next_generation(self, report: GenerationReport, slots: int, rng: np.random.Generator) -> list[ChildSpec]:
        parents = top_half(report.population)
        specs = []
        for i in range(slots):
            parent = parents[i % len(parents)]
            before = parent.genome_obj()
            after, curriculum, diagnosis = _diagnose(parent, before, rng)
            after = after.clamped()
            specs.append(ChildSpec(
                genome=after, parent_id=parent.id, inherit_beliefs=True, curriculum=curriculum,
                reason=f"{diagnosis}: {describe_edits(before, after)}; curriculum {curriculum_text(curriculum)}",
            ))
        return specs
