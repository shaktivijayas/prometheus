"""What the Creator network sees: fixed-size, scale-free, always-finite descriptions of candidates and population."""
from __future__ import annotations

import math

import numpy as np

from ..benchmark import CURRICULUM_PRESETS, LESSONS
from ..genome import GENE_NAMES, Genome
from ..registry import EntityRecord
from .base import ranked

N_GENES = len(GENE_NAMES)
N_PRESETS = len(CURRICULUM_PRESETS)
DEATHS = ("hazard", "starvation", "post_shift")

# layout of candidate_features (indices are part of the checkpoint contract)
_SCALARS = 1 + 1 + 3 + 1 + 1 + 1 + 1 + 1 + len(DEATHS) + LESSONS  # fitness..lesson scores
RANK_INDEX = N_GENES + _SCALARS
INHERIT_INDEX = RANK_INDEX + 1
CURRICULUM_SLICE = slice(INHERIT_INDEX + 1, INHERIT_INDEX + 1 + LESSONS * N_PRESETS)
CANDIDATE_DIM = CURRICULUM_SLICE.stop
POPULATION_DIM = 2 * N_GENES + 2 + 1 + 1


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b and math.isfinite(b) and b > 0 else 0.0


def _metric(rec: EntityRecord, key: str) -> float:
    m = rec.metrics or {}
    v = m.get(key, 0.0)
    return float(v) if v is not None and math.isfinite(float(v)) else 0.0


def candidate_features(rec: EntityRecord, population: list[EntityRecord]) -> np.ndarray:
    pop = population or [rec]
    maxes = {k: max(_metric(r, k) for r in pop) for k in ("fitness", "stable_score", "shift_score", "mean_score")}
    m = rec.metrics or {}
    deaths = m.get("death_causes", {}) or {}
    lessons = list(m.get("lesson_scores", []) or [])[:LESSONS]
    lessons += [0.0] * (LESSONS - len(lessons))
    best_lesson = max(lessons) if lessons else 0.0
    order = [r.id for r in ranked(pop)]
    rank = order.index(rec.id) / (len(order) - 1) if rec.id in order and len(order) > 1 else 0.0
    onehot = np.zeros((LESSONS, N_PRESETS))
    for i, preset in enumerate(list(rec.curriculum)[:LESSONS]):
        if preset in CURRICULUM_PRESETS:
            onehot[i, CURRICULUM_PRESETS.index(preset)] = 1.0
    parts = [
        Genome.from_dict(rec.genome).to_vector(),
        [_safe_div(_metric(rec, "fitness"), maxes["fitness"])],
        [math.log1p(max(_metric(rec, "fitness"), 0.0))],
        [_safe_div(_metric(rec, k), maxes[k]) for k in ("stable_score", "shift_score", "mean_score")],
        [_metric(rec, "survival_rate")],
        [_metric(rec, "adaptation_ratio") / 2.0],
        [_metric(rec, "learning_speed")],
        [min(_metric(rec, "mean_prediction_error") / 10.0, 3.0)],
        [_metric(rec, "perception_cost") / 100.0],
        [float(deaths.get(k, 0.0) or 0.0) for k in DEATHS],
        [_safe_div(x, best_lesson) for x in lessons],
        [rank],
        [1.0 if rec.inherit_beliefs else 0.0],
        onehot.ravel(),
    ]
    out = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
    assert out.shape == (CANDIDATE_DIM,)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def population_features(population: list[EntityRecord], generation: int, total_generations: int) -> np.ndarray:
    vecs = np.array([Genome.from_dict(r.genome).to_vector() for r in population]) if population \
        else np.zeros((1, N_GENES))
    fit = np.array([max(_metric(r, "fitness"), 0.0) for r in population]) if population else np.zeros(1)
    if len(vecs) > 1:
        dists = [np.linalg.norm(a - b) for i, a in enumerate(vecs) for b in vecs[i + 1:]]
        diversity = float(np.mean(dists))
    else:
        diversity = 0.0
    out = np.concatenate([
        vecs.mean(axis=0), vecs.std(axis=0),
        [math.log1p(float(fit.mean())), math.log1p(float(fit.max()))],
        [diversity],
        [_safe_div(generation, total_generations)],
    ])
    assert out.shape == (POPULATION_DIM,)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
