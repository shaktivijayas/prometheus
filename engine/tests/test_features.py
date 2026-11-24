import numpy as np

from prometheus.benchmark import CURRICULUM_PRESETS, DEFAULT_CURRICULUM
from prometheus.creators.base import ChildSpec
from prometheus.creators.features import (
    CANDIDATE_DIM, CURRICULUM_SLICE, POPULATION_DIM, candidate_features, population_features,
)
from prometheus.genome import Genome
from prometheus.registry import Registry


def metrics(fitness=1.0):
    return {"fitness": fitness, "mean_score": fitness, "stable_score": fitness, "shift_score": fitness / 2,
            "survival_rate": 0.5, "adaptation_ratio": 1.2, "learning_speed": 0.5, "mean_prediction_error": 4.0,
            "perception_cost": 20.0, "death_causes": {"hazard": 0.1, "starvation": 0.3, "post_shift": 0.05},
            "lesson_scores": [0.5, 1.0, 1.5, 2.0]}


def population(fitnesses, curricula=None):
    reg = Registry(clock=lambda: "t")
    recs = []
    for i, f in enumerate(fitnesses):
        cur = curricula[i] if curricula else DEFAULT_CURRICULUM
        r = reg.create(ChildSpec(Genome.random(np.random.default_rng(i)), None, i % 2 == 0, cur, "x"), 0)
        r.metrics = metrics(f)
        recs.append(r)
    return recs


def test_dimensions():
    assert CANDIDATE_DIM == 44 and POPULATION_DIM == 22
    pop = population([1.0, 2.0, 3.0])
    assert candidate_features(pop[0], pop).shape == (CANDIDATE_DIM,)
    assert population_features(pop, 2, 10).shape == (POPULATION_DIM,)


def test_finite_for_degenerate_populations():
    for pop in (population([0.0, 0.0, 0.0]), population([0.0])):
        for r in pop:
            assert np.isfinite(candidate_features(r, pop)).all()
        assert np.isfinite(population_features(pop, 0, 1)).all()
        assert np.isfinite(population_features(pop, 0, 0)).all()


def test_invariant_to_population_order():
    pop = population([3.0, 1.0, 2.0, 2.0])
    shuffled = [pop[2], pop[0], pop[3], pop[1]]
    for r in pop:
        assert np.array_equal(candidate_features(r, pop), candidate_features(r, shuffled))
    assert np.allclose(population_features(pop, 1, 5), population_features(shuffled, 1, 5))


def test_curriculum_one_hot_slice():
    a = population([1.0, 2.0], [DEFAULT_CURRICULUM, DEFAULT_CURRICULUM])
    b = population([1.0, 2.0], [("easy", "easy", "hard", "shift"), DEFAULT_CURRICULUM])
    fa, fb = candidate_features(a[0], a), candidate_features(b[0], b)
    diff = np.nonzero(fa != fb)[0]
    assert set(diff) <= set(range(CURRICULUM_SLICE.start, CURRICULUM_SLICE.stop)) and len(diff) > 0
    onehot = fb[CURRICULUM_SLICE].reshape(4, len(CURRICULUM_PRESETS))
    assert (onehot.sum(axis=1) == 1).all()
    assert onehot[0, CURRICULUM_PRESETS.index("easy")] == 1


def test_rank_feature_best_is_zero():
    pop = population([1.0, 5.0, 3.0])
    best = max(pop, key=lambda r: r.fitness)
    worst = min(pop, key=lambda r: r.fitness)
    from prometheus.creators.features import RANK_INDEX
    assert candidate_features(best, pop)[RANK_INDEX] == 0.0
    assert candidate_features(worst, pop)[RANK_INDEX] == 1.0
