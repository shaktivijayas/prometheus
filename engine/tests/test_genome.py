import math

import numpy as np

from prometheus.genome import GENE_BOUNDS, GENE_NAMES, INT_GENES, Genome


def within_bounds(g: Genome) -> bool:
    return all(GENE_BOUNDS[n][0] <= getattr(g, n) <= GENE_BOUNDS[n][1] for n in GENE_NAMES)


def test_gene_order_is_fixed():
    assert GENE_NAMES == (
        "perception_radius", "memory_capacity", "learning_rate", "exploration",
        "exploration_decay", "planning_horizon", "risk_aversion", "surprise_threshold",
        "curiosity_bonus",
    )


def test_default_is_within_bounds():
    assert within_bounds(Genome.default())


def test_clamp_rounds_ints_and_clips():
    g = Genome.default()
    bad = Genome(**{**g.to_dict(), "perception_radius": 7.6, "learning_rate": -1.0, "planning_horizon": 3.4})
    c = bad.clamped()
    assert c.perception_radius == 5 and isinstance(c.perception_radius, int)
    assert c.planning_horizon == 3 and isinstance(c.planning_horizon, int)
    assert c.learning_rate == 0.01


def test_clamp_replaces_nan_with_default():
    g = Genome(**{**Genome.default().to_dict(), "learning_rate": math.nan, "memory_capacity": math.inf})
    c = g.clamped()
    assert c.learning_rate == 0.3
    assert c.memory_capacity == 10


def test_vector_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = Genome.random(rng)
        v = g.to_vector()
        assert v.shape == (len(GENE_NAMES),) and (v >= 0).all() and (v <= 1).all()
        assert Genome.from_vector(v) == g


def test_random_is_seeded():
    assert Genome.random(np.random.default_rng(1)) == Genome.random(np.random.default_rng(1))
    g = Genome.random(np.random.default_rng(1))
    assert within_bounds(g)
    assert all(isinstance(getattr(g, n), int) for n in INT_GENES)


def test_dict_roundtrip():
    g = Genome.random(np.random.default_rng(5))
    assert Genome.from_dict(g.to_dict()) == g
