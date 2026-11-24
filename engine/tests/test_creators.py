from dataclasses import replace

import numpy as np
import pytest

from prometheus.benchmark import CURRICULUM_PRESETS, DEFAULT_CURRICULUM
from prometheus.creators import CREATOR_NAMES, get_creator
from prometheus.creators.base import ChildSpec, GenerationReport
from prometheus.genome import Genome
from prometheus.registry import Registry


def metrics(fitness=1.0, hazard=0.0, starvation=0.0, adaptation=1.0, speed=0.25):
    return {"fitness": fitness, "mean_score": fitness, "stable_score": fitness, "shift_score": fitness,
            "survival_rate": 0.5, "adaptation_ratio": adaptation, "learning_speed": speed,
            "mean_prediction_error": 1.0, "perception_cost": 10.0,
            "death_causes": {"hazard": hazard, "starvation": starvation, "post_shift": 0.0},
            "lesson_scores": [1.0] * 4}


def population(rows):
    reg = Registry(clock=lambda: "t")
    recs = []
    for i, m in enumerate(rows):
        g = Genome.random(np.random.default_rng(i))
        r = reg.create(ChildSpec(g, None, False, DEFAULT_CURRICULUM, "seed"), 0)
        r.metrics = m
        recs.append(r)
    return GenerationReport(generation=1, population=recs, history=[])


def test_registry_names():
    assert set(CREATOR_NAMES) >= {"random", "heuristic"}
    with pytest.raises(ValueError):
        get_creator("nope")


@pytest.mark.parametrize("name", ["random", "heuristic"])
def test_both_creators_return_exact_slot_count_and_valid_specs(name):
    creator = get_creator(name)
    rng = np.random.default_rng(0)
    seeds = creator.seed_population(8, rng)
    assert len(seeds) == 8
    report = population([metrics(fitness=f) for f in np.linspace(0.1, 2.0, 8)])
    specs = creator.next_generation(report, 6, rng)
    assert len(specs) == 6
    ids = {r.id for r in report.population}
    for s in seeds + specs:
        assert s.genome == s.genome.clamped()
        assert len(s.curriculum) == 4 and all(p in CURRICULUM_PRESETS for p in s.curriculum)
        assert s.reason
    assert all(s.parent_id in ids for s in specs)


def test_validated_fixes_bad_curriculum_and_genome():
    bad = ChildSpec(replace(Genome.default(), perception_radius=9), None, False, ("easy", "bogus"), "r")
    good = bad.validated()
    assert good.curriculum == DEFAULT_CURRICULUM
    assert good.genome.perception_radius == 5


def test_heuristic_targets_hazard_deaths():
    report = population([metrics(fitness=2.0, hazard=0.5)] + [metrics(fitness=0.1)] * 3)
    parent = report.population[0]
    spec = get_creator("heuristic").next_generation(report, 1, np.random.default_rng(0))[0]
    assert spec.parent_id == parent.id
    assert spec.genome.risk_aversion > parent.genome_obj().risk_aversion or parent.genome_obj().risk_aversion == 3.0
    assert "hazard" in spec.reason
    assert spec.curriculum == ("easy", "hard", "shift", "normal")
    assert spec.inherit_beliefs


def test_heuristic_targets_poor_adaptation():
    report = population([metrics(fitness=2.0, adaptation=0.3)] + [metrics(fitness=0.1)] * 3)
    parent = report.population[0].genome_obj()
    spec = get_creator("heuristic").next_generation(report, 1, np.random.default_rng(0))[0]
    assert spec.genome.surprise_threshold < parent.surprise_threshold or parent.surprise_threshold == 0.1
    assert "adapt" in spec.reason
    assert spec.curriculum == ("shift", "shift", "shift", "normal")


def test_random_creator_seeded():
    report = population([metrics(fitness=f) for f in (1, 2, 3, 4)])
    a = get_creator("random").next_generation(report, 3, np.random.default_rng(9))
    b = get_creator("random").next_generation(report, 3, np.random.default_rng(9))
    assert [(s.genome, s.curriculum, s.parent_id) for s in a] == [(s.genome, s.curriculum, s.parent_id) for s in b]


def test_random_parents_from_top_half():
    report = population([metrics(fitness=f) for f in (1, 2, 3, 4)])
    top = {r.id for r in report.population if r.metrics["fitness"] >= 3}
    specs = get_creator("random").next_generation(report, 20, np.random.default_rng(1))
    assert {s.parent_id for s in specs} <= top


def test_all_equal_fitness_no_crash():
    report = population([metrics(fitness=0.0)] * 4)
    for name in ("random", "heuristic"):
        assert len(get_creator(name).next_generation(report, 3, np.random.default_rng(0))) == 3


def test_clone_creator_copies_parents_exactly():
    report = population([metrics(fitness=f) for f in (1, 2, 3, 4)])
    specs = get_creator("clone").next_generation(report, 4, np.random.default_rng(0))
    by_id = {r.id: r for r in report.population}
    for s in specs:
        parent = by_id[s.parent_id]
        assert s.genome == parent.genome_obj() and list(s.curriculum) == parent.curriculum
        assert not s.inherit_beliefs


def test_heuristic_never_ends_training_on_a_rule_shift():
    """A final shift lesson leaves the child believing scrambled laws (reviewer finding #2)."""
    rows = [metrics(fitness=2.0, hazard=0.5), metrics(fitness=2.0, adaptation=0.3),
            metrics(fitness=2.0, starvation=0.5), metrics(fitness=2.0, speed=0.75), metrics(fitness=2.0)]
    for row in rows:
        report = population([row] + [metrics(fitness=0.1)] * 3)
        spec = get_creator("heuristic").next_generation(report, 1, np.random.default_rng(0))[0]
        assert spec.curriculum[-1] != "shift", spec.reason
