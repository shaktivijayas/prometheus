import math

import numpy as np
import pytest

import prometheus.benchmark as bm
from prometheus.benchmark import (
    CURRICULUM_PRESETS, DEFAULT_CURRICULUM, EvalSuite, adaptation_ratio, evaluate,
    train_and_evaluate, train_child,
)
from prometheus.child import ChildAgent
from prometheus.genome import Genome


def new_agent(seed=0):
    return ChildAgent(Genome.default(), 5, np.random.default_rng(seed))


def test_constants():
    assert CURRICULUM_PRESETS == ("easy", "normal", "hard", "shift")
    assert DEFAULT_CURRICULUM == ("normal", "normal", "shift", "normal")


def test_eval_suite_is_fixed():
    worlds = EvalSuite().worlds()
    assert worlds == EvalSuite().worlds()
    assert len(worlds) == 24
    assert [p for p, _ in worlds] == ["normal"] * 12 + ["shift"] * 12


def test_training_budget_identical_across_curricula(monkeypatch):
    calls = []
    real = bm.run_episode

    def counting(agent, world, record=False):
        calls.append(1)
        return real(agent, world, record)

    monkeypatch.setattr(bm, "run_episode", counting)
    for curriculum in [DEFAULT_CURRICULUM, ("easy",) * 4, ("hard", "shift", "easy", "normal")]:
        calls.clear()
        scores = train_child(new_agent(), curriculum, seed=1)
        assert len(calls) == 20 and len(scores) == 4


def test_invalid_preset_raises():
    with pytest.raises(ValueError):
        train_child(new_agent(), ("easy", "bogus", "easy", "easy"), seed=0)


def test_eval_order_independent():
    agent = new_agent()
    train_child(agent, DEFAULT_CURRICULUM, seed=2, episodes_per_lesson=1)
    suite = EvalSuite(n_worlds=6)
    m1 = evaluate(agent, suite, [1.0])
    reversed_suite = EvalSuite(n_worlds=6)
    reversed_suite_worlds = list(reversed(suite.worlds()))
    m2 = bm._evaluate_worlds(agent, reversed_suite_worlds, [1.0], suite.rules())
    assert m1.mean_score == pytest.approx(m2.mean_score)
    assert m1.fitness == pytest.approx(m2.fitness)


def test_evaluate_does_not_change_agent():
    agent = new_agent()
    train_child(agent, DEFAULT_CURRICULUM, seed=2, episodes_per_lesson=1)
    before = agent.beliefs.copy()
    evaluate(agent, EvalSuite(n_worlds=4), [1.0])
    assert np.array_equal(before.mean, agent.beliefs.mean)


def test_adaptation_ratio_zero_pre_shift():
    assert adaptation_ratio([0.0, 0.0], [0.0, 0.0]) == 1.0
    assert adaptation_ratio([0.0], [0.1]) == 2.0
    assert adaptation_ratio([0.2], [0.1]) == pytest.approx(0.5)
    assert adaptation_ratio([0.1], [0.9]) == 2.0  # clipped
    assert adaptation_ratio([], []) == 1.0


def test_metrics_fields():
    m = evaluate(new_agent(), EvalSuite(n_worlds=4), [0.5, 1.0, 1.0, 1.0])
    d = m.to_dict()
    assert set(d["death_causes"]) == {"hazard", "starvation", "post_shift"}
    assert m.learning_speed == pytest.approx(2 / 4)
    for k, v in d.items():
        if isinstance(v, float):
            assert math.isfinite(v), k
    assert m.fitness == pytest.approx(m.mean_score * (0.75 + 0.25 * min(m.adaptation_ratio, 1.0)))


def test_trained_child_beats_untrained():
    suite = EvalSuite()
    trained = new_agent(0)
    lessons = train_child(trained, DEFAULT_CURRICULUM, seed=0, rules=suite.rules())
    untrained = new_agent(0)
    assert evaluate(trained, suite, lessons).fitness > evaluate(untrained, suite, []).fitness


def test_train_and_evaluate_deterministic():
    suite = EvalSuite(n_worlds=4)
    a = train_and_evaluate(Genome.default(), None, DEFAULT_CURRICULUM, 5, suite, episodes_per_lesson=1)
    b = train_and_evaluate(Genome.default(), None, DEFAULT_CURRICULUM, 5, suite, episodes_per_lesson=1)
    assert a[0] == b[0]
    assert np.array_equal(a[1].mean, b[1].mean)


def test_inherited_beliefs_are_used():
    suite = EvalSuite(n_worlds=4)
    _, parent_beliefs = train_and_evaluate(Genome.default(), None, DEFAULT_CURRICULUM, 1, suite, episodes_per_lesson=2)
    _, child_beliefs = train_and_evaluate(Genome.default(), parent_beliefs, ("easy",) * 4, 2, suite, episodes_per_lesson=0)
    assert np.array_equal(parent_beliefs.mean, child_beliefs.mean)


def test_universe_is_shared_by_training_and_eval():
    suite = EvalSuite(n_worlds=4, rules_seed=7)
    assert suite.rules().roles == EvalSuite(n_worlds=4, rules_seed=7).rules().roles
    assert any(EvalSuite(rules_seed=s).rules().roles != suite.rules().roles for s in range(8, 20))
