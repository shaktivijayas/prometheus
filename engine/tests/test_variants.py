import numpy as np
import pytest

from prometheus.benchmark import TRAIN_UNIVERSE_BASE, EvalSuite, is_training_universe, train_and_evaluate
from prometheus.genome import Genome
from prometheus.runner import Evolution, RunConfig, _Executor, run_evolution, run_lockstep
from prometheus.creators import get_creator
from prometheus.world import PRESETS, World, WorldVariant, make_world

TINY = dict(generations=3, population=4, elite=1, episodes_per_lesson=1, eval_worlds=4, workers=1,
            replicates=1, verify_replicates=2)
CLOCK = lambda: "t"  # noqa: E731


def test_default_variant_changes_nothing():
    for preset, cfg in PRESETS.items():
        assert WorldVariant().apply(cfg) == cfg


def test_variant_scales_tiles_and_moves_shift():
    v = WorldVariant(food_scale=0.5, hazard_scale=2.0, shift_step=60)
    cfg = v.apply(PRESETS["shift"])
    assert cfg.n_food == 7 and cfg.n_hazard == 20 and cfg.shift_step == 60
    tiny = WorldVariant(food_scale=0.01).apply(PRESETS["easy"])
    assert tiny.n_food >= 1 and tiny.n_goal >= 1
    w = make_world("shift", 3, variant=WorldVariant(shift_step=60))
    w.energy = 1e9
    shifted_at = [w.t for _ in range(80) if w.step(4, 1).shifted]
    assert shifted_at == [61]


def test_random_variant_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = WorldVariant.random(rng)
        assert 0.6 <= v.food_scale <= 1.4 and 0.6 <= v.hazard_scale <= 1.6 and 60 <= v.shift_step <= 140


def test_suite_variant_is_used_for_training_and_eval():
    plain = train_and_evaluate(Genome.default(), None, ("normal",) * 4, 1, EvalSuite(n_worlds=4), episodes_per_lesson=1)
    lean = train_and_evaluate(Genome.default(), None, ("normal",) * 4, 1,
                              EvalSuite(n_worlds=4, variant=WorldVariant(food_scale=0.6)), episodes_per_lesson=1)
    assert plain[0] != lean[0]


def test_training_universes_are_disjoint_from_evaluation():
    assert is_training_universe(TRAIN_UNIVERSE_BASE) and not is_training_universe(TRAIN_UNIVERSE_BASE - 1)
    with pytest.raises(ValueError):
        run_evolution(RunConfig(creator="random", universe=TRAIN_UNIVERSE_BASE + 5, **TINY))


def test_lockstep_equals_separate_runs():
    configs = [RunConfig(creator=c, seed=s, **TINY) for c, s in (("random", 0), ("clone", 1), ("heuristic", 2))]
    separate = [run_evolution(c, clock=CLOCK) for c in configs]
    evolutions = [Evolution(c, get_creator(c.creator), clock=CLOCK) for c in configs]
    ex = _Executor(1)
    run_lockstep(evolutions, ex)
    ex.close()
    for a, evo in zip(separate, evolutions):
        b = evo.result()
        assert [r.to_dict() for r in a.registry.all()] == [r.to_dict() for r in b.registry.all()]
        assert a.generations == b.generations and a.verification == b.verification


def test_verification_can_be_skipped():
    res = run_evolution(RunConfig(creator="random", **{**TINY, "verify_replicates": 0}), clock=CLOCK)
    assert res.verification == {}
