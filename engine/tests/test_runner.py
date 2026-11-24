import numpy as np
import json
import math

import pytest

import prometheus.world as world_mod
from prometheus.output import write_run
from prometheus.runner import RunConfig, run_evolution

TINY = dict(generations=3, population=4, elite=1, episodes_per_lesson=1, eval_worlds=4, workers=1)
CLOCK = lambda: "2026-11-24T00:00:00Z"  # noqa: E731


@pytest.fixture(scope="module")
def tiny_run():
    return run_evolution(RunConfig(creator="heuristic", **TINY), clock=CLOCK)


def active_at(result, generation):
    return [r for r in result.registry.all()
            if r.generation <= generation and (r.retired_at_generation is None or r.retired_at_generation > generation)]


def test_population_size_constant(tiny_run):
    for g in range(3):
        assert len(active_at(tiny_run, g)) == 4
    assert len(tiny_run.registry.active()) == 4
    assert len(tiny_run.registry.all()) == 4 + 2 * 3


def test_every_child_was_evaluated(tiny_run):
    for r in tiny_run.registry.all():
        assert r.metrics is not None and r.beliefs is not None


def test_elites_survive_unchanged(tiny_run):
    gen0 = [r for r in tiny_run.registry.all() if r.generation == 0]
    best = sorted(gen0, key=lambda r: (-r.fitness, r.id))[0]
    assert best.retired_at_generation is None or best.retired_at_generation > 1


def test_lineage_links(tiny_run):
    ids = {r.id for r in tiny_run.registry.all()}
    for r in tiny_run.registry.all():
        if r.generation == 0:
            assert r.parent_id is None
        else:
            assert r.parent_id in ids


def test_actual_change_recorded(tiny_run):
    for r in tiny_run.registry.all():
        if r.parent_id:
            parent = tiny_run.registry.get(r.parent_id)
            assert r.actual_change == pytest.approx(r.fitness - parent.fitness, abs=1e-6)


def test_generation_stats(tiny_run):
    assert [g["generation"] for g in tiny_run.generations] == [0, 1, 2]
    for g in tiny_run.generations:
        assert g["best"] >= g["mean"] >= 0 and math.isfinite(g["diversity"])
    best = [g["best"] for g in tiny_run.generations]
    assert best == sorted(best)  # elitism: the best never gets worse


def test_events_cover_lifecycle(tiny_run):
    types = {e["type"] for e in tiny_run.events.events}
    assert {"created", "trained", "evaluated", "retired", "creator_decision", "generation_complete"} <= types


def test_deterministic_across_worker_counts():
    a = run_evolution(RunConfig(creator="random", **{**TINY, "workers": 1}), clock=CLOCK)
    b = run_evolution(RunConfig(creator="random", **{**TINY, "workers": 3}), clock=CLOCK)
    assert [r.to_dict() for r in a.registry.all()] == [r.to_dict() for r in b.registry.all()]


def test_same_seed_same_universe_for_every_arm():
    a = run_evolution(RunConfig(creator="random", **TINY), clock=CLOCK)
    b = run_evolution(RunConfig(creator="heuristic", **TINY), clock=CLOCK)
    assert a.universe == b.universe


def test_write_run_layout(tiny_run, tmp_path):
    write_run(tiny_run, tmp_path)
    for name in ("run.json", "entities.json", "events.jsonl", "generations.json"):
        assert (tmp_path / name).exists()
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["config"]["creator"] == "heuristic" and "summary" in run and "universe" in run
    entities = json.loads((tmp_path / "entities.json").read_text())
    assert len(entities) == 10
    for line in (tmp_path / "events.jsonl").read_text().splitlines():
        json.loads(line)
    replays = list((tmp_path / "replays").glob("*.json"))
    champions = {g["best_entity"] for g in tiny_run.generations}
    assert len(replays) == 2 * len(champions)  # one stable + one shift replay per distinct champion
    rep = json.loads(replays[0].read_text())
    assert {"entity_id", "preset", "seed", "generation", "trace"} <= set(rep)


def test_all_dying_population_finite(monkeypatch):
    monkeypatch.setattr(world_mod, "START_ENERGY", 0.5)
    res = run_evolution(RunConfig(creator="heuristic", **TINY), clock=CLOCK)
    for g in res.generations:
        assert all(math.isfinite(v) for k, v in g.items() if isinstance(v, float))


def test_replicates_average_several_upbringings():
    one = run_evolution(RunConfig(creator="clone", **{**TINY, "generations": 1, "replicates": 1}), clock=CLOCK)
    two = run_evolution(RunConfig(creator="clone", **{**TINY, "generations": 1, "replicates": 2}), clock=CLOCK)
    f1 = [r.fitness for r in one.registry.all()]
    f2 = [r.fitness for r in two.registry.all()]
    assert f1 != f2  # second upbringing changes the averaged score
    # replicate 0 is the child that lives on: same learned beliefs either way
    assert [r.beliefs for r in one.registry.all()] == [r.beliefs for r in two.registry.all()]


def test_default_replicates_is_three():
    assert RunConfig().replicates == 3


class _BadCreator:
    """Misbehaves on purpose: unknown parent ids and NaN predictions (reviewer finding #4)."""
    name = "bad"

    def seed_population(self, n, rng):
        from prometheus.creators import get_creator
        return get_creator("random").seed_population(n, rng)

    def next_generation(self, report, slots, rng):
        from prometheus.creators.base import ChildSpec
        from prometheus.genome import Genome
        return [ChildSpec(Genome.default(), "E9999", True, ("normal",) * 4, "bad", predicted_change=float("nan"),
                          decision={"x": np.array([1.0, np.inf])})
                for _ in range(slots)]


def test_bad_creator_output_is_cleaned_and_json_is_strict(tmp_path):
    def strict(text):
        return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    res = run_evolution(RunConfig(creator="random", **{**TINY, "generations": 2}), creator=_BadCreator(), clock=CLOCK)
    for r in res.registry.all():
        if r.generation > 0:
            assert r.parent_id is None and not r.inherit_beliefs and r.predicted_change is None
    write_run(res, tmp_path)
    strict((tmp_path / "entities.json").read_text())
    for line in (tmp_path / "events.jsonl").read_text().splitlines():
        strict(line)


def test_run_ends_with_unbiased_verification():
    """Selection picks lucky draws; the headline must come from fresh upbringings (re-review of finding #1)."""
    res = run_evolution(RunConfig(creator="random", **{**TINY, "verify_replicates": 3}), clock=CLOCK)
    v = res.verification
    assert v["gen0_best"]["entity_id"] == res.generations[0]["best_entity"]
    assert v["final_best"]["entity_id"] == res.generations[-1]["best_entity"]
    assert v["gen0_best"]["n"] == v["final_best"]["n"] == 3
    assert v["gain"] == pytest.approx(v["final_best"]["mean"] - v["gen0_best"]["mean"], abs=1e-6)


def test_verification_seeds_never_used_in_selection():
    from prometheus.runner import replicate_seeds, verification_seeds
    used = {s for i in range(1, 200) for s in replicate_seeds(0, f"E{i:04d}", 3)}
    assert not used & set(verification_seeds(0, 50))


def test_verification_written_to_run_json(tmp_path):
    res = run_evolution(RunConfig(creator="clone", **{**TINY, "verify_replicates": 2}), clock=CLOCK)
    write_run(res, tmp_path)
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["verification"]["final_best"]["n"] == 2
