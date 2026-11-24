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
