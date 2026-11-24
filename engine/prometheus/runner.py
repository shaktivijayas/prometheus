"""The generation loop: Creator designs and teaches, children learn, the benchmark judges, the best survive."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .benchmark import EvalSuite, replay_episodes, train_and_evaluate_replicated
from .child import Beliefs
from .creators import get_creator
from .creators.base import Creator, GenerationReport, ranked
from .genome import Genome
from .registry import EntityRecord, EventLog, Registry
from .world import PRESETS


@dataclass
class RunConfig:
    creator: str = "heuristic"
    generations: int = 10
    population: int = 8
    elite: int = 2
    seed: int = 0
    episodes_per_lesson: int = 5
    eval_worlds: int = 24
    workers: int = 0              # 0 = all cores, 1 = in-process
    record_replays: bool = True
    replicates: int = 3           # independent upbringings averaged per child, so fitness reflects the design
    universe: int | None = None   # which hidden laws; defaults to `seed` so every arm with a seed shares a universe

    def universe_seed(self) -> int:
        return self.seed if self.universe is None else self.universe


@dataclass
class RunResult:
    config: RunConfig
    registry: Registry
    events: EventLog
    generations: list[dict]
    replays: dict[str, dict]
    universe: list[str]
    timing: dict = field(default_factory=dict)


def child_seed(run_seed: int, entity_id: str) -> int:
    return int(np.random.SeedSequence([run_seed, int(entity_id[1:])]).generate_state(1)[0])


def replicate_seeds(run_seed: int, entity_id: str, replicates: int) -> list[int]:
    """Replicate 0 uses child_seed (it is the child that lives on); the rest are independent upbringings."""
    extra = [int(np.random.SeedSequence([run_seed, int(entity_id[1:]), r]).generate_state(1)[0])
             for r in range(1, replicates)]
    return [child_seed(run_seed, entity_id)] + extra


def _evaluate_job(job: tuple) -> tuple[dict, dict]:
    """Top-level (spawn-safe) worker: teach and score one child."""
    genome_d, beliefs_d, curriculum, seeds, suite, episodes_per_lesson = job
    beliefs = Beliefs.from_dict(beliefs_d) if beliefs_d else None
    metrics, learned = train_and_evaluate_replicated(Genome.from_dict(genome_d), beliefs, curriculum, seeds, suite,
                                                     episodes_per_lesson=episodes_per_lesson)
    return metrics.to_dict(), learned.to_dict()


class _Executor:
    def __init__(self, workers: int):
        self.workers = workers if workers > 0 else (os.cpu_count() or 1)
        self.pool = mp.get_context("spawn").Pool(self.workers) if self.workers > 1 else None

    def map(self, jobs: list[tuple]) -> list[tuple[dict, dict]]:
        if self.pool is None or len(jobs) <= 1:
            return [_evaluate_job(j) for j in jobs]
        return self.pool.map(_evaluate_job, jobs)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _diversity(records: list[EntityRecord]) -> float:
    vecs = [Genome.from_dict(r.genome).to_vector() for r in records]
    if len(vecs) < 2:
        return 0.0
    d = [float(np.linalg.norm(a - b)) for i, a in enumerate(vecs) for b in vecs[i + 1:]]
    return float(np.mean(d))


def _gen_stats(generation: int, active: list[EntityRecord]) -> dict:
    f = np.array([r.fitness for r in active]) if active else np.zeros(1)
    best = ranked(active)[0] if active else None
    return {"generation": generation, "best": round(float(f.max()), 6), "mean": round(float(f.mean()), 6),
            "median": round(float(np.median(f)), 6), "diversity": round(_diversity(active), 6),
            "best_entity": best.id if best else None}


def run_evolution(config: RunConfig, creator: Creator | None = None,
                  clock: Callable[[], str] | None = None,
                  executor: "_Executor | None" = None) -> RunResult:
    t0 = time.perf_counter()
    creator = creator or get_creator(config.creator)
    suite = EvalSuite(n_worlds=config.eval_worlds, rules_seed=config.universe_seed())
    registry, events = Registry(clock=clock), EventLog()
    rng = np.random.default_rng([config.seed, 1])
    generations: list[dict] = []
    replays: dict[str, dict] = {}
    own_executor = executor is None
    executor = executor or _Executor(config.workers)
    try:
        for gen in range(config.generations):
            if gen == 0:
                specs = creator.seed_population(config.population, rng)
            else:
                active = ranked(registry.active())
                for loser in active[config.elite:]:
                    registry.retire(loser.id, gen)
                    events.emit("retired", gen, loser.id, fitness=loser.fitness)
                report = GenerationReport(gen, active, list(generations))
                specs = creator.next_generation(report, config.population - config.elite, rng)
                if len(specs) != config.population - config.elite:
                    raise RuntimeError(f"creator {creator.name} returned {len(specs)} specs, "
                                       f"expected {config.population - config.elite}")
            new = []
            for spec in specs:
                spec = _checked_parent(spec.validated(), registry)
                rec = registry.create(spec, gen)
                new.append(rec)
                events.emit("created", gen, rec.id, name=rec.name, parent_id=rec.parent_id,
                            specialization=rec.specialization)
                events.emit("creator_decision", gen, rec.id, reason=rec.reason, parent_id=rec.parent_id,
                            predicted_change=rec.predicted_change, curriculum=rec.curriculum,
                            inherit_beliefs=rec.inherit_beliefs)
            _evaluate(new, registry, events, executor, suite, config, gen)
            active = registry.active()
            stats = _gen_stats(gen, active)
            generations.append(stats)
            events.emit("generation_complete", gen, **{k: v for k, v in stats.items() if k != "generation"})
            if config.record_replays and stats["best_entity"]:
                _record_replays(registry.get(stats["best_entity"]), gen, suite, config, registry, events, replays)
    finally:
        if own_executor:
            executor.close()
    return RunResult(config, registry, events, generations, replays, list(suite.rules().roles),
                     timing={"duration_s": round(time.perf_counter() - t0, 3)})


def _evaluate(new: list[EntityRecord], registry: Registry, events: EventLog, executor: _Executor,
              suite: EvalSuite, config: RunConfig, gen: int) -> None:
    jobs = []
    for rec in new:
        start = _start_beliefs(rec, registry)
        jobs.append((rec.genome, start.to_dict() if start else None, tuple(rec.curriculum),
                     replicate_seeds(config.seed, rec.id, max(1, config.replicates)), suite,
                     config.episodes_per_lesson))
    for rec, (metrics, beliefs) in zip(new, executor.map(jobs)):
        rec.metrics, rec.beliefs = metrics, beliefs
        if rec.parent_id:
            rec.actual_change = round(rec.fitness - registry.get(rec.parent_id).fitness, 6)
        events.emit("trained", gen, rec.id, lesson_scores=metrics["lesson_scores"], curriculum=rec.curriculum)
        events.emit("evaluated", gen, rec.id, fitness=metrics["fitness"], survival_rate=metrics["survival_rate"],
                    adaptation_ratio=metrics["adaptation_ratio"], actual_change=rec.actual_change,
                    predicted_change=rec.predicted_change)


def _checked_parent(spec, registry: Registry):
    """A parent must be an already-scored entity; otherwise the child is treated as parentless."""
    if spec.parent_id is None:
        return spec
    try:
        parent = registry.get(spec.parent_id)
    except KeyError:
        parent = None
    if parent is None or parent.metrics is None:
        spec.reason = f"{spec.reason} [invalid parent {spec.parent_id!r} dropped]"
        spec.parent_id, spec.inherit_beliefs = None, False
    return spec


def _start_beliefs(rec: EntityRecord, registry: Registry) -> Beliefs | None:
    """The beliefs a child began life with. Parents never retrain, so their stored beliefs are exactly these."""
    if rec.parent_id and rec.inherit_beliefs:
        parent = registry.get(rec.parent_id)
        return Beliefs.from_dict(parent.beliefs) if parent.beliefs else None
    return None


def _record_replays(best: EntityRecord, gen: int, suite: EvalSuite, config: RunConfig, registry: Registry,
                    events: EventLog, replays: dict[str, dict]) -> None:
    worlds = [0, suite.n_worlds // 2]
    shift_step = PRESETS["shift"].shift_step
    threshold = best.genome["surprise_threshold"]
    for preset, wseed, res in replay_episodes(Genome.from_dict(best.genome), _start_beliefs(best, registry),
                                             tuple(best.curriculum), child_seed(config.seed, best.id), suite,
                                             worlds, episodes_per_lesson=config.episodes_per_lesson):
        key = f"{best.id}_{preset}_{wseed}"
        replays[key] = {"entity_id": best.id, "name": best.name, "generation": gen, "preset": preset,
                        "seed": wseed, "score": round(res.score, 6), "goals": res.goals, "steps": res.steps,
                        "survived": res.survived, "death_cause": res.death_cause, "trace": res.trace}
        if preset == "shift":
            for step in res.trace[1:]:
                if step["t"] > shift_step and step["surprise"] > threshold:
                    events.emit("rule_shift_detected", gen, best.id, t=step["t"], kind=step["kind"],
                                surprise=step["surprise"], replay=key)
                    break
