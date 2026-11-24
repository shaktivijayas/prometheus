"""The generation loop: Creator designs and teaches, children learn, the benchmark judges, the best survive.

`Evolution` is a steppable state machine (propose jobs -> ingest results), so many evolutions can share one
worker pool in lockstep (`run_lockstep`), which is how the Creator is trained. `run_evolution` drives one.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .benchmark import EvalSuite, is_training_universe, replay_episodes, train_and_evaluate_replicated
from .child import Beliefs
from .creators import get_creator
from .creators.base import Creator, GenerationReport, ranked
from .genome import Genome
from .registry import EntityRecord, EventLog, Registry
from .world import PRESETS, WorldVariant


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
    verify_replicates: int = 10   # fresh upbringings re-scoring gen-0 best and final champion; 0 = skip
    variant: WorldVariant = field(default_factory=WorldVariant)

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
    verification: dict = field(default_factory=dict)


def child_seed(run_seed: int, entity_id: str) -> int:
    return int(np.random.SeedSequence([run_seed, int(entity_id[1:])]).generate_state(1)[0])


def replicate_seeds(run_seed: int, entity_id: str, replicates: int) -> list[int]:
    """Replicate 0 uses child_seed (it is the child that lives on); the rest are independent upbringings."""
    extra = [int(np.random.SeedSequence([run_seed, int(entity_id[1:]), r]).generate_state(1)[0])
             for r in range(1, replicates)]
    return [child_seed(run_seed, entity_id)] + extra


def verification_seeds(run_seed: int, n: int) -> list[int]:
    """Fresh upbringings never used during selection (a different seed-sequence shape from replicate_seeds)."""
    return [int(np.random.SeedSequence([run_seed, 2**31, 7, i]).generate_state(1)[0]) for i in range(n)]


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


def _job(rec: EntityRecord, registry: Registry, seeds: list[int], suite: EvalSuite, config: RunConfig) -> tuple:
    start = _start_beliefs(rec, registry)
    return (rec.genome, start.to_dict() if start else None, tuple(rec.curriculum), seeds, suite,
            config.episodes_per_lesson)


class Evolution:
    """One population evolving under one Creator. Drive it with pending_jobs() / ingest(results)."""

    def __init__(self, config: RunConfig, creator: Creator | None = None,
                 clock: Callable[[], str] | None = None, allow_training_universe: bool = False):
        if is_training_universe(config.universe_seed()) and not allow_training_universe:
            raise ValueError(f"universe {config.universe_seed()} is reserved for training the Creator; "
                             "evaluation runs must use a smaller universe seed")
        self.t0 = time.perf_counter()
        self.config = config
        self.creator = creator or get_creator(config.creator)
        self.suite = EvalSuite(n_worlds=config.eval_worlds, rules_seed=config.universe_seed(), variant=config.variant)
        self.registry, self.events = Registry(clock=clock), EventLog()
        self.rng = np.random.default_rng([config.seed, 1])
        self.generations: list[dict] = []
        self.replays: dict[str, dict] = {}
        self.verification: dict = {}
        self.gen = 0
        self.phase = "propose" if config.generations > 0 else "done"
        self._new: list[EntityRecord] = []
        self._verify_targets: dict[str, str] = {}

    @property
    def done(self) -> bool:
        return self.phase == "done"

    # ----- stepping -------------------------------------------------------
    def pending_jobs(self) -> list[tuple]:
        if self.phase == "propose":
            return self._propose()
        if self.phase == "verify":
            return self._verify_jobs()
        return []

    def ingest(self, results: list[tuple[dict, dict]]) -> None:
        if self.phase == "propose":
            self._ingest_generation(results)
        elif self.phase == "verify":
            self._ingest_verification(results)

    def result(self) -> RunResult:
        return RunResult(self.config, self.registry, self.events, self.generations, self.replays,
                         list(self.suite.rules().roles), timing={"duration_s": round(time.perf_counter() - self.t0, 3)},
                         verification=self.verification)

    # ----- generations ----------------------------------------------------
    def _propose(self) -> list[tuple]:
        cfg, gen, registry, events = self.config, self.gen, self.registry, self.events
        if gen == 0:
            specs = self.creator.seed_population(cfg.population, self.rng)
        else:
            active = ranked(registry.active())
            for loser in active[cfg.elite:]:
                registry.retire(loser.id, gen)
                events.emit("retired", gen, loser.id, fitness=loser.fitness)
            report = GenerationReport(gen, active, list(self.generations))
            specs = self.creator.next_generation(report, cfg.population - cfg.elite, self.rng)
            if len(specs) != cfg.population - cfg.elite:
                raise RuntimeError(f"creator {self.creator.name} returned {len(specs)} specs, "
                                   f"expected {cfg.population - cfg.elite}")
        self._new = []
        for spec in specs:
            spec = _checked_parent(spec.validated(), registry)
            rec = registry.create(spec, gen)
            self._new.append(rec)
            events.emit("created", gen, rec.id, name=rec.name, parent_id=rec.parent_id,
                        specialization=rec.specialization)
            events.emit("creator_decision", gen, rec.id, reason=rec.reason, parent_id=rec.parent_id,
                        predicted_change=rec.predicted_change, curriculum=rec.curriculum,
                        inherit_beliefs=rec.inherit_beliefs)
        return [_job(rec, registry, replicate_seeds(cfg.seed, rec.id, max(1, cfg.replicates)), self.suite, cfg)
                for rec in self._new]

    def _ingest_generation(self, results: list[tuple[dict, dict]]) -> None:
        gen, registry, events = self.gen, self.registry, self.events
        for rec, (metrics, beliefs) in zip(self._new, results):
            rec.metrics, rec.beliefs = metrics, beliefs
            if rec.parent_id:
                rec.actual_change = round(rec.fitness - registry.get(rec.parent_id).fitness, 6)
            events.emit("trained", gen, rec.id, lesson_scores=metrics["lesson_scores"], curriculum=rec.curriculum)
            events.emit("evaluated", gen, rec.id, fitness=metrics["fitness"], survival_rate=metrics["survival_rate"],
                        adaptation_ratio=metrics["adaptation_ratio"], actual_change=rec.actual_change,
                        predicted_change=rec.predicted_change)
        stats = _gen_stats(gen, registry.active())
        self.generations.append(stats)
        events.emit("generation_complete", gen, **{k: v for k, v in stats.items() if k != "generation"})
        if self.config.record_replays and stats["best_entity"]:
            self._record_replays(registry.get(stats["best_entity"]))
        self.gen += 1
        if self.gen >= self.config.generations:
            self.phase = "verify"

    # ----- verification ---------------------------------------------------
    def _verify_jobs(self) -> list[tuple]:
        """Re-score the gen-0 best and the final champion on fresh upbringings.

        Selection keeps the luckiest draws, so the in-run 'best' curve is biased upward (a clone arm that changes
        nothing still 'improves'). These numbers are not: the same fresh seeds are used for both designs.
        """
        n = self.config.verify_replicates
        if n <= 0 or not self.generations:
            self.phase = "done"
            return []
        seeds = verification_seeds(self.config.seed, n)
        self._verify_targets = {"gen0_best": self.generations[0]["best_entity"],
                                "final_best": self.generations[-1]["best_entity"]}
        jobs = []
        for entity_id in self._verify_targets.values():
            rec = self.registry.get(entity_id)
            jobs += [_job(rec, self.registry, [s], self.suite, self.config) for s in seeds]
        return jobs

    def _ingest_verification(self, results: list[tuple[dict, dict]]) -> None:
        n = self.config.verify_replicates
        out: dict = {}
        for i, (key, entity_id) in enumerate(self._verify_targets.items()):
            f = np.array([m["fitness"] for m, _ in results[i * n:(i + 1) * n]])
            out[key] = {"entity_id": entity_id, "name": self.registry.get(entity_id).name,
                        "mean": round(float(f.mean()), 6), "std": round(float(f.std()), 6), "n": n}
        out["gain"] = round(out["final_best"]["mean"] - out["gen0_best"]["mean"], 6)
        base = out["gen0_best"]["mean"]
        out["gain_pct"] = round(100.0 * out["gain"] / base, 2) if base > 0 else None
        self.verification = out
        self.phase = "done"

    # ----- replays --------------------------------------------------------
    def _record_replays(self, best: EntityRecord) -> None:
        suite, cfg = self.suite, self.config
        worlds = [0, suite.n_worlds // 2]
        shift_step = suite.variant.apply(PRESETS["shift"]).shift_step
        threshold = best.genome["surprise_threshold"]
        for preset, wseed, res in replay_episodes(Genome.from_dict(best.genome), _start_beliefs(best, self.registry),
                                                 tuple(best.curriculum), child_seed(cfg.seed, best.id), suite,
                                                 worlds, episodes_per_lesson=cfg.episodes_per_lesson):
            key = f"{best.id}_{preset}_{wseed}"
            self.replays[key] = {"entity_id": best.id, "name": best.name, "generation": self.gen, "preset": preset,
                                 "seed": wseed, "score": round(res.score, 6), "goals": res.goals,
                                 "steps": res.steps, "survived": res.survived, "death_cause": res.death_cause,
                                 "trace": res.trace}
            if preset == "shift":
                for step in res.trace[1:]:
                    if step["t"] > shift_step and step["surprise"] > threshold:
                        self.events.emit("rule_shift_detected", self.gen, best.id, t=step["t"], kind=step["kind"],
                                         surprise=step["surprise"], replay=key)
                        break


def run_lockstep(evolutions: list[Evolution], executor: _Executor) -> None:
    """Advance many evolutions together, sending all their children to the worker pool in one batch per step."""
    while True:
        batch: list[tuple[Evolution, int, int]] = []
        jobs: list[tuple] = []
        for evo in evolutions:
            if evo.done:
                continue
            pending = evo.pending_jobs()
            if evo.done:  # e.g. verification skipped
                continue
            batch.append((evo, len(jobs), len(pending)))
            jobs += pending
        if not batch:
            return
        results = executor.map(jobs) if jobs else []
        for evo, start, count in batch:
            evo.ingest(results[start:start + count])


def run_evolution(config: RunConfig, creator: Creator | None = None,
                  clock: Callable[[], str] | None = None,
                  executor: "_Executor | None" = None) -> RunResult:
    evo = Evolution(config, creator, clock)
    own_executor = executor is None
    executor = executor or _Executor(config.workers)
    try:
        run_lockstep([evo], executor)
    finally:
        if own_executor:
            executor.close()
    return evo.result()


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
