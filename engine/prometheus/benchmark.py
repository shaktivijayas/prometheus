"""Curriculum training and the fixed benchmark every child is scored on."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .child import Beliefs, ChildAgent, run_episode
from .genome import Genome
from .world import PRESETS, RuleSet, make_world

CURRICULUM_PRESETS = ("easy", "normal", "hard", "shift")
LESSONS = 4
EPISODES_PER_LESSON = 5
DEFAULT_CURRICULUM = ("normal", "normal", "shift", "normal")
EVAL_SEED = 10_000


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


@dataclass(frozen=True)
class EvalSuite:
    n_worlds: int = 24
    seed: int = EVAL_SEED
    rules_seed: int = EVAL_SEED  # which universe (hidden laws) training and evaluation happen in

    def rules(self) -> RuleSet:
        return RuleSet.random(PRESETS["normal"].n_kinds, np.random.default_rng(self.rules_seed))

    def worlds(self) -> list[tuple[str, int]]:
        half = self.n_worlds // 2
        stable = [("normal", self.seed + i) for i in range(half)]
        shift = [("shift", self.seed + half + i) for i in range(self.n_worlds - half)]
        return stable + shift


def train_child(agent: ChildAgent, curriculum: Sequence[str], seed: int,
                episodes_per_lesson: int = EPISODES_PER_LESSON, rules: RuleSet | None = None) -> list[float]:
    """Run the child through its lessons (it learns as it goes). Returns mean score per lesson."""
    for preset in curriculum:
        if preset not in CURRICULUM_PRESETS:
            raise ValueError(f"unknown lesson preset {preset!r}; valid: {CURRICULUM_PRESETS}")
    scores = []
    for lesson, preset in enumerate(curriculum):
        lesson_scores = [run_episode(agent, make_world(preset, _seed(seed, lesson, ep), rules)).score
                         for ep in range(episodes_per_lesson)]
        scores.append(float(np.mean(lesson_scores)) if lesson_scores else 0.0)
    return scores


def adaptation_ratio(pre_rates: Sequence[float], post_rates: Sequence[float]) -> float:
    """How well goal-finding holds up after the rules change (1.0 = fully adapted)."""
    if not pre_rates:
        return 1.0
    pre, post = float(np.mean(pre_rates)), float(np.mean(post_rates))
    if pre == 0.0:
        return 1.0 if post == 0.0 else 2.0
    return float(min(max(post / pre, 0.0), 2.0))


def shift_rates(results) -> tuple[list[float], list[float]]:
    """Pre/post-shift goal rates from the shift-world episodes where the child lived to see the shift."""
    seen = [r for r in results if r.pre_shift_rate is not None and r.post_shift_rate is not None]
    return [r.pre_shift_rate for r in seen], [r.post_shift_rate for r in seen]


def learning_speed(lesson_scores: Sequence[float]) -> float:
    """Fraction of the course needed to reach 90% of the best lesson score (lower = faster)."""
    if not lesson_scores:
        return 1.0
    best = max(lesson_scores)
    for i, s in enumerate(lesson_scores):
        if s >= 0.9 * best:
            return (i + 1) / len(lesson_scores)
    return 1.0


@dataclass
class ChildMetrics:
    fitness: float
    mean_score: float
    stable_score: float
    shift_score: float
    survival_rate: float
    adaptation_ratio: float
    learning_speed: float
    mean_prediction_error: float
    perception_cost: float
    death_causes: dict[str, float] = field(default_factory=dict)
    lesson_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
        d["death_causes"] = {k: round(v, 6) for k, v in self.death_causes.items()}
        d["lesson_scores"] = [round(v, 6) for v in self.lesson_scores]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChildMetrics":
        return cls(**d)


def _evaluate_worlds(agent: ChildAgent, worlds: list[tuple[str, int]], lesson_scores: list[float],
                     rules: RuleSet | None = None) -> ChildMetrics:
    results = []
    for preset, wseed in worlds:
        trial = copy.deepcopy(agent)  # every world starts from the same trained state
        results.append((preset, run_episode(trial, make_world(preset, wseed, rules))))
    scores = [r.score for _, r in results]
    stable = [r.score for p, r in results if not PRESETS[p].shift]
    shifted = [r for p, r in results if PRESETS[p].shift]
    n = len(results)
    mean_score = float(np.mean(scores)) if scores else 0.0
    ratio = adaptation_ratio(*shift_rates(shifted))
    return ChildMetrics(
        fitness=mean_score * (0.75 + 0.25 * min(ratio, 1.0)),
        mean_score=mean_score,
        stable_score=float(np.mean(stable)) if stable else 0.0,
        shift_score=float(np.mean([r.score for r in shifted])) if shifted else 0.0,
        survival_rate=sum(r.survived for _, r in results) / n if n else 0.0,
        adaptation_ratio=ratio,
        learning_speed=learning_speed(lesson_scores),
        mean_prediction_error=float(np.mean([r.mean_prediction_error for _, r in results])) if n else 0.0,
        perception_cost=float(np.mean([r.perception_cost for _, r in results])) if n else 0.0,
        death_causes={
            "hazard": sum(r.death_cause == "hazard" for _, r in results) / n if n else 0.0,
            "starvation": sum(r.death_cause == "starvation" for _, r in results) / n if n else 0.0,
            "post_shift": sum(r.died_after_shift for _, r in results) / n if n else 0.0,
        },
        lesson_scores=list(lesson_scores),
    )


def evaluate(agent: ChildAgent, suite: EvalSuite, lesson_scores: list[float]) -> ChildMetrics:
    return _evaluate_worlds(agent, suite.worlds(), lesson_scores, suite.rules())


def train_and_evaluate(genome: Genome, beliefs: Beliefs | None, curriculum: Sequence[str], seed: int,
                       suite: EvalSuite, n_kinds: int = 5,
                       episodes_per_lesson: int = EPISODES_PER_LESSON) -> tuple[ChildMetrics, Beliefs]:
    """Build a child, teach it its curriculum, then score it. Pure: same inputs, same outputs."""
    agent = ChildAgent(genome, n_kinds, np.random.default_rng(seed), beliefs)
    lessons = train_child(agent, curriculum, seed, episodes_per_lesson, suite.rules())
    return evaluate(agent, suite, lessons), agent.beliefs.copy()


def replay_episodes(genome: Genome, beliefs: Beliefs | None, curriculum: Sequence[str], seed: int,
                    suite: EvalSuite, world_indices: Sequence[int], n_kinds: int = 5,
                    episodes_per_lesson: int = EPISODES_PER_LESSON) -> list[tuple[str, int, "object"]]:
    """Re-create a child exactly as train_and_evaluate did and record step-by-step traces of chosen eval worlds."""
    agent = ChildAgent(genome, n_kinds, np.random.default_rng(seed), beliefs)
    train_child(agent, curriculum, seed, episodes_per_lesson, suite.rules())
    worlds = suite.worlds()
    out = []
    for i in world_indices:
        preset, wseed = worlds[i]
        trial = copy.deepcopy(agent)
        out.append((preset, wseed, run_episode(trial, make_world(preset, wseed, suite.rules()), record=True)))
    return out


def average_metrics(items: Sequence[ChildMetrics]) -> ChildMetrics:
    """Mean of several replicate scorings of the same design."""
    if len(items) == 1:
        return items[0]
    keys = ("fitness", "mean_score", "stable_score", "shift_score", "survival_rate", "adaptation_ratio",
            "learning_speed", "mean_prediction_error", "perception_cost")
    avg = {k: float(np.mean([getattr(m, k) for m in items])) for k in keys}
    causes = {k: float(np.mean([m.death_causes[k] for m in items])) for k in items[0].death_causes}
    lessons = [float(x) for x in np.mean([m.lesson_scores for m in items], axis=0)] if items[0].lesson_scores else []
    return ChildMetrics(**avg, death_causes=causes, lesson_scores=lessons)


def train_and_evaluate_replicated(genome: Genome, beliefs: Beliefs | None, curriculum: Sequence[str],
                                  seeds: Sequence[int], suite: EvalSuite, n_kinds: int = 5,
                                  episodes_per_lesson: int = EPISODES_PER_LESSON) -> tuple[ChildMetrics, Beliefs]:
    """Score a design over several independent upbringings (different training seeds) and average.

    One upbringing is a lottery (the same genome scores 0.06-2.24 depending on its training worlds);
    averaging replicates makes fitness reflect the design. The child that lives on (its beliefs, its
    replays) is replicate 0.
    """
    runs = [train_and_evaluate(genome, beliefs, curriculum, s, suite, n_kinds, episodes_per_lesson) for s in seeds]
    return average_metrics([m for m, _ in runs]), runs[0][1]
