"""Child intelligences: a learned world model (beliefs about each tile kind) plus a planner.

A child never sees what a tile kind does. It predicts the value of stepping on a kind
from its beliefs, observes the actual outcome, learns from the error, and (if its
genome allows) partially resets its beliefs when it is surprised - which is how it
adapts when the world's rules change.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .genome import SURPRISE_OFF, Genome
from .world import ACTIONS, PERCEPTION_COST, WALL, StepResult, World

GOAL_VALUE = 20.0
ENERGY_SCALE = 10.0
UNKNOWN_RISK = 0.5        # how much risk aversion discounts never-seen kinds (x ENERGY_SCALE)
DISTANCE_COST = 0.5
RESET_FACTOR = 0.25


@dataclass
class Beliefs:
    mean: np.ndarray
    count: np.ndarray

    @classmethod
    def empty(cls, n_kinds: int) -> "Beliefs":
        return cls(np.zeros(n_kinds), np.zeros(n_kinds))

    def copy(self) -> "Beliefs":
        return Beliefs(self.mean.copy(), self.count.copy())

    def to_dict(self) -> dict:
        return {"mean": [round(float(x), 6) for x in self.mean],
                "count": [round(float(x), 6) for x in self.count]}

    @classmethod
    def from_dict(cls, d: dict) -> "Beliefs":
        return cls(np.array(d["mean"], dtype=float), np.array(d["count"], dtype=float))


@dataclass
class EpisodeResult:
    score: float
    goals: int
    steps: int
    survived: bool
    death_cause: str | None
    died_after_shift: bool
    pre_shift_rate: float | None
    post_shift_rate: float | None
    mean_prediction_error: float
    perception_cost: float
    trace: list[dict] | None = None


class ChildAgent:
    def __init__(self, genome: Genome, n_kinds: int, rng: np.random.Generator, beliefs: Beliefs | None = None):
        self.genome = genome
        self.n_kinds = n_kinds
        self.rng = rng
        self.beliefs = beliefs.copy() if beliefs is not None else Beliefs.empty(n_kinds)
        self.epsilon = float(genome.exploration)
        self._errors: list[float] = []

    # ----- learning -------------------------------------------------------
    def start_episode(self) -> None:
        if self.genome.memory_capacity == 0:
            self.beliefs = Beliefs.empty(self.n_kinds)
        self._errors = []

    def end_episode(self) -> None:
        self.epsilon *= self.genome.exploration_decay

    def observe_outcome(self, result: StepResult) -> float:
        """Learn from stepping onto a tile. Returns the surprise (0 if nothing to compare)."""
        kind = result.kind
        if kind is None:
            return 0.0
        value = result.energy_delta + (GOAL_VALUE if result.reached_goal else 0.0)
        b = self.beliefs
        surprise = 0.0
        if b.count[kind] > 0:
            error = abs(value - b.mean[kind])
            self._errors.append(error)
            surprise = error / ENERGY_SCALE
        alpha = max(self.genome.learning_rate, 1.0 / (b.count[kind] + 1.0))
        b.mean[kind] += alpha * (value - b.mean[kind])
        b.count[kind] = min(b.count[kind] + 1.0, 5.0 * max(self.genome.memory_capacity, 1))
        threshold = self.genome.surprise_threshold
        if threshold < SURPRISE_OFF and surprise > threshold:
            b.count *= RESET_FACTOR
            b.mean[kind] = value
        return surprise

    @property
    def mean_prediction_error(self) -> float:
        return float(np.mean(self._errors)) if self._errors else 0.0

    # ----- acting ---------------------------------------------------------
    def cell_value(self, cell: int) -> float:
        """Believed value of stepping onto a cell, adjusted for risk. Walls are -inf."""
        if cell == WALL:
            return -math.inf
        if cell < 0:
            return 0.0
        g, b = self.genome, self.beliefs
        if b.count[cell] <= 0:
            return g.curiosity_bonus * ENERGY_SCALE - g.risk_aversion * UNKNOWN_RISK * ENERGY_SCALE
        return float(b.mean[cell] - g.risk_aversion * ENERGY_SCALE / math.sqrt(b.count[cell] + 1.0))

    def act(self, world: World) -> int:
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(ACTIONS)))
        r = self.genome.perception_radius
        view = world.observe(r)
        size = view.shape[0]
        # lookup table indexed by cell + 2: WALL(-2), FLOOR(-1), kinds 0..K-1
        table = np.array([self.cell_value(c) for c in range(WALL, self.n_kinds)])
        values = table[view.astype(np.int64) + 2]
        start = (r, r)
        first: dict[tuple[int, int], int] = {start: -1}
        dist = {start: 0}
        queue = deque([start])
        best_action, best_score = None, 0.0
        while queue:
            cur = queue.popleft()
            if dist[cur] >= self.genome.planning_horizon:
                continue
            for a, (dr, dc) in enumerate(ACTIONS[:4]):
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in dist:
                    continue
                v = values[nxt]
                if v < 0:  # walls, and tiles believed harmful, are never entered
                    continue
                dist[nxt] = dist[cur] + 1
                first[nxt] = a if cur == start else first[cur]
                score = v - DISTANCE_COST * dist[nxt]
                if v > 0 and score > best_score:
                    best_action, best_score = first[nxt], score
                queue.append(nxt)
        if best_action is not None:
            return best_action
        safe = [a for a, (dr, dc) in enumerate(ACTIONS[:4]) if values[r + dr, r + dc] >= 0]
        if not safe:
            return 4
        return int(safe[int(self.rng.integers(len(safe)))])


def run_episode(agent: ChildAgent, world: World, record: bool = False) -> EpisodeResult:
    agent.start_episode()
    radius = agent.genome.perception_radius
    trace: list[dict] | None = None
    if record:
        trace = [{"grid": world.grid.tolist(), "roles": list(world.rules.roles), "radius": radius,
                  "pos": list(world.pos), "energy": world.energy,
                  "shift_step": world.config.shift_step if world.config.shift else None}]
    goals = goals_pre = goals_post = 0
    steps_pre = steps_post = 0
    last_delta = 0.0
    while not world.done:
        action = agent.act(world)
        res = world.step(action, radius)
        predicted = None
        if res.kind is not None and agent.beliefs.count[res.kind] > 0:
            predicted = round(float(agent.beliefs.mean[res.kind]), 4)
        surprise = agent.observe_outcome(res)
        last_delta = res.energy_delta
        if res.reached_goal:
            goals += 1
        if world.shift_happened:
            steps_post += 1
            goals_post += int(res.reached_goal)
        else:
            steps_pre += 1
            goals_pre += int(res.reached_goal)
        if trace is not None:
            value = None
            if res.kind is not None:
                value = res.energy_delta + (GOAL_VALUE if res.reached_goal else 0.0)
            trace.append({"t": world.t, "pos": list(world.pos), "action": action, "kind": res.kind,
                          "predicted": predicted, "value": value, "energy": round(world.energy, 4),
                          "surprise": round(surprise, 4), "shifted": res.shifted})
    agent.end_episode()
    survived = world.energy > 0
    death_cause = None if survived else ("hazard" if last_delta < 0 else "starvation")
    max_steps = world.config.max_steps
    shift_world = world.config.shift
    return EpisodeResult(
        score=goals + 0.5 * world.t / max_steps,
        goals=goals,
        steps=world.t,
        survived=survived,
        death_cause=death_cause,
        died_after_shift=(not survived) and world.shift_happened,
        pre_shift_rate=(goals_pre / steps_pre if steps_pre else 0.0) if shift_world else None,
        post_shift_rate=(goals_post / steps_post if steps_post else 0.0) if shift_world else None,
        mean_prediction_error=agent.mean_prediction_error,
        perception_cost=world.t * PERCEPTION_COST * radius,
        trace=trace,
    )
