"""The hidden-rule grid world.

Tiles carry a *kind* id (0..K-1). What each kind does (its role) is hidden from
children: they only ever see kind ids and must learn the effects by experience.
In shift worlds, the roles are secretly permuted part-way through an episode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOOR = -1
WALL = -2

START_ENERGY = 30.0
STEP_COST = 1.0
PERCEPTION_COST = 0.1  # per unit of perception radius, per step
FOOD_VALUE = 8.0
HAZARD_VALUE = -12.0
GOAL_ENERGY = 5.0
NEUTRAL_VALUE = 0.0
MAX_STEPS = 200
SHIFT_STEP = 100

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))  # N, E, S, W, stay

ROLES = ("goal", "food", "food", "hazard", "neutral")
ROLE_EFFECT = {"goal": GOAL_ENERGY, "food": FOOD_VALUE, "hazard": HAZARD_VALUE, "neutral": NEUTRAL_VALUE}
CONSUMED_ROLES = {"goal", "food"}


@dataclass(frozen=True)
class WorldConfig:
    size: int = 15
    n_kinds: int = 5
    wall_density: float = 0.08
    n_food: int = 14
    n_hazard: int = 10
    n_goal: int = 2
    n_neutral: int = 10
    shift: bool = False
    shift_step: int = SHIFT_STEP
    max_steps: int = MAX_STEPS


PRESETS: dict[str, WorldConfig] = {
    "easy": WorldConfig(size=9, wall_density=0.05, n_food=8, n_hazard=3, n_goal=1, n_neutral=4),
    "normal": WorldConfig(),
    "hard": WorldConfig(wall_density=0.15, n_food=8, n_hazard=20),
    "shift": WorldConfig(shift=True),
}


@dataclass
class RuleSet:
    roles: list[str]

    @classmethod
    def random(cls, n_kinds: int, rng: np.random.Generator) -> "RuleSet":
        if n_kinds != len(ROLES):
            raise ValueError(f"n_kinds must be {len(ROLES)} (one per role in ROLES)")
        return cls([ROLES[i] for i in rng.permutation(len(ROLES))])

    def effect(self, kind: int) -> float:
        return ROLE_EFFECT[self.roles[kind]]

    def is_goal(self, kind: int) -> bool:
        return self.roles[kind] == "goal"

    def shifted(self, rng: np.random.Generator) -> "RuleSet":
        """A permutation of roles in which no food kind stays food."""
        food = [k for k, r in enumerate(self.roles) if r == "food"]
        while True:
            new = [self.roles[i] for i in rng.permutation(len(self.roles))]
            if all(new[k] != "food" for k in food):
                return RuleSet(new)


@dataclass
class StepResult:
    kind: int | None          # kind id of the tile stepped onto (None for floor)
    energy_delta: float       # tile effect only (excludes step and perception cost)
    reached_goal: bool
    shifted: bool
    done: bool


class World:
    def __init__(self, config: WorldConfig, seed: int, rules: RuleSet | None = None):
        """`rules` are the universe's hidden laws; when omitted the world draws its own from `seed`."""
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.rules = RuleSet(list(rules.roles)) if rules is not None else RuleSet.random(config.n_kinds, self.rng)
        n = config.size
        self.grid = np.full((n, n), FLOOR, dtype=np.int8)
        walls = self.rng.random((n, n)) < config.wall_density
        self.grid[walls] = WALL
        counts = {"food": config.n_food, "hazard": config.n_hazard,
                  "goal": config.n_goal, "neutral": config.n_neutral}
        for role, count in counts.items():
            kinds = [k for k, r in enumerate(self.rules.roles) if r == role]
            for _ in range(count):
                cell = self._random_floor()
                if cell is None:
                    break
                self.grid[cell] = kinds[int(self.rng.integers(len(kinds)))]
        start = self._random_floor()
        if start is None:  # pathological config: clear a cell
            start = (n // 2, n // 2)
            self.grid[start] = FLOOR
        self.pos: tuple[int, int] = start
        self.energy = START_ENERGY
        self.t = 0
        self.done = False
        self.shift_happened = False
        self.last_kind: int | None = None

    def _random_floor(self, exclude: tuple[int, int] | None = None) -> tuple[int, int] | None:
        cells = np.argwhere(self.grid == FLOOR)
        if exclude is not None and len(cells):
            cells = cells[~((cells[:, 0] == exclude[0]) & (cells[:, 1] == exclude[1]))]
        if len(cells) == 0:
            return None
        r, c = cells[int(self.rng.integers(len(cells)))]
        return int(r), int(c)

    def observe(self, radius: int) -> np.ndarray:
        n = self.config.size
        size = 2 * radius + 1
        window = np.full((size, size), WALL, dtype=np.int8)
        r0, c0 = self.pos[0] - radius, self.pos[1] - radius
        rs, re = max(r0, 0), min(r0 + size, n)
        cs, ce = max(c0, 0), min(c0 + size, n)
        window[rs - r0:re - r0, cs - c0:ce - c0] = self.grid[rs:re, cs:ce]
        return window

    def step(self, action: int, perception_radius: int) -> StepResult:
        if self.done:
            return StepResult(None, 0.0, False, False, True)
        shifted = False
        if self.config.shift and not self.shift_happened and self.t == self.config.shift_step:
            self.rules = self.rules.shifted(self.rng)
            self.shift_happened = True
            shifted = True

        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        n = self.config.size
        if 0 <= r < n and 0 <= c < n and self.grid[r, c] != WALL:
            self.pos = (r, c)

        self.energy -= STEP_COST + PERCEPTION_COST * perception_radius
        cell = int(self.grid[self.pos])
        kind = cell if cell >= 0 else None
        delta = 0.0
        reached_goal = False
        if kind is not None:
            role = self.rules.roles[kind]
            delta = ROLE_EFFECT[role]
            self.energy += delta
            reached_goal = role == "goal"
            if role in CONSUMED_ROLES:
                self.grid[self.pos] = FLOOR
                spot = self._random_floor(exclude=self.pos)
                if spot is not None:
                    self.grid[spot] = kind
        self.last_kind = kind
        self.t += 1
        self.done = self.energy <= 0 or self.t >= self.config.max_steps
        return StepResult(kind, delta, reached_goal, shifted, self.done)


def make_world(preset: str, seed: int, rules: RuleSet | None = None) -> World:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; valid: {sorted(PRESETS)}")
    return World(PRESETS[preset], seed, rules)


__all__ = [
    "FLOOR", "WALL", "START_ENERGY", "STEP_COST", "PERCEPTION_COST", "FOOD_VALUE",
    "HAZARD_VALUE", "GOAL_ENERGY", "NEUTRAL_VALUE", "MAX_STEPS", "SHIFT_STEP", "ACTIONS",
    "ROLES", "WorldConfig", "PRESETS", "RuleSet", "StepResult", "World", "make_world",
]
