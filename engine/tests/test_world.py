import pytest
import numpy as np

from prometheus.world import (
    FLOOR, WALL, FOOD_VALUE, HAZARD_VALUE, GOAL_ENERGY, STEP_COST, PERCEPTION_COST,
    PRESETS, RuleSet, World, WorldConfig,
)


def kind_of(world, role):
    return world.rules.roles.index(role)


def clear_around(world):
    """Put the agent at (5,5) of a cleared grid so tests can place tiles precisely."""
    world.grid[:, :] = FLOOR
    world.pos = (5, 5)


def test_same_seed_same_world():
    a, b, c = World(WorldConfig(), 3), World(WorldConfig(), 3), World(WorldConfig(), 4)
    assert np.array_equal(a.grid, b.grid) and a.pos == b.pos and a.rules.roles == b.rules.roles
    assert not (np.array_equal(a.grid, c.grid) and a.pos == c.pos)


def test_tile_counts_match_config():
    cfg = WorldConfig()
    w = World(cfg, 0)
    roles = w.rules.roles
    counts = {r: 0 for r in set(roles)}
    for k, r in enumerate(roles):
        counts[r] += int((w.grid == k).sum())
    assert counts["food"] == cfg.n_food
    assert counts["hazard"] == cfg.n_hazard
    assert counts["goal"] == cfg.n_goal
    assert counts["neutral"] == cfg.n_neutral
    assert w.grid[w.pos] == FLOOR


def test_wall_blocks_movement():
    w = World(WorldConfig(), 0)
    clear_around(w)
    w.grid[4, 5] = WALL
    w.step(0, perception_radius=1)
    assert w.pos == (5, 5)


def test_food_consumed_and_respawned():
    w = World(WorldConfig(), 0)
    food = kind_of(w, "food")
    total_food_before = sum(int((w.grid == k).sum()) for k, r in enumerate(w.rules.roles) if r == "food")
    w.grid[w.pos[0], :] = np.where(w.grid[w.pos[0], :] == WALL, FLOOR, w.grid[w.pos[0], :])
    r, c = w.pos
    c2 = c + 1 if c + 1 < w.config.size else c - 1
    if w.grid[r, c2] >= 0 and w.rules.roles[w.grid[r, c2]] == "food":
        total_food_before -= 1
    w.grid[r, c2] = food
    total_food_before += 1
    action = 1 if c2 > c else 3
    e0 = w.energy
    res = w.step(action, perception_radius=2)
    assert res.kind == food
    assert w.energy - e0 == pytest.approx(FOOD_VALUE - STEP_COST - PERCEPTION_COST * 2)
    total_food_after = sum(int((w.grid == k).sum()) for k, rr in enumerate(w.rules.roles) if rr == "food")
    assert total_food_after == total_food_before


def test_hazard_persists():
    w = World(WorldConfig(), 0)
    clear_around(w)
    hz = kind_of(w, "hazard")
    w.grid[5, 6] = hz
    res = w.step(1, perception_radius=1)
    assert res.energy_delta == HAZARD_VALUE
    assert w.grid[5, 6] == hz


def test_goal_sets_flag():
    w = World(WorldConfig(), 0)
    clear_around(w)
    w.grid[6, 5] = kind_of(w, "goal")
    res = w.step(2, perception_radius=1)
    assert res.reached_goal and res.energy_delta == GOAL_ENERGY


def test_shift_at_step_100():
    w = World(PRESETS["shift"], 0)
    w.energy = 1e9
    before = list(w.rules.roles)
    shifted_at = []
    for _ in range(150):
        res = w.step(4, perception_radius=1)
        if res.shifted:
            shifted_at.append(w.t)
    assert shifted_at == [101]  # shift happens at the start of the step when t == 100
    assert w.shift_happened and w.rules.roles != before


def test_shift_changes_every_food_kind():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        rules = RuleSet.random(5, rng)
        new = rules.shifted(rng)
        for k, role in enumerate(rules.roles):
            if role == "food":
                assert new.roles[k] != "food"
        assert sorted(new.roles) == sorted(rules.roles)


def test_observe_shape_and_edges():
    w = World(WorldConfig(), 0)
    w.pos = (0, 0)
    obs = w.observe(2)
    assert obs.shape == (5, 5)
    assert (obs[:2, :] == WALL).all() and (obs[:, :2] == WALL).all()


def test_episode_ends_on_energy_or_max_steps():
    w = World(WorldConfig(), 0)
    clear_around(w)
    while not w.done:
        w.step(4, perception_radius=1)
    assert w.energy <= 0 and w.t < 200

    w2 = World(WorldConfig(), 0)
    w2.energy = 1e9
    while not w2.done:
        w2.step(4, perception_radius=1)
    assert w2.t == 200
