import math
from dataclasses import replace

import numpy as np
import pytest

from prometheus.child import GOAL_VALUE, Beliefs, ChildAgent, run_episode
from prometheus.genome import SURPRISE_OFF, Genome
from prometheus.world import FLOOR, WALL, PRESETS, StepResult, World, WorldConfig


def agent(**genes):
    g = replace(Genome.default(), **genes)
    return ChildAgent(g, 5, np.random.default_rng(0))


def outcome(kind, delta, goal=False):
    return StepResult(kind=kind, energy_delta=delta, reached_goal=goal, shifted=False, done=False)


def blank_world():
    w = World(WorldConfig(), 0)
    w.grid[:, :] = FLOOR
    w.pos = (5, 5)
    return w


def kind_of(world, role):
    return world.rules.roles.index(role)


def test_beliefs_learn_food_value():
    a = agent()
    for _ in range(30):
        a.observe_outcome(outcome(1, 8.0))
    assert a.beliefs.mean[1] == pytest.approx(8.0, abs=0.01)
    assert a.beliefs.count[1] > 0


def test_goal_value_includes_bonus():
    a = agent()
    a.observe_outcome(outcome(0, 5.0, goal=True))
    assert a.beliefs.mean[0] == pytest.approx(5.0 + GOAL_VALUE)


def test_surprise_resets_counts():
    a = agent(surprise_threshold=0.5, memory_capacity=50)
    for _ in range(20):
        a.observe_outcome(outcome(1, 8.0))
    a.observe_outcome(outcome(2, 8.0))
    counts_before = a.beliefs.count.copy()
    s = a.observe_outcome(outcome(1, -12.0))
    assert s == pytest.approx(2.0, abs=0.05)
    assert a.beliefs.mean[1] == -12.0
    assert a.beliefs.count[2] == pytest.approx(counts_before[2] * 0.25)


def test_surprise_off_never_resets():
    a = agent(surprise_threshold=SURPRISE_OFF, memory_capacity=50)
    for _ in range(20):
        a.observe_outcome(outcome(1, 8.0))
    a.observe_outcome(outcome(2, 8.0))
    c2 = a.beliefs.count[2]
    a.observe_outcome(outcome(1, -12.0))
    assert a.beliefs.count[2] == c2
    assert a.beliefs.mean[1] > -12.0


def test_unknown_kind_has_no_surprise():
    a = agent(surprise_threshold=0.1)
    assert a.observe_outcome(outcome(3, -12.0)) == 0.0


def test_memory_zero_forgets_between_episodes():
    a = agent(memory_capacity=0)
    a.observe_outcome(outcome(1, 8.0))
    a.end_episode()
    a.start_episode()
    assert a.beliefs.count.sum() == 0


def test_memory_keeps_beliefs_between_episodes():
    a = agent(memory_capacity=10)
    a.observe_outcome(outcome(1, 8.0))
    a.end_episode()
    a.start_episode()
    assert a.beliefs.count[1] > 0


def test_count_capped_by_memory_capacity():
    a = agent(memory_capacity=2)
    for _ in range(50):
        a.observe_outcome(outcome(1, 8.0))
    assert a.beliefs.count[1] == 10  # 5 * memory_capacity


def test_planner_moves_toward_known_food():
    w = blank_world()
    food = kind_of(w, "food")
    w.grid[5, 7] = food
    a = agent(exploration=0.0)
    a.beliefs.mean[food], a.beliefs.count[food] = 8.0, 10
    assert a.act(w) == 1  # East


def test_planner_avoids_believed_hazard_path():
    w = blank_world()
    food, hz = kind_of(w, "food"), kind_of(w, "hazard")
    w.grid[5, 6] = hz
    w.grid[5, 7] = food
    a = agent(exploration=0.0)
    a.beliefs.mean[food], a.beliefs.count[food] = 8.0, 10
    a.beliefs.mean[hz], a.beliefs.count[hz] = -12.0, 10
    assert a.act(w) in (0, 2)  # goes around via North or South


def test_risk_aversion_changes_behaviour_on_unknown_tiles():
    w = blank_world()
    unknown = kind_of(w, "neutral")
    w.grid[5, 6] = unknown
    bold = agent(exploration=0.0, risk_aversion=0.0, curiosity_bonus=1.0)
    timid = agent(exploration=0.0, risk_aversion=3.0, curiosity_bonus=1.0)
    assert bold.act(w) == 1  # curious and bold: investigates the unknown tile
    assert timid.act(w) != 1


def test_boxed_in_agent_stays():
    w = blank_world()
    for r, c in ((4, 5), (6, 5), (5, 4), (5, 6)):
        w.grid[r, c] = WALL
    a = agent(exploration=0.0)
    assert a.act(w) == 4
    res = run_episode(a, w)
    assert not res.survived and res.death_cause == "starvation"


def test_run_episode_deterministic():
    def go():
        a = ChildAgent(Genome.default(), 5, np.random.default_rng(7))
        return run_episode(a, World(PRESETS["shift"], 3))
    assert go() == go()


def test_episode_result_fields_finite():
    w = blank_world()  # nothing to eat: the child must die on its first step
    w.energy = 0.5
    res = run_episode(agent(), w)
    assert not res.survived and res.death_cause in ("hazard", "starvation")
    for v in (res.score, res.mean_prediction_error, res.perception_cost):
        assert math.isfinite(v)
    assert res.pre_shift_rate is None and res.post_shift_rate is None


def test_shift_world_reports_shift_rates():
    a = ChildAgent(Genome.default(), 5, np.random.default_rng(1))
    w = World(PRESETS["shift"], 2)
    w.energy = 1e6
    res = run_episode(a, w)
    assert res.pre_shift_rate is not None and res.post_shift_rate is not None


def test_trace_recorded():
    w = World(WorldConfig(), 4)
    res = run_episode(agent(), w, record=True)
    assert "grid" in res.trace[0] and "roles" in res.trace[0]
    assert len(res.trace) == res.steps + 1
    assert {"t", "pos", "action", "kind", "predicted", "value", "energy", "surprise", "shifted"} <= set(res.trace[1])


def test_beliefs_dict_roundtrip():
    b = Beliefs(np.array([1.0, 2, 3, 4, 5]), np.array([0.0, 1, 2, 3, 4]))
    b2 = Beliefs.from_dict(b.to_dict())
    assert np.array_equal(b.mean, b2.mean) and np.array_equal(b.count, b2.count)
