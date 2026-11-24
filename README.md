# PROMETHEUS

**One intelligence creates another. Another learns. Another evolves.**

PROMETHEUS is an experimental AI ecosystem built around one question: *can an AI (the Creator) design, teach and evolve other AIs that measurably improve over generations, and can the Creator itself be trained to get better at creating?*

The Creator is a neural network we build and train ourselves with reinforcement learning. It does not call Claude or any other external model or API. Each child intelligence has its own identity, genome, learned memory and lineage. Children live in a grid world whose laws are hidden: they find out by experience which tiles feed them, hurt them or count as goals, and the laws sometimes change in the middle of an episode. The Creator decides each child's design (its genome), whether it inherits its parent's knowledge, and how it is taught (its curriculum of lessons). Every claim is checked against a control arm that uses random mutation and gets the same budget.

> **Status:** Plan 1 of 3 (the engine) is done. Plan 2 (the trained neural Creator) and Plan 3 (the lab dashboard) are next.
> Design: [`docs/superpowers/specs/2026-11-24-prometheus-core-design.md`](docs/superpowers/specs/2026-11-24-prometheus-core-design.md)

## How it works

```
We train the CREATOR  ──►  Creator DESIGNS a child (genome), passes on knowledge, TEACHES it (picks 4 lessons)
        ▲                                           │
        │                                           ▼
 Creator is rewarded by               Child LEARNS from experience in hidden-rule worlds
 how much its children improve  ◄──   then is scored on a fixed benchmark of 24 worlds
```

- **World:** 15×15 grid with 5 tile kinds. Roles (goal, food, food, hazard, neutral) are hidden and fixed per *universe*. Shift worlds secretly permute them at step 100.
- **Child:** a learned world model (a belief about each kind's value) and a bounded planner. Nine genes control it: perception, memory, learning rate, exploration, planning horizon, risk aversion, surprise threshold (adapting to rule changes), curiosity, and exploration decay.
- **Teaching:** 4 lessons × 5 episodes, each lesson chosen from `easy`, `normal`, `hard` or `shift`. Every child gets the same total practice.
- **Benchmark:** 12 stable and 12 shift worlds, identical for every child, generation and Creator. Fitness rewards goals, survival and adapting after a shift.
- **Fairness:** every Creator arm gets the same population, generations, training budget, evaluation suite and universe.

## Quickstart

```bash
cd engine
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # on macOS/Linux use .venv/bin/python
.venv/Scripts/python -m pytest                          # 80 tests

# evolve one population with one Creator
.venv/Scripts/prometheus run --creator random --seed 0

# compare Creators on identical budgets across seeds
.venv/Scripts/prometheus compare --creators heuristic,random --seeds 0,1,2 --name baseline --out ..
```

Creators available now: `random` (control) and `heuristic` (hand-written rules). The trained `neural` Creator arrives in Plan 2.

## Baseline results

The defaults are population 8, 10 generations and 3 seeds. The run took 37.6 s on a 12-core laptop.

| Creator | Gen 0 best | Final best (mean ± std) | Gain |
|---|---|---|---|
| heuristic (hand-written rules) | 1.356 | 1.967 ± 0.080 | +45% |
| random mutation (control) | 1.916 | 10.678 ± 1.858 | +458% |

Evolution works: selection plus mutation raised the best child's score more than fivefold. The hand-written rules got stuck. They kept "fixing" adaptation by making children wipe their memory more often, and never checked whether that helped. Random mutation is the bar the trained Creator has to beat. See [`docs/engine-speed.md`](docs/engine-speed.md) for details.

## Run output

Each run writes a self-contained folder, which the dashboard (Plan 3) will read:

```
runs/<run>/
  run.json          config, universe, timing, summary
  entities.json     every child ever created: identity, lineage, genome, curriculum, metrics, beliefs, reason
  events.jsonl      timeline: created, creator_decision, trained, evaluated, retired, rule_shift_detected, generation_complete
  generations.json  best / mean / median fitness and genome diversity per generation
  replays/          step-by-step traces of each champion in one stable and one shift world
comparisons/<name>.json   per-arm curves (mean ± std across seeds)
```

## License

MIT
