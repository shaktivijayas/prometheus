# PROMETHEUS

**One intelligence creates another. Another learns. Another evolves.**

PROMETHEUS is an experimental AI ecosystem built around one question: *can an AI (the Creator) design, teach and evolve other AIs that measurably improve over generations, and can the Creator itself be trained to get better at creating?*

The Creator is a neural network we build and train ourselves with reinforcement learning. It does not call Claude or any other external model or API. Each child intelligence has its own identity, genome, learned memory and lineage. Children live in a grid world whose laws are hidden: they find out by experience which tiles feed them, hurt them or count as goals, and the laws sometimes change in the middle of an episode. The Creator decides each child's design (its genome), whether it inherits its parent's knowledge, and how it is taught (its curriculum of lessons). Every claim is checked against a control arm that uses random mutation and gets the same budget.

> **Status:** the core is done. The engine, the trained neural Creator and the lab dashboard all work.
> **Live lab dashboard:** https://shaktivijayas.github.io/prometheus/
> Full results: [`docs/creator-results.md`](docs/creator-results.md) · Design: [`docs/superpowers/specs/2026-11-24-prometheus-core-design.md`](docs/superpowers/specs/2026-11-24-prometheus-core-design.md)

## Headline result

Every number below was measured in universes the Creator never trained in, with the same budget for every arm. Results are verified, meaning champions are re-scored on 10 fresh upbringings, across 5 universes.

| Creator | Verified final champion | **Verified gain** over gen-0 |
|---|---|---|
| **trained Creator** (PPO, 156 iterations) | **15.85 ± 0.97** | **+12.37 ± 1.38** |
| random mutation (control) | 10.22 ± 0.82 | +6.75 ± 1.51 |
| untrained Creator (same network, random weights) | 8.27 ± 1.87 | +4.79 ± 2.16 |
| trained Creator with its teaching switched off | 5.10 ± 2.26 | +1.62 ± 2.18 |
| clone (null: changes nothing) | 3.48 ± 0.97 | +0.00 ± 0.00 |
| hand-written rules | 1.26 ± 0.16 | −0.06 ± 0.20 |

- **The trained Creator gets 1.8× random mutation's improvement.** Its worst held-out universe beats random's best.
- **Training is what does it.** The same network untrained is no better than random.
- **Teaching is half the skill.** Without its lesson plans the trained Creator's gain collapses to +1.6. It learned to almost never teach a law-change lesson (0.2% of lessons) and to favour hard lessons. It also designs fast-learning, far-seeing children that drop their beliefs as soon as the world surprises them.

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
.venv/Scripts/python -m pytest                          # 129 tests

# evolve one population with one Creator
.venv/Scripts/prometheus run --creator random --seed 0

# compare Creators on identical budgets across seeds
.venv/Scripts/prometheus compare --creators clone,heuristic,random --seeds 0,1,2 --name baseline --out ..
```

Creators: `clone` (null baseline), `random` (control), `heuristic` (hand-written rules) and `neural` (the trained network). `neural` takes a version: `neural:latest`, `neural:v000` … `neural:v006`, and `:noteach` switches teaching off. Trained checkpoints are in `checkpoints/`.

```bash
# train the Creator (or continue: --resume --iterations <more>)
.venv/Scripts/prometheus train-creator --iterations 300 --workers 6
# score every checkpoint on held-out universes, then export everything for the dashboard
.venv/Scripts/prometheus creator-progress --versions all --seeds 0,1,2 --out ..
.venv/Scripts/prometheus export-dashboard --out ../dashboard/public/data --runs "../runs/final_*_s0" --comparisons ../comparisons/final.json --creator ../creator
```

The dashboard (`dashboard/`, React + Vite) runs with `npm install && npm run dev`.

## Baseline results (before the Creator)

The defaults are population 8, 10 generations, each child scored as the average of 3 upbringings, and 3 seeds. The run took 176 s on a 12-core laptop.

| Creator | Verified gen-0 best | Verified final champion | **Verified gain** (mean ± std, 3 seeds) | In-run best* |
|---|---|---|---|---|
| clone (null: copies parents, changes nothing) | 4.131 | 4.131 ± 0.450 | **+0.000 ± 0.000** | 6.514 |
| heuristic (hand-written rules) | 1.325 | 1.192 ± 0.134 | **−0.132 ± 0.210** | 1.938 |
| random mutation (control) | 4.131 | 9.818 ± 0.840 | **+5.687 ± 0.862** | 10.672 |

\* The in-run best is selection-biased. Selection keeps whichever copy of a design got lucky with its training worlds, so even the clone arm, which never changes anything, "improves" from 3.16 to 6.51 in-run. That's why every run ends by **re-scoring the generation-0 best design and the final champion on 10 fresh upbringings** that played no part in selection. The verified gain is the honest headline, and the clone arm's verified gain of exactly 0 shows it works.

**What the numbers say:**
- **Evolution is real.** Random mutation plus selection more than doubled the verified quality of the best design (4.13 → 9.82).
- **The hand-written rules don't improve anything.** In the first version, all of their lesson plans ended on a rule-shift lesson, which leaves a child believing scrambled laws. The engine review caught this. Even with that fixed, the rules make no measurable progress.
- **The bar for the trained Creator is a verified gain clearly above random mutation's +5.7.**

See [`docs/engine-speed.md`](docs/engine-speed.md) for details.

## Run output

Each run writes a self-contained folder, which the dashboard reads:

```
runs/<run>/
  run.json          config, universe, timing, summary, verification (unbiased re-scoring)
  entities.json     every child ever created: identity, lineage, genome, curriculum, metrics, beliefs, reason
  events.jsonl      timeline: created, creator_decision, trained, evaluated, retired, rule_shift_detected, generation_complete
  generations.json  best / mean / median fitness and genome diversity per generation
  replays/          step-by-step traces of each champion in one stable and one shift world
comparisons/<name>.json   per-arm curves (mean ± std across seeds)
```

## License

MIT
