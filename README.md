# PROMETHEUS

**One intelligence creates another. Another learns. Another evolves.**

PROMETHEUS is an experimental AI ecosystem built around one question: *can an AI (the Creator) design, teach and evolve other AIs that measurably improve over generations, and can the Creator itself be trained to get better at creating?*

The Creator is a neural network we build and train ourselves with reinforcement learning. It does not call Claude or any other external model or API. Each child intelligence has its own identity, genome, learned memory and lineage. Children live in a grid world whose laws are hidden: they find out by experience which tiles feed them, hurt them or count as goals, and the laws sometimes change in the middle of an episode. The Creator decides each child's design (its genome), whether it inherits its parent's knowledge, and how it is taught (its curriculum of lessons). Every claim is checked against a control arm that uses random mutation and gets the same budget.

> **Status:** the core is done. The engine, the trained neural Creator and the lab dashboard all work.
> **Live lab dashboard:** https://shaktivijayas.github.io/prometheus/
> Full results: [`docs/creator-results.md`](docs/creator-results.md) · Design: [`docs/superpowers/specs/2026-11-24-prometheus-core-design.md`](docs/superpowers/specs/2026-11-24-prometheus-core-design.md)

## Headline result

Every number below was measured in universes the Creator never trained in, with the same budget for every arm. Results are verified: champions are re-scored on 10 fresh upbringings, in world layouts selection never saw, across 5 universes.

| Creator | Verified final champion | **Verified gain** over gen-0 |
|---|---|---|
| **trained Creator** (PPO, 156 iterations) | **16.29 ± 0.85** | **+11.51 ± 1.37** |
| trained Creator's designs, random lessons | 16.21 ± 0.70 | +11.43 ± 0.83 |
| random mutation (control) | 12.57 ± 1.14 | +7.80 ± 1.78 |
| untrained Creator (same network, random weights) | 11.02 ± 2.53 | +6.25 ± 2.92 |
| trained Creator's designs, one fixed default lesson plan | 6.94 ± 2.05 | +2.16 ± 1.93 |
| clone (null: changes nothing) | 4.78 ± 1.11 | +0.00 ± 0.00 |
| hand-written rules | 1.84 ± 0.28 | −0.00 ± 0.20 |

- **The trained Creator beats random mutation clearly.** The gap is about 6 standard errors, and its worst held-out universe beats random's best.
- **Training is what does it.** The same network untrained is no better than random.
- **Its skill is in design, not teaching.** It learned to build far-seeing, fast-learning children that drop their beliefs as soon as the world surprises them. Its learned lesson plans, though, are *no better than random lessons*. What children are taught still matters: one fixed plan wrecks the same designs. An earlier version of this README claimed the opposite, and a control arm added after code review corrected it. Details are in [`docs/creator-results.md`](docs/creator-results.md).

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
.venv/Scripts/python -m pytest                          # 132 tests

# evolve one population with one Creator
.venv/Scripts/prometheus run --creator random --seed 0

# compare Creators on identical budgets across seeds
.venv/Scripts/prometheus compare --creators clone,heuristic,random --seeds 0,1,2 --name baseline --out ..
```

Creators: `clone` (null baseline), `random` (control), `heuristic` (hand-written rules) and `neural` (the trained network). `neural` takes a version: `neural:latest`, `neural:v000` … `neural:v006`, `:noteach` gives every child one fixed default lesson plan, and `:randomteach` gives random lessons (both are control arms). Trained checkpoints are in `checkpoints/`.

```bash
# train the Creator (or continue: --resume --iterations <more>)
.venv/Scripts/prometheus train-creator --iterations 300 --workers 6
# score every checkpoint on held-out universes, then export everything for the dashboard
.venv/Scripts/prometheus creator-progress --versions all --seeds 0,1,2 --out ..
.venv/Scripts/prometheus export-dashboard --out ../dashboard/public/data --runs "../runs/final_*_s0" --comparisons ../comparisons/final.json --creator ../creator
```

The dashboard (`dashboard/`, React + Vite) runs with `npm install && npm run dev`.

## Baseline results (before the Creator)

The defaults are population 8, 10 generations, each child scored as the average of 3 upbringings, and 3 seeds. The run took 178 s on a 12-core laptop.

| Creator | Verified gen-0 best | Verified final champion | **Verified gain** (mean ± std, 3 seeds) | In-run best* |
|---|---|---|---|---|
| clone (null: copies parents, changes nothing) | 5.534 | 5.534 ± 0.507 | **+0.000 ± 0.000** | 6.514 |
| heuristic (hand-written rules) | 1.892 | 1.758 ± 0.273 | **−0.135 ± 0.095** | 1.938 |
| random mutation (control) | 5.534 | 12.433 ± 0.997 | **+6.900 ± 0.679** | 10.672 |

\* The in-run best is selection-biased. Selection keeps whichever copy of a design got lucky with its training worlds, so even the clone arm, which never changes anything, "improves" to 6.51 in-run. That's why every run ends by **re-scoring the generation-0 best design and the final champion on 10 fresh upbringings, in world layouts that played no part in selection**. The verified gain is the honest headline, and the clone arm's verified gain of exactly 0 shows it works.

**What the numbers say:**
- **Evolution is real.** Random mutation plus selection more than doubled the verified quality of the best design (5.53 → 12.43).
- **The hand-written rules don't improve anything.** In the first version, all of their lesson plans ended on a rule-shift lesson, which leaves a child believing scrambled laws. The engine review caught this. Even with that fixed, the rules make no measurable progress.

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
