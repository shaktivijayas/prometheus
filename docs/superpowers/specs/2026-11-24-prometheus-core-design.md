# PROMETHEUS Core — Design

Date: 2026-11-24
Status: draft, awaiting approval

## 1. Intent

PROMETHEUS asks one question: **can an AI (the Creator) design other AIs that measurably improve over generations?** The full vision (see the original concept) spans world models, RL, evolution, tool creation, uncertainty, scientific discovery and artificial civilizations. This document covers only the **core**: the smallest system that can answer the central question honestly.

The defining rule (from the concept, §19): the system must show that the evolved population is *measurably better* than the original one, and that the Creator's design decisions beat a no-intelligence baseline. More agents or prettier graphs do not count as success.

## 2. Goals

1. A simulated world with **hidden rules** that agents must discover by experimenting, and that **change mid-run** so adaptation can be measured.
2. **Child intelligences** with persistent identity, genome (architecture), memory (learned beliefs), lineage and history.
3. **The Creator** analyses performance and failures and designs new children / successors, recording *why* each was created.
4. **A fixed benchmark** so every generation is scored on identical challenges.
5. **A control arm** (random mutation, no Creator reasoning) under an identical budget, to prove whether the Creator adds value.
6. **A lab dashboard** to inspect everything: genealogy, timeline, entity profiles, generation curves, arm comparison, episode replay.

## 3. Non-goals (for the core)

- ORACLE, GENESIS, ARGUS, artificial civilization (later add-ons).
- Multiple agents interacting in the same world (one child per episode in the core).
- LLM-powered children. Children are cheap numerical programs.
- Live-streaming the dashboard while a run executes (dashboard replays recorded runs).
- Training deep neural networks.

## 4. Anti-goals (would be failure even if it "works")

- Claiming improvement without the control-arm comparison, or from a single seed.
- The Creator arm winning only because it got more evaluations/compute than the control arm.
- Narrative theater: entity names and "reasons" that don't correspond to real genome changes.
- Any claim of consciousness or life.

## 5. Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **A. Python engine + React dashboard replaying recorded runs** | Python is the research standard; runs are reproducible files; dashboard can be hosted statically (GitHub Pages) with no API key exposed. Two codebases. | **Chosen by user.** |
| B. Everything in the browser (TypeScript) | Flashy live demo; weaker for real learning; Creator still needs a server for the key. | Rejected. |
| C. Python + Streamlit | Fastest; generic look; poor for genealogy and replay. | Rejected. |

Child type: **hybrid** — Creator (LLM or heuristic) designs; children are small learning programs. Chosen over LLM children because evolution needs thousands of cheap episodes and real learning, not prompt edits.

## 6. Architecture

```
prometheus/
  engine/                     Python package `prometheus`
    world.py                  GridWorld, hidden rule sets, rule shift
    genome.py                 Genome dataclass + bounds/validation
    child.py                  ChildAgent: world-model learner + planner
    benchmark.py              Fixed seeded benchmark suite + metrics
    creators/
      base.py                 Creator interface
      claude.py               Claude-backed Creator
      heuristic.py            Rule-based diagnostic Creator
      random_mutation.py      Control arm
    registry.py               Entity registry, lineage, event log
    runner.py                 Generation loop
    cli.py                    `prometheus run`, `prometheus compare`
  tests/
  dashboard/                  Vite + React + TypeScript
  runs/                       Output (gitignored except sample runs)
```

### 6.1 World

- Grid (default 15×15) with walls and cells of **tile kinds** `0..K-1` (default K=5). Agents see only kind IDs, never semantics.
- A **rule set** maps each kind to an effect: energy delta (food / hazard / neutral), passable or not, or "goal" (+reward, respawns elsewhere).
- Agent has energy; each step costs energy (+ extra cost proportional to perception radius). Episode ends at energy ≤ 0 or `max_steps` (default 200).
- **Rule shift:** in shift episodes, at step `shift_step` the kind→effect mapping is permuted (e.g. food kind becomes hazard). Unannounced.
- Actions: N/E/S/W/stay. Stepping onto a kind applies its effect.
- Fully deterministic given a seed.

### 6.2 Child (genome + learner)

**Genome** (the architecture the Creator designs), all bounded and validated:

| Gene | Meaning | Range |
|---|---|---|
| `perception_radius` | how far it sees | 1–5 (costs energy) |
| `memory_capacity` | episodes of belief carried over | 0–50 |
| `learning_rate` | belief update speed | 0.01–1.0 |
| `exploration` / `exploration_decay` | curiosity vs exploitation | 0–1 / 0.9–1.0 |
| `planning_horizon` | path-search depth | 1–12 |
| `risk_aversion` | weight on uncertain/negative kinds | 0–3 |
| `surprise_threshold` | prediction error that triggers belief reset (adaptation) | 0.1–5, or off |
| `curiosity_bonus` | value of visiting unknown kinds | 0–2 |

**Learner:** a small explicit world model — per-kind belief of energy effect (mean + count), updated from observed outcomes. Each step it *predicts* the effect of its move, observes the actual result, and records prediction error. A **planner** does bounded BFS within the visible area using believed values, risk aversion and curiosity. Surprise above threshold → partially reset beliefs (this is how it adapts to rule shifts).

**Memory:** beliefs persist across training episodes (up to `memory_capacity`); successors may **inherit** a parent's beliefs (an inheritance gene/decision the Creator controls).

**Identity record:** id, name (e.g. `NAVIGATOR-003`), parent id, generation, created_at, specialization label, creation reason, genome, status (active / retired), metrics history, belief snapshot, notable events.

### 6.3 Benchmark

- **Training:** each child runs `train_episodes` (default 20) on randomly seeded worlds (seeds derived from run seed + child id).
- **Evaluation:** a **fixed suite** of `eval_worlds` (default 24) seeded worlds: half stable, half with a rule shift. Same suite for every child, generation and arm.
- **Metrics per child:** mean score (goals + survival), survival rate, pre-shift vs post-shift score (**adaptation ratio**), learning speed (episodes to reach X% of final), mean prediction error, energy spent on perception.
- **Fitness:** a single scalar (weighted mean score with adaptation term) used for selection; all component metrics stored.

### 6.4 Creators

Common interface:

```python
class Creator:
    def seed_population(self, n: int, objective: str) -> list[ChildSpec]
    def next_generation(self, report: GenerationReport, budget: int) -> list[ChildSpec]
```

`ChildSpec` = genome + parent_id (optional) + inherit_beliefs flag + name + specialization + **reason** (text).

- **ClaudeCreator:** sends the generation report (metrics, failure summaries such as "died on hazard kind 2 after shift", genome table) to Claude; receives structured JSON child specs; validates/clamps genomes; falls back to heuristic on API failure. Model configurable (default Sonnet-class for cost); call count and tokens logged and capped.
- **HeuristicCreator:** rule-based diagnosis (low adaptation → raise surprise sensitivity; dying to hazards → raise risk aversion; slow learning → raise learning rate; etc.) producing targeted changes with generated reasons. Lets the whole system run with no API key.
- **RandomMutationCreator (control):** selects top-k by fitness and applies random Gaussian mutations. No reasoning.

**Selection** (shared, not Creator-controlled, for fairness): keep top `elite` (default 2) unchanged; the Creator fills the remaining slots. Population size fixed (default 8). Retired entities keep their full records.

**Fairness rule:** every arm gets the same population size, generations, training episodes and eval suite. Only the design decisions differ.

### 6.5 Runner and CLI

- `prometheus run --creator {claude,heuristic,random} --generations 10 --population 8 --seed 0 --out runs/<name>`
- `prometheus compare --creators heuristic,random[,claude] --seeds 0,1,2 --generations 10` → runs all arms × seeds and writes a comparison summary (mean ± std fitness per generation per arm).
- Human oversight: config caps on max children, max API calls/tokens; Claude arm prompts for confirmation before spending past a threshold (can be pre-approved with a flag).

### 6.6 Run output format (contract between engine and dashboard)

```
runs/<run_id>/
  run.json          config, creator, seed, timing, API usage, final summary
  entities.json     every child ever created (identity record, genome, metrics, lineage)
  events.jsonl      timeline: created, trained, evaluated, retired, rule_shift_detected, creator_reasoning
  generations.json  per-generation stats (best/mean/median fitness, diversity)
  replays/<entity_id>_<world>.json   step-by-step episode traces for the best child per generation (one stable, one shift world)
comparisons/<name>.json  arm × seed × generation curves
```

### 6.7 Dashboard

Vite + React + TypeScript, static. Loads an index of runs from `public/runs/`.

- **Overview:** run summary, population status, headline result ("Gen 10 best vs Gen 1 best: +X%; Creator vs control: …").
- **Genealogy:** tree of all entities (colored by fitness, retired greyed), click to open profile.
- **Entity profile:** identity, parent, reason for creation, genome (diff vs parent highlighted), metrics, belief table, events.
- **Timeline:** filterable event stream including Creator reasoning.
- **Evolution charts:** fitness per generation, metric breakdown.
- **Comparison:** arms side-by-side with error bands across seeds.
- **Replay:** canvas grid playback of an episode, showing agent, visible area, beliefs, predicted vs actual effect, and the moment of rule shift.

Deployable to GitHub Pages with bundled sample runs.

## 7. Testing

- pytest: world determinism and rule shift; genome validation/clamping; learner belief updates and surprise reset; benchmark reproducibility (same seed → same scores); each Creator returns valid specs (Claude mocked); runner end-to-end on a tiny config; output schema.
- Dashboard: type-checked against a shared schema; smoke build.

## 8. Success criteria for the core

1. `prometheus compare` with heuristic vs random over ≥3 seeds runs on a laptop in minutes with no API key.
2. Dashboard shows all views for a recorded run.
3. The report states honestly whether later generations beat earlier ones and whether the Creator beat control — including if it didn't.

## 9. Builder's calls (decided by Claude, open to correction)

- Default sizes above (15×15, K=5, population 8, 10 generations, 20 train / 24 eval episodes) — tuned for minutes-scale runs.
- Claude model default Sonnet-class; configurable.
- Repo location `~/Projects/prometheus`; Python ≥3.11, numpy only for the engine; React + a charting lib + a tree/graph lib for the dashboard.
- If no `ANTHROPIC_API_KEY` is set, the Claude arm is skipped with a clear message.
