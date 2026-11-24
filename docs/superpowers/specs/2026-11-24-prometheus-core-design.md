# PROMETHEUS Core — Design

Date: 2026-11-24
Status: draft, awaiting approval

## 1. Intent

PROMETHEUS asks one question: **can an AI (the Creator) design other AIs that measurably improve over generations — and can the Creator itself be trained to get better at creating?**

**The Creator is a neural network we build and train ourselves with reinforcement learning. It is not Claude or any external/API model.** No API keys, no external model calls anywhere in the system. The full vision (see the original concept) spans world models, RL, evolution, tool creation, uncertainty, scientific discovery and artificial civilizations. This document covers only the **core**: the smallest system that can answer the central question honestly.

The defining rule (from the concept, §19): the system must show that the evolved population is *measurably better* than the original one, and that the Creator's design decisions beat a no-intelligence baseline. More agents or prettier graphs do not count as success.

## 2. Goals

1. A simulated world with **hidden rules** that agents must discover by experimenting, and that **change mid-run** so adaptation can be measured.
2. **Child intelligences** with persistent identity, genome (architecture), memory (learned beliefs), lineage and history.
3. **The Creator** — a trained neural network — analyses performance and failures and designs new children / successors, recording *why* each was created (which features drove the decision) and *what it predicted* would happen.
3a. **The Creator teaches.** Besides designing a child, it plans the child's training: which kinds of worlds it practices in, in what order, and how hard. The parent raises the child, and is rewarded for good teaching as well as good design.
3b. **Creator training** across many runs, with saved, versioned checkpoints, so we can measure whether the parent itself improves (research question 8).
4. **A fixed benchmark** so every generation is scored on identical challenges.
5. **A control arm** (random mutation, no Creator reasoning) under an identical budget, to prove whether the Creator adds value.
6. **A lab dashboard** to inspect everything: genealogy, timeline, entity profiles, generation curves, arm comparison, episode replay.

## 3. Non-goals (for the core)

- ORACLE, GENESIS, ARGUS, artificial civilization (later add-ons).
- Multiple agents interacting in the same world (one child per episode in the core).
- LLM-powered children or Creator. No external AI models or APIs.
- Creator inventing **new abilities/architectures** (concept §3.2). The core Creator tunes designs within a fixed gene set; open-ended architecture invention is the planned next stage.
- Live-streaming the dashboard while a run executes (dashboard replays recorded runs).
- Training deep neural networks.

## 4. Anti-goals (would be failure even if it "works")

- Claiming improvement without the control-arm comparison, or from a single seed.
- The Creator arm winning only because it got more evaluations/compute than the control arm.
- Narrative theater: entity names and "reasons" that don't correspond to real genome changes.
- A "trained" Creator that only looks better because it was evaluated on the same worlds it trained on (Creator evaluation uses held-out world configs and seeds).
- Any claim of consciousness or life.

## 5. Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **A. Python engine + React dashboard replaying recorded runs** | Python is the research standard; runs are reproducible files; dashboard can be hosted statically (GitHub Pages) with no API key exposed. Two codebases. | **Chosen by user.** |
| B. Everything in the browser (TypeScript) | Flashy live demo; weaker for real learning; Creator still needs a server for the key. | Rejected. |
| C. Python + Streamlit | Fastest; generic look; poor for genealogy and replay. | Rejected. |

Child type: small learning programs (numpy). Chosen over LLM children because evolution needs thousands of cheap episodes and real learning, not prompt edits.

Creator type:

| Option | Decision |
|---|---|
| **Neural network trained with RL (PyTorch), checkpoints carried across runs** | **Chosen by user.** The parent is genuinely trained. |
| Claude / LLM API as Creator | Rejected by user: cannot be trained, not our intelligence. |
| Fixed rules only | Kept only as a hand-designed baseline arm. |

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
      neural.py               The Creator: PyTorch policy + prediction head
      features.py             Population/child -> observation tensors
      training.py             RL training loop for the Creator, checkpoints
      heuristic.py            Hand-designed baseline arm
      random_mutation.py      Control arm
    registry.py               Entity registry, lineage, event log
    runner.py                 Generation loop
    cli.py                    `prometheus run`, `train-creator`, `compare`
  checkpoints/                Versioned Creator weights + training history
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

- **Training:** each child runs `train_episodes` (default 20) split into 4 **lessons** of 5 episodes. The child's **curriculum** (set by its Creator, see 6.4) chooses a difficulty preset for each lesson:
  - `easy`: small grid, few hazards, plentiful food, no rule shift
  - `normal`: default world, no rule shift
  - `hard`: dense hazards, scarce food, more walls
  - `shift`: default world with a mid-episode rule shift
  Seeds are derived from run seed + child id + lesson. The total training budget is the same for every child; only its composition and order change.
- **Evaluation:** a **fixed suite** of `eval_worlds` (default 24) seeded worlds: half stable, half with a rule shift. Same suite for every child, generation and arm.
- **Metrics per child:** mean score (goals + survival), survival rate, pre-shift vs post-shift score (**adaptation ratio**), learning speed (episodes to reach X% of final), mean prediction error, energy spent on perception.
- **Fitness:** a single scalar (weighted mean score with adaptation term) used for selection; all component metrics stored.

### 6.4 Creators

Common interface:

```python
class Creator:
    def seed_population(self, n: int, rng) -> list[ChildSpec]
    def next_generation(self, report: GenerationReport, slots: int) -> list[ChildSpec]
```

`ChildSpec` = genome + parent_id (optional) + inherit_beliefs flag + **curriculum** (4 lesson presets) + name + specialization + **reason** + **predicted fitness change**.

#### The Creator (trained neural network) — `neural.py`

**Observation** (per candidate parent, built by `features.py`): normalized genome, fitness and all component metrics, failure profile (share of deaths by cause: hazard / starvation / post-shift), adaptation ratio, learning speed, rank in population, generation index, plus population summary stats (mean/std of each gene and metric, diversity). Fixed-size vectors, so the network works for any population.

**Network:** small MLP encoder (about 2×128) shared across candidates, with five heads:

1. **Parent choice:** a score per candidate; softmax picks which child to build on.
2. **Gene edits:** mean and log-std of a Gaussian per gene (how much to change each gene, in normalized units); sampled, then clamped to genome bounds.
3. **Inheritance:** Bernoulli; pass the parent's learned beliefs on or not.
4. **Teaching (curriculum):** for each of the 4 lessons, a categorical choice over `easy / normal / hard / shift`, conditioned on the chosen parent and the gene edits (so it can teach a cautious child differently from a curious one).
5. **Prediction:** expected fitness change of the child vs its parent (trained by regression; also the RL value baseline).

The decisions are sampled in order (parent, then genes, inheritance, curriculum), so the Creator learns how design and teaching work together.

**Reward:** the child's benchmark fitness minus the parent's, normalized per run. Small penalty for producing near-duplicates of existing children (keeps diversity).

**Algorithm:** policy gradient with a learned baseline (PPO-style clipped updates), batched across all slots in a generation and across many parallel evolution runs.

**Reasons:** each child's `reason` is generated from what the network actually did (the largest gene edits, the parent's dominant failure mode, and the prediction), e.g. *"Parent died to hazards after the rule shift in 60% of worlds: surprise_threshold -0.8, risk_aversion +0.5; curriculum easy → normal → shift → shift; predicted +12% fitness."* No free text that isn't grounded in real decisions.

#### Training the Creator — `training.py`

- `prometheus train-creator --iterations N` runs many short evolutions (smaller train/eval budgets for speed) on **varied world configurations** (grid size, number of tile kinds, shift timing and probability, energy economics) so the Creator learns general design skill, not one world.
- After each iteration the policy updates; every M iterations a **versioned checkpoint** is saved (`checkpoints/creator_v001.pt`, ...) with its training history (reward curve, prediction error curve).
- Training continues from the latest checkpoint, so the Creator keeps accumulating experience across sessions.
- **Held-out evaluation:** Creator quality is always measured on world configs and seeds never seen in training.
- Runs on CPU; uses the GPU (RTX 3050) when available. Child simulation is the bottleneck, so the world and learner are written for speed (numpy, precomputed paths, multiprocessing across runs).

#### Baseline arms

- **Untrained Creator:** same network, random initial weights (checkpoint v000). The "fresh parent".
- **Trained Creator, teaching off:** the trained network designs children but every child gets the fixed default curriculum (`normal, normal, shift, normal`). Isolates **how much the teaching itself adds**.
- **HeuristicCreator:** hand-written rules (low adaptation: raise surprise sensitivity, more `shift` lessons; dying to hazards: raise risk aversion, start with `easy`; ...). Shows whether the trained Creator beats a human designer's rules.
- **RandomMutationCreator (control):** top-k parents, random Gaussian mutations, random curriculum. No intelligence.

**Selection** (shared, not Creator-controlled, for fairness): keep top `elite` (default 2) unchanged; the Creator fills the remaining slots. Population size fixed (default 8). Retired entities keep their full records.

**Fairness rule:** every arm gets the same population size, generations, training episodes and eval suite. Only the design decisions differ.

### 6.5 Runner and CLI

- `prometheus run --creator {neural[:vNNN],heuristic,random} --generations 10 --population 8 --seed 0 --out runs/<name>`
- `prometheus train-creator --iterations N [--resume]`
- `prometheus compare --creators neural:latest,neural:latest:noteach,neural:v000,heuristic,random --seeds 0,1,2 --generations 10`: all arms × seeds on held-out worlds; writes mean ± std fitness per generation per arm.
- `prometheus creator-progress`: evaluates a series of checkpoints (v000 ... latest) on the same held-out benchmark. **Does the parent get better as it trains?**
- Human oversight: config caps on max children per run and training compute; training is always started explicitly by the user.

### 6.6 Run output format (contract between engine and dashboard)

```
runs/<run_id>/
  run.json          config, creator + checkpoint version, seed, timing, final summary
  entities.json     every child ever created (identity record, genome, metrics, lineage, predicted vs actual change)
  events.jsonl      timeline: created, trained, evaluated, retired, rule_shift_detected, creator_decision
  generations.json  per-generation stats (best/mean/median fitness, diversity, Creator prediction error)
  replays/<entity_id>_<world>.json   step-by-step traces for the best child per generation (one stable, one shift world)
comparisons/<name>.json       arm × seed × generation curves
creator/progress.json         checkpoint version × held-out performance + prediction accuracy
creator/training_history.json reward and loss curves over training
```

### 6.7 Dashboard

Vite + React + TypeScript, static. Loads an index of runs from `public/runs/`.

- **Overview:** run summary, population status, headline result ("Gen 10 best vs Gen 1 best: +X%; trained Creator vs control: ...").
- **Creator:** training curves, checkpoint-progress chart (fresh to experienced), prediction accuracy over time, decision breakdown (which genes it tends to edit for which failure types), and teaching patterns (which curricula it chooses for which kinds of child, and how that changed during training).
- **Genealogy:** tree of all entities (colored by fitness, retired greyed), click to open profile.
- **Entity profile:** identity, parent, reason for creation, curriculum it was taught (with score after each lesson), predicted vs actual improvement, genome (diff vs parent highlighted), metrics, belief table, events.
- **Timeline:** filterable event stream including Creator decisions.
- **Evolution charts:** fitness per generation, metric breakdown.
- **Comparison:** arms side-by-side with error bands across seeds.
- **Replay:** canvas grid playback of an episode, showing agent, visible area, beliefs, predicted vs actual effect, and the moment of rule shift.

Deployable to GitHub Pages with bundled sample runs.

## 7. Testing

- pytest: world determinism and rule shift; genome validation/clamping; learner belief updates and surprise reset; benchmark reproducibility (same seed, same scores); every Creator returns valid specs; curriculum presets generate the intended worlds and the training budget is identical across curricula; Creator network shapes, sampling within bounds, a policy-gradient step improves a toy problem, checkpoint save/load round-trip; runner end-to-end on a tiny config; output schema.
- Dashboard: type-checked against a shared schema; smoke build.

## 8. Success criteria for the core

1. `train-creator` runs on the laptop and produces checkpoints with training curves.
2. `compare` and `creator-progress` run over at least 3 seeds on held-out worlds.
3. Dashboard shows all views for recorded runs and Creator training.
4. The report states honestly: do later generations beat earlier ones, does the trained Creator beat random and heuristic, does its teaching add value over the default curriculum, and does an experienced Creator beat a fresh one, including if any answer is no.

## 9. Builder's calls (open to correction)

- Default sizes (15×15, K=5, population 8, 10 generations, 20 train / 24 eval episodes); smaller budgets during Creator training for speed.
- Creator network size (about 2×128 MLP) and PPO-style training; may be tuned once training curves are visible.
- Repo location `~/Projects/prometheus`; Python 3.11+, numpy for the world and children, PyTorch for the Creator; React + a charting lib + a tree/graph lib for the dashboard.
- Main risk: simulation speed limits how much the Creator can train. Mitigation: fast world code, multiprocessing, small training budgets; measured early in the build.
