# PROMETHEUS Engine Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the simulation engine: hidden-rule world with rule shifts, child intelligences that learn, curriculum training, fixed benchmark, entity registry, the heuristic and random Creators, the generation runner with run output, and `prometheus run` / `prometheus compare`.

**Architecture:** Pure-Python package `prometheus` under `engine/`, numpy only. Children are explicit world-model learners plus a bounded planner. The runner evolves a population generation by generation through a `Creator` interface; the trained neural Creator (Plan 2) plugs into the same interface. Every run writes a self-contained directory that the dashboard (Plan 3) reads.

**Tech Stack:** Python 3.13 (3.11+ supported), numpy, pytest, argparse, multiprocessing.

**Spec:** `docs/superpowers/specs/2026-11-24-prometheus-core-design.md`

**Later plans:** Plan 2 builds the trained neural Creator (`creators/neural.py`, `features.py`, `training.py`, `train-creator`, `creator-progress`). Plan 3 builds the dashboard. Both are written once this engine exists and its simulation speed is measured (Task 4).

## Global Constraints

- No external AI models or APIs anywhere. The Creator is ours (Plan 2); Plan 1 has heuristic and random Creators only.
- Engine dependencies: numpy only (pytest for development). PyTorch arrives in Plan 2.
- Everything is deterministic given a seed: same config and seed give byte-identical run output (except timestamps and timing fields).
- Defaults from the spec: grid 15×15, K=5 tile kinds, population 8, elite 2, 10 generations, 4 lessons × 5 episodes = 20 training episodes, 24 eval worlds (12 stable, 12 shift), max 200 steps per episode.
- Curriculum presets are exactly `easy`, `normal`, `hard`, `shift`. Default curriculum: `("normal", "normal", "shift", "normal")`.
- Fairness: every arm gets identical population, generations, training budget and eval suite.
- Commits: small and frequent, one per task step that says commit. No AI co-author or attribution trailers in commit messages.
- Commands below run from `engine/` using the venv: `.venv/Scripts/python` on Windows (written as `python` below once the venv is active).

## Review Focus

1. **A child with no reachable target** (boxed in by walls or believed hazards) must keep acting (stay or random move), never crash or loop forever. Test in Task 3.
2. **Observation at grid edges:** cells outside the grid read as walls, and the window is always `(2r+1)×(2r+1)`. Test in Task 1.
3. **Out-of-range genomes from a Creator** (floats for int genes, values past bounds, NaN) are clamped and rounded, never propagated. Test in Task 2.
4. **Degenerate scores:** zero pre-shift score (adaptation ratio division), all-equal fitness in selection, a population that all dies at step 1. These must give finite numbers. Tests in Tasks 4 and 7.
5. **Determinism under multiprocessing:** a run with `workers=1` and `workers=4` produces identical entities and fitness. Test in Task 7.

---

### Task 1: Project scaffold and the World

**Files:**
- Create: `engine/pyproject.toml`, `engine/prometheus/__init__.py`, `engine/prometheus/world.py`, `engine/tests/test_world.py`, `.gitignore`

**Interfaces:**
- Produces (in `prometheus.world`):
  - Constants: `FLOOR = -1`, `WALL = -2`, `START_ENERGY = 30.0`, `STEP_COST = 1.0`, `PERCEPTION_COST = 0.1` (per unit of radius per step), `FOOD_VALUE = 8.0`, `HAZARD_VALUE = -12.0`, `GOAL_ENERGY = 5.0`, `NEUTRAL_VALUE = 0.0`, `MAX_STEPS = 200`, `SHIFT_STEP = 100`, `ACTIONS = ((-1,0),(0,1),(1,0),(0,-1),(0,0))` (N, E, S, W, stay).
  - `ROLES = ("goal", "food", "food", "hazard", "neutral")`: role multiset for K=5.
  - `@dataclass(frozen=True) WorldConfig(size=15, n_kinds=5, wall_density=0.08, n_food=14, n_hazard=10, n_goal=2, n_neutral=10, shift=False, shift_step=SHIFT_STEP, max_steps=MAX_STEPS)`
  - `PRESETS: dict[str, WorldConfig]`: `easy=WorldConfig(size=9, wall_density=0.05, n_food=8, n_hazard=3, n_goal=1, n_neutral=4)`, `normal=WorldConfig()`, `hard=WorldConfig(wall_density=0.15, n_food=8, n_hazard=20)`, `shift=WorldConfig(shift=True)`.
  - `@dataclass RuleSet(roles: list[str])` with `effect(kind) -> float` (energy delta) and `is_goal(kind) -> bool`. `RuleSet.random(n_kinds, rng)` shuffles `ROLES`; `shifted(rng) -> RuleSet` returns a permutation in which **no food kind keeps the food role** (assert-tested).
  - `@dataclass StepResult(kind: int | None, energy_delta: float, reached_goal: bool, shifted: bool, done: bool)`
  - `class World(config: WorldConfig, seed: int)` with attributes `grid: np.ndarray[int8]`, `pos: tuple[int,int]`, `energy: float`, `t: int`, `done: bool`, `rules: RuleSet`, `shift_happened: bool`, `last_kind: int | None`; methods `observe(radius: int) -> np.ndarray` (window centered on agent, out-of-bounds = WALL) and `step(action: int, perception_radius: int) -> StepResult`.

World rules (the implementer cannot infer these):
- Grid starts FLOOR; walls placed at `wall_density`; then tiles placed on random empty cells: `n_food` tiles split randomly between the two food-role kinds, `n_hazard` of the hazard kind, `n_goal` of the goal kind, `n_neutral` of the neutral kind. The agent starts on a random empty floor cell.
- `step`: moving into WALL or off-grid leaves position unchanged. Energy -= `STEP_COST + PERCEPTION_COST * perception_radius`. If the new cell holds a kind, apply its effect: food → `+FOOD_VALUE`, tile consumed (becomes FLOOR) and a tile of the same kind respawns on a random empty floor cell; goal → `+GOAL_ENERGY`, `reached_goal=True`, consumed and respawned; hazard → `HAZARD_VALUE`, tile stays; neutral → 0, tile stays.
- Shift: if `config.shift` and `t == shift_step` (checked at the start of the step), `rules = rules.shifted(rng)`, `shift_happened=True`, and this step's `StepResult.shifted=True`.
- `done` when energy ≤ 0 or `t >= max_steps`.
- All randomness from `np.random.default_rng(seed)`.

- [ ] **Step 1: Scaffold.** `pyproject.toml` (name `prometheus`, requires-python `>=3.11`, deps `numpy>=1.26`, optional `dev = ["pytest>=8"]`, console script `prometheus = "prometheus.cli:main"`). `.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `runs/`, `comparisons/`, `node_modules/`, `dist/`. Create venv: `python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"`. Expected: install succeeds.

- [ ] **Step 2: Write failing tests** in `tests/test_world.py`:

```python
def test_same_seed_same_world():  # grid, pos, rules identical for seed 3; differ for seed 4
def test_tile_counts_match_config():  # count of each role's tiles == config numbers (food summed over both food kinds)
def test_wall_blocks_movement():  # place WALL north of agent, step(0) -> pos unchanged
def test_food_consumed_and_respawned():  # step onto food: energy delta == FOOD_VALUE - STEP_COST - PERCEPTION_COST*r ; total food tile count unchanged
def test_hazard_persists():  # step onto hazard: delta includes HAZARD_VALUE; tile still there
def test_goal_sets_flag():  # reached_goal True
def test_shift_at_step_100():  # PRESETS["shift"], step 100 times: result.shifted True exactly once, at t==100; rules changed
def test_shift_changes_every_food_kind():  # for 50 seeds: no kind with role food before keeps role food after
def test_observe_shape_and_edges():  # agent at (0,0), observe(2) shape (5,5), top-left 2 rows/cols are WALL
def test_episode_ends_on_energy_or_max_steps():  # stay forever -> done by starvation before 200; energy <= 0
```

- [ ] **Step 3: Run** `python -m pytest tests/test_world.py -v`. Expected: FAIL (module missing).
- [ ] **Step 4: Implement** `prometheus/world.py` per Interfaces. Tests place tiles by writing `world.grid[...]` directly, so keep `grid` a plain mutable array.
- [ ] **Step 5: Run** the tests. Expected: all PASS.
- [ ] **Step 6: Commit** scaffold (`git commit -m "Scaffold engine package"`), then the world (`git commit -m "Add hidden-rule world with rule shift"`).

---

### Task 2: Genome

**Files:**
- Create: `engine/prometheus/genome.py`, `engine/tests/test_genome.py`

**Interfaces:**
- Produces:
  - `GENE_BOUNDS: dict[str, tuple[float, float]]` in this exact order: `perception_radius (1,5)`, `memory_capacity (0,50)`, `learning_rate (0.01,1.0)`, `exploration (0.0,1.0)`, `exploration_decay (0.9,1.0)`, `planning_horizon (1,12)`, `risk_aversion (0.0,3.0)`, `surprise_threshold (0.1,5.0)`, `curiosity_bonus (0.0,2.0)`.
  - `INT_GENES = {"perception_radius", "memory_capacity", "planning_horizon"}`; `GENE_NAMES = tuple(GENE_BOUNDS)`.
  - `SURPRISE_OFF = 5.0`: a threshold at the upper bound means adaptation reset is disabled.
  - `@dataclass(frozen=True) Genome` with those nine fields; `Genome.default()` (radius 2, memory 10, lr 0.3, exploration 0.3, decay 0.97, horizon 6, risk 1.0, surprise 1.5, curiosity 0.5); `Genome.random(rng)`; `clamped() -> Genome` (clip to bounds, round int genes, NaN → default value); `to_vector() -> np.ndarray` (normalized 0..1, order `GENE_NAMES`); `Genome.from_vector(v) -> Genome` (inverse, clamped); `to_dict()/from_dict()`.

- [ ] **Step 1: Failing tests:**

```python
def test_default_is_within_bounds()
def test_clamp_rounds_ints_and_clips():  # Genome(perception_radius=7.6, learning_rate=-1, ...).clamped() -> radius 5 (int), lr 0.01
def test_clamp_replaces_nan_with_default():  # learning_rate=nan -> 0.3
def test_vector_roundtrip():  # g == Genome.from_vector(g.to_vector()) for 20 random genomes
def test_random_is_seeded():  # Genome.random(default_rng(1)) equal twice
```

- [ ] **Step 2: Run** `python -m pytest tests/test_genome.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `genome.py`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `"Add child genome with bounds and normalization"`.

---

### Task 3: Child agent (learner, planner, episodes)

**Files:**
- Create: `engine/prometheus/child.py`, `engine/tests/test_child.py`

**Interfaces:**
- Consumes: `World`, `StepResult`, constants (Task 1); `Genome`, `SURPRISE_OFF` (Task 2).
- Produces:
  - `GOAL_VALUE = 20.0`, `ENERGY_SCALE = 10.0`.
  - `@dataclass Beliefs(mean: np.ndarray, count: np.ndarray)` with `Beliefs.empty(n_kinds)`, `copy()`, `to_dict()/from_dict()`.
  - `class ChildAgent(genome: Genome, n_kinds: int, rng: np.random.Generator, beliefs: Beliefs | None = None)` with attributes `beliefs`, `epsilon`; methods `start_episode()`, `act(world: World) -> int`, `observe_outcome(result: StepResult) -> float` (returns surprise), `end_episode()`.
  - `@dataclass EpisodeResult(score, goals, steps, survived: bool, death_cause: str | None, died_after_shift: bool, pre_shift_rate: float | None, post_shift_rate: float | None, mean_prediction_error: float, perception_cost: float, trace: list[dict] | None)`
  - `run_episode(agent: ChildAgent, world: World, record: bool = False) -> EpisodeResult`

Behaviour (not determined by tests alone):
- **Value of an outcome** = `energy_delta_from_tile + (GOAL_VALUE if reached_goal else 0)`, excluding step and perception cost. Only steps that land on a kind update beliefs.
- **Prediction** before each move: `beliefs.mean[kind]` for the target cell's kind (0 for floor). Prediction error = |value − predicted| for kinds with count > 0.
- **Update:** `alpha = max(lr, 1/(count+1))`; `mean += alpha*(value-mean)`; `count = min(count+1, 5*memory_capacity)` (if memory_capacity is 0, `start_episode` resets beliefs to empty).
- **Surprise** = error / ENERGY_SCALE. If `surprise_threshold < SURPRISE_OFF` and surprise > threshold: `count *= 0.25` for all kinds and `mean[kind] = value` (partial reset: adaptation).
- **Planner** (`act`): with probability `epsilon` take a uniformly random action. Otherwise BFS from the agent within `observe(perception_radius)` up to `planning_horizon` steps, never entering walls and never passing *through* a cell whose believed value is negative (it may still target nothing). Cell value: known kind → `mean` × `(1 + risk_aversion)` if negative, else `mean`; unknown kind (count 0) → `curiosity_bonus * ENERGY_SCALE`; floor → 0. Pick the reachable cell maximizing `value − 0.5*distance` with value > 0; move along the first step of its path. If none, pick a random non-wall neighbour, and stay if there is none.
- `epsilon` starts at `genome.exploration` at construction; `end_episode()` multiplies it by `exploration_decay`.
- **Episode score** = `goals + 0.5 * steps/max_steps`. `death_cause`: `"hazard"` if the final step's tile effect was negative, else `"starvation"`; None if survived. `pre/post_shift_rate` = goals per step before/after the shift (None in stable worlds). `perception_cost` = total energy spent on perception.
- `trace` (when `record`): per step `{"t", "pos", "action", "kind", "predicted", "value", "energy", "surprise", "shifted"}`, preceded by an initial entry `{"grid": world.grid.tolist(), "roles": rules.roles, "radius": r}`.

- [ ] **Step 1: Failing tests:**

```python
def test_beliefs_learn_food_value():  # repeatedly feed observe_outcome(StepResult(kind=1, energy_delta=8, ...)) -> mean[1] approx 8
def test_surprise_resets_counts():  # after learning kind 1 ≈ +8, observe -12 with threshold 0.5 -> counts scaled by 0.25, mean[1] == -12
def test_surprise_off_never_resets():  # threshold SURPRISE_OFF -> counts not scaled
def test_memory_zero_forgets_between_episodes():
def test_planner_moves_toward_known_food():  # tiny 5x5 grid via world.grid edits, food 2 cells east, epsilon 0 -> action E
def test_planner_avoids_believed_hazard_path():  # hazard directly east between agent and food; path goes around
def test_boxed_in_agent_stays():  # walls on all 4 sides, epsilon 0 -> act returns 4 and run_episode terminates
def test_run_episode_deterministic():  # same seeds -> identical EpisodeResult
def test_episode_result_fields_finite():  # all-dying population at step 1 (energy 0.5 start via world.energy) -> finite fields, death_cause set
def test_trace_recorded():  # record=True -> trace[0] has grid; len(trace) == steps + 1
```

- [ ] **Step 2: Run** `python -m pytest tests/test_child.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `child.py`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `"Add learning child agent with planner"`.

---

### Task 4: Benchmark: curriculum training, eval suite, metrics, speed check

**Files:**
- Create: `engine/prometheus/benchmark.py`, `engine/tests/test_benchmark.py`, `engine/scripts/measure_speed.py`, `docs/engine-speed.md`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `CURRICULUM_PRESETS = ("easy", "normal", "hard", "shift")`, `LESSONS = 4`, `EPISODES_PER_LESSON = 5`, `DEFAULT_CURRICULUM = ("normal", "normal", "shift", "normal")`, `EVAL_SEED = 10_000`.
  - `@dataclass(frozen=True) EvalSuite(n_worlds: int = 24, seed: int = EVAL_SEED)` with `worlds() -> list[tuple[str, int]]`: `(preset, seed)` pairs, first half `"normal"`, second half `"shift"`.
  - `train_child(agent: ChildAgent, curriculum: Sequence[str], seed: int, episodes_per_lesson: int = EPISODES_PER_LESSON) -> list[float]` returns mean score per lesson (and leaves the agent trained). Invalid preset names raise `ValueError`.
  - `@dataclass ChildMetrics(fitness, mean_score, stable_score, shift_score, survival_rate, adaptation_ratio, learning_speed, mean_prediction_error, perception_cost, death_causes: dict[str, float], lesson_scores: list[float])` with `to_dict()`.
  - `evaluate(agent: ChildAgent, suite: EvalSuite, lesson_scores: list[float]) -> ChildMetrics`: each eval world runs on a **deep copy** of the trained agent (beliefs and rng copied, epsilon kept), so the order of worlds never matters.
  - `fitness = mean_score * (0.75 + 0.25 * min(adaptation_ratio, 1.0))`.
  - `adaptation_ratio = mean(post_shift_rate) / mean(pre_shift_rate)` over shift worlds, clipped to `[0, 2]`; if the pre-shift mean is 0, the ratio is `1.0` when post is also 0, else `2.0`.
  - `learning_speed` = (index of first lesson whose score ≥ 90% of the best lesson score + 1) / LESSONS; lower is faster.
  - `death_causes` keys exactly `"hazard"`, `"starvation"`, `"post_shift"`: fraction of eval episodes.
  - `train_and_evaluate(genome: Genome, beliefs: Beliefs | None, curriculum, seed: int, suite: EvalSuite, n_kinds: int = 5) -> tuple[ChildMetrics, Beliefs]`: pure function used by the runner's worker pool.

- [ ] **Step 1: Failing tests:**

```python
def test_eval_suite_is_fixed():  # EvalSuite().worlds() identical twice; 12 normal + 12 shift
def test_training_budget_identical_across_curricula():  # count episodes run for each curriculum == 20 (patch run_episode with a counter)
def test_invalid_preset_raises()
def test_eval_order_independent():  # evaluate on suite vs reversed suite -> same metrics
def test_adaptation_ratio_zero_pre_shift():  # pre 0 & post 0 -> 1.0 ; pre 0 & post > 0 -> 2.0
def test_trained_child_beats_untrained():  # default genome, 20 training episodes, fitness > same genome with 0 training (seed 0)
def test_train_and_evaluate_deterministic()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_benchmark.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `benchmark.py`.
- [ ] **Step 4: Run.** Expected: PASS. If `test_trained_child_beats_untrained` fails, the learner is broken: fix the learner, never the test.
- [ ] **Step 5: Speed check.** `scripts/measure_speed.py` times `train_and_evaluate` for the default genome (5 repeats) and prints seconds per child and steps per second. Run it and write the numbers plus the machine into `docs/engine-speed.md`, with the projected time for one default run (8 + 9×6 = 62 children). Expected: prints numbers. Plan 2's training budget is sized from this.
- [ ] **Step 6: Commit** `"Add curriculum training and fixed benchmark"`, then `"Record engine speed baseline"`.

---

### Task 5: Registry, events and naming

**Files:**
- Create: `engine/prometheus/registry.py`, `engine/tests/test_registry.py`

**Interfaces:**
- Consumes: `Genome` (Task 2), `ChildMetrics`, `Beliefs`.
- Produces:
  - `@dataclass EntityRecord(id: str, name: str, parent_id: str | None, generation: int, created_at: str, specialization: str, reason: str, genome: dict, curriculum: list[str], inherit_beliefs: bool, predicted_change: float | None, status: str = "active", metrics: dict | None = None, beliefs: dict | None = None, actual_change: float | None = None, retired_at_generation: int | None = None)` with `to_dict()`.
  - `specialization_for(genome: Genome) -> str`: the gene furthest above its default (normalized) decides: perception_radius → `"SCOUT"`, risk_aversion → `"SENTINEL"`, surprise_threshold *below* default → `"ADAPTER"`, curiosity_bonus or exploration → `"EXPLORER"`, planning_horizon → `"NAVIGATOR"`, learning_rate → `"LEARNER"`; if no deviation exceeds 0.1 → `"GENERALIST"`.
  - `class Registry` with `create(spec: ChildSpec, generation: int) -> EntityRecord` (ids `E0001`, `E0002`, …; names `<SPECIALIZATION>-<NNN>` with a per-prefix counter; a non-empty `spec.specialization` overrides `specialization_for`), `get(id)`, `active() -> list[EntityRecord]`, `retire(id, generation)`, `all() -> list[EntityRecord]`.
  - `class EventLog` with `emit(type: str, generation: int, entity_id: str | None = None, **data)` and `events: list[dict]`. Each event has `{"seq", "type", "generation", "entity_id", "data"}`. Types used: `created`, `trained`, `evaluated`, `retired`, `rule_shift_detected`, `creator_decision`, `generation_complete`.
  - `created_at` comes from an injectable clock `Registry(clock=lambda: "...")` so tests and determinism checks can fix it.
- Note: `ChildSpec` is defined in Task 6's `creators/base.py`; to avoid a cycle, `registry.py` imports it only under `TYPE_CHECKING`.

- [ ] **Step 1: Failing tests:** ids and names increment per prefix; the specialization override works; `retire` sets status and generation and drops the record from `active()`; events get sequential `seq`; `specialization_for(Genome.default()) == "GENERALIST"`; a high-radius genome gives `"SCOUT"`.
- [ ] **Step 2: Run** `python -m pytest tests/test_registry.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `"Add entity registry and event log"`.

---

### Task 6: Creator interface, random and heuristic Creators

**Files:**
- Create: `engine/prometheus/creators/__init__.py`, `creators/base.py`, `creators/random_mutation.py`, `creators/heuristic.py`, `engine/tests/test_creators.py`

**Interfaces:**
- Consumes: `Genome`, `CURRICULUM_PRESETS`, `DEFAULT_CURRICULUM`, `EntityRecord`.
- Produces (in `creators/base.py`):
  - `@dataclass ChildSpec(genome: Genome, parent_id: str | None, inherit_beliefs: bool, curriculum: tuple[str, ...], reason: str, specialization: str = "", predicted_change: float | None = None)`; `validated() -> ChildSpec` clamps the genome and replaces invalid curriculum entries or a wrong length with `DEFAULT_CURRICULUM`.
  - `@dataclass GenerationReport(generation: int, population: list[EntityRecord], history: list[dict])`, where history holds per-generation stats from earlier generations.
  - `class Creator(Protocol)`: `name: str`; `seed_population(n: int, rng) -> list[ChildSpec]`; `next_generation(report: GenerationReport, slots: int, rng) -> list[ChildSpec]` (must return exactly `slots` specs).
  - `get_creator(name: str) -> Creator` registry in `creators/__init__.py` with `"random"` and `"heuristic"`; Plan 2 adds `"neural"`.
- `RandomMutationCreator`: seed population = random genomes with random curricula, reason `"Random initial design"`. Next generation: parent uniform from the top half by fitness (ties broken by id), each normalized gene + N(0, 0.1), `inherit_beliefs` 50%, random curriculum, reason `"Random mutation of <parent name>"`.
- `HeuristicCreator`: seed population = default genome plus small random variation, `DEFAULT_CURRICULUM`. Next generation: parents taken round-robin from the top half; diagnose the parent's dominant weakness from metrics, in this order:
  1. `death_causes["hazard"] > 0.3` → risk_aversion +0.5, perception_radius +1; curriculum `("easy","hard","normal","shift")`
  2. `adaptation_ratio < 0.6` → surprise_threshold −0.5, learning_rate +0.1; curriculum `("normal","shift","shift","shift")`
  3. `death_causes["starvation"] > 0.3` → exploration +0.1, curiosity +0.3, perception_radius −1 if > 2; curriculum `("easy","normal","normal","shift")`
  4. `learning_speed > 0.5` → learning_rate +0.15, memory_capacity +5; curriculum `("easy","easy","normal","shift")`
  5. otherwise → planning_horizon +1, plus a small random nudge; default curriculum

  Always inherit beliefs. The reason names the diagnosis and the edits, e.g. `"SENTINEL-002 died to hazards in 45% of worlds: risk_aversion +0.5, perception_radius +1; curriculum easy→hard→normal→shift"`. `predicted_change = None`, since only the neural Creator predicts.

- [ ] **Step 1: Failing tests:**

```python
def test_both_creators_return_exact_slot_count()
def test_specs_are_valid():  # all genomes clamped-equal; curriculum length 4, names in presets
def test_validated_fixes_bad_curriculum():  # ("easy","bogus") -> DEFAULT_CURRICULUM
def test_heuristic_targets_hazard_deaths():  # parent with hazard 0.5 -> child risk_aversion > parent's, reason mentions "hazard"
def test_heuristic_targets_poor_adaptation()
def test_random_creator_seeded():  # same rng seed -> identical specs
def test_all_equal_fitness_no_crash():  # population with identical fitness -> slots specs
```

- [ ] **Step 2: Run** `python -m pytest tests/test_creators.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `"Add Creator interface and random mutation control"`, then `"Add heuristic Creator baseline"`.

---

### Task 7: Generation runner and run output

**Files:**
- Create: `engine/prometheus/runner.py`, `engine/prometheus/output.py`, `engine/tests/test_runner.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `@dataclass RunConfig(creator: str = "heuristic", generations: int = 10, population: int = 8, elite: int = 2, seed: int = 0, episodes_per_lesson: int = 5, eval_worlds: int = 24, workers: int = 0, record_replays: bool = True)`; `workers=0` means `os.cpu_count()`, and 1 means in-process.
  - `@dataclass RunResult(config, registry: Registry, events: EventLog, generations: list[dict], replays: dict[str, list[dict]])`
  - `run_evolution(config: RunConfig, creator: Creator | None = None, clock=None) -> RunResult` (the creator is resolved from `config.creator` when None; injectable for Plan 2).
  - `write_run(result: RunResult, out_dir: Path) -> None` (in `output.py`) writes exactly the layout from spec §6.6: `run.json`, `entities.json`, `events.jsonl`, `generations.json`, `replays/<entity_id>_<preset>_<seed>.json`.

Generation loop:
1. Gen 0: `creator.seed_population(population, rng)` → registry `create` → evaluate all.
2. Gen g ≥ 1: sort the active population by fitness (desc, ties by id). Keep the top `elite` as they are (no retraining). Retire the rest (`retired` events). Call `creator.next_generation(report, population - elite, rng)`, validate each spec, create the records, evaluate.
3. **Evaluate** = `train_and_evaluate(genome, parent beliefs if inherit_beliefs else None, curriculum, child_seed, suite)` run through a `multiprocessing.Pool` (`spawn`-safe top-level function). `child_seed = hash of (run seed, entity id)` via `np.random.SeedSequence([seed, int(id[1:])])`. Store metrics, the beliefs snapshot, and `actual_change = fitness - parent fitness`. Emit `trained` (with lesson scores) and `evaluated`. Emit `creator_decision` for each spec, with its reason and prediction.
4. After each generation, append to `generations`: `{generation, best, mean, median, diversity}`, where diversity is the mean pairwise L2 distance of normalized genomes. Emit `generation_complete`.
5. If `record_replays`: for the generation's best child, re-run eval world 0 (stable) and world 12 (shift) with `record=True` on copies, and store their traces. Emit `rule_shift_detected` for any step in a shift-world replay whose surprise exceeds the child's threshold after the shift.
6. The Creator rng is `np.random.default_rng([seed, 1])`; registry and world rngs are separate seeds, so arms with the same seed get the same eval suite and seed-population randomness streams.

- [ ] **Step 1: Failing tests** (tiny config: generations=3, population=4, elite=1, episodes_per_lesson=1, eval_worlds=4):

```python
def test_population_size_constant():  # active count == 4 after each generation
def test_elites_survive_unchanged():  # top-1 id of gen g is active in gen g+1 with same metrics
def test_lineage_links():  # every non-gen-0 entity has a parent_id that exists in the registry
def test_actual_change_recorded()
def test_deterministic_across_worker_counts():  # workers=1 vs workers=4, fixed clock -> identical entities.json content
def test_write_run_layout(tmp_path):  # files exist; events.jsonl lines parse; replays present; entities json loads
def test_all_dying_population_finite():  # monkeypatch START_ENERGY tiny -> generations stats finite, no exception
```

- [ ] **Step 2: Run** `python -m pytest tests/test_runner.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `runner.py` and `output.py`.
- [ ] **Step 4: Run** the whole suite: `python -m pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `"Add generation runner"`, then `"Write run output for the dashboard"`.

---

### Task 8: Compare, CLI and README

**Files:**
- Create: `engine/prometheus/compare.py`, `engine/prometheus/cli.py`, `engine/tests/test_cli.py`, `README.md`

**Interfaces:**
- Consumes: `RunConfig`, `run_evolution`, `write_run`, `get_creator`.
- Produces:
  - `compare(creators: list[str], seeds: list[int], base: RunConfig, out_dir: Path, name: str) -> dict` runs each creator × seed (writing each run to `out_dir/runs/<name>_<creator>_s<seed>/`) and writes `out_dir/comparisons/<name>.json`: `{"name", "creators", "seeds", "generations", "arms": {creator: {"best_mean": [...], "best_std": [...], "mean_mean": [...], "mean_std": [...], "runs": [run dir names]}}}`, with one value per generation.
  - CLI `prometheus run --creator --generations --population --elite --seed --episodes-per-lesson --eval-worlds --workers --out` (default out `runs/<creator>_s<seed>`) prints a per-generation best/mean table and the final line `Gen 0 best X -> Gen N best Y (+Z%)`.
  - CLI `prometheus compare --creators heuristic,random --seeds 0,1,2 [same sizing flags] --out . --name <name>` prints a final table: arm, final best mean ± std, improvement over gen 0.
  - Unknown creator names exit with code 2 and list the valid names.

- [ ] **Step 1: Failing tests:** the `compare` JSON has one value per generation per arm and std 0 for a single seed; the CLI `run` with tiny flags exits 0 and creates `run.json`; an unknown creator exits 2 (use `subprocess.run([sys.executable, "-m", "prometheus.cli", ...])`, so `cli.py` needs an `if __name__ == "__main__"` guard).
- [ ] **Step 2: Run** `python -m pytest tests/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `compare.py` and `cli.py`.
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Real run.** `prometheus compare --creators heuristic,random --seeds 0,1,2 --name baseline`. Record the printed table and the wall time in `docs/engine-speed.md`. Report honestly whether the heuristic beat random and whether the best fitness rose over generations. Expected: completes, table printed.
- [ ] **Step 6: README.md** at the repo root: what PROMETHEUS is (two paragraphs from spec §1), that the Creator is a network trained in-house (Plan 2, coming), quickstart (venv, install, `prometheus run`, `prometheus compare`), the run output layout, and the baseline results table from Step 5.
- [ ] **Step 7: Commit** `"Add compare command and CLI"`, then `"Add README with baseline results"`, then push: `git push`.
