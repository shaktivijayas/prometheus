# PROMETHEUS Trained Creator Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Creator as a neural network that we train with reinforcement learning to design, endow and teach children. Save it as versioned checkpoints, and measure on held-out universes whether it beats random mutation, whether its teaching adds value, and whether an experienced Creator beats a fresh one.

**Architecture:** A PyTorch policy network (`creators/neural.py`) reads per-candidate and population features (`creators/features.py`). It samples a parent, gene edits, an inheritance flag and a 4-lesson curriculum, and predicts the child's fitness change. Training (`creators/training.py`) runs batches of short evolutions in lockstep across all cores, on varied training universes and world variants, and updates the policy with PPO. The prediction head doubles as the value baseline. The engine's runner is refactored into a steppable `Evolution` so many runs can share one worker pool.

**Tech Stack:** Python 3.11+, numpy, PyTorch (CPU build; the network is small and child simulation is the bottleneck), pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-prometheus-core-design.md` (section 6.4 onward). It builds on Plan 1: `docs/superpowers/plans/2026-11-24-prometheus-engine.md` and `docs/engine-speed.md`.

## Global Constraints

- **Lessons from the engine review (binding):** a single upbringing's score is mostly luck (std 1.7 on mean 3.6), and the in-run "best" curve is selection-biased (a clone arm that changes nothing shows +106%). So: (a) children in training are scored as the average of `TrainConfig.replicates = 2` upbringings; (b) every Creator is judged on **verified** results only: the gen-0 best and the final champion re-scored on fresh upbringings (`RunResult.verification`). `creator-progress` and the final report use verified numbers, and the `clone` arm is included in the final comparison as the null baseline.

- The Creator is our own network, trained by us. No external AI models or APIs.
- **Fair comparisons:** in a given seed, the neural arm's generation 0 is identical to the random arm's (same `seed_population` draws), so their curves differ only by Creator decisions. All arms share population, generations, budget, eval suite and universe.
- **Held-out evaluation:** training uses universes `rules_seed >= TRAIN_UNIVERSE_BASE = 1_000_000` with randomized world variants. Evaluation (`run`, `compare`, `creator-progress`) uses universes `< 1_000_000` and the default variant. A test enforces that the ranges are disjoint.
- Selection stays shared and outside Creator control: elites are the top `elite` by fitness.
- Same seed and checkpoint give an identical run. Neural sampling uses a `torch.Generator` seeded from the run's numpy rng.
- Checkpoints go in `checkpoints/creator_vNNN.pt` (committed; small) with `checkpoints/index.json`. Training history goes in `creator/training_history.json`, and checkpoint progress in `creator/progress.json`.
- Commits: small and frequent. No AI co-author or attribution trailers.
- Commands run from `engine/` with `.venv/Scripts/python` (written as `python`).

## Review Focus

1. **Population with all-zero fitness, or one candidate:** features stay finite, and softmax over one candidate works. Test in Task 2.
2. **Checkpoint from an older feature size or missing file:** `neural:v999` exits 2 with a clear message, and a shape mismatch raises a clear error. Test in Task 5.
3. **Resume after interruption:** resuming reloads weights, optimizer and iteration count, and never overwrites `v000`. Test in Task 4.
4. **`noteach` really disables teaching:** every child gets `DEFAULT_CURRICULUM` and the prediction is made with that curriculum. Test in Task 3.
5. **Training universes leaking into evaluation:** the seed ranges are disjoint, and a test fails if `compare` ever touches a training universe. Test in Task 1.

---

### Task 1: World variants, universes and a steppable runner

**Files:**
- Modify: `engine/prometheus/world.py`, `engine/prometheus/benchmark.py`, `engine/prometheus/runner.py`
- Test: `engine/tests/test_variants.py`; the existing `tests/test_runner.py` must stay green.

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) WorldVariant(food_scale: float = 1.0, hazard_scale: float = 1.0, shift_step: int = SHIFT_STEP)`, with `apply(config: WorldConfig) -> WorldConfig` (tile counts scaled and rounded, minimum 1 for food and goal; `shift_step` replaced) and `WorldVariant.random(rng)` (food 0.6–1.4, hazard 0.6–1.6, shift_step 60–140).
  - `make_world(preset, seed, rules=None, variant=None)`.
  - `EvalSuite` gains `variant: WorldVariant = WorldVariant()`, used for both training and evaluation of a child.
  - `TRAIN_UNIVERSE_BASE = 1_000_000` in `benchmark.py`; `is_training_universe(rules_seed) -> bool`.
  - `RunConfig` gains `variant: WorldVariant = WorldVariant()`.
  - `class Evolution(config: RunConfig, creator: Creator, clock=None)` in `runner.py`, with `pending_jobs() -> list[tuple]` (proposes the next generation's children and returns their worker jobs; empty when finished), `ingest(results: list[tuple[dict, dict]]) -> None` (stores metrics, emits events, records stats and replays), `done: bool`, and `result() -> RunResult`. `run_evolution` becomes a loop over a single `Evolution` with the same output. `RunConfig.verify_replicates = 0` skips verification (used in training runs only).
  - `run_lockstep(evolutions: list[Evolution], executor) -> None` advances many evolutions together, sending all their jobs to the pool in one `map` per generation.
  - `run_evolution` and `compare` raise `ValueError` when given a training universe.

- [ ] **Step 1: Failing tests:** `WorldVariant()` applied to every preset leaves it unchanged; a scaled variant changes tile counts as specified; `shift_step` moves the shift; `run_lockstep` over 3 configs gives entities identical to 3 separate `run_evolution` calls; `run_evolution(RunConfig(universe=TRAIN_UNIVERSE_BASE + 5))` raises `ValueError`.
- [ ] **Step 2: Run** `python -m pytest tests/test_variants.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Refactor the runner without changing any output for existing configs.
- [ ] **Step 4: Run** the full suite. Expected: all PASS, including every Plan 1 test unchanged.
- [ ] **Step 5: Commit** `"Add world variants and training universes"`, then `"Make evolution steppable for lockstep runs"`.

---

### Task 2: Creator features

**Files:**
- Create: `engine/prometheus/creators/features.py`, `engine/tests/test_features.py`

**Interfaces:**
- Produces:
  - `CANDIDATE_DIM` and `POPULATION_DIM` (ints, asserted by tests).
  - `candidate_features(rec: EntityRecord, population: list[EntityRecord]) -> np.ndarray[CANDIDATE_DIM]`, in this order: normalized genome (9); fitness / population max (1); `log1p(fitness)` (1); stable, shift and mean score each / population max (3); survival rate (1); adaptation ratio / 2 (1); learning speed (1); `min(mean_prediction_error / 10, 3)` (1); `perception_cost / 100` (1); death causes hazard, starvation, post_shift (3); lesson scores / max lesson score (4); rank fraction, 0 = best (1); inherited flag (1); own curriculum one-hot, 4 lessons × 4 presets (16). Total 44.
  - `population_features(population, generation, total_generations) -> np.ndarray[POPULATION_DIM]`: mean and std of normalized genomes (18), `log1p` of mean and best fitness (2), genome diversity (1), generation / total (1). Total 22.
  - Every division is guarded so the output is always finite.

- [ ] **Step 1: Failing tests:** dimensions match; all features finite for an all-zero-fitness population and for a single-candidate population; candidate features are invariant to population order; changing a parent's curriculum changes exactly its one-hot slice.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement. Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `"Add Creator observation features"`.

---

### Task 3: The Creator network and NeuralCreator

**Files:**
- Modify: `engine/pyproject.toml` (add `torch>=2.2`; install the CPU wheel with `pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- Create: `engine/prometheus/creators/neural.py`, `engine/tests/test_neural.py`
- Modify: `engine/prometheus/creators/__init__.py` (register `"neural"`)

**Interfaces:**
- Produces:
  - `EDIT_SCALE = 0.25` (largest mean edit per gene, in normalized units); `HIDDEN = 128`.
  - `class CreatorNet(nn.Module)`:
    - encoder: `Linear(CANDIDATE_DIM + POPULATION_DIM, 128) → Tanh → Linear(128, 128) → Tanh`, applied to each candidate, giving `h`.
    - `parent_head: Linear(128, 1)`: a score per candidate, softmax over candidates.
    - `edit_mean: Linear(128, 9)`, with `mean = EDIT_SCALE * tanh(·)`; `edit_log_std`, a learned parameter of shape (9,) initialised to `log(0.08)`.
    - `inherit_head: Linear(128, 1)`: Bernoulli logit.
    - `teach_head: Linear(128 + 9, 64) → Tanh → Linear(64, 16)`: reshaped to 4 lessons × 4 preset logits, conditioned on the parent's `h` and the sampled edit.
    - `predict_head: Linear(128 + 9 + 1 + 16, 64) → Tanh → Linear(64, 1)`: predicted normalized fitness change, taking `h`, the edit, the inherit flag and the curriculum one-hot.
  - `@dataclass Decision(cand: np.ndarray, pop: np.ndarray, parent_index: int, edit: np.ndarray, inherit: int, curriculum: np.ndarray[4], log_prob: float, prediction: float)`, holding everything PPO needs to re-score a decision.
  - `CreatorNet.sample(cand: Tensor[N, C], pop: Tensor[P], generator, teach: bool = True) -> Decision` samples in order: parent, edit, inherit, curriculum. When `teach=False`, the curriculum is fixed to `DEFAULT_CURRICULUM` and adds no log-prob.
  - `CreatorNet.evaluate(decisions: list[Decision], teach=True) -> (log_probs, entropy, predictions)` for PPO.
  - `class NeuralCreator(net: CreatorNet, teach: bool = True, total_generations: int = 10, name="neural")` implements `Creator`. `seed_population` delegates to `RandomMutationCreator().seed_population`, so generation 0 is identical to the random arm's. `next_generation` builds features, samples one `Decision` per slot (the torch generator is seeded from `rng.integers(2**63)`), and turns each into a `ChildSpec`: child genome = `clip(parent_vector + edit, 0, 1)`; `predicted_change = prediction * report-scale` (see Task 4); `decision = {"parent_index", "edit", "inherit", "curriculum", "log_prob", "prediction"}`; the reason is built from the decision.
  - `NeuralCreator.decisions: list[Decision]` accumulates every decision, for training.
  - Reason format, generated only from real decisions: `"{parent} ({dominant failure: 'died to hazards in 40% of worlds' | 'adapted poorly after shifts (0.45)' | 'starved in 60% of worlds' | 'strong all-round'}): {top-3 gene edits by size, e.g. 'curiosity_bonus +0.31, risk_aversion -0.20'}; {'inherits knowledge' | 'starts fresh'}; curriculum easy > shift > shift > normal; predicts {+x.xx} fitness"`.
  - Factory registration: `"neural"` accepts `""` / `"latest"` / `"vNNN"`, optionally followed by `":noteach"`, and loads the checkpoint through Task 5's `load_checkpoint`. Until Task 5 exists, `""` gives a fresh `torch.manual_seed(0)` network.

- [ ] **Step 1: Failing tests:**

```python
def test_sample_shapes_and_valid_spec()          # specs validate unchanged; curriculum presets valid; parent in population
def test_sampling_deterministic_given_seed()     # same numpy rng seed -> identical specs
def test_logprob_recompute_matches_sample()      # evaluate([d]).log_probs ≈ d.log_prob (atol 1e-5)
def test_noteach_forces_default_curriculum()     # all specs DEFAULT_CURRICULUM; decision log_prob excludes teach head
def test_generation0_matches_random_arm()        # seed_population specs == RandomMutationCreator's for same rng seed
def test_reason_mentions_real_edits()            # the largest |edit| gene name appears in the reason
def test_single_candidate_population()           # N=1 works
```

- [ ] **Step 2: Run** `python -m pytest tests/test_neural.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement. Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `"Add Creator policy network"`, then `"Register the neural Creator"`.

---

### Task 4: Reinforcement learning (PPO) training loop

**Files:**
- Create: `engine/prometheus/creators/training.py`, `engine/tests/test_training.py`

**Interfaces:**
- Produces:
  - `@dataclass TrainConfig(iterations: int = 100, runs_per_iteration: int = 12, generations: int = 6, population: int = 8, elite: int = 2, episodes_per_lesson: int = 3, eval_worlds: int = 12, lr: float = 3e-4, ppo_epochs: int = 4, clip: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01, duplicate_penalty: float = 0.1, checkpoint_every: int = 10, seed: int = 0, workers: int = 0, replicates: int = 2)`
  - **Reward** for each decision = `(child_fitness - parent_fitness) / scale`, where `scale = max(mean gen-0 fitness of that run, 0.25)`, minus `duplicate_penalty` if the child's normalized genome is within L2 0.05 of any other active genome. `predicted_change` in ChildSpec = `prediction * scale`, where the scale is computed from the report's history (gen-0 mean) for the same normalization.
  - **Advantage** = reward − prediction (the prediction head is the baseline), normalized per batch.
  - **Loss** = PPO clipped surrogate + `value_coef` × MSE(prediction, reward) − `entropy_coef` × entropy. Adam, `ppo_epochs` passes over the batch in minibatches of 256, and grad-norm clip 1.0.
  - `ppo_update(net, optimizer, decisions, rewards, cfg) -> dict` returns `{"policy_loss", "value_loss", "entropy", "approx_kl"}`.
  - `collect(net, cfg, iteration, executor) -> (decisions, rewards, stats)`. It runs `runs_per_iteration` evolutions in lockstep (`run_lockstep`). Each uses its own `NeuralCreator(net)`, universe `TRAIN_UNIVERSE_BASE + iteration * runs_per_iteration + i`, and variant `WorldVariant.random(rng)`. `stats` = `{"mean_reward", "final_best_mean", "gen0_best_mean", "improvement_mean", "prediction_mae"}`.
  - `train(cfg, checkpoint_dir: Path, history_path: Path, resume: bool) -> None`. On a fresh start it saves `v000` (random init, `torch.manual_seed(cfg.seed)`) before any update. Each iteration runs `collect` then `ppo_update`, appends one history row `{iteration, version, *stats, *losses, seconds}`, and saves `vNNN` every `checkpoint_every` iterations plus at the end. Resume loads the latest checkpoint's weights, optimizer and iteration.

- [ ] **Step 1: Failing tests:**

```python
def test_ppo_learns_toy_bandit():
    # reward = 1 if decision.curriculum[0] == index of "shift" else 0, on fixed random features;
    # 60 ppo_update calls on fresh samples -> P(first lesson == shift) rises from ~0.25 to > 0.6
def test_ppo_learns_edit_direction():
    # reward = edit[risk_aversion index]; after updates the mean edit for that gene is > 0.05
def test_prediction_head_fits_rewards():          # value loss falls by > 50% on a fixed batch
def test_rewards_use_parent_delta_and_penalty()   # hand-built records -> exact reward values
def test_train_tiny_end_to_end(tmp_path):         # iterations=2, runs=2, generations=2, population=4, elite=1,
                                                  # episodes_per_lesson=1, eval_worlds=4, checkpoint_every=1
                                                  # -> v000, v001, v002 exist; history has 2 rows with finite numbers
def test_resume_continues(tmp_path)               # train 1 iteration, resume 1 more -> history rows 1,2; v000 unchanged (hash)
def test_training_never_uses_eval_universes()     # every universe collect() touches is >= TRAIN_UNIVERSE_BASE
```

- [ ] **Step 2: Run** `python -m pytest tests/test_training.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement. Step 4: Run** the full suite. Expected: PASS. If the toy bandit does not learn, the PPO math is wrong: fix the code, never the threshold.
- [ ] **Step 5: Commit** `"Add PPO training for the Creator"`.

---

### Task 5: Checkpoints, CLI commands and Creator progress

**Files:**
- Create: `engine/prometheus/creators/checkpoints.py`, `engine/prometheus/progress.py`, `engine/tests/test_checkpoints.py`
- Modify: `engine/prometheus/cli.py`, `engine/prometheus/creators/__init__.py`, `engine/tests/test_cli.py`

**Interfaces:**
- Produces:
  - `save_checkpoint(dir, version: int, net, optimizer, iteration: int, meta: dict) -> Path`, which writes `creator_v{version:03d}.pt` (`state_dict`, optimizer state, iteration, `CANDIDATE_DIM`, `POPULATION_DIM`, meta) and updates `index.json` (`[{version, iteration, file, created_at, mean_reward}]`).
  - `load_checkpoint(dir, which: str = "latest") -> tuple[CreatorNet, dict]`, where `which` is `"latest"` or `"vNNN"`. A missing file raises `FileNotFoundError` naming the available versions. A feature-dimension mismatch raises `ValueError("checkpoint built for feature dims X/Y, code has A/B")`.
  - `DEFAULT_CHECKPOINT_DIR` = repo-root `checkpoints/`, overridable with the env var `PROMETHEUS_CHECKPOINTS`.
  - CLI `prometheus train-creator --iterations N [--resume] [--runs-per-iteration] [--checkpoint-every] [--workers] [--seed]` prints one line per iteration: iteration, mean reward, improvement, prediction MAE, and seconds.
  - CLI `prometheus creator-progress [--versions v000,v010,...|all] --seeds 0,1,2,3,4 --out .` runs each checkpoint at default run sizes on the held-out seeds. It writes `creator/progress.json`: `[{version, iteration, verified_final_mean, verified_final_std, verified_gain_mean, verified_gain_std, prediction_mae}]`, and prints a table.
  - `compare` and `run` accept `neural:latest`, `neural:v000`, `neural:latest:noteach`.
  - `train-creator` copies `creator/training_history.json` next to the checkpoints so the dashboard can read it.

- [ ] **Step 1: Failing tests:** a save/load round-trip produces identical outputs for the same input; `v999` raises `FileNotFoundError` listing versions; a dimension mismatch raises `ValueError`; the CLI `run --creator neural:v999` exits 2; the CLI `train-creator` with tiny flags exits 0 and creates checkpoints; `creator-progress` on two tiny checkpoints writes `progress.json` with 2 rows.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement. Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `"Add Creator checkpoints"`, then `"Add train-creator and creator-progress commands"`.

---

### Task 6: Train the Creator for real and report honestly

**Files:**
- Create: `docs/creator-results.md`; `checkpoints/*` and `creator/*.json` (committed)
- Modify: `README.md`

- [ ] **Step 1: Train.** `prometheus train-creator --iterations 300 --checkpoint-every 25` runs in the background. Watch the per-iteration lines. Expected: mean reward and prediction MAE trend better over iterations. If reward is flat after about 100 iterations, stop and debug with superpowers:systematic-debugging: check reward scale, advantage normalization and entropy. Do not just train longer.
- [ ] **Step 2: Creator progress.** `prometheus creator-progress --versions all --seeds 0,1,2,3,4 --out ..` answers whether the parent gets better as it trains.
- [ ] **Step 3: Final comparison on held-out universes.** `prometheus compare --creators neural:latest,neural:latest:noteach,neural:v000,random,clone,heuristic --seeds 0,1,2,3,4 --name final --out ..`. All answers use the verified gain columns.
- [ ] **Step 4: Write `docs/creator-results.md`:** the training curves (table of history rows every 25 iterations), the progress table, the final comparison table (mean ± std), and the four honest answers from spec §8: do later generations beat earlier ones; does the trained Creator beat random and heuristic; does teaching add value (latest vs noteach); does an experienced Creator beat a fresh one (latest vs v000). Include what the Creator learned to do (most common edits per failure type, curriculum patterns, from the final comparison's entities). If any answer is "no", say so and give the likely reason.
- [ ] **Step 5: Update the README** status and results. **Commit** `"Train the Creator"` (checkpoints and history), then `"Report Creator results"`, then push.
