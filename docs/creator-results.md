# Creator results

The Creator is a small neural network (an MLP with about 46,000 parameters) that we train ourselves with PPO, on a CPU. It never calls Claude or any other external model or API. Each time it makes a child it does five things:

1. picks a parent from the living population
2. edits the parent's 9 genes
3. decides whether the child inherits the parent's learned beliefs
4. **teaches** the child by choosing 4 lessons from `easy`, `normal`, `hard` and `shift`
5. predicts how much better the child will be than its parent

The child then learns in its universe from those lessons, and the benchmark scores it. The Creator's reward is how much the child beat its parent, normalised, with a small penalty for near-duplicate children.

Every number below comes from **universes the Creator never trained in**. Training universes are numbered from 1,000,000 upward, evaluation universes are below that, and the code refuses to mix them. Every result is **verified**: the generation-0 best design and the final champion are re-scored on 10 fresh upbringings, in world layouts that played no part in selection (same universe, same hidden laws, different maps). That removes the "lucky draw" bias.

## 1. Training

156 PPO iterations. Each iteration ran 12 evolution runs (6 generations × 8 children), for 360 Creator decisions per iteration. Training took 2.9 hours on a 12-core laptop. It was planned for 300 iterations, but Claude Code's low-memory guard killed the process at iteration 156. By then the curves had been flat for about 100 iterations, and checkpoint v006 (iteration 150) had been saved.

Every value in this table is an average over the 10-iteration window, measured on training universes:

| Iterations | Mean reward | Final population mean fitness | Final best fitness | Prediction error (MAE) |
|---|---|---|---|---|
| 1–10 | +0.34 | 4.63 | 8.73 | 1.32 |
| 26–35 | +3.47 | 13.72 | 16.14 | 3.05 |
| 51–60 | +4.05 | 14.24 | 15.99 | 2.80 |
| 91–100 | +3.88 | 14.74 | 16.27 | 2.53 |
| 147–156 | +3.79 | 14.85 | 16.24 | 2.12 |

Most of the learning happens in the first 25 iterations. After that the Creator mainly gets better at **predicting** its children.

## 2. Checkpoint progress (held-out)

Each checkpoint ran 3 full evolution runs (10 generations × 8 children) in held-out universes 0, 1 and 2. The figures are mean ± std across those 3 universes.

| Checkpoint | Training iteration | Verified champion | Verified gain | Prediction MAE |
|---|---|---|---|---|
| v000 (untrained) | 0 | 10.17 ± 2.90 | +4.64 ± 2.78 | 1.60 |
| v001 | 25 | 17.57 ± 1.11 | +12.03 ± 0.81 | 2.49 |
| v002 | 50 | 16.79 ± 1.19 | +11.25 ± 0.74 | 2.10 |
| v003 | 75 | 16.59 ± 1.11 | +11.06 ± 0.70 | 1.38 |
| v004 | 100 | 17.51 ± 1.31 | +11.98 ± 1.52 | 1.30 |
| v005 | 125 | 16.87 ± 1.73 | +11.34 ± 1.67 | 1.17 |
| v006 | 150 | 16.30 ± 1.08 | +10.77 ± 1.09 | 1.07 |

Training more than doubles the Creator's verified gain, and almost all of that arrives by v001. After that, champion quality wanders within the noise. It doesn't keep climbing, and v006 is a little below the peak. The prediction error rises at first (1.60 → 2.49, because the trained Creator's children change a lot more than random ones). It then falls steadily to 1.07 as the Creator learns to forecast its own children.

## 3. Final comparison (held-out)

The comparison has 7 arms, each run on 5 held-out universes (0–4). Every arm gets the same budget: population 8, 10 generations, each child scored as the average of 3 upbringings, and 10 fresh verification upbringings in unseen world layouts. It took 20 minutes.

| Creator | Verified gen-0 best | Verified final champion | **Verified gain** (mean ± std, 5 universes) | In-run best* |
|---|---|---|---|---|
| **trained Creator** (`neural:latest`, v006) | 4.78 | **16.29 ± 0.85** | **+11.51 ± 1.37** | 16.88 |
| trained designs, random lessons (`neural:latest:randomteach`) | 4.78 | 16.21 ± 0.70 | +11.43 ± 0.83 | 16.23 |
| random mutation (control) | 4.78 | 12.57 ± 1.14 | +7.80 ± 1.78 | 10.65 |
| untrained Creator (`neural:v000`) | 4.78 | 11.02 ± 2.53 | +6.25 ± 2.92 | 9.51 |
| trained designs, one fixed default plan (`neural:latest:noteach`) | 4.78 | 6.94 ± 2.05 | +2.16 ± 1.93 | 9.44 |
| clone (null: changes nothing) | 4.78 | 4.78 ± 1.11 | +0.00 ± 0.00 | 6.30 |
| heuristic (hand-written rules) | 1.84 | 1.84 ± 0.28 | −0.00 ± 0.20 | 1.91 |

\* The in-run best is selection-biased. The clone arm, which never changes anything, still "improves" to 6.30 in-run. Its verified gain of exactly 0 shows that verification removes that bias.

## The four questions

**Do generations improve?** Yes. With the trained Creator, verified champion quality rises from 4.78 to 16.29 over 10 generations, about 3.4×. It improves in all 5 held-out universes.

**Does the trained Creator beat random mutation and the hand-written rules?** Yes, on both counts.
- Its verified champions average 16.29, against 12.57 for random mutation. The gap is about 6 standard errors.
- Its gain is +11.5 against +7.8, which is 1.5× random's.
- Its worst held-out universe (15.21) beats random's best (14.09).
- The hand-written rules make no progress at all.

**Does teaching help? Partly, and the honest answer is more interesting than "yes".**
- **What children are taught matters a great deal.** The same trained designs, all taught one fixed default plan (`normal > normal > shift > normal`), collapse from 16.29 to 6.94. That is worse than random mutation, and about 9 standard errors below.
- **But the Creator's own lesson choices are no better than random lessons.** Give each child of those same designs 4 lessons drawn uniformly at random, and it reaches 16.21, the same as the trained Creator's 16.29 (0.2 standard errors apart).
- So the trained Creator's advantage comes from **how it designs children**, not from a learned teaching skill. The one thing its teaching clearly learned is to avoid what hurts. It almost never picks a `shift` lesson (0.2% of lessons). But varied random lessons turn out to be just as good, so that choice doesn't improve on random.
- We don't yet know why that one fixed plan is so harmful. The likely cause is that selection then favours designs tuned to a single lesson sequence, which carry over badly. That is the next experiment.
- An earlier draft of this report claimed "teaching is half the skill". That claim compared only against the fixed default plan. A reviewer pointed out the confound, and the random-lessons arm shows the claim was wrong.

**Does an experienced Creator beat a fresh one?** Yes. The untrained network (v000, random weights) reaches 11.02. That is no better than random mutation (12.57, about 1 standard error apart), which makes sense because its edits are just noise. After training it reaches 16.29, about 4 standard errors above where it started. Training is what makes the difference.

## What the Creator learned

These come from the 270 children the trained Creator made across the 5 held-out final runs, compared with random mutation.

**Teaching.** It almost never teaches a `shift` lesson: 0.2% of lessons, against 25% if lessons were picked at random. It prefers `hard` lessons (42%), then `normal` (40%), then `easy` (18%). Its favourite plans are `normal > hard > hard > normal`, `normal > easy > hard > normal` and `hard > hard > hard > hard`.

It found this without being told. A shift lesson leaves a child believing scrambled laws right before it is tested, and the hand-written rules originally made exactly that mistake. **Still, the random-lessons control (section 3) does just as well, so these preferences are a learned habit rather than an advantage.**

**Design.** The average designs its children ended up with, compared with random mutation's children:

| Gene | Trained Creator | Random mutation | What it means |
|---|---|---|---|
| perception radius | 4.9 | 3.9 | sees almost the whole local area |
| memory capacity | 44 | 18 | keeps long-run evidence about each kind |
| learning rate | 0.81 | 0.44 | updates beliefs fast |
| exploration | 0.04 | 0.52 | almost never moves randomly |
| planning horizon | 11.1 | 8.0 | plans long routes |
| risk aversion | 0.04 | 1.06 | does not avoid uncertain tiles |
| surprise threshold | 1.17 | 3.05 | **doubts its beliefs quickly** when a law changes |
| curiosity bonus | 0.65 | 1.12 | moderate interest in unseen kinds |

Put together: a far-seeing, fast-learning planner that exploits what it knows and resets its beliefs quickly when the world surprises it. That resetting is how it copes with law changes in the benchmark, without ever practising them in a lesson. This is a coherent strategy, not a pile of random edits, and the Creator found it only through reward.

## Limits, stated plainly

- **The learned teaching adds nothing over random lessons** (section 3). The Creator's measurable skill is in design.
- **Known training flaw.** The PPO advantage subtracts the Creator's own *action-conditioned* forecast (an estimate of Q(s, a)), not a state-only value estimate (V(s)). As its forecasts get better, its learning signal shrinks. This probably explains why it plateaued after about 25 iterations, and could also explain why the teaching head never got past "random is as good". The fix is a separate state-only value head plus retraining. It is the first next step. A final code review flagged this, and it wasn't fixed here because a full retrain takes about 3 hours and the machine's memory limit already killed one training run.
- 5 held-out universes is a small sample. The trained-vs-random gap is large compared with the spread, but the exact ratio will move with more seeds.
- The trained Creator converges, so its children are very similar to one another. That suits this benchmark, but it would be risky in a world whose laws change between generations.
- Training was cut short at 156 of 300 planned iterations by the machine's memory limit, not by choice.

## Reproduce

```bash
cd engine
.venv/Scripts/prometheus train-creator --iterations 300 --checkpoint-every 25 --workers 6 --checkpoints ../checkpoints-new
# or continue the shipped run from v006 (iteration 150) to 300:
#   .venv/Scripts/prometheus train-creator --resume --iterations 150 --checkpoint-every 25 --workers 6
.venv/Scripts/prometheus creator-progress --versions all --seeds 0,1,2 --workers 6 --out ..
.venv/Scripts/prometheus compare --creators neural:latest,neural:latest:randomteach,neural:latest:noteach,neural:v000,random,clone,heuristic \
    --seeds 0,1,2,3,4 --workers 6 --name final --out ..
```
