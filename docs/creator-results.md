# Creator results

The Creator is a small neural network (an MLP with about 46,000 parameters) that we train ourselves with PPO, on a CPU. It never calls Claude or any other external model or API. Each time it makes a child it does five things:

1. picks a parent from the living population
2. edits the parent's 9 genes
3. decides whether the child inherits the parent's learned beliefs
4. **teaches** the child by choosing 4 lessons from `easy`, `normal`, `hard` and `shift`
5. predicts how much better the child will be than its parent

The child then learns in its universe from those lessons, and the benchmark scores it. The Creator's reward is how much the child beat its parent, normalised, with a small penalty for near-duplicate children.

Every number below comes from **universes the Creator never trained in**. Training universes are numbered from 1,000,000 upward, evaluation universes are below that, and the code refuses to mix them. Every result is **verified**: the generation-0 best design and the final champion are re-scored on 10 fresh upbringings that played no part in selection. That removes the "lucky draw" bias.

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
| v000 (untrained) | 0 | 7.82 ± 2.25 | +3.69 ± 2.17 | 1.60 |
| v001 | 25 | 15.83 ± 0.75 | +11.70 ± 0.64 | 2.49 |
| v002 | 50 | 15.81 ± 0.50 | +11.68 ± 0.45 | 2.10 |
| v003 | 75 | 15.66 ± 0.57 | +11.53 ± 0.61 | 1.38 |
| v004 | 100 | 15.71 ± 0.70 | +11.58 ± 0.69 | 1.30 |
| v005 | 125 | 15.41 ± 0.72 | +11.28 ± 0.54 | 1.17 |
| v006 | 150 | 15.44 ± 0.70 | +11.31 ± 0.54 | 1.07 |

After training, the Creator produces champions that are twice as good. Champion quality plateaus by v001. The small drift from v004 to v006 is inside the noise. The prediction error on held-out universes keeps falling the whole time (2.49 → 1.07): the Creator keeps learning to forecast what its children will do even after it stops making them better.

## 3. Final comparison (held-out)

The comparison has 6 arms, each run on 5 held-out universes (0–4). Every arm gets the same budget: population 8, 10 generations, each child scored as the average of 3 upbringings, and 10 fresh upbringings for verification. It took 23.6 minutes.

| Creator | Verified gen-0 best | Verified final champion | **Verified gain** (mean ± std, 5 universes) | In-run best* |
|---|---|---|---|---|
| **trained Creator** (`neural:latest`, v006) | 3.48 | **15.85 ± 0.97** | **+12.37 ± 1.38** | 16.88 |
| random mutation (control) | 3.48 | 10.22 ± 0.82 | +6.75 ± 1.51 | 10.65 |
| untrained Creator (`neural:v000`) | 3.48 | 8.27 ± 1.87 | +4.79 ± 2.16 | 9.51 |
| trained Creator, teaching disabled | 3.48 | 5.10 ± 2.26 | +1.62 ± 2.18 | 9.44 |
| clone (null: changes nothing) | 3.48 | 3.48 ± 0.97 | +0.00 ± 0.00 | 6.30 |
| heuristic (hand-written rules) | 1.33 | 1.26 ± 0.16 | −0.06 ± 0.20 | 1.91 |

\* The in-run best is selection-biased. The clone arm, which never changes anything, still "improves" to 6.30 in-run. Its verified gain of exactly 0 shows that verification removes that bias.

## The four questions

**Do generations improve?** Yes. With the trained Creator, the verified champion quality rises from 3.48 to 15.85 over 10 generations, about 4.6×, and every one of the 5 held-out universes improves.

**Does the trained Creator beat random mutation and the hand-written rules?** Yes, on both counts. Its verified gain is +12.37 ± 1.38, against +6.75 ± 1.51 for random mutation. That is 1.8× random's gain, and the gap is about 6 standard errors. Its worst held-out universe (verified champion 14.81) beats random's best (10.95). The hand-written rules make no progress at all.

**Does teaching help?** Yes, and it is the biggest single effect. When we switch off the trained Creator's teaching head and give every child the default lesson plan (`normal > normal > shift > normal`), the verified gain collapses from +12.37 to +1.62. That is worse than random mutation. The design choices and the teaching were learned together: the Creator learned to build children that are fast-learning, far-seeing and doubt quickly, then to teach them in a way that suits that design. The same designs taught the default way do badly. **The Creator's skill at teaching is a large part of why it works.**

**Does an experienced Creator beat a fresh one?** Yes. The untrained Creator (v000, random weights) gains +4.79 ± 2.16. That is about the same as random mutation, which makes sense, because its edits are just noise. The trained Creator gains +12.37 ± 1.38. Training is what makes the difference.

## What the Creator learned

These come from the 270 children the trained Creator made across the 5 held-out final runs, compared with random mutation.

**Teaching.** It almost never teaches a `shift` lesson: 0.2% of lessons, against 25% if lessons were picked at random. It prefers `hard` lessons (42%), then `normal` (40%), then `easy` (18%). Its favourite plans are `normal > hard > hard > normal`, `normal > easy > hard > normal` and `hard > hard > hard > hard`.

It found this without being told: a shift lesson leaves a child believing scrambled laws right before it is tested. The hand-written rules originally made exactly that mistake, and the default curriculum still contains a shift lesson, which is why the no-teaching arm suffers. `hard` worlds, with more hazards and less food, teach the costly lessons (which kind hurts) quickly.

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

- The plateau after about 25 iterations means this Creator has found one good strategy and is refining its forecasts, not its children. A harder world or a larger action space for the Creator (the planned stage-two "invent new abilities" Creator) is where more training would pay off.
- 5 held-out universes is a small sample. The gap between the trained Creator and random mutation is large compared with the spread, but the exact ratio (1.8×) will move with more seeds.
- The trained Creator converges: its children are very similar to one another. That is fine for this benchmark, but it would be risky in a world whose laws change between generations.
- Training was cut short at 156 of 300 planned iterations by the machine's memory limit, not by choice. To resume from v006 to 300 iterations: `prometheus train-creator --resume --iterations 150`.

## Reproduce

```bash
cd engine
.venv/Scripts/prometheus train-creator --iterations 300 --workers 6             # or: --resume --iterations <more>
.venv/Scripts/prometheus creator-progress --versions all --seeds 0,1,2 --workers 6 --out ..
.venv/Scripts/prometheus compare --creators neural:latest,neural:latest:noteach,neural:v000,random,clone,heuristic \
    --seeds 0,1,2,3,4 --workers 6 --name final --out ..
```
