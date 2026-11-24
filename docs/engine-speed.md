# Engine speed baseline

Measured 2026-11-24 with `engine/scripts/measure_speed.py`.

| | |
|---|---|
| Machine | Intel Core (Family 6 Model 141), 12 logical cores, Windows 11, Python 3.13.9 |
| One child, one upbringing (20 training + 24 eval episodes) | 0.179 s (×3 upbringings by default) |
| Children in a default run (8 + 9 generations × 6) | 62 |
| Projected default run | 11.1 s on 1 core; about 1 s of compute on 12 cores, plus process overhead |

Most children die well before 200 steps, which keeps episodes short. Better children survive longer and take more time per episode.

**What this means for Creator training (Plan 2):** one evolution run costs about 11 core-seconds, so on 12 cores the Creator can experience roughly 4,000 runs, or about 240,000 child designs, per hour. That is enough for real reinforcement learning on a laptop.

## Baseline comparison (after the engine review fixes)

`prometheus compare --creators clone,heuristic,random --seeds 0,1,2 --name baseline`, with population 8, 10 generations, each child averaged over 3 upbringings (20 training + 24 eval episodes each), and 10 fresh upbringings for verification. Wall time: 176 s on 12 cores.

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

### How noisy is one score?

The final clone champion's design (SCOUT, fitness 6.40 in-run) was re-scored on fresh seeds:
- **one upbringing:** mean 3.64, std 1.68, range 1.67–6.49
- **average of 3 upbringings:** mean 3.30, std 0.91

So a single score is mostly luck, and the in-run 6.40 was the luckiest of about 30 identical copies. This matters for training the Creator: its reward must use replicate-averaged scores and be judged by verified results.

### Superseded first run

The first comparison reported heuristic +45% and random +458%. Both numbers came from the selection-biased in-run curve with single upbringings, and are wrong.
