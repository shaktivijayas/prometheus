# Engine speed baseline

Measured 2026-11-24 with `engine/scripts/measure_speed.py`.

| | |
|---|---|
| Machine | Intel Core (Family 6 Model 141), 12 logical cores, Windows 11, Python 3.13.9 |
| One child (20 training + 24 eval episodes) | 0.179 s |
| Children in a default run (8 + 9 generations × 6) | 62 |
| Projected default run | 11.1 s on 1 core; about 1 s of compute on 12 cores, plus process overhead |

Most children die well before 200 steps, which keeps episodes short. Better children survive longer and take more time per episode.

**What this means for Creator training (Plan 2):** one evolution run costs about 11 core-seconds, so on 12 cores the Creator can experience roughly 4,000 runs, or about 240,000 child designs, per hour. That is enough for real reinforcement learning on a laptop.

## Baseline comparison (Plan 1, Task 8)

`prometheus compare --creators heuristic,random --seeds 0,1,2 --name baseline` with default sizes (population 8, 10 generations, 20 training + 24 eval episodes per child). Wall time: 37.6 s on 12 cores.

| Creator | Gen 0 best | Final best (mean ± std over 3 seeds) | Gain |
|---|---|---|---|
| heuristic (hand-written rules) | 1.356 | 1.967 ± 0.080 | +45% |
| random mutation (control) | 1.916 | 10.678 ± 1.858 | +458% |

**What happened.** Evolution clearly works: random mutation plus selection raised the best child's score more than fivefold. The hand-written rules did much worse. Their "failed to adapt" rule fired 48 times in a row, repeatedly lowering the surprise threshold so children kept wiping their own knowledge, and the rules never checked whether a fix helped. The winning random-arm design went the other way: wide perception (4), high curiosity (1.6), low risk aversion (0.44), a *high* surprise threshold (3.7), and shift lessons first.

**What this means for the trained Creator.** Random mutation (about 10.7) is the real bar. The hand-written baseline is kept as it is, as evidence that plausible human rules can fail. Beating it alone proves little.
