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
