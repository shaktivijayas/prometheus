"""Time how long one child takes to be taught and scored (the unit of all evolution and Creator training)."""
import os
import platform
import time

from prometheus.benchmark import DEFAULT_CURRICULUM, EvalSuite, train_and_evaluate
from prometheus.genome import Genome

REPEATS = 5
STEPS_PER_EPISODE_MAX = 200


def main() -> None:
    suite = EvalSuite()
    times, steps = [], 0
    for i in range(REPEATS):
        t = time.perf_counter()
        metrics, _ = train_and_evaluate(Genome.default(), None, DEFAULT_CURRICULUM, i, suite)
        times.append(time.perf_counter() - t)
    per_child = sum(times) / len(times)
    children_per_run = 8 + 9 * 6
    print(f"machine: {platform.processor() or platform.machine()} ({os.cpu_count()} logical cores), Python {platform.python_version()}")
    print(f"seconds per child (20 training + 24 eval episodes): {per_child:.3f}")
    print(f"children per default run: {children_per_run}")
    print(f"projected default run, 1 core: {per_child * children_per_run:.1f}s; "
          f"{os.cpu_count()} cores: ~{per_child * children_per_run / (os.cpu_count() or 1):.1f}s + overhead")


if __name__ == "__main__":
    main()
