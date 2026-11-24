"""Write dashboard test fixtures produced by the real engine."""
import json
import sys
from pathlib import Path

import numpy as np

from prometheus.benchmark import EvalSuite
from prometheus.child import ChildAgent, run_episode
from prometheus.genome import Genome
from prometheus.runner import RunConfig, run_evolution
from prometheus.world import make_world

if __name__ == "__main__":
    out = Path(sys.argv[1])
    suite = EvalSuite(rules_seed=3)
    agent = ChildAgent(Genome.default(), 5, np.random.default_rng(4))
    world = make_world("shift", 77, suite.rules())
    world.energy = 1e6
    res = run_episode(agent, world, record=True)
    (out / "replay.json").write_text(json.dumps({
        "entity_id": "E0001", "name": "FIXTURE-001", "generation": 0, "preset": "shift", "seed": 77,
        "score": res.score, "goals": res.goals, "steps": res.steps, "survived": res.survived,
        "death_cause": res.death_cause, "trace": res.trace,
        "final_grid": world.grid.tolist(), "final_roles": list(world.rules.roles)}))
    run = run_evolution(RunConfig(creator="heuristic", generations=3, population=4, elite=1, episodes_per_lesson=1,
                                  eval_worlds=4, workers=1, replicates=1, verify_replicates=0), clock=lambda: "t")
    (out / "entities.json").write_text(json.dumps([r.to_dict() for r in run.registry.all()]))
    print("fixtures written", res.steps, "steps,", len(run.registry.all()), "entities")
