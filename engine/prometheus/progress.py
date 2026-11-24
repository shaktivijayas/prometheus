"""Does the parent get better as it trains? Evaluate a series of Creator checkpoints on held-out universes."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from .creators.checkpoints import default_checkpoint_dir, load_checkpoint, read_index
from .creators.neural import NeuralCreator
from .runner import RunConfig, _Executor, run_evolution


def resolve_versions(spec: str, checkpoint_dir: Path) -> list[str]:
    available = [e["name"] for e in read_index(checkpoint_dir)]
    if spec == "all":
        return available
    wanted = [v.strip() for v in spec.split(",") if v.strip()]
    missing = [v for v in wanted if v not in available]
    if missing:
        raise FileNotFoundError(f"checkpoints {missing} not found; available: {', '.join(available) or 'none'}")
    return wanted


def creator_progress(versions: list[str], seeds: list[int], base: RunConfig, out_dir: Path,
                     checkpoint_dir: Path | None = None, progress=None) -> list[dict]:
    checkpoint_dir = Path(checkpoint_dir or default_checkpoint_dir())
    out_dir = Path(out_dir)
    (out_dir / "creator").mkdir(parents=True, exist_ok=True)
    rows = []
    executor = _Executor(base.workers)
    try:
        for version in versions:
            finals, gains, errors, in_run = [], [], [], []
            iteration = None
            for seed in seeds:
                net, meta = load_checkpoint(checkpoint_dir, version)
                iteration = meta["iteration"]
                creator = NeuralCreator(net, teach=True, name=f"neural:{version}", version=version)
                result = run_evolution(replace(base, creator=f"neural:{version}", seed=seed, record_replays=False),
                                       creator=creator, executor=executor)
                v = result.verification
                finals.append(v["final_best"]["mean"])
                gains.append(v["gain"])
                in_run.append(result.generations[-1]["best"])
                errors += [abs(r.predicted_change - r.actual_change) for r in result.registry.all()
                           if r.predicted_change is not None and r.actual_change is not None]
                if progress:
                    progress(version, seed, result)
            rows.append({
                "version": version, "iteration": iteration, "seeds": list(seeds),
                "verified_final_mean": round(float(np.mean(finals)), 6),
                "verified_final_std": round(float(np.std(finals)), 6),
                "verified_gain_mean": round(float(np.mean(gains)), 6),
                "verified_gain_std": round(float(np.std(gains)), 6),
                "in_run_best_mean": round(float(np.mean(in_run)), 6),
                "prediction_mae": round(float(np.mean(errors)), 6) if errors else None,
            })
    finally:
        executor.close()
    (out_dir / "creator" / "progress.json").write_text(json.dumps(rows, indent=1, allow_nan=False), encoding="utf-8")
    history = checkpoint_dir / "training_history.json"
    if history.exists():
        shutil.copyfile(history, out_dir / "creator" / "training_history.json")
    return rows
