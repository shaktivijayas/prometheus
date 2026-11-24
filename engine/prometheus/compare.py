"""Run several Creators on identical budgets and seeds, and summarise who evolves better children."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .output import write_run
from .runner import RunConfig, _Executor, run_evolution


def run_dir_name(name: str, creator: str, seed: int) -> str:
    return f"{name}_{creator.replace(':', '-')}_s{seed}"


def compare(creators: list[str], seeds: list[int], base: RunConfig, out_dir: Path, name: str,
            progress=None) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "comparisons").mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict] = {}
    executor = _Executor(base.workers)
    try:
        for creator in creators:
            best_curves, mean_curves, runs, verified = [], [], [], []
            for seed in seeds:
                cfg = replace(base, creator=creator, seed=seed)
                result = run_evolution(cfg, executor=executor)
                run_name = run_dir_name(name, creator, seed)
                write_run(result, out_dir / "runs" / run_name)
                best_curves.append([g["best"] for g in result.generations])
                mean_curves.append([g["mean"] for g in result.generations])
                runs.append(run_name)
                verified.append(result.verification)
                if progress:
                    progress(creator, seed, result)
            best, mean = np.array(best_curves), np.array(mean_curves)
            arms[creator] = {
                "best_mean": [round(float(x), 6) for x in best.mean(axis=0)],
                "best_std": [round(float(x), 6) for x in best.std(axis=0)],
                "mean_mean": [round(float(x), 6) for x in mean.mean(axis=0)],
                "mean_std": [round(float(x), 6) for x in mean.std(axis=0)],
                "runs": runs,
                # unbiased headline: champions re-scored on fresh upbringings (see runner._verify)
                "verified_final_mean": round(float(np.mean([v["final_best"]["mean"] for v in verified])), 6),
                "verified_final_std": round(float(np.std([v["final_best"]["mean"] for v in verified])), 6),
                "verified_gen0_mean": round(float(np.mean([v["gen0_best"]["mean"] for v in verified])), 6),
                "verified_gain_mean": round(float(np.mean([v["gain"] for v in verified])), 6),
                "verified_gain_std": round(float(np.std([v["gain"] for v in verified])), 6),
            }
    finally:
        executor.close()
    data = {"name": name, "creators": list(creators), "seeds": list(seeds), "generations": base.generations,
            "arms": arms}
    (out_dir / "comparisons" / f"{name}.json").write_text(json.dumps(data, indent=1, allow_nan=False), encoding="utf-8")
    return data
