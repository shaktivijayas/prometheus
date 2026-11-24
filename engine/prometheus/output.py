"""Write a run to disk in the layout the dashboard reads (spec section 6.6)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .creators.base import ranked
from .runner import RunResult


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, allow_nan=False), encoding="utf-8")


def run_summary(result: RunResult) -> dict:
    gens = result.generations
    first, last = (gens[0]["best"], gens[-1]["best"]) if gens else (0.0, 0.0)
    best = ranked(result.registry.all())[0] if result.registry.all() else None
    predicted = [(r.predicted_change, r.actual_change) for r in result.registry.all()
                 if r.predicted_change is not None and r.actual_change is not None]
    pred_error = (sum(abs(p - a) for p, a in predicted) / len(predicted)) if predicted else None
    return {
        "gen0_best": first,
        "final_best": last,
        "improvement": round(last - first, 6),
        "improvement_pct": round(100.0 * (last - first) / first, 2) if first > 0 else None,
        "best_entity": best.id if best else None,
        "best_name": best.name if best else None,
        "children_created": len(result.registry.all()),
        "mean_prediction_error": None if pred_error is None else round(pred_error, 6),
    }


def write_run(result: RunResult, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / "replays").mkdir(parents=True, exist_ok=True)
    _dump(out_dir / "run.json", {
        "config": asdict(result.config),
        "creator": result.config.creator,
        "universe": result.universe,
        "timing": result.timing,
        "summary": run_summary(result),
        "verification": result.verification,
    })
    _dump(out_dir / "entities.json", [r.to_dict() for r in result.registry.all()])
    with open(out_dir / "events.jsonl", "w", encoding="utf-8") as f:
        for e in result.events.events:
            f.write(json.dumps(e, allow_nan=False) + "\n")
    _dump(out_dir / "generations.json", result.generations)
    for key, replay in result.replays.items():
        (out_dir / "replays" / f"{key}.json").write_text(json.dumps(replay, allow_nan=False), encoding="utf-8")
