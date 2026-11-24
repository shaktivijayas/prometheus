"""Collect runs, comparisons and Creator data into the folder the static dashboard reads."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def _export_run(run_dir: Path, out_dir: Path) -> dict:
    name = run_dir.name
    dest = out_dir / "runs" / name
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "replays").mkdir(parents=True)
    run = _read(run_dir / "run.json")
    entities = _read(run_dir / "entities.json")
    for fname in ("run.json", "entities.json", "generations.json"):
        shutil.copyfile(run_dir / fname, dest / fname)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    _write(dest / "events.json", events)
    replays = []
    for rp in sorted((run_dir / "replays").glob("*.json")):
        shutil.copyfile(rp, dest / "replays" / rp.name)
        r = _read(rp)
        replays.append({"file": rp.name, "key": rp.stem, "entity_id": r["entity_id"], "name": r.get("name"),
                        "generation": r["generation"], "preset": r["preset"], "seed": r["seed"],
                        "score": r.get("score"), "steps": r.get("steps")})
    _write(dest / "replays" / "index.json", replays)
    cfg = run["config"]
    return {"name": name, "creator": cfg["creator"], "seed": cfg["seed"], "generations": cfg["generations"],
            "children": len(entities), "verification": run.get("verification", {}), "summary": run.get("summary", {})}


def export_dashboard(out_dir: Path, runs: list[Path], comparisons: list[Path], creator_dir: Path | None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "runs": [], "comparisons": [], "creator": {"progress": False, "history": False}}
    for run_dir in runs:
        index["runs"].append(_export_run(Path(run_dir), out_dir))
    for comp in comparisons:
        data = _read(comp)
        _write(out_dir / "comparisons" / Path(comp).name, data)
        index["comparisons"].append({"name": data["name"], "file": f"comparisons/{Path(comp).name}",
                                     "creators": data["creators"], "seeds": data["seeds"]})
    if creator_dir is not None:
        for key, fname in (("progress", "progress.json"), ("history", "training_history.json")):
            src = Path(creator_dir) / fname
            if src.exists():
                _write(out_dir / "creator" / fname, _read(src))
                index["creator"][key] = True
    _write(out_dir / "index.json", index)
    return index
