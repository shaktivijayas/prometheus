"""Versioned Creator checkpoints: creator_v000.pt (untrained) ... creator_vNNN.pt, listed in index.json."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from .features import CANDIDATE_DIM, POPULATION_DIM
from .neural import CreatorNet

_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_checkpoint_dir() -> Path:
    return Path(os.environ.get("PROMETHEUS_CHECKPOINTS", _REPO_ROOT / "checkpoints"))


def _index_path(directory: Path) -> Path:
    return Path(directory) / "index.json"


def read_index(directory: Path | None = None) -> list[dict]:
    p = _index_path(directory or default_checkpoint_dir())
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def version_name(version: int) -> str:
    return f"v{version:03d}"


def save_checkpoint(directory: Path, version: int, net: CreatorNet, optimizer, iteration: int, meta: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"creator_{version_name(version)}.pt"
    torch.save({"state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "iteration": iteration, "version": version,
                "dims": [CANDIDATE_DIM, POPULATION_DIM], "meta": meta}, path)
    index = [e for e in read_index(directory) if e["version"] != version]
    index.append({"version": version, "name": version_name(version), "iteration": iteration, "file": path.name,
                  "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "mean_reward": meta.get("mean_reward")})
    index.sort(key=lambda e: e["version"])
    _index_path(directory).write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path


def load_checkpoint(directory: Path | None = None, which: str = "latest") -> tuple[CreatorNet, dict]:
    """Returns (net, meta). meta has version, version_name, iteration, optimizer_state and the saved meta."""
    directory = Path(directory or default_checkpoint_dir())
    index = read_index(directory)
    available = [e["name"] for e in index]
    if which == "latest":
        if not index:
            raise FileNotFoundError(f"no Creator checkpoints in {directory}; run `prometheus train-creator` first")
        entry = index[-1]
    else:
        m = re.fullmatch(r"v(\d+)", which)
        entry = next((e for e in index if m and e["version"] == int(m.group(1))), None)
        if entry is None:
            raise FileNotFoundError(f"checkpoint {which!r} not found in {directory}; available: "
                                    f"{', '.join(available) or 'none'}")
    blob = torch.load(directory / entry["file"], map_location="cpu", weights_only=False)
    dims = blob.get("dims")
    if dims != [CANDIDATE_DIM, POPULATION_DIM]:
        raise ValueError(f"checkpoint built for feature dims {dims[0]}/{dims[1]}, "
                         f"code has {CANDIDATE_DIM}/{POPULATION_DIM}")
    net = CreatorNet()
    net.load_state_dict(blob["state_dict"])
    meta = dict(blob.get("meta", {}))
    meta.update({"version": blob["version"], "version_name": version_name(blob["version"]),
                 "iteration": blob["iteration"], "optimizer_state": blob.get("optimizer")})
    return net, meta
