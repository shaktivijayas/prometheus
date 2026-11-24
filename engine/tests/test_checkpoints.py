import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from prometheus.creators.checkpoints import load_checkpoint, save_checkpoint
from prometheus.creators.features import CANDIDATE_DIM, POPULATION_DIM
from prometheus.creators.neural import CreatorNet

TINY_TRAIN = ["--iterations", "1", "--runs-per-iteration", "2", "--generations", "2", "--population", "4",
              "--elite", "1", "--episodes-per-lesson", "1", "--eval-worlds", "4", "--checkpoint-every", "1",
              "--workers", "1", "--replicates", "1"]
TINY_RUN = ["--generations", "2", "--population", "4", "--elite", "1", "--episodes-per-lesson", "1",
            "--eval-worlds", "4", "--workers", "1", "--replicates", "1", "--verify-replicates", "2"]


def cli(*args, cwd, ckpt):
    env = {**os.environ, "PROMETHEUS_CHECKPOINTS": str(ckpt)}
    return subprocess.run([sys.executable, "-m", "prometheus.cli", *args], cwd=cwd, capture_output=True, text=True,
                          env=env)


def test_roundtrip_identical_outputs(tmp_path):
    torch.manual_seed(3)
    net = CreatorNet()
    save_checkpoint(tmp_path, 0, net, None, 0, {})
    loaded, meta = load_checkpoint(tmp_path, "v000")
    rng = np.random.default_rng(0)
    cand, pop = rng.random((5, CANDIDATE_DIM)), rng.random(POPULATION_DIM)
    a = net.sample(cand, pop, torch.Generator().manual_seed(1))
    b = loaded.sample(cand, pop, torch.Generator().manual_seed(1))
    assert a.parent_index == b.parent_index and np.allclose(a.edit, b.edit) and a.log_prob == pytest.approx(b.log_prob)
    assert meta["version_name"] == "v000" and meta["iteration"] == 0


def test_missing_version_lists_available(tmp_path):
    save_checkpoint(tmp_path, 0, CreatorNet(), None, 0, {})
    with pytest.raises(FileNotFoundError, match="v000"):
        load_checkpoint(tmp_path, "v999")
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "empty", "latest")


def test_dimension_mismatch_is_clear(tmp_path):
    path = save_checkpoint(tmp_path, 0, CreatorNet(), None, 0, {})
    blob = torch.load(path, weights_only=False)
    blob["dims"] = [1, 2]
    torch.save(blob, path)
    with pytest.raises(ValueError, match="feature dims 1/2"):
        load_checkpoint(tmp_path, "v000")


def test_cli_train_then_run_then_progress(tmp_path):
    ckpt = tmp_path / "ckpt"
    p = cli("train-creator", *TINY_TRAIN, cwd=tmp_path, ckpt=ckpt)
    assert p.returncode == 0, p.stderr
    assert (ckpt / "creator_v000.pt").exists() and (ckpt / "creator_v001.pt").exists()
    assert (ckpt / "training_history.json").exists()

    bad = cli("run", "--creator", "neural:v999", *TINY_RUN, cwd=tmp_path, ckpt=ckpt)
    assert bad.returncode == 2 and "v999" in bad.stderr

    ok = cli("run", "--creator", "neural:latest", *TINY_RUN, "--out", str(tmp_path / "r"), cwd=tmp_path, ckpt=ckpt)
    assert ok.returncode == 0, ok.stderr

    prog = cli("creator-progress", "--versions", "all", "--seeds", "0", *TINY_RUN, "--out", str(tmp_path),
               cwd=tmp_path, ckpt=ckpt)
    assert prog.returncode == 0, prog.stderr
    rows = json.loads((tmp_path / "creator" / "progress.json").read_text())
    assert [r["version"] for r in rows] == ["v000", "v001"]
    for r in rows:
        assert {"iteration", "verified_final_mean", "verified_gain_mean", "prediction_mae"} <= set(r)
    assert (tmp_path / "creator" / "training_history.json").exists()
