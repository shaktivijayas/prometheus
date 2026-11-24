import json
import subprocess
import sys
from pathlib import Path

from prometheus.compare import compare
from prometheus.runner import RunConfig

TINY_FLAGS = ["--generations", "2", "--population", "4", "--elite", "1", "--episodes-per-lesson", "1",
              "--eval-worlds", "4", "--workers", "1"]


def cli(*args, cwd):
    return subprocess.run([sys.executable, "-m", "prometheus.cli", *args], cwd=cwd, capture_output=True, text=True)


def test_compare_shapes(tmp_path):
    base = RunConfig(generations=2, population=4, elite=1, episodes_per_lesson=1, eval_worlds=4, workers=1)
    out = compare(["heuristic", "random"], [0], base, tmp_path, "t")
    data = json.loads((tmp_path / "comparisons" / "t.json").read_text())
    assert data == out
    for arm in ("heuristic", "random"):
        a = data["arms"][arm]
        assert len(a["best_mean"]) == len(a["best_std"]) == len(a["mean_mean"]) == 2
        assert a["best_std"] == [0.0, 0.0]
        assert (tmp_path / "runs" / a["runs"][0] / "run.json").exists()


def test_compare_std_across_seeds(tmp_path):
    base = RunConfig(generations=2, population=4, elite=1, episodes_per_lesson=1, eval_worlds=4, workers=1)
    data = compare(["random"], [0, 1], base, tmp_path, "s")
    assert len(data["arms"]["random"]["runs"]) == 2


def test_cli_run(tmp_path):
    out = tmp_path / "r"
    p = cli("run", "--creator", "heuristic", *TINY_FLAGS, "--out", str(out), cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert (out / "run.json").exists()
    assert "Verified on" in p.stdout


def test_cli_compare(tmp_path):
    p = cli("compare", "--creators", "heuristic,random", "--seeds", "0", *TINY_FLAGS, "--out", str(tmp_path),
            "--name", "c", cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert (Path(tmp_path) / "comparisons" / "c.json").exists()
    assert "heuristic" in p.stdout and "random" in p.stdout


def test_cli_unknown_creator(tmp_path):
    p = cli("run", "--creator", "nope", *TINY_FLAGS, cwd=tmp_path)
    assert p.returncode == 2
    assert "heuristic" in p.stderr and "random" in p.stderr


def test_compare_reports_verified_numbers(tmp_path):
    base = RunConfig(generations=2, population=4, elite=1, episodes_per_lesson=1, eval_worlds=4, workers=1,
                     verify_replicates=2)
    arm = compare(["clone"], [0, 1], base, tmp_path, "v")["arms"]["clone"]
    for key in ("verified_final_mean", "verified_final_std", "verified_gen0_mean", "verified_gain_mean"):
        assert isinstance(arm[key], float)
