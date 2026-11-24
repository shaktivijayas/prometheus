import hashlib
import json
import math

import numpy as np
import pytest
import torch

from prometheus.benchmark import CURRICULUM_PRESETS, DEFAULT_CURRICULUM, TRAIN_UNIVERSE_BASE
from prometheus.creators.base import ChildSpec
from prometheus.creators.checkpoints import load_checkpoint
from prometheus.creators.features import CANDIDATE_DIM, POPULATION_DIM
from prometheus.creators.neural import CreatorNet
from prometheus.creators.training import TrainConfig, decision_rewards, ppo_update, train
from prometheus.genome import GENE_NAMES, Genome
from prometheus.registry import Registry

SHIFT = CURRICULUM_PRESETS.index("shift")
RISK = GENE_NAMES.index("risk_aversion")


def toy_inputs(n=6):
    rng = np.random.default_rng(0)
    return rng.random((n, CANDIDATE_DIM)).astype(np.float32), rng.random(POPULATION_DIM).astype(np.float32)


def sample_batch(net, k, seed):
    cand, pop = toy_inputs()
    g = torch.Generator().manual_seed(seed)
    return [net.sample(cand, pop, g) for _ in range(k)]


def test_ppo_learns_toy_bandit():
    """Reward only when the first lesson is a rule-shift lesson: the teaching head must learn that."""
    torch.manual_seed(0)
    net = CreatorNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    cfg = TrainConfig()
    before = np.mean([d.curriculum[0] == SHIFT for d in sample_batch(net, 400, 999)])
    for i in range(60):
        ds = sample_batch(net, 64, i)
        ppo_update(net, opt, ds, np.array([1.0 if d.curriculum[0] == SHIFT else 0.0 for d in ds]), cfg)
    after = np.mean([d.curriculum[0] == SHIFT for d in sample_batch(net, 400, 998)])
    assert before < 0.4 and after > 0.6, (before, after)


def test_ppo_learns_edit_direction():
    """Reward = how much risk aversion was raised: the design head must learn to raise it."""
    torch.manual_seed(1)
    net = CreatorNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    cfg = TrainConfig()
    for i in range(60):
        ds = sample_batch(net, 64, i)
        ppo_update(net, opt, ds, np.array([d.edit[RISK] for d in ds]) * 10, cfg)
    mean_edit = np.mean([d.edit[RISK] for d in sample_batch(net, 400, 997)])
    assert mean_edit > 0.05, mean_edit


def test_prediction_head_fits_rewards():
    torch.manual_seed(2)
    net = CreatorNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    ds = sample_batch(net, 128, 5)
    rewards = np.array([2.0 * d.edit[0] + (0.5 if d.inherit else -0.5) for d in ds])
    first = ppo_update(net, opt, ds, rewards, TrainConfig())["value_loss"]
    for _ in range(80):
        last = ppo_update(net, opt, ds, rewards, TrainConfig(entropy_coef=0.0))["value_loss"]
    assert last < 0.5 * first, (first, last)


def _rec(reg, genome, gen, parent=None, fitness=1.0, scale=2.0):
    r = reg.create(ChildSpec(genome, parent, False, DEFAULT_CURRICULUM, "x",
                             decision={"scale": scale} if parent else {}), gen)
    r.metrics = {"fitness": fitness}
    return r


def test_rewards_use_parent_delta_and_penalty():
    reg = Registry(clock=lambda: "t")
    g = Genome.default()
    far = Genome.from_vector(np.clip(g.to_vector() + 0.3, 0, 1))
    p = _rec(reg, g, 0, fitness=2.0)
    a = _rec(reg, far, 1, parent=p.id, fitness=3.0)   # improves by 1.0 -> 0.5 normalized
    b = _rec(reg, g, 1, parent=p.id, fitness=1.0)     # worse by 1.0 and an exact duplicate of p
    r = decision_rewards(reg, TrainConfig(duplicate_penalty=0.1))
    assert r == pytest.approx([0.5, -0.5 - 0.1])


TINY = dict(iterations=2, runs_per_iteration=2, generations=2, population=4, elite=1, episodes_per_lesson=1,
            eval_worlds=4, checkpoint_every=1, workers=1, replicates=1)


def test_train_tiny_end_to_end(tmp_path):
    universes = []
    train(TrainConfig(**TINY), tmp_path / "ckpt", tmp_path / "history.json", resume=False,
          on_run=lambda evo: universes.append(evo.config.universe_seed()))
    for v in ("v000", "v001", "v002"):
        assert (tmp_path / "ckpt" / f"creator_{v}.pt").exists()
    hist = json.loads((tmp_path / "history.json").read_text())
    assert [h["iteration"] for h in hist] == [1, 2]
    for h in hist:
        for k in ("mean_reward", "policy_loss", "value_loss", "entropy", "prediction_mae", "final_mean_fitness"):
            assert math.isfinite(h[k]), k
    assert universes and all(u >= TRAIN_UNIVERSE_BASE for u in universes)
    index = json.loads((tmp_path / "ckpt" / "index.json").read_text())
    assert [e["version"] for e in index] == [0, 1, 2]


def test_resume_continues(tmp_path):
    cfg = TrainConfig(**{**TINY, "iterations": 1})
    train(cfg, tmp_path / "c", tmp_path / "h.json", resume=False)
    v0 = hashlib.sha256((tmp_path / "c" / "creator_v000.pt").read_bytes()).hexdigest()
    train(cfg, tmp_path / "c", tmp_path / "h.json", resume=True)
    assert [h["iteration"] for h in json.loads((tmp_path / "h.json").read_text())] == [1, 2]
    assert hashlib.sha256((tmp_path / "c" / "creator_v000.pt").read_bytes()).hexdigest() == v0
    net, meta = load_checkpoint(tmp_path / "c", "latest")
    assert meta["iteration"] == 2


def test_fresh_training_refuses_existing_checkpoints(tmp_path):
    """Final review #3: a fresh run must not mix with an older run's checkpoints."""
    cfg = TrainConfig(**{**TINY, "iterations": 1})
    train(cfg, tmp_path / "c", tmp_path / "h.json", resume=False)
    with pytest.raises(FileExistsError):
        train(cfg, tmp_path / "c", tmp_path / "h.json", resume=False)
