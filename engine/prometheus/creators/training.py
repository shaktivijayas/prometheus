"""Training the Creator with reinforcement learning (PPO).

Each iteration the Creator raises many short populations in lockstep, in training universes it has never seen
(randomized laws, food/hazard economy and shift timing). Every child it designs earns a reward: how much better the
child scored than its parent (normalized by the run's generation-0 mean fitness), minus a small penalty for
near-duplicate designs. The prediction head learns to forecast that reward and serves as the PPO baseline.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..benchmark import TRAIN_UNIVERSE_BASE
from ..genome import Genome
from ..registry import Registry
from ..runner import Evolution, RunConfig, _Executor, run_lockstep
from ..world import WorldVariant
from .checkpoints import load_checkpoint, read_index, save_checkpoint
from .neural import CreatorNet, Decision, NeuralCreator

DUPLICATE_DISTANCE = 0.05
MINIBATCH = 256


@dataclass
class TrainConfig:
    iterations: int = 100
    runs_per_iteration: int = 12
    generations: int = 6
    population: int = 8
    elite: int = 2
    episodes_per_lesson: int = 3
    eval_worlds: int = 12
    lr: float = 3e-4
    ppo_epochs: int = 4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    duplicate_penalty: float = 0.1
    checkpoint_every: int = 10
    seed: int = 0
    workers: int = 0
    replicates: int = 2


# ----- rewards -------------------------------------------------------------------------------------------------
def _active_at(registry: Registry, generation: int):
    return [r for r in registry.all() if r.generation <= generation
            and (r.retired_at_generation is None or r.retired_at_generation > generation)]


def decision_rewards(registry: Registry, cfg: TrainConfig) -> list[float]:
    """One reward per Creator-designed child, in creation order (the order the Creator made its decisions)."""
    rewards = []
    for rec in registry.all():
        if rec.generation == 0 or not rec.decision or rec.parent_id is None:
            continue
        parent = registry.get(rec.parent_id)
        scale = float(rec.decision.get("scale", 1.0)) or 1.0
        reward = (rec.fitness - parent.fitness) / scale
        mine = Genome.from_dict(rec.genome).to_vector()
        for other in _active_at(registry, rec.generation):
            if other.id != rec.id and np.linalg.norm(mine - Genome.from_dict(other.genome).to_vector()) < DUPLICATE_DISTANCE:
                reward -= cfg.duplicate_penalty
                break
        rewards.append(float(reward))
    return rewards


# ----- PPO ------------------------------------------------------------------------------------------------------
def ppo_update(net: CreatorNet, optimizer: torch.optim.Optimizer, decisions: list[Decision], rewards,
               cfg: TrainConfig, teach: bool = True) -> dict:
    net.train()
    R = torch.as_tensor(np.asarray(rewards, dtype=np.float32))
    old_lp = torch.tensor([d.log_prob for d in decisions], dtype=torch.float32)
    old_pred = torch.tensor([d.prediction for d in decisions], dtype=torch.float32)
    adv = R - old_pred
    if len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(decisions)
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    batches = 0
    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, MINIBATCH):
            mb = perm[start:start + MINIBATCH]
            lp, ent, pred = net.evaluate([decisions[int(i)] for i in mb], teach=teach)
            ratio = torch.exp(lp - old_lp[mb])
            a = adv[mb]
            policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            value_loss = torch.nn.functional.mse_loss(pred, R[mb])
            entropy = ent.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            sums["policy_loss"] += policy_loss.item()
            sums["value_loss"] += value_loss.item()
            sums["entropy"] += entropy.item()
            sums["approx_kl"] += (old_lp[mb] - lp).mean().item()
            batches += 1
    return {k: v / max(batches, 1) for k, v in sums.items()}


# ----- experience collection ----------------------------------------------------------------------------------
def collect(net: CreatorNet, cfg: TrainConfig, iteration: int, executor: _Executor,
            on_run: Callable[[Evolution], None] | None = None) -> tuple[list[Decision], list[float], dict]:
    rng = np.random.default_rng([cfg.seed, iteration])
    pairs = []
    for i in range(cfg.runs_per_iteration):
        universe = TRAIN_UNIVERSE_BASE + cfg.seed * 10_000_000 + (iteration - 1) * cfg.runs_per_iteration + i
        rc = RunConfig(creator="neural", generations=cfg.generations, population=cfg.population, elite=cfg.elite,
                       seed=universe, episodes_per_lesson=cfg.episodes_per_lesson, eval_worlds=cfg.eval_worlds,
                       workers=cfg.workers, record_replays=False, replicates=cfg.replicates, universe=universe,
                       verify_replicates=0, variant=WorldVariant.random(rng))
        creator = NeuralCreator(net, teach=True, name="neural:training")
        pairs.append((Evolution(rc, creator, clock=lambda: "", allow_training_universe=True), creator))
    run_lockstep([e for e, _ in pairs], executor)

    decisions, rewards, abs_err = [], [], []
    gen0_mean, final_mean, final_best = [], [], []
    for evo, creator in pairs:
        if on_run:
            on_run(evo)
        r = decision_rewards(evo.registry, cfg)
        if len(r) != len(creator.decisions):
            raise RuntimeError(f"reward/decision mismatch: {len(r)} vs {len(creator.decisions)}")
        decisions += creator.decisions
        rewards += r
        for rec in evo.registry.all():
            if rec.predicted_change is not None and rec.actual_change is not None:
                abs_err.append(abs(rec.predicted_change - rec.actual_change))
        gen0_mean.append(evo.generations[0]["mean"])
        final_mean.append(evo.generations[-1]["mean"])
        final_best.append(evo.generations[-1]["best"])
    stats = {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "decisions": len(decisions),
        "gen0_mean_fitness": float(np.mean(gen0_mean)),
        "final_mean_fitness": float(np.mean(final_mean)),
        "final_best_fitness": float(np.mean(final_best)),
        "improvement": float(np.mean(np.array(final_mean) - np.array(gen0_mean))),
        "prediction_mae": float(np.mean(abs_err)) if abs_err else 0.0,
    }
    return decisions, rewards, stats


# ----- the training loop ----------------------------------------------------------------------------------------
def train(cfg: TrainConfig, checkpoint_dir: Path, history_path: Path, resume: bool,
          on_run: Callable[[Evolution], None] | None = None, log: Callable[[dict], None] | None = None) -> None:
    checkpoint_dir, history_path = Path(checkpoint_dir), Path(history_path)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    history: list[dict] = []
    if resume:
        net, meta = load_checkpoint(checkpoint_dir, "latest")
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        if meta.get("optimizer_state"):
            optimizer.load_state_dict(meta["optimizer_state"])
        start_iteration, version = int(meta["iteration"]), int(meta["version"])
        if history_path.exists():
            history = [h for h in json.loads(history_path.read_text(encoding="utf-8"))
                       if h["iteration"] <= start_iteration]
    else:
        if read_index(checkpoint_dir):
            raise FileExistsError(f"{checkpoint_dir} already holds checkpoints from another training run; "
                                  "use --resume to continue it, or choose an empty --checkpoints directory")
        torch.manual_seed(cfg.seed)
        net = CreatorNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        start_iteration, version = 0, 0
        save_checkpoint(checkpoint_dir, 0, net, optimizer, 0, {"config": asdict(cfg), "note": "untrained"})

    executor = _Executor(cfg.workers)
    try:
        last = start_iteration + cfg.iterations
        for it in range(start_iteration + 1, last + 1):
            t0 = time.perf_counter()
            decisions, rewards, stats = collect(net, cfg, it, executor, on_run)
            torch.manual_seed(cfg.seed * 1_000_003 + it)
            losses = ppo_update(net, optimizer, decisions, rewards, cfg)
            row = {"iteration": it, "version": None, **stats, **losses, "seconds": round(time.perf_counter() - t0, 2)}
            if it % cfg.checkpoint_every == 0 or it == last:
                version += 1
                save_checkpoint(checkpoint_dir, version, net, optimizer, it,
                                {"config": asdict(cfg), "mean_reward": stats["mean_reward"]})
                row["version"] = version
            history.append({k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v)
                            for k, v in row.items()})
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(history, indent=1), encoding="utf-8")
            if log:
                log(row)
    finally:
        executor.close()
