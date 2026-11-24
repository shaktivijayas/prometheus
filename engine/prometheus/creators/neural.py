"""The Creator: a neural network we train ourselves (reinforcement learning).

For each new child it decides, in order:
  1. which existing child to build on (parent),
  2. how to change each gene (design),
  3. whether to pass on the parent's learned knowledge (inheritance),
  4. which 4 lessons to teach it (curriculum),
and it predicts how much better the child will be than its parent (also the RL value baseline).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..benchmark import CURRICULUM_PRESETS, DEFAULT_CURRICULUM, LESSONS
from ..genome import GENE_NAMES, Genome
from .base import ChildSpec, GenerationReport, curriculum_text, describe_edits
from .features import CANDIDATE_DIM, POPULATION_DIM, candidate_features, population_features
from .random_mutation import RandomMutationCreator

EDIT_SCALE = 0.25
HIDDEN = 128
N_GENES = len(GENE_NAMES)
N_PRESETS = len(CURRICULUM_PRESETS)
DEFAULT_CURRICULUM_INDEX = [CURRICULUM_PRESETS.index(p) for p in DEFAULT_CURRICULUM]
MIN_PREDICTION_SCALE = 0.25


@dataclass
class Decision:
    """Everything needed to re-score one Creator decision during training."""
    cand: np.ndarray          # [N, CANDIDATE_DIM]
    pop: np.ndarray           # [POPULATION_DIM]
    parent_index: int
    edit: np.ndarray          # [N_GENES] normalized gene changes
    inherit: int
    curriculum: np.ndarray    # [LESSONS] preset indices
    log_prob: float
    prediction: float         # normalized predicted fitness change


class CreatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(CANDIDATE_DIM + POPULATION_DIM, HIDDEN), nn.Tanh(),
                                     nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.parent_head = nn.Linear(HIDDEN, 1)
        self.edit_mean = nn.Linear(HIDDEN, N_GENES)
        self.edit_log_std = nn.Parameter(torch.full((N_GENES,), math.log(0.08)))
        self.inherit_head = nn.Linear(HIDDEN, 1)
        self.teach_head = nn.Sequential(nn.Linear(HIDDEN + N_GENES, 64), nn.Tanh(),
                                        nn.Linear(64, LESSONS * N_PRESETS))
        self.predict_head = nn.Sequential(nn.Linear(HIDDEN + N_GENES + 1 + LESSONS * N_PRESETS, 64), nn.Tanh(),
                                          nn.Linear(64, 1))

    # ----- pieces -------------------------------------------------------------
    def encode(self, cand: torch.Tensor, pop: torch.Tensor) -> torch.Tensor:
        """cand [B, N, C], pop [B, P] -> h [B, N, H]."""
        pop_b = pop.unsqueeze(1).expand(-1, cand.shape[1], -1)
        return self.encoder(torch.cat([cand, pop_b], dim=-1))

    def _edit_dist(self, hp: torch.Tensor) -> torch.distributions.Normal:
        mean = EDIT_SCALE * torch.tanh(self.edit_mean(hp))
        return torch.distributions.Normal(mean, self.edit_log_std.exp().expand_as(mean))

    def _teach_logits(self, hp: torch.Tensor, edit: torch.Tensor) -> torch.Tensor:
        return self.teach_head(torch.cat([hp, edit], dim=-1)).view(-1, LESSONS, N_PRESETS)

    def _predict(self, hp, edit, inherit, curriculum) -> torch.Tensor:
        onehot = nn.functional.one_hot(curriculum, N_PRESETS).float().view(curriculum.shape[0], -1)
        return self.predict_head(torch.cat([hp, edit, inherit.float().unsqueeze(-1), onehot], dim=-1)).squeeze(-1)

    # ----- acting -------------------------------------------------------------
    @torch.no_grad()
    def sample(self, cand: np.ndarray, pop: np.ndarray, generator: torch.Generator, teach: bool = True) -> Decision:
        c = torch.as_tensor(cand, dtype=torch.float32).unsqueeze(0)
        p = torch.as_tensor(pop, dtype=torch.float32).unsqueeze(0)
        h = self.encode(c, p)[0]                                       # [N, H]
        parent_dist = torch.distributions.Categorical(logits=self.parent_head(h).squeeze(-1))
        idx = int(torch.multinomial(parent_dist.probs, 1, generator=generator))
        hp = h[idx:idx + 1]                                            # [1, H]
        edit_dist = self._edit_dist(hp)
        noise = torch.randn(edit_dist.mean.shape, generator=generator)
        edit = edit_dist.mean + edit_dist.stddev * noise
        inherit_p = torch.sigmoid(self.inherit_head(hp)).squeeze(-1)
        inherit = (torch.rand(1, generator=generator) < inherit_p).long()
        if teach:
            teach_dist = torch.distributions.Categorical(logits=self._teach_logits(hp, edit))
            probs = teach_dist.probs[0]
            curriculum = torch.stack([torch.multinomial(probs[i], 1, generator=generator)[0]
                                      for i in range(LESSONS)]).unsqueeze(0)
        else:
            curriculum = torch.tensor([DEFAULT_CURRICULUM_INDEX])
        log_prob = (parent_dist.log_prob(torch.tensor(idx)) + edit_dist.log_prob(edit).sum()
                    + torch.distributions.Bernoulli(probs=inherit_p).log_prob(inherit.float()).sum())
        if teach:
            log_prob = log_prob + teach_dist.log_prob(curriculum).sum()
        prediction = self._predict(hp, edit, inherit, curriculum)
        return Decision(np.asarray(cand, dtype=np.float32), np.asarray(pop, dtype=np.float32), idx,
                        edit[0].numpy().astype(np.float64), int(inherit[0]), curriculum[0].numpy().copy(),
                        float(log_prob), float(prediction[0]))

    # ----- learning -----------------------------------------------------------
    def evaluate(self, decisions: list[Decision], teach: bool = True):
        """Re-score past decisions under the current weights: (log_probs, entropy, predictions), each [B]."""
        out_lp, out_ent, out_pred = [None] * len(decisions), [None] * len(decisions), [None] * len(decisions)
        groups: dict[int, list[int]] = {}
        for i, d in enumerate(decisions):
            groups.setdefault(d.cand.shape[0], []).append(i)
        for _, idxs in groups.items():
            cand = torch.as_tensor(np.stack([decisions[i].cand for i in idxs]), dtype=torch.float32)
            pop = torch.as_tensor(np.stack([decisions[i].pop for i in idxs]), dtype=torch.float32)
            parent = torch.tensor([decisions[i].parent_index for i in idxs])
            edit = torch.as_tensor(np.stack([decisions[i].edit for i in idxs]), dtype=torch.float32)
            inherit = torch.tensor([decisions[i].inherit for i in idxs])
            curriculum = torch.as_tensor(np.stack([decisions[i].curriculum for i in idxs]), dtype=torch.long)
            h = self.encode(cand, pop)                                  # [B, N, H]
            parent_dist = torch.distributions.Categorical(logits=self.parent_head(h).squeeze(-1))
            hp = h[torch.arange(len(idxs)), parent]                     # [B, H]
            edit_dist = self._edit_dist(hp)
            inherit_dist = torch.distributions.Bernoulli(logits=self.inherit_head(hp).squeeze(-1))
            lp = parent_dist.log_prob(parent) + edit_dist.log_prob(edit).sum(-1) + inherit_dist.log_prob(inherit.float())
            ent = parent_dist.entropy() + edit_dist.entropy().sum(-1) + inherit_dist.entropy()
            if teach:
                teach_dist = torch.distributions.Categorical(logits=self._teach_logits(hp, edit))
                lp = lp + teach_dist.log_prob(curriculum).sum(-1)
                ent = ent + teach_dist.entropy().sum(-1)
            pred = self._predict(hp, edit, inherit, curriculum)
            for j, i in enumerate(idxs):
                out_lp[i], out_ent[i], out_pred[i] = lp[j], ent[j], pred[j]
        return torch.stack(out_lp), torch.stack(out_ent), torch.stack(out_pred)


def _dominant_failure(m: dict) -> str:
    deaths = m.get("death_causes", {}) or {}
    hazard, starve = deaths.get("hazard", 0.0) or 0.0, deaths.get("starvation", 0.0) or 0.0
    adapt = m.get("adaptation_ratio", 1.0)
    if hazard >= 0.3 and hazard >= starve:
        return f"died to hazards in {round(100 * hazard)}% of worlds"
    if adapt is not None and adapt < 0.6:
        return f"adapted poorly after shifts ({adapt:.2f})"
    if starve >= 0.3:
        return f"starved in {round(100 * starve)}% of worlds"
    return "strong all-round"


def _top_edits(before: Genome, after: Genome, edit: np.ndarray, k: int = 3) -> str:
    """The k largest intended edits (by size), described as the actual change to the gene."""
    order = np.argsort(-np.abs(edit))[:k]
    full = dict(part.rsplit(" ", 1) for part in describe_edits(before, after).split(", ") if " " in part)
    parts = []
    for i in order:
        name = GENE_NAMES[int(i)]
        if name in full:
            parts.append(f"{name} {full[name]}")
        else:  # the gene did not change: clipped at its bound, or too small to move a whole-number gene
            parts.append(f"{name} unchanged (edit {edit[i]:+.2f} had no effect)")
    return ", ".join(parts)


class NeuralCreator:
    def __init__(self, net: CreatorNet, teach: bool = True, name: str = "neural", version: str = "fresh"):
        self.net = net
        self.teach = teach
        self.name = name
        self.version = version
        self.decisions: list[Decision] = []

    def seed_population(self, n: int, rng: np.random.Generator) -> list[ChildSpec]:
        # identical to the random arm's generation 0, so arms differ only in Creator decisions
        return RandomMutationCreator().seed_population(n, rng)

    def next_generation(self, report: GenerationReport, slots: int, rng: np.random.Generator) -> list[ChildSpec]:
        pop = list(report.population)
        cand = np.stack([candidate_features(r, pop) for r in pop])
        popf = population_features(pop, report.generation, report.total_generations)
        generator = torch.Generator().manual_seed(int(rng.integers(2**62)))
        scale = prediction_scale(report.history)
        self.net.eval()
        specs = []
        for _ in range(slots):
            d = self.net.sample(cand, popf, generator, teach=self.teach)
            self.decisions.append(d)
            parent = pop[d.parent_index]
            before = parent.genome_obj()
            after = Genome.from_vector(np.clip(before.to_vector() + d.edit, 0.0, 1.0))
            curriculum = tuple(CURRICULUM_PRESETS[int(i)] for i in d.curriculum)
            predicted = d.prediction * scale
            reason = (f"{parent.name} ({_dominant_failure(parent.metrics or {})}): "
                      f"{_top_edits(before, after, d.edit)}; "
                      f"{'inherits knowledge' if d.inherit else 'starts fresh'}; "
                      f"curriculum {curriculum_text(curriculum)}; predicts {predicted:+.2f} fitness")
            specs.append(ChildSpec(
                genome=after, parent_id=parent.id, inherit_beliefs=bool(d.inherit), curriculum=curriculum,
                reason=reason, predicted_change=predicted,
                decision={"parent_index": d.parent_index, "edit": d.edit.tolist(), "inherit": d.inherit,
                          "curriculum": d.curriculum.tolist(), "log_prob": d.log_prob, "prediction": d.prediction,
                          "scale": scale, "creator_version": self.version, "teach": self.teach},
            ))
        return specs


def prediction_scale(history: list[dict]) -> float:
    """Fitness changes are normalized by the run's generation-0 mean fitness (scale-free across universes)."""
    if history:
        return max(float(history[0].get("mean", 0.0)), MIN_PREDICTION_SCALE)
    return MIN_PREDICTION_SCALE


def make_neural(arg: str) -> NeuralCreator:
    """Factory for "neural[:<which>][:noteach]" where which is "", "fresh", "latest" or "vNNN"."""
    which, _, flag = arg.partition(":")
    if flag not in ("", "noteach"):
        raise ValueError(f"unknown neural creator flag {flag!r}; use ':noteach'")
    teach = flag != "noteach"
    name = "neural" + (f":{which}" if which else "") + (":noteach" if not teach else "")
    if which in ("", "fresh"):
        torch.manual_seed(0)
        return NeuralCreator(CreatorNet(), teach=teach, name=name, version="fresh")
    from .checkpoints import load_checkpoint
    net, meta = load_checkpoint(which=which)
    return NeuralCreator(net, teach=teach, name=name, version=meta.get("version_name", which))
