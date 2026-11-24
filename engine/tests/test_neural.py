import numpy as np
import pytest
import torch

from prometheus.benchmark import CURRICULUM_PRESETS, DEFAULT_CURRICULUM
from prometheus.creators import get_creator
from prometheus.creators.base import ChildSpec, GenerationReport
from prometheus.creators.features import CANDIDATE_DIM, POPULATION_DIM
from prometheus.creators.neural import CreatorNet, NeuralCreator
from prometheus.creators.random_mutation import RandomMutationCreator
from prometheus.genome import GENE_NAMES, Genome
from prometheus.registry import Registry


def metrics(fitness):
    return {"fitness": fitness, "mean_score": fitness, "stable_score": fitness, "shift_score": fitness,
            "survival_rate": 0.4, "adaptation_ratio": 0.8, "learning_speed": 0.5, "mean_prediction_error": 3.0,
            "perception_cost": 30.0, "death_causes": {"hazard": 0.4, "starvation": 0.3, "post_shift": 0.1},
            "lesson_scores": [1.0, 1.5, 2.0, 2.0]}


def report(n=6, generation=2):
    reg = Registry(clock=lambda: "t")
    recs = []
    for i in range(n):
        r = reg.create(ChildSpec(Genome.random(np.random.default_rng(i)), None, False, DEFAULT_CURRICULUM, "x"), 0)
        r.metrics = metrics(1.0 + i)
        recs.append(r)
    return GenerationReport(generation, recs, [{"generation": 0, "best": 6.0, "mean": 3.5}], total_generations=10)


def creator(teach=True, seed=0):
    torch.manual_seed(seed)
    return NeuralCreator(CreatorNet(), teach=teach)


def test_sample_shapes_and_valid_spec():
    rep = report()
    specs = creator().next_generation(rep, 5, np.random.default_rng(0))
    ids = {r.id for r in rep.population}
    assert len(specs) == 5
    for s in specs:
        assert s.validated().genome == s.genome
        assert len(s.curriculum) == 4 and all(p in CURRICULUM_PRESETS for p in s.curriculum)
        assert s.parent_id in ids and s.predicted_change is not None
        assert {"parent_index", "edit", "inherit", "curriculum", "log_prob", "prediction"} <= set(s.decision)


def test_sampling_deterministic_given_seed():
    a = creator().next_generation(report(), 4, np.random.default_rng(3))
    b = creator().next_generation(report(), 4, np.random.default_rng(3))
    assert [(s.genome, s.curriculum, s.parent_id, s.inherit_beliefs, s.predicted_change) for s in a] == \
           [(s.genome, s.curriculum, s.parent_id, s.inherit_beliefs, s.predicted_change) for s in b]


def test_logprob_recompute_matches_sample():
    c = creator()
    c.next_generation(report(), 6, np.random.default_rng(1))
    log_probs, entropy, predictions = c.net.evaluate(c.decisions)
    assert torch.allclose(log_probs, torch.tensor([d.log_prob for d in c.decisions], dtype=torch.float32), atol=1e-5)
    assert torch.allclose(predictions, torch.tensor([d.prediction for d in c.decisions], dtype=torch.float32), atol=1e-5)
    assert (entropy > -100).all() and entropy.shape == (6,)


def test_noteach_forces_default_curriculum():
    c = creator(teach=False)
    specs = c.next_generation(report(), 6, np.random.default_rng(2))
    assert all(s.curriculum == DEFAULT_CURRICULUM for s in specs)
    lp_noteach, _, _ = c.net.evaluate(c.decisions, teach=False)
    lp_teach, _, _ = c.net.evaluate(c.decisions, teach=True)
    assert torch.allclose(lp_noteach, torch.tensor([d.log_prob for d in c.decisions], dtype=torch.float32), atol=1e-5)
    assert not torch.allclose(lp_noteach, lp_teach)  # teaching choices carry log-prob only when teaching


def test_generation0_matches_random_arm():
    a = creator().seed_population(8, np.random.default_rng(4))
    b = RandomMutationCreator().seed_population(8, np.random.default_rng(4))
    assert [(s.genome, s.curriculum) for s in a] == [(s.genome, s.curriculum) for s in b]


def test_child_genome_is_parent_plus_edit():
    rep = report()
    c = creator()
    spec = c.next_generation(rep, 1, np.random.default_rng(5))[0]
    parent = next(r for r in rep.population if r.id == spec.parent_id)
    expected = Genome.from_vector(np.clip(parent.genome_obj().to_vector() + np.array(spec.decision["edit"]), 0, 1))
    assert spec.genome == expected


def test_reason_mentions_real_edits():
    c = creator()
    spec = c.next_generation(report(), 1, np.random.default_rng(6))[0]
    edit = np.array(spec.decision["edit"])
    biggest = GENE_NAMES[int(np.argmax(np.abs(edit)))]
    assert biggest in spec.reason
    assert " > ".join(spec.curriculum) in spec.reason


def test_single_candidate_population():
    specs = creator().next_generation(report(n=1), 3, np.random.default_rng(7))
    assert len(specs) == 3


def test_feature_dims_wired():
    net = CreatorNet()
    assert net.encoder[0].in_features == CANDIDATE_DIM + POPULATION_DIM


def test_registered_as_neural():
    c = get_creator("neural")
    assert isinstance(c, NeuralCreator) and c.teach
    assert get_creator("neural::noteach").teach is False


def test_randomteach_gives_uniform_random_lessons():
    """Final review #4: a fair control for teaching keeps the learned designs but draws lessons at random."""
    from prometheus.creators.neural import make_neural
    c = make_neural("fresh:randomteach")
    assert c.name == "neural:fresh:randomteach"
    specs = c.next_generation(report(), 40, np.random.default_rng(3))
    lessons = [l for s in specs for l in s.curriculum]
    assert set(lessons) == set(CURRICULUM_PRESETS)            # all four presets appear
    assert len({s.curriculum for s in specs}) > 10           # not one fixed plan
    assert all(not s.decision["teach"] for s in specs)
