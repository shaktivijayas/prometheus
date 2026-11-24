from dataclasses import replace

from prometheus.benchmark import DEFAULT_CURRICULUM
from prometheus.creators.base import ChildSpec
from prometheus.genome import Genome
from prometheus.registry import EventLog, Registry, specialization_for


def spec(genome=None, parent=None, specialization=""):
    return ChildSpec(genome=genome or Genome.default(), parent_id=parent, inherit_beliefs=False,
                     curriculum=DEFAULT_CURRICULUM, reason="test", specialization=specialization)


def fixed_registry():
    return Registry(clock=lambda: "2026-11-24T00:00:00Z")


def test_ids_and_names_increment_per_prefix():
    reg = fixed_registry()
    a = reg.create(spec(), 0)
    b = reg.create(spec(), 0)
    c = reg.create(spec(specialization="SCOUT"), 1)
    assert (a.id, b.id, c.id) == ("E0001", "E0002", "E0003")
    assert (a.name, b.name, c.name) == ("GENERALIST-001", "GENERALIST-002", "SCOUT-001")
    assert c.generation == 1 and a.created_at == "2026-11-24T00:00:00Z"


def test_record_copies_spec_fields():
    reg = fixed_registry()
    parent = reg.create(spec(), 0)
    child = reg.create(ChildSpec(Genome.default(), parent.id, True, ("easy",) * 4, "because",
                                 predicted_change=0.12), 1)
    assert child.parent_id == parent.id and child.inherit_beliefs
    assert child.curriculum == ["easy"] * 4 and child.reason == "because"
    assert child.predicted_change == 0.12
    assert child.genome == Genome.default().to_dict()


def test_retire():
    reg = fixed_registry()
    a, b = reg.create(spec(), 0), reg.create(spec(), 0)
    reg.retire(a.id, 3)
    assert reg.get(a.id).status == "retired" and reg.get(a.id).retired_at_generation == 3
    assert [r.id for r in reg.active()] == [b.id]
    assert len(reg.all()) == 2


def test_specialization_for():
    g = Genome.default()
    assert specialization_for(g) == "GENERALIST"
    assert specialization_for(replace(g, perception_radius=5)) == "SCOUT"
    assert specialization_for(replace(g, risk_aversion=3.0)) == "SENTINEL"
    assert specialization_for(replace(g, surprise_threshold=0.1)) == "ADAPTER"
    assert specialization_for(replace(g, curiosity_bonus=2.0)) == "EXPLORER"
    assert specialization_for(replace(g, planning_horizon=12)) == "NAVIGATOR"
    assert specialization_for(replace(g, learning_rate=1.0)) == "LEARNER"


def test_event_log_sequence():
    log = EventLog()
    log.emit("created", 0, "E0001", reason="x")
    log.emit("generation_complete", 0, best=1.0)
    assert [e["seq"] for e in log.events] == [1, 2]
    assert log.events[0] == {"seq": 1, "type": "created", "generation": 0, "entity_id": "E0001", "data": {"reason": "x"}}
    assert log.events[1]["entity_id"] is None


def test_to_dict_is_json_ready():
    import json
    reg = fixed_registry()
    json.dumps(reg.create(spec(), 0).to_dict())
