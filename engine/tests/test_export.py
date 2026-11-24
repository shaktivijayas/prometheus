import json

from prometheus.export import export_dashboard
from prometheus.output import write_run
from prometheus.runner import RunConfig, run_evolution

TINY = dict(generations=2, population=4, elite=1, episodes_per_lesson=1, eval_worlds=4, workers=1, replicates=1,
            verify_replicates=2)


def strict(path):
    return json.loads(path.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def make_run(tmp_path, name="final_random_s0", creator="random"):
    res = run_evolution(RunConfig(creator=creator, **TINY), clock=lambda: "t")
    write_run(res, tmp_path / "runs" / name)
    return tmp_path / "runs" / name


def test_export_run_and_index(tmp_path):
    run = make_run(tmp_path)
    comp = tmp_path / "comparisons" / "c.json"
    comp.parent.mkdir()
    comp.write_text(json.dumps({"name": "c", "creators": ["random"], "seeds": [0], "arms": {}}))
    out = tmp_path / "data"
    index = export_dashboard(out, [run], [comp], None)
    assert strict(out / "index.json") == index
    entry = index["runs"][0]
    run_json = strict(run / "run.json")
    assert entry["name"] == "final_random_s0" and entry["creator"] == "random" and entry["seed"] == 0
    assert entry["verification"] == run_json["verification"] and entry["summary"] == run_json["summary"]
    assert entry["generations"] == 2 and entry["children"] == len(strict(run / "entities.json"))
    events = strict(out / "runs" / "final_random_s0" / "events.json")
    assert isinstance(events, list) and events[0]["seq"] == 1
    replays = strict(out / "runs" / "final_random_s0" / "replays" / "index.json")
    assert sorted(r["file"] for r in replays) == sorted(p.name for p in (run / "replays").glob("*.json"))
    assert index["comparisons"] == [{"name": "c", "file": "comparisons/c.json", "creators": ["random"], "seeds": [0]}]
    assert index["creator"] == {"progress": False, "history": False}


def test_export_creator_data(tmp_path):
    creator = tmp_path / "creator"
    creator.mkdir()
    (creator / "progress.json").write_text("[]")
    (creator / "training_history.json").write_text("[]")
    index = export_dashboard(tmp_path / "data", [], [], creator)
    assert index["creator"] == {"progress": True, "history": True}
    assert (tmp_path / "data" / "creator" / "progress.json").exists()
