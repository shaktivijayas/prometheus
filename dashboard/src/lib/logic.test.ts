import { describe, expect, it } from "vitest";
import replayFixture from "./__fixtures__/replay.json";
import entitiesFixture from "./__fixtures__/entities.json";
import { frames } from "./replay";
import { buildForest, layoutForest } from "./tree";
import { curriculumPatterns, failureType, geneDiff, predictionPairs } from "./stats";
import type { Entity, Replay } from "./types";

const replay = replayFixture as unknown as Replay & { final_grid: number[][]; final_roles: string[] };
const entities = entitiesFixture as unknown as Entity[];

describe("replay reconstruction", () => {
  it("rebuilds the engine's exact final grid and laws from the trace alone", () => {
    const fs = frames(replay);
    expect(fs.length).toBe(replay.steps + 1);
    const last = fs[fs.length - 1];
    expect(last.grid).toEqual(replay.final_grid);
    expect(last.roles).toEqual(replay.final_roles);
  });

  it("marks the step where the laws change and keeps the old laws before it", () => {
    const fs = frames(replay);
    const shiftAt = fs.findIndex((f) => f.event === "shift");
    expect(shiftAt).toBeGreaterThan(0);
    expect(fs[shiftAt - 1].roles).toEqual(replay.trace[0].roles);
    expect(fs[shiftAt].roles).not.toEqual(replay.trace[0].roles);
  });

  it("does not mutate earlier frames", () => {
    const fs = frames(replay);
    expect(fs[0].grid).toEqual(replay.trace[0].grid);
  });
});

describe("genealogy layout", () => {
  it("roots are generation 0 and every child hangs under its parent", () => {
    const forest = buildForest(entities);
    expect(forest.every((n) => n.entity.generation === 0)).toBe(true);
    const count = (ns: typeof forest): number => ns.reduce((s, n) => s + 1 + count(n.children), 0);
    expect(count(forest)).toBe(entities.length);
  });

  it("lays out nodes without overlaps, one column per generation", () => {
    const { nodes, links } = layoutForest(buildForest(entities));
    const seen = new Set(nodes.map((n) => `${n.x},${n.y}`));
    expect(seen.size).toBe(nodes.length);
    for (const n of nodes) expect(n.x).toBe(n.entity.generation);
    expect(links.length).toBe(entities.filter((e) => e.parent_id).length);
  });

  it("treats a child whose parent is missing as a root instead of dropping it", () => {
    const orphan = { ...entities[entities.length - 1], id: "E9999", parent_id: "E7777", generation: 2 };
    const forest = buildForest([...entities, orphan]);
    expect(forest.some((n) => n.entity.id === "E9999")).toBe(true);
  });
});

describe("stats", () => {
  it("geneDiff reports signed changes from parent", () => {
    const parent = entities.find((e) => e.parent_id === null)!;
    const child = { ...parent, genome: { ...parent.genome, risk_aversion: parent.genome.risk_aversion + 0.5 } };
    const d = geneDiff(child, parent).find((g) => g.gene === "risk_aversion")!;
    expect(d.delta).toBeCloseTo(0.5);
  });

  it("classifies failure types like the engine does", () => {
    const m = entities[0].metrics!;
    expect(failureType({ ...m, death_causes: { hazard: 0.5, starvation: 0.1, post_shift: 0 } })).toBe("hazard");
    expect(failureType({ ...m, death_causes: { hazard: 0, starvation: 0.1, post_shift: 0 }, adaptation_ratio: 0.3 })).toBe("adaptation");
    expect(failureType({ ...m, death_causes: { hazard: 0, starvation: 0.6, post_shift: 0 }, adaptation_ratio: 1 })).toBe("starvation");
    expect(failureType(null)).toBe("unknown");
  });

  it("counts lesson choices per slot by the parent's failure type", () => {
    const children = entities.filter((e) => e.parent_id);
    const patterns = curriculumPatterns(entities);
    const total = Object.values(patterns).reduce((s, slots) => s + Object.values(slots[0]).reduce((a, b) => a + b, 0), 0);
    expect(total).toBe(children.length);
  });

  it("pairs only children that have both a prediction and an outcome", () => {
    const withPred = entities.map((e, i) => (e.parent_id ? { ...e, predicted_change: i * 0.1 } : e));
    const pairs = predictionPairs(withPred);
    expect(pairs.length).toBe(entities.filter((e) => e.parent_id && e.actual_change !== null).length);
    expect(predictionPairs(entities).length).toBe(0); // heuristic children never predict
  });
});
