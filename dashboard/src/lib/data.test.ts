import { afterEach, describe, expect, it, vi } from "vitest";
import { clearCache, loadIndex, loadRun } from "./data";

function stubFetch(files: Record<string, unknown>) {
  vi.stubGlobal("fetch", async (u: string) => {
    const key = Object.keys(files).find((k) => u.endsWith(k));
    if (!key) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => files[key] };
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  clearCache();
});

describe("data loading", () => {
  it("reports a missing index instead of throwing", async () => {
    stubFetch({});
    const r = await loadIndex();
    expect(r.ok).toBe(false);
  });

  it("loads a run whose optional files are missing, with empty lists", async () => {
    stubFetch({ "runs/a/run.json": { creator: "random" }, "runs/a/entities.json": [] });
    const r = await loadRun("a");
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.value.events).toEqual([]);
      expect(r.value.replays).toEqual([]);
      expect(r.value.generations).toEqual([]);
    }
  });

  it("fails a run with no run.json", async () => {
    stubFetch({ "runs/b/entities.json": [] });
    expect((await loadRun("b")).ok).toBe(false);
  });

  it("survives a network error", async () => {
    vi.stubGlobal("fetch", async () => { throw new Error("offline"); });
    const r = await loadIndex();
    expect(r.ok).toBe(false);
  });
});
