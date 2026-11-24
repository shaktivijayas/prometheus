import type {
  Comparison, DataIndex, Entity, EventRow, GenerationRow, HistoryRow, ProgressRow, Replay, ReplayRef, RunBundle,
  RunJson,
} from "./types";

export type Loaded<T> = { ok: true; value: T } | { ok: false; missing: true; reason: string };

const cache = new Map<string, Promise<unknown>>();

function url(path: string): string {
  const base = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? "/";
  return `${base.replace(/\/$/, "")}/data/${path}`;
}

async function getJson<T>(path: string): Promise<Loaded<T>> {
  if (!cache.has(path)) {
    cache.set(path, (async () => {
      try {
        const res = await fetch(url(path));
        if (!res.ok) return { ok: false, missing: true, reason: `${path} was not found (HTTP ${res.status}).` } as const;
        const type = res.headers?.get?.("content-type") ?? "application/json";
        if (!type.includes("json")) return { ok: false, missing: true, reason: `${path} was not found.` } as const;
        return { ok: true, value: (await res.json()) as T } as const;
      } catch (err) {
        return { ok: false, missing: true, reason: `${path}: ${String(err)}` } as const;
      }
    })());
  }
  return cache.get(path) as Promise<Loaded<T>>;
}

export const loadIndex = () => getJson<DataIndex>("index.json");
export const loadComparison = (file: string) => getJson<Comparison>(file);
export const loadReplay = (run: string, file: string) => getJson<Replay>(`runs/${run}/replays/${file}`);
export const loadProgress = () => getJson<ProgressRow[]>("creator/progress.json");
export const loadHistory = () => getJson<HistoryRow[]>("creator/training_history.json");

export async function loadRun(name: string): Promise<Loaded<RunBundle>> {
  const [run, entities, events, generations, replays] = await Promise.all([
    getJson<RunJson>(`runs/${name}/run.json`),
    getJson<Entity[]>(`runs/${name}/entities.json`),
    getJson<EventRow[]>(`runs/${name}/events.json`),
    getJson<GenerationRow[]>(`runs/${name}/generations.json`),
    getJson<ReplayRef[]>(`runs/${name}/replays/index.json`),
  ]);
  if (!run.ok) return run;
  if (!entities.ok) return entities;
  return {
    ok: true,
    value: {
      run: run.value,
      entities: entities.value,
      events: events.ok ? events.value : [],
      generations: generations.ok ? generations.value : [],
      replays: replays.ok ? replays.value : [],
    },
  };
}

/** Test hook: forget cached responses. */
export function clearCache(): void {
  cache.clear();
}
