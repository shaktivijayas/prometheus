import { GENE_BOUNDS, GENE_NAMES, PRESETS, type Entity, type Metrics } from "./types";

export type FailureType = "hazard" | "adaptation" | "starvation" | "strong" | "unknown";

/** Same rule the engine's Creator uses to describe a parent's dominant weakness. */
export function failureType(m: Metrics | null | undefined): FailureType {
  if (!m) return "unknown";
  const hazard = m.death_causes?.hazard ?? 0;
  const starve = m.death_causes?.starvation ?? 0;
  if (hazard >= 0.3 && hazard >= starve) return "hazard";
  if (m.adaptation_ratio !== null && m.adaptation_ratio < 0.6) return "adaptation";
  if (starve >= 0.3) return "starvation";
  return "strong";
}

export interface GeneChange {
  gene: string;
  from: number;
  to: number;
  delta: number;
  normDelta: number; // as a fraction of the gene's full range
}

export function geneDiff(child: Entity, parent: Entity | null | undefined): GeneChange[] {
  return GENE_NAMES.map((gene) => {
    const to = child.genome[gene];
    const from = parent ? parent.genome[gene] : to;
    const [lo, hi] = GENE_BOUNDS[gene];
    return { gene, from, to, delta: to - from, normDelta: (to - from) / (hi - lo) };
  });
}

export type LessonCounts = Record<string, number>;

/** For each parent failure type: how often each preset was chosen in each of the 4 lesson slots. */
export function curriculumPatterns(entities: Entity[]): Record<string, LessonCounts[]> {
  const byId = new Map(entities.map((e) => [e.id, e]));
  const out: Record<string, LessonCounts[]> = {};
  for (const e of entities) {
    if (!e.parent_id) continue;
    const parent = byId.get(e.parent_id);
    const type = failureType(parent?.metrics);
    out[type] ??= [0, 1, 2, 3].map(() => Object.fromEntries(PRESETS.map((p) => [p, 0])));
    e.curriculum.forEach((preset, slot) => {
      if (slot < 4) out[type][slot][preset] = (out[type][slot][preset] ?? 0) + 1;
    });
  }
  return out;
}

export interface PredictionPair {
  predicted: number;
  actual: number;
  entity: Entity;
}

export function predictionPairs(entities: Entity[]): PredictionPair[] {
  return entities
    .filter((e) => e.parent_id && e.predicted_change !== null && e.actual_change !== null)
    .map((e) => ({ predicted: e.predicted_change as number, actual: e.actual_change as number, entity: e }));
}

export function fmt(x: number | null | undefined, digits = 2, signed = false): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "n/a";
  const s = x.toFixed(digits);
  return signed && x > 0 ? `+${s}` : s;
}
