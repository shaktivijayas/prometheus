// Mirrors the engine's run-output contract (design spec section 6.6).

export interface Metrics {
  fitness: number;
  mean_score: number;
  stable_score: number;
  shift_score: number;
  survival_rate: number;
  adaptation_ratio: number;
  learning_speed: number;
  mean_prediction_error: number;
  perception_cost: number;
  death_causes: { hazard: number; starvation: number; post_shift: number };
  lesson_scores: number[];
}

export type GenomeDict = Record<string, number>;

export interface Entity {
  id: string;
  name: string;
  parent_id: string | null;
  generation: number;
  created_at: string;
  specialization: string;
  reason: string;
  genome: GenomeDict;
  curriculum: string[];
  inherit_beliefs: boolean;
  predicted_change: number | null;
  status: "active" | "retired";
  metrics: Metrics | null;
  beliefs: { mean: number[]; count: number[] } | null;
  actual_change: number | null;
  retired_at_generation: number | null;
  decision: Record<string, unknown> | null;
}

export interface Verified {
  entity_id: string;
  name: string;
  mean: number;
  std: number;
  n: number;
}

export interface Verification {
  gen0_best?: Verified;
  final_best?: Verified;
  gain?: number;
  gain_pct?: number | null;
}

export interface RunJson {
  config: Record<string, unknown> & { creator: string; seed: number; generations: number };
  creator: string;
  universe: string[];
  timing: { duration_s?: number };
  summary: Record<string, unknown>;
  verification: Verification;
}

export interface EventRow {
  seq: number;
  type: string;
  generation: number;
  entity_id: string | null;
  data: Record<string, unknown>;
}

export interface GenerationRow {
  generation: number;
  best: number;
  mean: number;
  median: number;
  diversity: number;
  best_entity: string | null;
}

export interface TraceStart {
  grid: number[][];
  roles: string[];
  radius: number;
  pos: [number, number];
  energy: number;
  shift_step: number | null;
}

export interface TraceStep {
  t: number;
  pos: [number, number];
  action: number;
  kind: number | null;
  predicted: number | null;
  value: number | null;
  energy: number;
  surprise: number;
  shifted: boolean;
  consumed: boolean;
  respawn: [number, number, number] | null;
  beliefs: number[];
  roles?: string[];
}

export interface Replay {
  entity_id: string;
  name: string;
  generation: number;
  preset: string;
  seed: number;
  score: number;
  goals: number;
  steps: number;
  survived: boolean;
  death_cause: string | null;
  trace: [TraceStart, ...TraceStep[]];
}

export interface ReplayRef {
  file: string;
  key: string;
  entity_id: string;
  name: string;
  generation: number;
  preset: string;
  seed: number;
  score: number;
  steps: number;
}

export interface RunIndexEntry {
  name: string;
  creator: string;
  seed: number;
  generations: number;
  children: number;
  verification: Verification;
  summary: Record<string, unknown>;
}

export interface ComparisonRef {
  name: string;
  file: string;
  creators: string[];
  seeds: number[];
}

export interface DataIndex {
  generated_at: string;
  runs: RunIndexEntry[];
  comparisons: ComparisonRef[];
  creator: { progress: boolean; history: boolean };
}

export interface Arm {
  best_mean: number[];
  best_std: number[];
  mean_mean: number[];
  mean_std: number[];
  runs: string[];
  verified_final_mean: number;
  verified_final_std: number;
  verified_gen0_mean: number;
  verified_gain_mean: number;
  verified_gain_std: number;
}

export interface Comparison {
  name: string;
  creators: string[];
  seeds: number[];
  generations: number;
  arms: Record<string, Arm>;
}

export interface ProgressRow {
  version: string;
  iteration: number;
  seeds: number[];
  verified_final_mean: number;
  verified_final_std: number;
  verified_gain_mean: number;
  verified_gain_std: number;
  in_run_best_mean: number;
  prediction_mae: number | null;
}

export interface HistoryRow {
  iteration: number;
  version: number | null;
  mean_reward: number;
  decisions: number;
  gen0_mean_fitness: number;
  final_mean_fitness: number;
  final_best_fitness: number;
  improvement: number;
  prediction_mae: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
  approx_kl: number;
  seconds: number;
}

export interface RunBundle {
  run: RunJson;
  entities: Entity[];
  events: EventRow[];
  generations: GenerationRow[];
  replays: ReplayRef[];
}

export const PRESETS = ["easy", "normal", "hard", "shift"] as const;
export const FLOOR = -1;
export const WALL = -2;
export const GENE_NAMES = [
  "perception_radius", "memory_capacity", "learning_rate", "exploration", "exploration_decay",
  "planning_horizon", "risk_aversion", "surprise_threshold", "curiosity_bonus",
] as const;
export const GENE_BOUNDS: Record<string, [number, number]> = {
  perception_radius: [1, 5], memory_capacity: [0, 50], learning_rate: [0.01, 1], exploration: [0, 1],
  exploration_decay: [0.9, 1], planning_horizon: [1, 12], risk_aversion: [0, 3], surprise_threshold: [0.1, 5],
  curiosity_bonus: [0, 2],
};
