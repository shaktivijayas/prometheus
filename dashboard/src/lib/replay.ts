import { FLOOR, type Replay, type TraceStep } from "./types";

export type FrameEvent = "start" | "shift" | "goal" | "food" | "hazard" | "surprise" | null;

export interface Frame {
  t: number;
  grid: number[][];
  pos: [number, number];
  energy: number;
  roles: string[];
  beliefs: number[] | null;
  radius: number;
  event: FrameEvent;
  step: TraceStep | null;
}

const GOAL_VALUE = 20;

function classify(step: TraceStep, roles: string[]): FrameEvent {
  if (step.shifted) return "shift";
  if (step.kind === null) return null;
  const role = roles[step.kind];
  if (role === "goal" && step.value !== null && step.value >= GOAL_VALUE) return "goal";
  if (role === "food") return "food";
  if (role === "hazard") return "hazard";
  return null;
}

/** Rebuild every frame of an episode from its trace alone (grid changes: consumption, respawns, law shifts). */
export function frames(replay: Replay): Frame[] {
  const [start, ...steps] = replay.trace;
  let grid = start.grid.map((row) => row.slice());
  let roles = start.roles.slice();
  const out: Frame[] = [{
    t: 0, grid, pos: start.pos, energy: start.energy, roles, beliefs: null, radius: start.radius,
    event: "start", step: null,
  }];
  for (const step of steps) {
    if (step.consumed || step.respawn || step.roles) grid = grid.map((row) => row.slice());
    if (step.roles) roles = step.roles.slice();
    if (step.consumed) grid[step.pos[0]][step.pos[1]] = FLOOR;
    if (step.respawn) {
      const [r, c, k] = step.respawn;
      grid[r][c] = k;
    }
    out.push({
      t: step.t, grid, pos: step.pos, energy: step.energy, roles, beliefs: step.beliefs, radius: start.radius,
      event: classify(step, roles), step,
    });
  }
  return out;
}
