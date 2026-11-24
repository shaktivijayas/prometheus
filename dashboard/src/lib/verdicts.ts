import type { Arm, Comparison } from "./types";

export type Verdict = "yes" | "no" | "inconclusive";

export interface Judgement {
  verdict: Verdict;
  diff: number;
  se: number;
}

function judge(diff: number, se: number): Verdict {
  if (diff > 2 * se) return "yes";
  if (diff + 2 * se <= 0) return "no";
  return "inconclusive";
}

/** Does arm `a` produce better verified champions than arm `b`? (n = seeds per arm) */
export function beats(a: Arm, b: Arm, n: number): Judgement {
  const diff = a.verified_final_mean - b.verified_final_mean;
  const se = Math.sqrt((a.verified_final_std ** 2 + b.verified_final_std ** 2) / Math.max(n, 1));
  return { verdict: judge(diff, se), diff, se };
}

/** Did this arm's verified champion beat its own generation-0 best? */
export function improves(a: Arm, n: number): Judgement {
  const se = a.verified_gain_std / Math.sqrt(Math.max(n, 1));
  return { verdict: judge(a.verified_gain_mean, se), diff: a.verified_gain_mean, se };
}

export type Role = "trained" | "noteach" | "untrained" | "random" | "clone" | "heuristic";

export function findArm(c: Comparison, role: Role): [string, Arm] | undefined {
  const entries = Object.entries(c.arms);
  const match = (name: string) => {
    if (role === "noteach") return name.startsWith("neural") && name.endsWith(":noteach");
    if (role === "untrained") return name.startsWith("neural") && name.includes(":v000") && !name.endsWith(":noteach");
    if (role === "trained") return name.startsWith("neural") && !name.endsWith(":noteach") && !name.includes(":v000");
    return name === role;
  };
  return entries.find(([name]) => match(name));
}
