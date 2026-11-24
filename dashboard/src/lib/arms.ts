// Human names and colours for Creator arms, by the creator string used on the command line.
export interface ArmInfo {
  label: string;
  short: string;
  color: string;
  note: string;
}

export function armInfo(creator: string): ArmInfo {
  if (creator.startsWith("neural")) {
    const noteach = creator.endsWith(":noteach");
    const v000 = creator.includes(":v000");
    if (v000) return { label: "Untrained Creator", short: "untrained", color: "var(--arm-fresh)", note: "same network, random initial weights (v000)" };
    if (creator.endsWith(":randomteach")) return { label: "Trained Creator, random lessons", short: "random lessons", color: "var(--arm-randomteach)", note: "same learned designs, but each child's 4 lessons are drawn at random" };
    if (noteach) return { label: "Trained Creator, fixed lessons", short: "fixed lessons", color: "var(--arm-noteach)", note: "same learned designs, but every child gets one fixed default lesson plan" };
    return { label: "Trained Creator", short: "trained", color: "var(--arm-trained)", note: "our network after reinforcement-learning training" };
  }
  if (creator === "random") return { label: "Random mutation", short: "random", color: "var(--arm-random)", note: "control: random gene changes and lessons" };
  if (creator === "clone") return { label: "Clone (null)", short: "clone", color: "var(--arm-clone)", note: "copies parents exactly; any gain is luck" };
  if (creator === "heuristic") return { label: "Hand-written rules", short: "rules", color: "var(--arm-heuristic)", note: "a human's if-then design rules" };
  return { label: creator, short: creator, color: "var(--ink-3)", note: "" };
}

export const ROLE_COLOR: Record<string, string> = {
  food: "var(--tile-food)",
  hazard: "var(--tile-hazard)",
  goal: "var(--tile-goal)",
  neutral: "var(--tile-neutral)",
};

export const TRUE_VALUE: Record<string, number> = { food: 8, hazard: -12, goal: 25, neutral: 0 };
