import { armInfo } from "./arms";
import { describe, expect, it } from "vitest";
import { beats, improves, findArm } from "./verdicts";
import type { Arm, Comparison } from "./types";

const arm = (final: number, std: number, gain = 0, gainStd = 0): Arm => ({
  best_mean: [], best_std: [], mean_mean: [], mean_std: [], runs: [],
  verified_final_mean: final, verified_final_std: std, verified_gen0_mean: final - gain,
  verified_gain_mean: gain, verified_gain_std: gainStd,
});

describe("verdicts", () => {
  it("says yes only when the gap clears two standard errors", () => {
    expect(beats(arm(10, 1), arm(5, 1), 5).verdict).toBe("yes");
    expect(beats(arm(5.5, 1), arm(5, 1), 5).verdict).toBe("inconclusive");
    expect(beats(arm(3, 0.5), arm(5, 0.5), 5).verdict).toBe("no");
  });

  it("reports the difference and its standard error", () => {
    const r = beats(arm(10, 2), arm(4, 2), 4);
    expect(r.diff).toBeCloseTo(6);
    expect(r.se).toBeCloseTo(Math.sqrt(4 / 4 + 4 / 4));
  });

  it("judges improvement over generations from the verified gain", () => {
    expect(improves(arm(9, 1, 5, 0.5), 3).verdict).toBe("yes");
    expect(improves(arm(4, 1, 0, 0), 3).verdict).toBe("no");
    expect(improves(arm(4, 1, 0.2, 1), 3).verdict).toBe("inconclusive");
  });

  it("finds arms by role regardless of exact checkpoint name", () => {
    const c = { arms: { "neural:latest": arm(1, 0), "neural:v000": arm(1, 0), "neural:latest:noteach": arm(1, 0), random: arm(1, 0) } } as unknown as Comparison;
    expect(findArm(c, "trained")?.[0]).toBe("neural:latest");
    expect(findArm(c, "untrained")?.[0]).toBe("neural:v000");
    expect(findArm(c, "noteach")?.[0]).toBe("neural:latest:noteach");
    expect(findArm(c, "random")?.[0]).toBe("random");
    expect(findArm(c, "heuristic")).toBeUndefined();
  });

  it("tells the random-lessons control apart from the trained arm", () => {
    const c = { arms: { "neural:latest:randomteach": arm(1, 0), "neural:latest": arm(2, 0), "neural:latest:noteach": arm(1, 0) } } as unknown as Comparison;
    expect(findArm(c, "trained")?.[0]).toBe("neural:latest");
    expect(findArm(c, "randomteach")?.[0]).toBe("neural:latest:randomteach");
    expect(findArm(c, "noteach")?.[0]).toBe("neural:latest:noteach");
    expect(armInfo("neural:latest:randomteach").label).toBe("Trained Creator, random lessons");
  });
});
