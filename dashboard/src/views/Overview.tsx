import { useState } from "react";
import { BarChart, Legend } from "../components/Chart";
import { Block, Loading, Missing, PageHead, VerdictBadge } from "../components/UI";
import { armInfo } from "../lib/arms";
import { loadComparison, loadIndex } from "../lib/data";
import { href } from "../lib/router";
import { fmt } from "../lib/stats";
import type { Comparison, DataIndex } from "../lib/types";
import { useLoad } from "../lib/useLoad";
import { beats, findArm, improves, type Judgement } from "../lib/verdicts";

export function Overview() {
  const index = useLoad(loadIndex, []);
  return (
    <>
      <PageHead
        title="Can an AI learn to raise better AIs?"
        lede={
          <>
            PROMETHEUS trains a <strong>Creator</strong> network with reinforcement learning. It designs child
            intelligences, decides what they inherit and plans their lessons. The children then learn on their own in
            worlds whose laws are hidden and sometimes change. Every result here is re-scored on fresh upbringings, so luck
            can't pass for progress.
          </>
        }
      />
      <Loop />
      {index.status === "loading" && <Loading />}
      {index.status === "missing" && (
        <Missing title="No exported data found">
          <p>Run the engine, then export results into the dashboard:</p>
          <code>prometheus export-dashboard --runs "runs/*" --comparisons "comparisons/*.json" --creator creator</code>
        </Missing>
      )}
      {index.status === "ready" && <Results index={index.value} />}
    </>
  );
}

function Loop() {
  const steps = [
    ["Creator designs", "picks a parent, edits its 9 genes, decides inheritance"],
    ["Creator teaches", "chooses 4 lessons: easy, normal, hard or rule-shift"],
    ["Child learns", "discovers hidden tile laws by trial and error"],
    ["Benchmark scores", "24 fixed worlds, half with a mid-episode law change"],
    ["Creator is rewarded", "by how much its child beat the parent"],
  ];
  return (
    <ol className="loop" aria-label="The training loop">
      {steps.map(([title, text], i) => (
        <li key={title}>
          <span className="loop-n mono">{i + 1}</span>
          <span><strong>{title}</strong><span className="small muted"> · {text}</span></span>
        </li>
      ))}
    </ol>
  );
}

function Results({ index }: { index: DataIndex }) {
  const comps = index.comparisons;
  const preferred = comps.find((c) => c.name === "final") ?? comps[0];
  const [file, setFile] = useState(preferred?.file);
  const comparison = useLoad(() => (file ? loadComparison(file) : Promise.resolve({ ok: false as const, missing: true as const, reason: "none" })), [file]);
  if (!comps.length) return <Missing title="No comparisons exported yet." />;
  return (
    <>
      {comps.length > 1 && (
        <div className="chips" role="group" aria-label="Comparison" style={{ marginTop: "var(--space-6)" }}>
          {comps.map((c) => (
            <button key={c.file} className="chip" aria-pressed={c.file === file} onClick={() => setFile(c.file)}>
              {c.name} · {c.creators.length} arms · {c.seeds.length} seeds
            </button>
          ))}
        </div>
      )}
      {comparison.status === "loading" && <Loading height={300} />}
      {comparison.status === "missing" && <Missing title="Comparison file missing">{comparison.reason}</Missing>}
      {comparison.status === "ready" && <ComparisonResults c={comparison.value} index={index} />}
    </>
  );
}

function evidence(j: Judgement, unit = "") {
  return `difference ${fmt(j.diff, 2, true)}${unit}, standard error ${fmt(j.se, 2)}`;
}

function ComparisonResults({ c, index }: { c: Comparison; index: DataIndex }) {
  const n = c.seeds.length;
  const trained = findArm(c, "trained");
  const random = findArm(c, "random");
  const heuristic = findArm(c, "heuristic");
  const noteach = findArm(c, "noteach");
  const untrained = findArm(c, "untrained");
  const subject = trained ?? random;

  const rows: { q: string; j?: Judgement; text: string }[] = [];
  if (subject) {
    const j = improves(subject[1], n);
    rows.push({ q: "Do later generations beat earlier ones?", j,
      text: `${armInfo(subject[0]).label}: verified champion ${fmt(subject[1].verified_final_mean)} vs its generation-0 best ${fmt(subject[1].verified_gen0_mean)} (${evidence(j)}).` });
  }
  if (trained && random) {
    const j = beats(trained[1], random[1], n);
    rows.push({ q: "Does the trained Creator beat random mutation?", j, text: `Verified champions ${fmt(trained[1].verified_final_mean)} vs ${fmt(random[1].verified_final_mean)} (${evidence(j)}).` });
  } else rows.push({ q: "Does the trained Creator beat random mutation?", text: "Needs a comparison that includes the trained Creator." });
  if (trained && heuristic) {
    const j = beats(trained[1], heuristic[1], n);
    rows.push({ q: "Does it beat hand-written design rules?", j, text: `Verified champions ${fmt(trained[1].verified_final_mean)} vs ${fmt(heuristic[1].verified_final_mean)} (${evidence(j)}).` });
  }
  if (trained && noteach) {
    const j = beats(trained[1], noteach[1], n);
    rows.push({ q: "Does its teaching add value?", j, text: `Same trained designer with and without its own lesson plans: ${fmt(trained[1].verified_final_mean)} vs ${fmt(noteach[1].verified_final_mean)} (${evidence(j)}).` });
  } else rows.push({ q: "Does its teaching add value?", text: "Needs the trained and no-teaching arms." });
  if (trained && untrained) {
    const j = beats(trained[1], untrained[1], n);
    rows.push({ q: "Does an experienced Creator beat a fresh one?", j, text: `After training vs the same network at v000: ${fmt(trained[1].verified_final_mean)} vs ${fmt(untrained[1].verified_final_mean)} (${evidence(j)}).` });
  } else rows.push({ q: "Does an experienced Creator beat a fresh one?", text: "Needs the trained and untrained (v000) arms." });

  const arms = Object.entries(c.arms).sort((a, b) => b[1].verified_final_mean - a[1].verified_final_mean);
  const runsByName = new Set(index.runs.map((r) => r.name));
  return (
    <>
      <Block title="The answers" intro={<>From the <strong>{c.name}</strong> comparison: {Object.keys(c.arms).length} Creators, {n} held-out universes each, {c.generations} generations. "Yes" needs the gap to clear two standard errors.</>}>
        <div className="verdicts">
          {rows.map((r) => (
            <div className="verdict" key={r.q}>
              <VerdictBadge verdict={r.j?.verdict ?? "pending"} />
              <div>
                <div className="q">{r.q}</div>
                <div className="evidence">{r.text}</div>
              </div>
            </div>
          ))}
        </div>
      </Block>

      <Block title="Verified champions" intro="Each run's final champion, re-scored on 10 fresh upbringings that played no part in selection. Bars show the mean across universes, whiskers ± one standard deviation.">
        <div className="grid-2">
          <div className="panel">
            <h3>Champion quality</h3>
            <BarChart bars={arms.map(([name, a]) => ({ label: armInfo(name).label, sublabel: name, value: a.verified_final_mean, err: a.verified_final_std, color: armInfo(name).color }))} signed={false} />
          </div>
          <div className="panel">
            <h3>Gain over the generation-0 best</h3>
            <BarChart bars={arms.map(([name, a]) => ({ label: armInfo(name).label, sublabel: armInfo(name).short, value: a.verified_gain_mean, err: a.verified_gain_std, color: armInfo(name).color }))} />
          </div>
        </div>
      </Block>

      <Block title="All arms" intro="The in-run best column is selection-biased: selection keeps whichever copy got lucky, so even the clone arm, which changes nothing, rises there. Use the verified columns.">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Creator</th><th className="num">Gen-0 best (verified)</th><th className="num">Champion (verified)</th>
                <th className="num">Verified gain</th><th className="num">In-run best (biased)</th><th>Runs</th>
              </tr>
            </thead>
            <tbody>
              {arms.map(([name, a]) => (
                <tr key={name}>
                  <td><span className="swatch" style={{ background: armInfo(name).color }} />{armInfo(name).label} <span className="muted mono small">{name}</span></td>
                  <td className="num">{fmt(a.verified_gen0_mean)}</td>
                  <td className="num">{fmt(a.verified_final_mean)} <span className="muted">±{fmt(a.verified_final_std)}</span></td>
                  <td className="num">{fmt(a.verified_gain_mean, 2, true)} <span className="muted">±{fmt(a.verified_gain_std)}</span></td>
                  <td className="num muted">{fmt(a.best_mean[a.best_mean.length - 1])}</td>
                  <td>{a.runs.filter((r) => runsByName.has(r)).map((r) => <a key={r} href={href.run(r)} style={{ marginRight: 8 }}>{r.split("_").pop()}</a>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: "var(--space-3)" }}>
          <Legend items={arms.map(([name]) => ({ label: `${armInfo(name).label}: ${armInfo(name).note}`, color: armInfo(name).color }))} />
        </div>
      </Block>
    </>
  );
}
