import { useEffect, useState } from "react";
import { BarChart, LineChart, Legend, Scatter } from "../components/Chart";
import { Block, Loading, Missing, PageHead } from "../components/UI";
import { armInfo } from "../lib/arms";
import { loadHistory, loadIndex, loadProgress, loadRun } from "../lib/data";
import { href } from "../lib/router";
import { curriculumPatterns, fmt, predictionPairs, type FailureType } from "../lib/stats";
import type { Entity, HistoryRow, ProgressRow, RunIndexEntry } from "../lib/types";
import { PRESETS } from "../lib/types";
import { useLoad } from "../lib/useLoad";

export function CreatorView() {
  const history = useLoad(loadHistory, []);
  const progress = useLoad(loadProgress, []);
  const index = useLoad(loadIndex, []);
  return (
    <>
      <PageHead
        crumbs={[{ label: "Results", href: href.overview() }, { label: "The Creator" }]}
        title="The Creator"
        lede="A small neural network (two 128-unit layers, five output heads) trained with PPO. Each child it designs earns a reward: how much the child beat its parent, scaled by the run's starting fitness. Training happens only in universes reserved for training. Everything on the Results page is measured in universes it never saw."
      />
      <Block title="Training" intro="Each iteration the Creator raises 12 short populations in new training universes, each with randomized laws, food and hazard levels, and shift timing, and then updates its weights. Markers show saved checkpoints.">
        {history.status === "loading" && <Loading />}
        {history.status === "missing" && <Missing title="No training history exported yet."><p>Train with <code>prometheus train-creator</code>, then export with <code>--creator creator</code>.</p></Missing>}
        {history.status === "ready" && <TrainingCharts rows={history.value} />}
      </Block>
      <Block title="Does the parent get better as it trains?" intro="Each saved checkpoint raises full-size populations in the same held-out universes. Its final champion is re-scored on fresh upbringings, so the line shows verified quality.">
        {progress.status === "loading" && <Loading />}
        {progress.status === "missing" && <Missing title="Checkpoint progress not measured yet."><p>Run <code>prometheus creator-progress --versions all</code>.</p></Missing>}
        {progress.status === "ready" && <ProgressCharts rows={progress.value} />}
      </Block>
      {index.status === "ready" && <Behaviour runs={index.value.runs} />}
    </>
  );
}

function TrainingCharts({ rows }: { rows: HistoryRow[] }) {
  const markers = rows.filter((r) => r.version).map((r) => ({ x: r.iteration, label: `v${String(r.version).padStart(3, "0")}` }));
  const smooth = (key: keyof HistoryRow) => {
    const k = Math.max(1, Math.round(rows.length / 25));
    return rows.map((r, i) => {
      const w = rows.slice(Math.max(0, i - k), i + k + 1).map((q) => q[key] as number);
      return { x: r.iteration, y: w.reduce((s, v) => s + v, 0) / w.length };
    });
  };
  const raw = (key: keyof HistoryRow) => rows.map((r) => ({ x: r.iteration, y: r[key] as number }));
  const first = rows.slice(0, 10), last = rows.slice(-10);
  const avg = (rs: HistoryRow[], k: keyof HistoryRow) => rs.reduce((s, r) => s + (r[k] as number), 0) / Math.max(rs.length, 1);
  return (
    <>
      <p className="small" style={{ marginBottom: "var(--space-4)", maxWidth: "78ch" }}>
        {rows.length} iterations, {rows.reduce((s, r) => s + r.decisions, 0).toLocaleString()} design decisions.
        Mean reward went from <strong className="num">{fmt(avg(first, "mean_reward"), 2, true)}</strong> (first 10 iterations) to{" "}
        <strong className="num">{fmt(avg(last, "mean_reward"), 2, true)}</strong> (last 10), and the average fitness of the populations it raised went
        from <strong className="num">{fmt(avg(first, "final_mean_fitness"))}</strong> to <strong className="num">{fmt(avg(last, "final_mean_fitness"))}</strong>.
      </p>
      <div className="grid-2">
        <div className="panel">
          <h3>Reward per decision</h3>
          <LineChart zero markers={markers} xLabel="iteration" series={[
            { name: "raw", color: "var(--line-strong)", points: raw("mean_reward") },
            { name: "smoothed", color: "var(--primary)", points: smooth("mean_reward") },
          ]} />
          <p className="caption">The child's fitness (averaged over 2 upbringings) minus its parent's, divided by the run's starting fitness. Grey: each iteration; teal: smoothed.</p>
        </div>
        <div className="panel">
          <h3>Fitness of the populations it raised</h3>
          <LineChart markers={markers} xLabel="iteration" series={[
            { name: "final generation mean", color: "var(--primary)", points: smooth("final_mean_fitness") },
            { name: "generation 0 mean", color: "var(--arm-clone)", points: smooth("gen0_mean_fitness"), dashed: true },
          ]} />
          <Legend items={[{ label: "final generation (mean)", color: "var(--primary)" }, { label: "generation 0 (random designs)", color: "var(--arm-clone)" }]} />
        </div>
        <div className="panel">
          <h3>Prediction error</h3>
          <LineChart markers={markers} xLabel="iteration" series={[{ name: "MAE", color: "var(--arm-fresh)", points: smooth("prediction_mae") }]} />
          <p className="caption">Mean absolute gap between the fitness change the Creator predicted for each child and what actually happened.</p>
        </div>
        <div className="panel">
          <h3>Policy entropy</h3>
          <LineChart markers={markers} xLabel="iteration" series={[{ name: "entropy", color: "var(--arm-random)", points: smooth("entropy") }]} />
          <p className="caption">How exploratory its decisions are. A steady fall means it is settling on strategies it trusts.</p>
        </div>
      </div>
    </>
  );
}

function ProgressCharts({ rows }: { rows: ProgressRow[] }) {
  if (!rows.length) return <Missing title="No checkpoints evaluated." />;
  const x = (r: ProgressRow) => r.iteration;
  return (
    <div className="grid-2">
      <div className="panel">
        <h3>Verified champion by checkpoint</h3>
        <LineChart xLabel="training iteration of the checkpoint" series={[{
          name: "champion", color: "var(--primary)", marker: true,
          points: rows.map((r) => ({ x: x(r), y: r.verified_final_mean, lo: r.verified_final_mean - r.verified_final_std, hi: r.verified_final_mean + r.verified_final_std })),
        }]} />
        <p className="caption">Mean over {rows[0].seeds.length} held-out universes; band ± one standard deviation.</p>
      </div>
      <div className="panel">
        <h3>Checkpoints</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Version</th><th className="num">Iteration</th><th className="num">Champion</th><th className="num">Gain</th><th className="num">Pred. error</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.version}>
                  <td className="mono">{r.version}</td>
                  <td className="num">{r.iteration}</td>
                  <td className="num">{fmt(r.verified_final_mean)} <span className="muted">±{fmt(r.verified_final_std)}</span></td>
                  <td className="num">{fmt(r.verified_gain_mean, 2, true)}</td>
                  <td className="num">{fmt(r.prediction_mae)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

const FAILURE_LABEL: Record<FailureType, string> = {
  hazard: "Parent died to hazards",
  adaptation: "Parent adapted poorly to shifts",
  starvation: "Parent starved",
  strong: "Parent was strong all-round",
  unknown: "Unknown",
};
const PRESET_COLOR: Record<string, string> = {
  easy: "var(--tile-food)", normal: "var(--arm-clone)", hard: "var(--tile-hazard)", shift: "var(--primary)",
};

function Behaviour({ runs }: { runs: RunIndexEntry[] }) {
  const neural = runs.filter((r) => r.creator.startsWith("neural") && !r.creator.includes(":v000") && !r.creator.endsWith(":noteach"));
  const [entities, setEntities] = useState<{ run: string; e: Entity }[] | null>(null);
  useEffect(() => {
    let alive = true;
    Promise.all(neural.map((r) => loadRun(r.name).then((b) => (b.ok ? b.value.entities.map((e) => ({ run: r.name, e })) : []))))
      .then((all) => { if (alive) setEntities(all.flat()); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs]);
  if (!neural.length) {
    return (
      <Block title="What it learned to do">
        <Missing title="No runs by the trained Creator exported yet." />
      </Block>
    );
  }
  if (!entities) return <Block title="What it learned to do"><Loading /></Block>;
  const patterns = curriculumPatterns(entities.map((x) => x.e));
  const pairs = predictionPairs(entities.map((x) => x.e));
  const runOf = new Map(entities.map((x) => [x.e, x.run]));
  const inherit = entities.filter((x) => x.e.parent_id);
  const inheritRate = inherit.filter((x) => x.e.inherit_beliefs).length / Math.max(inherit.length, 1);
  return (
    <Block title="What it learned to do" intro={`From ${inherit.length} children the trained Creator designed in ${neural.length} evaluation run${neural.length > 1 ? "s" : ""}. It passed its parent's knowledge on to ${Math.round(inheritRate * 100)}% of them.`}>
      <div className="grid-2">
        <div className="panel">
          <h3>Lesson plans, by what went wrong with the parent</h3>
          {Object.entries(patterns).map(([type, slots]) => {
            const total = Object.values(slots[0]).reduce((a, b) => a + b, 0);
            return (
              <div key={type} style={{ marginBottom: "var(--space-4)" }}>
                <div className="small" style={{ marginBottom: 6 }}><strong>{FAILURE_LABEL[type as FailureType]}</strong> <span className="muted">· {total} children</span></div>
                <div className="lessons">
                  {slots.map((counts, i) => (
                    <div key={i} className="lesson-col" aria-label={`Lesson ${i + 1}`}>
                      <div className="lesson-bar">
                        {PRESETS.map((p) => counts[p] ? (
                          <span key={p} title={`${p}: ${counts[p]}`} style={{ flexGrow: counts[p], background: PRESET_COLOR[p] }} />
                        ) : null)}
                      </div>
                      <span className="mono small muted">L{i + 1}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
          <Legend items={PRESETS.map((p) => ({ label: p, color: PRESET_COLOR[p] }))} />
        </div>
        <div className="panel">
          <h3>Predicted vs actual fitness change</h3>
          <Scatter
            xLabel="predicted change"
            yLabel="actual change"
            points={pairs.map((p) => ({
              x: p.predicted, y: p.actual, color: armInfo("neural").color,
              title: `${p.entity.name}: predicted ${fmt(p.predicted, 2, true)}, actual ${fmt(p.actual, 2, true)}`,
              href: href.child(runOf.get(p.entity)!, p.entity.id),
            }))}
          />
          <p className="caption">Each dot is one child (click to open it). Points on the dashed diagonal are perfect predictions.</p>
        </div>
      </div>
      <div className="panel" style={{ marginTop: "var(--space-4)" }}>
        <h3>Gene edits it favours</h3>
        <GeneEdits entities={entities.map((x) => x.e)} />
      </div>
    </Block>
  );
}

function GeneEdits({ entities }: { entities: Entity[] }) {
  const sums: Record<string, number[]> = {};
  for (const e of entities) {
    const edit = (e.decision?.edit as number[] | undefined) ?? null;
    if (!edit) continue;
    ["perception_radius", "memory_capacity", "learning_rate", "exploration", "exploration_decay", "planning_horizon",
      "risk_aversion", "surprise_threshold", "curiosity_bonus"].forEach((g, i) => (sums[g] ??= []).push(edit[i]));
  }
  const bars = Object.entries(sums).map(([g, xs]) => {
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
    return { label: g.replace(/_/g, " "), value: m, err: sd, color: m >= 0 ? "var(--primary)" : "var(--arm-heuristic)" };
  });
  if (!bars.length) return <p className="muted small">No decision records.</p>;
  return (
    <>
      <BarChart bars={bars} digits={3} />
      <p className="caption">Mean intended edit per gene, as a fraction of the gene's full range (± spread across children).</p>
    </>
  );
}
