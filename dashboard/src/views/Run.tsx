import { useMemo, useState } from "react";
import { LineChart, Legend, useWidth } from "../components/Chart";
import { Block, Loading, Missing, PageHead } from "../components/UI";
import { armInfo, ROLE_COLOR } from "../lib/arms";
import { loadIndex, loadRun } from "../lib/data";
import { href } from "../lib/router";
import { fmt } from "../lib/stats";
import { buildForest, layoutForest } from "../lib/tree";
import type { Entity, EventRow, RunBundle } from "../lib/types";
import { useLoad } from "../lib/useLoad";

export function RunsList() {
  const index = useLoad(loadIndex, []);
  return (
    <>
      <PageHead crumbs={[{ label: "Results", href: href.overview() }, { label: "Runs" }]} title="Runs"
        lede="Every exported evolution run. Each one is a single population raised by one Creator in one universe." />
      {index.status === "loading" && <Loading />}
      {index.status === "missing" && <Missing title="No exported runs." />}
      {index.status === "ready" && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Run</th><th>Creator</th><th className="num">Universe</th><th className="num">Children</th><th className="num">Gen-0 best (verified)</th><th className="num">Champion (verified)</th><th className="num">Gain</th></tr></thead>
            <tbody>
              {index.value.runs.map((r) => (
                <tr key={r.name}>
                  <td><a href={href.run(r.name)}>{r.name}</a></td>
                  <td><span className="swatch" style={{ background: armInfo(r.creator).color }} />{armInfo(r.creator).label}</td>
                  <td className="num">{r.seed}</td>
                  <td className="num">{r.children}</td>
                  <td className="num">{fmt(r.verification.gen0_best?.mean)}</td>
                  <td className="num">{fmt(r.verification.final_best?.mean)}</td>
                  <td className="num">{fmt(r.verification.gain, 2, true)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

export function RunView({ name }: { name: string }) {
  const bundle = useLoad(() => loadRun(name), [name]);
  return (
    <>
      {bundle.status === "loading" && <><PageHead title={name} /><Loading height={400} /></>}
      {bundle.status === "missing" && <><PageHead title={name} /><Missing title="This run isn't in the exported data.">{bundle.reason}</Missing></>}
      {bundle.status === "ready" && <RunBody name={name} b={bundle.value} />}
    </>
  );
}

function RunBody({ name, b }: { name: string; b: RunBundle }) {
  const arm = armInfo(b.run.creator);
  const v = b.run.verification;
  const byId = useMemo(() => new Map(b.entities.map((e) => [e.id, e])), [b.entities]);
  return (
    <>
      <PageHead
        crumbs={[{ label: "Runs", href: href.runs() }, { label: name }]}
        title={<><span className="swatch" style={{ background: arm.color, width: 14, height: 14 }} />{arm.label} <span className="muted mono" style={{ fontSize: "var(--step-1)" }}>· universe {String(b.run.config.seed)}</span></>}
        lede={`${b.entities.length} children over ${b.run.config.generations} generations. ${arm.note ? arm.note[0].toUpperCase() + arm.note.slice(1) + "." : ""}`}
      />
      <div className="grid-3">
        <div className="panel tinted">
          <h3>Verified result</h3>
          {v.final_best && v.gen0_best ? (
            <dl className="kv">
              <dt>Gen-0 best design</dt><dd><a href={href.child(name, v.gen0_best.entity_id)}>{v.gen0_best.name}</a> <span className="num">{fmt(v.gen0_best.mean)} ± {fmt(v.gen0_best.std)}</span></dd>
              <dt>Final champion</dt><dd><a href={href.child(name, v.final_best.entity_id)}>{v.final_best.name}</a> <span className="num">{fmt(v.final_best.mean)} ± {fmt(v.final_best.std)}</span></dd>
              <dt>Verified gain</dt><dd className="num"><strong>{fmt(v.gain, 3, true)}</strong>{v.gain_pct !== null && v.gain_pct !== undefined ? ` (${fmt(v.gain_pct, 1, true)}%)` : ""}</dd>
              <dt>Fresh upbringings</dt><dd className="num">{v.final_best.n}</dd>
            </dl>
          ) : <p className="muted small">This run was not verified.</p>}
        </div>
        <div className="panel tinted">
          <h3>Hidden laws of this universe</h3>
          <div className="laws">
            {b.run.universe.map((role, k) => (
              <span key={k} className="law"><span className="tile" style={{ background: ROLE_COLOR[role] }} /> kind {k} <span className="muted">is</span> <strong>{role}</strong></span>
            ))}
          </div>
          <p className="caption">Children only ever see the kind numbers. In shift worlds these roles are secretly reshuffled mid-episode.</p>
        </div>
        <div className="panel tinted">
          <h3>Replays</h3>
          {b.replays.length ? (
            <ul className="plain">
              {b.replays.map((r) => (
                <li key={r.file}><a href={href.replay(name, r.file)}>{r.name}</a> <span className="muted small">gen {r.generation} · {r.preset} world · score {fmt(r.score)}</span></li>
              ))}
            </ul>
          ) : <p className="muted small">No replays recorded for this run.</p>}
        </div>
      </div>

      {b.generations.length > 0 && (
        <Block title="Generations" intro="In-run fitness of the living population after each generation. The best line is selection-biased, since it keeps lucky draws; the verified result above is not.">
          <div className="panel">
            <LineChart xLabel="generation" xTicks={b.generations.map((g) => g.generation)} series={[
              { name: "best (biased)", color: arm.color, points: b.generations.map((g) => ({ x: g.generation, y: g.best })), marker: true },
              { name: "mean", color: "var(--ink-3)", dashed: true, points: b.generations.map((g) => ({ x: g.generation, y: g.mean })) },
            ]} />
            <Legend items={[{ label: "best in population (selection-biased)", color: arm.color }, { label: "population mean", color: "var(--ink-3)", dashed: true }]} />
          </div>
        </Block>
      )}

      <Block title="Family tree" intro="Columns are generations. Lines connect parent to child, dot size and colour show fitness, faded dots were retired, and rings mark each generation's champion. Click any child to open its profile.">
        <Genealogy run={name} entities={b.entities} champions={new Set(b.generations.map((g) => g.best_entity ?? ""))} />
      </Block>

      <Block title="Timeline" intro="Everything that happened in this run, in order.">
        <Timeline run={name} events={b.events} byId={byId} />
      </Block>
    </>
  );
}

function Genealogy({ run, entities, champions }: { run: string; entities: Entity[]; champions: Set<string> }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const layout = useMemo(() => layoutForest(buildForest(entities)), [entities]);
  const fit = entities.map((e) => e.metrics?.fitness ?? 0);
  const [lo, hi] = [Math.min(...fit), Math.max(...fit)];
  const rowH = 16;
  const pad = 28;
  const colW = Math.max(56, (width - pad * 2) / Math.max(layout.width - 1, 1));
  const W = pad * 2 + colW * Math.max(layout.width - 1, 1);
  const H = pad * 2 + rowH * Math.max(layout.height - 1, 1);
  const X = (g: number) => pad + g * colW;
  const Y = (r: number) => pad + r * rowH;
  const t = (e: Entity) => (hi > lo ? ((e.metrics?.fitness ?? 0) - lo) / (hi - lo) : 0.5);
  return (
    <div ref={ref} className="panel tree-panel">
      <svg width={W} height={H} className="chart" role="img" aria-label="Family tree of every child in this run">
        {Array.from({ length: layout.width }, (_, g) => (
          <text key={g} x={X(g)} y={12} textAnchor="middle">G{g}</text>
        ))}
        {layout.links.map((l) => (
          <path key={l.to.id} d={`M${X(l.from.x)},${Y(l.from.y)} C${X(l.from.x) + colW / 2},${Y(l.from.y)} ${X(l.to.x) - colW / 2},${Y(l.to.y)} ${X(l.to.x)},${Y(l.to.y)}`}
            fill="none" stroke="var(--line-strong)" strokeWidth={1} />
        ))}
        {layout.nodes.map((n) => {
          const e = n.entity;
          const r = 3 + t(e) * 4;
          return (
            <a key={n.id} href={href.child(run, e.id)}>
              <title>{`${e.name} · gen ${e.generation} · fitness ${fmt(e.metrics?.fitness)}${e.status === "retired" ? " · retired" : ""}`}</title>
              {champions.has(e.id) && <circle cx={X(n.x)} cy={Y(n.y)} r={r + 3.5} fill="none" stroke="var(--primary)" strokeWidth={1.5} />}
              <circle cx={X(n.x)} cy={Y(n.y)} r={r}
                fill={`color-mix(in oklch, var(--primary) ${Math.round(45 + t(e) * 55)}%, var(--surface-2))`}
                opacity={e.status === "retired" ? 0.45 : 1} />
            </a>
          );
        })}
      </svg>
    </div>
  );
}

const EVENT_LABEL: Record<string, string> = {
  created: "Created", creator_decision: "Creator's reasoning", trained: "Taught", evaluated: "Scored",
  retired: "Retired", generation_complete: "Generation done", rule_shift_detected: "Noticed a law change",
};

function describe(e: EventRow): string {
  const d = e.data as Record<string, unknown>;
  switch (e.type) {
    case "created": return `${d.name} (${d.specialization})${d.parent_id ? ` from ${d.parent_id}` : ""}`;
    case "creator_decision": return String(d.reason ?? "");
    case "trained": return `lessons ${(d.curriculum as string[]).join(" > ")} · lesson scores ${(d.lesson_scores as number[]).map((x) => fmt(x, 1)).join(", ")}`;
    case "evaluated": return `fitness ${fmt(d.fitness as number)}${d.actual_change !== null && d.actual_change !== undefined ? ` · vs parent ${fmt(d.actual_change as number, 2, true)}` : ""}${d.predicted_change !== null && d.predicted_change !== undefined ? ` · predicted ${fmt(d.predicted_change as number, 2, true)}` : ""}`;
    case "retired": return `fitness ${fmt(d.fitness as number)}`;
    case "generation_complete": return `best ${fmt(d.best as number)} · mean ${fmt(d.mean as number)} · diversity ${fmt(d.diversity as number)}`;
    case "rule_shift_detected": return `at step ${d.t}, surprise ${fmt(d.surprise as number)} on kind ${d.kind}`;
    default: return JSON.stringify(d);
  }
}

function Timeline({ run, events, byId }: { run: string; events: EventRow[]; byId: Map<string, Entity> }) {
  const types = Array.from(new Set(events.map((e) => e.type)));
  const [on, setOn] = useState<Set<string>>(() => new Set(["creator_decision", "generation_complete", "rule_shift_detected", "retired"]));
  const [limit, setLimit] = useState(120);
  const shown = events.filter((e) => on.has(e.type));
  if (!events.length) return <p className="muted small">No events recorded.</p>;
  return (
    <>
      <div className="chips" role="group" aria-label="Event types" style={{ marginBottom: "var(--space-3)" }}>
        {types.map((t) => (
          <button key={t} className="chip" aria-pressed={on.has(t)}
            onClick={() => setOn((s) => { const n = new Set(s); if (n.has(t)) n.delete(t); else n.add(t); return n; })}>
            {EVENT_LABEL[t] ?? t} <span className="muted">{events.filter((e) => e.type === t).length}</span>
          </button>
        ))}
      </div>
      <ol className="timeline">
        {shown.slice(0, limit).map((e) => (
          <li key={e.seq} data-type={e.type}>
            <span className="mono small muted">G{e.generation}</span>
            <span className="small"><strong>{EVENT_LABEL[e.type] ?? e.type}</strong>
              {e.entity_id && <> · <a href={href.child(run, e.entity_id)}>{byId.get(e.entity_id)?.name ?? e.entity_id}</a></>}
            </span>
            <span className="small tl-detail">{describe(e)}</span>
          </li>
        ))}
      </ol>
      {shown.length > limit && <button className="btn" onClick={() => setLimit((l) => l + 200)}>Show more ({shown.length - limit} left)</button>}
    </>
  );
}
