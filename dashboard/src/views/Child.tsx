import { useMemo } from "react";
import { Block, Loading, Missing, PageHead } from "../components/UI";
import { armInfo, ROLE_COLOR, TRUE_VALUE } from "../lib/arms";
import { loadRun } from "../lib/data";
import { href } from "../lib/router";
import { failureType, fmt, geneDiff } from "../lib/stats";
import type { Entity, RunBundle } from "../lib/types";
import { GENE_BOUNDS } from "../lib/types";
import { useLoad } from "../lib/useLoad";

export function ChildView({ run, id }: { run: string; id: string }) {
  const bundle = useLoad(() => loadRun(run), [run]);
  if (bundle.status === "loading") return <><PageHead title={id} /><Loading height={400} /></>;
  if (bundle.status === "missing") return <><PageHead title={id} /><Missing title="Run not found.">{bundle.reason}</Missing></>;
  const e = bundle.value.entities.find((x) => x.id === id);
  if (!e) return <><PageHead title={id} /><Missing title={`No child ${id} in ${run}.`} /></>;
  return <ChildBody run={run} b={bundle.value} e={e} />;
}

const FAILURE_TEXT = { hazard: "kept dying to hazards", adaptation: "struggled after law changes", starvation: "often starved", strong: "was strong all-round", unknown: "" };

function ChildBody({ run, b, e }: { run: string; b: RunBundle; e: Entity }) {
  const byId = useMemo(() => new Map(b.entities.map((x) => [x.id, x])), [b.entities]);
  const parent = e.parent_id ? byId.get(e.parent_id) : undefined;
  const kids = b.entities.filter((x) => x.parent_id === e.id);
  const m = e.metrics;
  const replays = b.replays.filter((r) => r.entity_id === e.id);
  const events = b.events.filter((x) => x.entity_id === e.id);
  const arm = armInfo(b.run.creator);
  return (
    <>
      <PageHead
        crumbs={[{ label: "Runs", href: href.runs() }, { label: run, href: href.run(run) }, { label: e.name }]}
        title={<>{e.name} <span className={`status ${e.status}`}>{e.status === "retired" ? `retired at G${e.retired_at_generation}` : "alive at the end"}</span></>}
        lede={<>Generation {e.generation} · {e.specialization.toLowerCase()} · designed by {arm.label.toLowerCase()}
          {parent ? <> from <a href={href.child(run, parent.id)}>{parent.name}</a></> : " as part of the founding population"}.</>}
      />

      <div className="quote">
        <div className="small muted" style={{ marginBottom: 4 }}>The Creator's reasoning</div>
        {e.reason}
      </div>

      <div className="grid-3" style={{ marginTop: "var(--space-5)" }}>
        <div className="panel">
          <h3>Outcome</h3>
          <dl className="kv">
            <dt>Fitness</dt><dd className="num"><strong>{fmt(m?.fitness)}</strong></dd>
            <dt>vs parent</dt><dd className="num">{fmt(e.actual_change, 2, true)}</dd>
            <dt>Predicted</dt><dd className="num">{fmt(e.predicted_change, 2, true)}</dd>
            <dt>Survived</dt><dd className="num">{m ? `${Math.round(m.survival_rate * 100)}% of worlds` : "n/a"}</dd>
            <dt>Adaptation</dt><dd className="num">{fmt(m?.adaptation_ratio)} <span className="muted">(1.0 = no drop after a law change)</span></dd>
            <dt>Deaths</dt><dd className="num">{m ? `hazard ${Math.round(m.death_causes.hazard * 100)}% · starved ${Math.round(m.death_causes.starvation * 100)}%` : "n/a"}</dd>
          </dl>
          {m && <p className="caption">In short: it {FAILURE_TEXT[failureType(m)]}.</p>}
        </div>
        <div className="panel">
          <h3>What it was taught</h3>
          <ol className="curriculum">
            {e.curriculum.map((p, i) => {
              const score = m?.lesson_scores[i];
              const max = Math.max(...(m?.lesson_scores ?? [1]), 0.0001);
              return (
                <li key={i}>
                  <span className="mono small muted">L{i + 1}</span>
                  <span className={`preset ${p}`}>{p}</span>
                  <span className="meter" aria-hidden="true"><span style={{ width: `${((score ?? 0) / max) * 100}%` }} /></span>
                  <span className="num small">{fmt(score)}</span>
                </li>
              );
            })}
          </ol>
          <p className="caption">Mean score in each lesson's 5 practice episodes. {e.inherit_beliefs ? "Started with its parent's knowledge." : "Started with no knowledge."}</p>
        </div>
        <div className="panel">
          <h3>Family</h3>
          <dl className="kv">
            <dt>Parent</dt><dd>{parent ? <a href={href.child(run, parent.id)}>{parent.name}</a> : "none (founder)"}</dd>
            <dt>Children</dt><dd>{kids.length ? kids.map((k) => <a key={k.id} href={href.child(run, k.id)} style={{ marginRight: 8 }}>{k.name}</a>) : "none"}</dd>
            <dt>Replays</dt><dd>{replays.length ? replays.map((r) => <a key={r.file} href={href.replay(run, r.file)} style={{ marginRight: 8 }}>{r.preset} world</a>) : "none (only champions are replayed)"}</dd>
          </dl>
        </div>
      </div>

      <Block title="Design" intro={parent ? `Each gene's position in its allowed range. The hollow marker is ${parent.name}'s value; the bar is this child's.` : "Each gene's position in its allowed range."}>
        <div className="panel">
          <div className="genes">
            {geneDiff(e, parent).map((g) => {
              const [lo, hi] = GENE_BOUNDS[g.gene];
              const pos = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
              return (
                <div className="gene" key={g.gene}>
                  <span className="small">{g.gene.replace(/_/g, " ")}</span>
                  <span className="gene-track">
                    <span className="gene-fill" style={{ width: pos(g.to) }} />
                    {parent && <span className="gene-parent" style={{ left: pos(g.from) }} title={`parent: ${g.from}`} />}
                  </span>
                  <span className="num small">{fmt(g.to, Number.isInteger(g.to) ? 0 : 2)}{parent && g.delta !== 0 && <span className={g.delta > 0 ? "up" : "down"}> {fmt(g.delta, Number.isInteger(g.delta) ? 0 : 2, true)}</span>}</span>
                </div>
              );
            })}
          </div>
        </div>
      </Block>

      <Block title="What it believes about the world" intro="After its lessons, this is what the child thinks stepping on each tile kind is worth. Compare that with the universe's true laws, which the child never sees.">
        {e.beliefs ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Kind</th><th>True role</th><th className="num">True value</th><th className="num">Believed value</th><th className="num">Evidence (count)</th><th>Verdict</th></tr></thead>
              <tbody>
                {b.run.universe.map((role, k) => {
                  const believed = e.beliefs!.mean[k];
                  const count = e.beliefs!.count[k];
                  const truth = TRUE_VALUE[role];
                  const known = count > 0;
                  const close = known && Math.abs(believed - truth) <= 3;
                  return (
                    <tr key={k}>
                      <td className="mono">{k}</td>
                      <td><span className="tile" style={{ background: ROLE_COLOR[role], marginRight: 6, verticalAlign: -2 }} />{role}</td>
                      <td className="num">{fmt(truth, 0, true)}</td>
                      <td className="num">{known ? fmt(believed, 1, true) : "never seen"}</td>
                      <td className="num">{fmt(count, 1)}</td>
                      <td className="small">{!known ? <span className="muted">unknown</span> : close ? <span className="up">correct</span> : <span className="down">wrong</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : <p className="muted small">No beliefs recorded.</p>}
      </Block>

      {events.length > 0 && (
        <Block title="Its events">
          <ul className="plain">
            {events.map((x) => <li key={x.seq}><span className="mono small muted">G{x.generation}</span> {x.type.replace(/_/g, " ")}</li>)}
          </ul>
        </Block>
      )}
    </>
  );
}
