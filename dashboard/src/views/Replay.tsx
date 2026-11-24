import { useEffect, useMemo, useRef, useState } from "react";
import { useWidth } from "../components/Chart";
import { Loading, Missing, PageHead } from "../components/UI";
import { ROLE_COLOR, TRUE_VALUE } from "../lib/arms";
import { loadReplay } from "../lib/data";
import { frames, type Frame } from "../lib/replay";
import { href } from "../lib/router";
import { fmt } from "../lib/stats";
import { FLOOR, WALL, type Replay } from "../lib/types";
import { useLoad } from "../lib/useLoad";

export function ReplayView({ run, file }: { run: string; file: string }) {
  const replay = useLoad(() => loadReplay(run, file), [run, file]);
  if (replay.status === "loading") return <><PageHead title="Replay" /><Loading height={420} /></>;
  if (replay.status === "missing") return <><PageHead title="Replay" /><Missing title="Replay not found.">{replay.reason}</Missing></>;
  return <Player run={run} replay={replay.value} />;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

const SPEEDS = [1, 4, 12, 30];
const ACTION = ["north", "east", "south", "west", "stay"];

function Player({ run, replay }: { run: string; replay: Replay }) {
  const fs = useMemo(() => frames(replay), [replay]);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const shiftAt = fs.findIndex((f) => f.event === "shift");
  const f = fs[Math.min(t, fs.length - 1)];

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setT((x) => {
        if (x >= fs.length - 1) { setPlaying(false); return x; }
        return x + 1;
      });
    }, 1000 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, fs.length]);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement) return;
      if (ev.key === " ") { ev.preventDefault(); setPlaying((p) => !p); }
      if (ev.key === "ArrowRight") setT((x) => Math.min(fs.length - 1, x + 1));
      if (ev.key === "ArrowLeft") setT((x) => Math.max(0, x - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fs.length]);

  const goals = fs.filter((x) => x.event === "goal").map((x) => x.t);
  return (
    <>
      <PageHead
        crumbs={[{ label: "Runs", href: href.runs() }, { label: run, href: href.run(run) }, { label: replay.name, href: href.child(run, replay.entity_id) }, { label: `${replay.preset} world` }]}
        title={<>{replay.name} in a {replay.preset === "shift" ? "law-changing" : "stable"} world</>}
        lede={<>Generation {replay.generation} champion. {replay.goals} goals in {replay.steps} steps, {replay.survived ? "survived" : `died (${replay.death_cause})`}.
          {shiftAt > 0 && <> The laws change at step {fs[shiftAt].t}.</>} Space plays and pauses; the arrow keys step.</>}
      />
      <div className="replay">
        <div className="panel replay-stage">
          <Board frame={f} />
          <div className="controls">
            <button className="btn primary" onClick={() => { if (t >= fs.length - 1) setT(0); setPlaying((p) => !p); }} aria-label={playing ? "Pause" : "Play"}>
              {playing ? "Pause" : t >= fs.length - 1 ? "Replay" : "Play"}
            </button>
            <button className="btn" onClick={() => setT((x) => Math.max(0, x - 1))} aria-label="Previous step" disabled={t === 0}>‹</button>
            <button className="btn" onClick={() => setT((x) => Math.min(fs.length - 1, x + 1))} aria-label="Next step" disabled={t >= fs.length - 1}>›</button>
            {shiftAt > 0 && <button className="btn" onClick={() => { setPlaying(false); setT(Math.max(0, shiftAt - 1)); }}>Go to law change</button>}
            <label className="small muted speed">
              Speed
              <select value={speed} onChange={(ev) => setSpeed(Number(ev.target.value))}>
                {SPEEDS.map((s) => <option key={s} value={s}>{s} steps/s</option>)}
              </select>
            </label>
          </div>
          <div className="scrub">
            <input type="range" min={0} max={fs.length - 1} value={t} onChange={(ev) => { setPlaying(false); setT(Number(ev.target.value)); }} aria-label="Step" />
            <div className="scrub-marks" aria-hidden="true">
              {goals.map((g) => <span key={g} className="mark goal" style={{ left: `${(g / (fs.length - 1)) * 100}%` }} />)}
              {shiftAt > 0 && <span className="mark shift" style={{ left: `${(shiftAt / (fs.length - 1)) * 100}%` }} />}
            </div>
          </div>
          <Energy fs={fs} t={t} />
        </div>
        <aside className="panel replay-side">
          <StepInfo f={f} shiftAt={shiftAt} />
          <Beliefs f={f} />
          <div className="legend" style={{ marginTop: "var(--space-4)" }}>
            {["food", "goal", "hazard", "neutral"].map((r) => <span key={r}><span className="swatch" style={{ background: ROLE_COLOR[r] }} />{r}</span>)}
            <span><span className="swatch" style={{ background: "var(--tile-wall)" }} />wall</span>
          </div>
        </aside>
      </div>
    </>
  );
}

function Board({ frame }: { frame: Frame }) {
  const [wrap, width] = useWidth<HTMLDivElement>(520);
  const canvas = useRef<HTMLCanvasElement>(null);
  const n = frame.grid.length;
  const size = Math.min(width, 560);
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size * dpr;
    c.height = size * dpr;
    const ctx = c.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const cell = size / n;
    const colors = {
      floor: cssVar("--tile-floor"), wall: cssVar("--tile-wall"), agent: cssVar("--agent"), line: cssVar("--line"),
      roles: Object.fromEntries(Object.entries(ROLE_COLOR).map(([r, v]) => [r, cssVar(v.slice(4, -1))])),
    };
    ctx.fillStyle = colors.floor;
    ctx.fillRect(0, 0, size, size);
    for (let r = 0; r < n; r++) {
      for (let col = 0; col < n; col++) {
        const v = frame.grid[r][col];
        if (v === FLOOR) continue;
        ctx.fillStyle = v === WALL ? colors.wall : colors.roles[frame.roles[v]] ?? colors.wall;
        const inset = v === WALL ? 0 : cell * 0.12;
        ctx.beginPath();
        ctx.roundRect(col * cell + inset, r * cell + inset, cell - inset * 2, cell - inset * 2, v === WALL ? 1 : cell * 0.18);
        ctx.fill();
        if (v >= 0 && cell > 18) {
          ctx.fillStyle = "rgba(0,0,0,0.55)";
          ctx.font = `${Math.round(cell * 0.34)}px ui-monospace, monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(v), col * cell + cell / 2, r * cell + cell / 2 + 1);
        }
      }
    }
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1;
    for (let i = 0; i <= n; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell); ctx.stroke();
    }
    const [pr, pc] = frame.pos;
    const rad = frame.radius;
    ctx.strokeStyle = colors.agent;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect((pc - rad) * cell, (pr - rad) * cell, (2 * rad + 1) * cell, (2 * rad + 1) * cell);
    ctx.setLineDash([]);
    ctx.fillStyle = colors.agent;
    ctx.beginPath();
    ctx.arc(pc * cell + cell / 2, pr * cell + cell / 2, cell * 0.32, 0, Math.PI * 2);
    ctx.fill();
  }, [frame, size, n]);
  return (
    <div ref={wrap} className="board-wrap">
      <canvas ref={canvas} style={{ width: size, height: size }} role="img"
        aria-label={`World at step ${frame.t}: agent at row ${frame.pos[0]}, column ${frame.pos[1]}, energy ${fmt(frame.energy, 1)}`} />
    </div>
  );
}

function StepInfo({ f, shiftAt }: { f: Frame; shiftAt: number }) {
  const s = f.step;
  const msg = !s ? "Episode start." :
    s.shifted ? "The laws just changed. Tile kinds now have different roles." :
    s.kind === null ? `Moved ${ACTION[s.action]} onto empty floor.` :
    `Stepped on kind ${s.kind} (${f.roles[s.kind]}): expected ${s.predicted === null ? "nothing (never seen)" : fmt(s.predicted, 1, true)}, got ${fmt(s.value, 1, true)}.`;
  return (
    <div>
      <div className="step-head"><span className="num">step {f.t}</span><span className="num">energy {fmt(f.energy, 1)}</span></div>
      <p className="small" style={{ minHeight: "3.2em" }}>{msg}</p>
      {s && s.surprise > 0 && <p className="small muted">Surprise {fmt(s.surprise, 2)}{s.surprise >= 1 ? " (big enough to make a careful child doubt what it knew)" : ""}</p>}
      {shiftAt > 0 && f.t < shiftAt && <p className="small muted">Laws unchanged so far.</p>}
    </div>
  );
}

function Beliefs({ f }: { f: Frame }) {
  return (
    <div style={{ marginTop: "var(--space-4)" }}>
      <h3>Belief vs truth, right now</h3>
      <table className="beliefs">
        <thead><tr><th>Kind</th><th>Truth now</th><th className="num">Value</th><th className="num">Believed</th></tr></thead>
        <tbody>
          {f.roles.map((role, k) => (
            <tr key={k}>
              <td className="mono">{k}</td>
              <td><span className="tile" style={{ background: ROLE_COLOR[role], marginRight: 6, verticalAlign: -2 }} />{role}</td>
              <td className="num">{fmt(TRUE_VALUE[role], 0, true)}</td>
              <td className={`num ${f.beliefs ? (Math.abs(f.beliefs[k] - TRUE_VALUE[role]) <= 3 ? "up" : "down") : ""}`}>
                {f.beliefs ? fmt(f.beliefs[k], 1, true) : "n/a"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Energy({ fs, t }: { fs: Frame[]; t: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const h = 56;
  const max = Math.max(...fs.map((f) => f.energy), 1);
  const X = (i: number) => (i / Math.max(fs.length - 1, 1)) * width;
  const Y = (e: number) => h - 4 - (Math.max(e, 0) / max) * (h - 8);
  const d = fs.map((f, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(f.energy).toFixed(1)}`).join("");
  return (
    <div ref={ref} style={{ marginTop: "var(--space-2)" }}>
      <svg width={width} height={h} className="chart" role="img" aria-label="Energy over the episode">
        <path d={d} fill="none" stroke="var(--primary)" strokeWidth={1.5} />
        <line x1={X(t)} x2={X(t)} y1={0} y2={h} stroke="var(--ink)" strokeWidth={1} />
        <text x={4} y={11}>energy</text>
      </svg>
    </div>
  );
}
