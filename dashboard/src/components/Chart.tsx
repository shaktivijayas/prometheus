import { useEffect, useRef, useState, type ReactNode } from "react";

/** Measure the container so charts lay out in real pixels (labels stay legible on phones). */
export function useWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(240, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0];
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

const fmtTick = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1));

export interface Series {
  name: string;
  color: string;
  points: { x: number; y: number; lo?: number; hi?: number }[];
  dashed?: boolean;
  marker?: boolean;
}

export function LineChart({ series, height = 240, xLabel, yLabel, xTicks, markers, zero = false }: {
  series: Series[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xTicks?: number[];
  markers?: { x: number; label: string }[];
  zero?: boolean;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const pad = { l: 44, r: 12, t: 12, b: 34 };
  const all = series.flatMap((s) => s.points);
  if (!all.length) return <div ref={ref} className="muted small">No data yet.</div>;
  const xs = all.map((p) => p.x);
  const ys = all.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]);
  if (zero) ys.push(0);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const span = y1 - y0 || 1;
  y0 -= span * 0.06; y1 += span * 0.06;
  const X = (v: number) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (width - pad.l - pad.r);
  const Y = (v: number) => pad.t + (1 - (v - y0) / (y1 - y0)) * (height - pad.t - pad.b);
  const yt = niceTicks(y0, y1, 4);
  const xt = xTicks ?? niceTicks(x0, x1, Math.min(6, Math.max(2, Math.floor(width / 110))));
  return (
    <div ref={ref}>
      <svg className="chart" width={width} height={height} role="img" aria-label={`${yLabel ?? "value"} by ${xLabel ?? "x"}`}>
        <g className="grid">{yt.map((t) => <line key={t} x1={pad.l} x2={width - pad.r} y1={Y(t)} y2={Y(t)} />)}</g>
        {zero && y0 < 0 && y1 > 0 && <line className="zero" x1={pad.l} x2={width - pad.r} y1={Y(0)} y2={Y(0)} />}
        {yt.map((t) => <text key={t} x={pad.l - 6} y={Y(t) + 4} textAnchor="end">{fmtTick(t)}</text>)}
        {xt.filter((t) => t >= x0 && t <= x1).map((t) => <text key={t} x={X(t)} y={height - pad.b + 16} textAnchor="middle">{fmtTick(t)}</text>)}
        {xLabel && <text className="label" x={(pad.l + width - pad.r) / 2} y={height - 2} textAnchor="middle">{xLabel}</text>}
        {markers?.map((m) => (
          <g key={m.x}>
            <line x1={X(m.x)} x2={X(m.x)} y1={pad.t} y2={height - pad.b} stroke="var(--line-strong)" strokeDasharray="2 3" />
            <text x={X(m.x) + 3} y={pad.t + 10}>{m.label}</text>
          </g>
        ))}
        {series.map((s) => {
          const band = s.points.filter((p) => p.lo !== undefined && p.hi !== undefined);
          const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
          const bandD = band.length > 1
            ? band.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.hi!)}`).join("") +
              band.slice().reverse().map((p) => `L${X(p.x)},${Y(p.lo!)}`).join("") + "Z"
            : "";
          return (
            <g key={s.name}>
              {bandD && <path d={bandD} fill={s.color} opacity={0.14} />}
              <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" />
              {s.marker && s.points.map((p) => <circle key={p.x} cx={X(p.x)} cy={Y(p.y)} r={3} fill={s.color} />)}
            </g>
          );
        })}
        {yLabel && <text className="label" x={pad.l + 4} y={pad.t + 10}>{yLabel}</text>}
      </svg>
    </div>
  );
}

export interface Bar {
  label: string;
  sublabel?: string;
  value: number;
  err?: number;
  color: string;
  href?: string;
}

/** Horizontal bars with ± error whiskers; handles negative values around a zero line. */
export function BarChart({ bars, unit = "", digits = 2, signed = true }: { bars: Bar[]; unit?: string; digits?: number; signed?: boolean }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const labelW = Math.min(190, Math.max(120, width * 0.32));
  const rowH = 40;
  const pad = { r: 64, t: 6, b: 22 };
  const height = pad.t + bars.length * rowH + pad.b;
  const lo = Math.min(0, ...bars.map((b) => b.value - (b.err ?? 0)));
  const hi = Math.max(0, ...bars.map((b) => b.value + (b.err ?? 0)));
  const X = (v: number) => labelW + ((v - lo) / (hi - lo || 1)) * (width - labelW - pad.r);
  const ticks = niceTicks(lo, hi, Math.max(2, Math.floor((width - labelW) / 90)));
  return (
    <div ref={ref}>
      <svg className="chart" width={width} height={height} role="img" aria-label="bar chart">
        <g className="grid">{ticks.map((t) => <line key={t} x1={X(t)} x2={X(t)} y1={pad.t} y2={height - pad.b} />)}</g>
        {ticks.map((t) => <text key={t} x={X(t)} y={height - 6} textAnchor="middle">{fmtTick(t)}</text>)}
        <line className="zero" x1={X(0)} x2={X(0)} y1={pad.t} y2={height - pad.b} />
        {bars.map((b, i) => {
          const y = pad.t + i * rowH + 8;
          const x0 = X(Math.min(0, b.value));
          const w = Math.abs(X(b.value) - X(0));
          const label = (
            <>
              <text className="label" x={0} y={y + 10}>{b.label}</text>
              {b.sublabel && <text x={0} y={y + 24}>{b.sublabel}</text>}
            </>
          );
          return (
            <g key={b.label}>
              {b.href ? <a href={b.href}>{label}</a> : label}
              <rect x={x0} y={y} width={Math.max(w, 1)} height={18} rx={3} fill={b.color} />
              {b.err !== undefined && b.err > 0 && (
                <g stroke="var(--ink-2)" strokeWidth={1.2}>
                  <line x1={X(b.value - b.err)} x2={X(b.value + b.err)} y1={y + 9} y2={y + 9} />
                  <line x1={X(b.value - b.err)} x2={X(b.value - b.err)} y1={y + 4} y2={y + 14} />
                  <line x1={X(b.value + b.err)} x2={X(b.value + b.err)} y1={y + 4} y2={y + 14} />
                </g>
              )}
              <text className="value" x={Math.max(X(b.value), X(b.value + (b.err ?? 0))) + 6} y={y + 13}>
                {signed && b.value > 0 ? "+" : ""}{b.value.toFixed(digits)}{unit}{b.err ? ` ±${b.err.toFixed(digits)}` : ""}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function Scatter({ points, xLabel, yLabel, diagonal = true, height = 280 }: {
  points: { x: number; y: number; color: string; title?: string; href?: string }[];
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const pad = { l: 44, r: 12, t: 12, b: 36 };
  if (!points.length) return <div ref={ref} className="muted small">No predictions recorded yet.</div>;
  const vals = points.flatMap((p) => [p.x, p.y]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1;
  lo -= span * 0.05; hi += span * 0.05;
  const X = (v: number) => pad.l + ((v - lo) / (hi - lo)) * (width - pad.l - pad.r);
  const Y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (height - pad.t - pad.b);
  const t = niceTicks(lo, hi, 5);
  return (
    <div ref={ref}>
      <svg className="chart" width={width} height={height} role="img" aria-label={`${yLabel} against ${xLabel}`}>
        <g className="grid">{t.map((v) => <line key={v} x1={pad.l} x2={width - pad.r} y1={Y(v)} y2={Y(v)} />)}</g>
        {t.map((v) => <text key={v} x={pad.l - 6} y={Y(v) + 4} textAnchor="end">{fmtTick(v)}</text>)}
        {t.map((v) => <text key={`x${v}`} x={X(v)} y={height - pad.b + 16} textAnchor="middle">{fmtTick(v)}</text>)}
        {diagonal && <line className="zero" x1={X(lo)} y1={Y(lo)} x2={X(hi)} y2={Y(hi)} />}
        <text className="label" x={(pad.l + width) / 2} y={height - 2} textAnchor="middle">{xLabel}</text>
        <text className="label" x={pad.l + 4} y={pad.t + 10}>{yLabel}</text>
        {points.map((p, i) => {
          const dot = <circle cx={X(p.x)} cy={Y(p.y)} r={3.5} fill={p.color} opacity={0.75}>{p.title && <title>{p.title}</title>}</circle>;
          return p.href ? <a key={i} href={p.href}>{dot}</a> : <g key={i}>{dot}</g>;
        })}
      </svg>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }): ReactNode {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label}>
          <span className="swatch" style={{ background: i.color, opacity: i.dashed ? 0.6 : 1 }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}
