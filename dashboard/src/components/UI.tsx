import type { ReactNode } from "react";
import type { Verdict } from "../lib/verdicts";

export function Loading({ height = 160 }: { height?: number }) {
  return <div className="skeleton" style={{ minHeight: height }} aria-busy="true" aria-label="Loading" />;
}

export function Missing({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty" role="status">
      <strong>{title}</strong>
      {children}
    </div>
  );
}

const VERDICT_TEXT: Record<Verdict | "pending", string> = {
  yes: "Yes",
  no: "No",
  inconclusive: "Inconclusive",
  pending: "Not measured yet",
};
const VERDICT_CLASS: Record<Verdict | "pending", string> = { yes: "yes", no: "no", inconclusive: "maybe", pending: "pending" };

export function VerdictBadge({ verdict }: { verdict: Verdict | "pending" }) {
  const icon = verdict === "yes" ? "✓" : verdict === "no" ? "✕" : verdict === "inconclusive" ? "≈" : "…";
  return (
    <span className={`badge ${VERDICT_CLASS[verdict]}`}>
      <span aria-hidden="true">{icon}</span>
      {VERDICT_TEXT[verdict]}
    </span>
  );
}

export function PageHead({ crumbs, title, lede }: { crumbs?: { label: string; href?: string }[]; title: ReactNode; lede?: ReactNode }) {
  return (
    <header className="page-head">
      {crumbs && (
        <nav className="crumbs" aria-label="Breadcrumb">
          {crumbs.map((c, i) => (
            <span key={i}>
              {c.href ? <a href={c.href}>{c.label}</a> : c.label}
              {i < crumbs.length - 1 && <span aria-hidden="true"> / </span>}
            </span>
          ))}
        </nav>
      )}
      <h1>{title}</h1>
      {lede && <p className="lede">{lede}</p>}
    </header>
  );
}

export function Block({ title, intro, children, id }: { title: string; intro?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section className="block" id={id}>
      <header>
        <h2>{title}</h2>
        {intro && <p>{intro}</p>}
      </header>
      {children}
    </section>
  );
}

export function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="8.5" r="4.2" fill="currentColor" />
      <circle cx="7.5" cy="23.5" r="3.6" fill="currentColor" opacity="0.75" />
      <circle cx="24.5" cy="23.5" r="3.6" fill="currentColor" opacity="0.75" />
      <path d="M16 12.5 8.5 20M16 12.5 23.5 20" stroke="currentColor" strokeWidth="2" fill="none" />
    </svg>
  );
}
