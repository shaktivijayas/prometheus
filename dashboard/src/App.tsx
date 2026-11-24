import { useEffect, useState } from "react";
import { Logo } from "./components/UI";
import { href, useRoute } from "./lib/router";
import { Overview } from "./views/Overview";
import { CreatorView } from "./views/Creator";
import { RunsList, RunView } from "./views/Run";
import { ChildView } from "./views/Child";
import { ReplayView } from "./views/Replay";

type Theme = "system" | "light" | "dark";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem("prometheus-theme") as Theme) || "system";
    } catch {
      return "system";
    }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try { localStorage.setItem("prometheus-theme", theme); } catch { /* private mode */ }
  }, [theme]);
  const next = () => setTheme((t) => (t === "system" ? "light" : t === "light" ? "dark" : "system"));
  return [theme, next];
}

export function App() {
  const route = useRoute();
  const [theme, nextTheme] = useTheme();
  const section = route.page === "child" || route.page === "replay" || route.page === "run" ? "runs" : route.page;
  const link = (page: string, label: string, to: string) => (
    <a href={to} aria-current={section === page ? "page" : undefined}>{label}</a>
  );
  return (
    <>
      <div className="topbar">
        <div className="topbar-inner">
          <a className="brand" href={href.overview()}><Logo /> PROMETHEUS</a>
          <nav className="nav" aria-label="Main">
            {link("overview", "Results", href.overview())}
            {link("creator", "The Creator", href.creator())}
            {link("runs", "Runs", href.runs())}
          </nav>
          <span className="spacer" />
          <a className="small muted" href="https://github.com/shaktivijayas/prometheus">Source</a>
          <button className="theme-toggle" onClick={nextTheme} aria-label={`Theme: ${theme}. Change theme`}>
            {theme === "system" ? "Auto" : theme === "light" ? "Light" : "Dark"}
          </button>
        </div>
      </div>
      <main className="page">
        {route.page === "overview" && <Overview />}
        {route.page === "creator" && <CreatorView />}
        {route.page === "runs" && <RunsList />}
        {route.page === "run" && <RunView name={route.run} />}
        {route.page === "child" && <ChildView run={route.run} id={route.id} />}
        {route.page === "replay" && <ReplayView run={route.run} file={route.file} />}
      </main>
    </>
  );
}
