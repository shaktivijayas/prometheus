import { useEffect, useState } from "react";

export type Route =
  | { page: "overview" }
  | { page: "creator" }
  | { page: "runs" }
  | { page: "run"; run: string }
  | { page: "child"; run: string; id: string }
  | { page: "replay"; run: string; file: string };

export function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  if (parts[0] === "creator") return { page: "creator" };
  if (parts[0] === "runs" && parts.length === 1) return { page: "runs" };
  if (parts[0] === "runs" && parts[1]) {
    if (parts[2] === "child" && parts[3]) return { page: "child", run: parts[1], id: parts[3] };
    if (parts[2] === "replay" && parts[3]) return { page: "replay", run: parts[1], file: parts[3] };
    return { page: "run", run: parts[1] };
  }
  return { page: "overview" };
}

export const href = {
  overview: () => "#/",
  creator: () => "#/creator",
  runs: () => "#/runs",
  run: (run: string) => `#/runs/${encodeURIComponent(run)}`,
  child: (run: string, id: string) => `#/runs/${encodeURIComponent(run)}/child/${encodeURIComponent(id)}`,
  replay: (run: string, file: string) => `#/runs/${encodeURIComponent(run)}/replay/${encodeURIComponent(file)}`,
};

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => {
      setRoute(parse(window.location.hash));
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
