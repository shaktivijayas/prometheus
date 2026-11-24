import { useEffect, useState } from "react";
import type { Loaded } from "./data";

export type State<T> = { status: "loading" } | { status: "ready"; value: T } | { status: "missing"; reason: string };

export function useLoad<T>(fn: () => Promise<Loaded<T>>, deps: unknown[]): State<T> {
  const [state, setState] = useState<State<T>>({ status: "loading" });
  useEffect(() => {
    let alive = true;
    setState({ status: "loading" });
    fn().then((r) => {
      if (!alive) return;
      setState(r.ok ? { status: "ready", value: r.value } : { status: "missing", reason: r.reason });
    });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
