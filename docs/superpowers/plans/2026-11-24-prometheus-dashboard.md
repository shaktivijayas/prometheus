# PROMETHEUS Lab Dashboard Implementation Plan (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static lab dashboard that makes the whole ecosystem inspectable from real recorded data. It shows the headline verified results, how the Creator was trained and whether it improved, every run's family tree, each child's profile (design, lessons, reason, predicted vs actual), the event timeline, generation charts, and a replay of any champion's episode, including the moment the laws change. It deploys to GitHub Pages.

**Architecture:** `prometheus export-dashboard` (Python) collects the chosen runs, comparisons and Creator data into `dashboard/public/data/` with one `index.json`. The dashboard (Vite + React + TypeScript) fetches those JSON files and renders everything client-side. Charts are hand-built SVG (no chart library) so they look consistent and stay small. Pure logic (tree layout, replay reconstruction, aggregations) lives in `src/lib/` and is unit-tested with Vitest. A GitHub Actions workflow builds and deploys `main` to Pages.

**Tech Stack:** Vite 5, React 18, TypeScript 5, Vitest, plain CSS with design tokens (light and dark), GitHub Actions + Pages.

**Spec:** `docs/superpowers/specs/2026-11-24-prometheus-core-design.md` §6.6 (data contract) and §6.7 (views).

## Global Constraints

- **Real data only.** Every number and picture comes from exported run files; nothing is hand-entered or mocked in the shipped app (test fixtures live under `src/lib/__fixtures__`).
- **Honest framing:** the headline compares Creators by **verified gain** (fresh-upbringing re-scores). Any in-run "best" curve is labelled "selection-biased".
- The dashboard is read-only and static: no backend, no API keys, no external network calls beyond its own files.
- Works at phone width (16 px gutters, no horizontal page scroll; wide tables and trees scroll inside their own panels). Light and dark themes come from CSS tokens.
- npm cache and temp files go on D: (`npm_config_cache=D:\npm-cache`), because C: is full.
- Commits: small and frequent. No AI co-author or attribution trailers.

## Review Focus

1. **Missing optional data:** a run without replays, a comparison without a neural arm, or no Creator training data yet should show an explanatory empty state, never crash. Tested in Task 2 (loader) and Task 4 (views render with partial index).
2. **Replay fidelity:** the replay grid at step *t* must equal the world's real grid. Reconstruction applies `consumed`, `respawn` and shifted `roles` exactly, tested against a fixture made by the engine (Task 2).
3. **Large runs:** trees with 60+ nodes and traces of 200 steps stay responsive. The layout is O(n), and replay frames are computed incrementally.
4. **Null fields** (`predicted_change: null`, `parent_id: null`, `adaptation_ratio` edge values) render as "n/a", not as `NaN` or `null` text.
5. **Pages base path:** asset and data URLs work under `/prometheus/` on GitHub Pages and at `/` locally (Vite `base` from the env).

---

### Task 1: Export command

**Files:** Create `engine/prometheus/export.py` and `engine/tests/test_export.py`; modify `engine/prometheus/cli.py`.

**Interfaces:**
- `export_dashboard(out_dir: Path, runs: list[Path], comparisons: list[Path], creator_dir: Path | None) -> dict` copies each run directory to `out_dir/runs/<run name>/` (run.json, entities.json, events.jsonl → converted to `events.json` as an array, generations.json, replays/*.json plus a `replays/index.json` listing them). It copies comparisons to `out_dir/comparisons/`, and `creator/progress.json` + `creator/training_history.json` to `out_dir/creator/`. It writes `out_dir/index.json`: `{"generated_at", "runs": [{"name", "creator", "seed", "generations", "children", "verification", "summary"}], "comparisons": [{"name", "file", "creators", "seeds"}], "creator": {"progress": bool, "history": bool}}`.
- CLI `prometheus export-dashboard --out dashboard/public/data --runs runs/final_*_s0 ... --comparisons comparisons/final.json comparisons/baseline.json --creator creator`. Globs are expanded by Python, not the shell.

- [ ] Failing tests: export of a tiny run (made with `run_evolution`) produces an index entry whose fields match run.json; events become a JSON array; replay index lists every replay; a missing creator dir gives `creator: {progress: false, history: false}`; strict JSON throughout.
- [ ] Implement, run the full suite, commit `"Add dashboard export command"`.

### Task 2: Dashboard scaffold, data types and pure logic

**Files:** `dashboard/` (Vite React TS scaffold: `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`), `src/lib/types.ts`, `src/lib/data.ts`, `src/lib/tree.ts`, `src/lib/replay.ts`, `src/lib/stats.ts`, with tests `src/lib/*.test.ts` and fixtures `src/lib/__fixtures__/` (a tiny real run exported by Task 1).

**Interfaces:**
- `types.ts`: TypeScript types mirroring the §6.6 contract: `RunIndex`, `RunJson`, `Entity`, `Metrics`, `EventRow`, `GenerationRow`, `Replay`, `TraceStep`, `Comparison`, `ProgressRow`, `HistoryRow`.
- `data.ts`: `loadIndex()`, `loadRun(name) -> {run, entities, events, generations, replays: ReplayRef[]}`, `loadReplay(run, key)`, `loadComparison(file)`, `loadCreator()`. Each uses `import.meta.env.BASE_URL` and returns typed data or a typed `{missing: true}`.
- `tree.ts`: `buildForest(entities) -> TreeNode[]` (roots = gen-0) and `layoutForest(forest) -> {nodes: {id, x, y, entity}[], links, width, height}`, where x = generation column and y = leaf order (tidy, no overlaps).
- `replay.ts`: `frames(replay) -> Frame[]`, where frame *t* holds `{grid, pos, energy, beliefs, roles, visibleRadius, event}`, built incrementally from the trace (apply `consumed`, `respawn`, shifted `roles`).
- `stats.ts`: `curriculumPatterns(entities)` (lesson-choice frequency by parent failure type), `predictionPairs(entities)` (predicted vs actual), `geneDiff(child, parent)`.

- [ ] Scaffold with npm (cache on D:). Add Vitest. Failing tests: the replay reconstruction's final frame equals a grid the engine computed (the fixture carries `final_grid`, added to the fixture by the Task 1 test helper); the tree layout has no overlapping nodes and the right parent links; `curriculumPatterns` counts; `geneDiff` signs. Implement, `npm test` and `npx tsc --noEmit`, then commit `"Scaffold dashboard with typed data layer"` and `"Add tree, replay and stats logic"`.

### Task 3: Design system and app shell

**Files:** `src/styles/tokens.css`, `src/styles/app.css`, `src/App.tsx`, `src/components/{Nav,Panel,Stat,EmptyState,Chart}.tsx`.

- Hash-based routes (works on Pages without server rewrites): `#/` (Overview), `#/creator`, `#/runs/:name`, `#/runs/:name/child/:id`, `#/runs/:name/replay/:key`.
- `Chart.tsx`: minimal SVG primitives (axes, line with error band, bars with error bars, scatter) used by every chart.
- Tokens: a neutral surface palette with one accent for "Creator" and a categorical palette for arms (trained, noteach, fresh, random, clone, heuristic). Light and dark.
- [ ] Build the shell with an empty state when `index.json` is missing. `npm run build` passes. Commit `"Add dashboard shell and design tokens"`.

### Task 4: Views

**Files:** `src/views/{Overview,Creator,Run,Child,Replay}.tsx` and their small components.

- **Overview:** the headline. A verified-gain bar chart per arm with ± std across seeds, from the `final` comparison (falling back to `baseline`). Four answer cards (generations improve? trained beats random/heuristic? teaching adds value? experienced beats fresh?), computed from the comparison data with the numbers shown. Links to runs.
- **Creator:** training curves from history (mean reward, final mean fitness, prediction MAE, entropy) with checkpoint markers; checkpoint progress (verified champion and gain per version, ± std); teaching patterns (stacked bars of lesson presets per lesson slot, by parent failure type); prediction scatter (predicted vs actual change), coloured by arm.
- **Run:** summary and verification; the generation chart (best/mean, best labelled selection-biased); the genealogy tree (zoom and pan, nodes coloured by fitness, retired greyed, champions ringed); a filterable timeline.
- **Child profile:** identity, parent link, reason, curriculum with per-lesson scores, predicted vs actual, a genome diff vs parent (bars), metrics, the belief table vs the universe's true laws, and events.
- **Replay:** canvas grid with tile kinds coloured, the agent, its visible area, a belief panel, an energy line, and play/pause/scrub/speed controls. Marks the step where the laws change and shows the new laws.
- [ ] Each view handles missing data (Review Focus 1, 4). `npm run build`; check visually in the browser at desktop and phone width, light and dark. Commit one view per commit.

### Task 5: Publish

- [ ] `.github/workflows/pages.yml`: Node 22, `npm ci`, `npm test`, `npm run build` with `BASE=/prometheus/`, deploy `dashboard/dist`. Enable Pages (Actions source) via `gh api`.
- [ ] Export the final data (after Plan 2 Task 6), commit `dashboard/public/data`, push `main`, wait for the deploy, and confirm the live URL returns 200 and renders. Put the live link in the README and the repo homepage. Commit `"Publish the lab dashboard"`.
