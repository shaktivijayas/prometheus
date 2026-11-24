"""Command line: `prometheus run` and `prometheus compare`."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .creators import CREATOR_NAMES, get_creator
from .runner import RunConfig


def _sizing(p: argparse.ArgumentParser) -> None:
    d = RunConfig()
    p.add_argument("--generations", type=int, default=d.generations)
    p.add_argument("--population", type=int, default=d.population)
    p.add_argument("--elite", type=int, default=d.elite)
    p.add_argument("--episodes-per-lesson", type=int, default=d.episodes_per_lesson)
    p.add_argument("--eval-worlds", type=int, default=d.eval_worlds)
    p.add_argument("--workers", type=int, default=d.workers, help="0 = all cores")


def _config(args, creator: str, seed: int) -> RunConfig:
    return RunConfig(creator=creator, generations=args.generations, population=args.population, elite=args.elite,
                     seed=seed, episodes_per_lesson=args.episodes_per_lesson, eval_worlds=args.eval_worlds,
                     workers=args.workers)


def _check_creators(names: list[str]) -> None:
    for name in names:
        try:
            get_creator(name)
        except ValueError:
            print(f"error: unknown creator {name!r}. Valid creators: {', '.join(sorted(CREATOR_NAMES))}",
                  file=sys.stderr)
            raise SystemExit(2)
        except Exception as exc:  # e.g. a neural checkpoint that does not exist
            print(f"error: cannot load creator {name!r}: {exc}", file=sys.stderr)
            raise SystemExit(2)


def cmd_run(args) -> None:
    from .output import write_run
    from .runner import run_evolution
    _check_creators([args.creator])
    out = Path(args.out or f"runs/{args.creator.replace(':', '-')}_s{args.seed}")
    t = time.perf_counter()
    result = run_evolution(_config(args, args.creator, args.seed))
    write_run(result, out)
    print(f"{'gen':>4} {'best':>8} {'mean':>8} {'diversity':>9}")
    for g in result.generations:
        print(f"{g['generation']:>4} {g['best']:>8.3f} {g['mean']:>8.3f} {g['diversity']:>9.3f}")
    first, last = result.generations[0]["best"], result.generations[-1]["best"]
    pct = f"{100 * (last - first) / first:+.1f}%" if first > 0 else "n/a"
    print(f"Gen 0 best {first:.3f} -> Gen {len(result.generations) - 1} best {last:.3f} ({pct})")
    print(f"wrote {out} in {time.perf_counter() - t:.1f}s")


def cmd_compare(args) -> None:
    from .compare import compare
    creators = [c.strip() for c in args.creators.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    _check_creators(creators)
    t = time.perf_counter()

    def progress(creator, seed, result):
        print(f"  {creator} seed {seed}: best {result.generations[-1]['best']:.3f}", flush=True)

    data = compare(creators, seeds, _config(args, creators[0], seeds[0]), Path(args.out), args.name, progress)
    print(f"\n{'creator':<24} {'final best (mean +/- std)':>24} {'gen 0 best':>11} {'gain':>8}")
    for creator, arm in data["arms"].items():
        final, std, first = arm["best_mean"][-1], arm["best_std"][-1], arm["best_mean"][0]
        gain = f"{100 * (final - first) / first:+.1f}%" if first > 0 else "n/a"
        print(f"{creator:<24} {final:>13.3f} +/- {std:<6.3f} {first:>11.3f} {gain:>8}")
    print(f"wrote {Path(args.out) / 'comparisons' / (args.name + '.json')} in {time.perf_counter() - t:.1f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prometheus", description="PROMETHEUS: Creator-driven evolution of child intelligences")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evolve one population with one Creator")
    run.add_argument("--creator", default="heuristic")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", default=None)
    _sizing(run)
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="compare Creators across seeds on identical budgets")
    cmp_.add_argument("--creators", default="heuristic,random")
    cmp_.add_argument("--seeds", default="0,1,2")
    cmp_.add_argument("--out", default=".")
    cmp_.add_argument("--name", default="comparison")
    _sizing(cmp_)
    cmp_.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
