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
    p.add_argument("--replicates", type=int, default=d.replicates,
                   help="independent upbringings averaged per child (reduces luck)")
    p.add_argument("--verify-replicates", type=int, default=d.verify_replicates,
                   help="fresh upbringings used to verify gen-0 best and final champion (0 = skip)")


def _config(args, creator: str, seed: int) -> RunConfig:
    return RunConfig(creator=creator, generations=args.generations, population=args.population, elite=args.elite,
                     seed=seed, episodes_per_lesson=args.episodes_per_lesson, eval_worlds=args.eval_worlds,
                     workers=args.workers, replicates=args.replicates,
                     verify_replicates=args.verify_replicates)


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
    print(f"In-run curve (selection-biased): Gen 0 best {first:.3f} -> "
          f"Gen {len(result.generations) - 1} best {last:.3f} ({pct})")
    v = result.verification
    vpct = f"{v['gain_pct']:+.1f}%" if v.get("gain_pct") is not None else "n/a"
    print(f"Verified on {v['final_best']['n']} fresh upbringings: gen-0 best design {v['gen0_best']['mean']:.3f}"
          f" -> final champion {v['final_best']['mean']:.3f} (gain {v['gain']:+.3f}, {vpct})")
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
    print("\nVerified results (champions re-scored on fresh upbringings; mean +/- std across seeds)")
    print(f"{'creator':<24} {'final champion':>18} {'gen-0 best':>11} {'verified gain':>19} {'in-run best*':>13}")
    for creator, arm in data["arms"].items():
        print(f"{creator:<24} {arm['verified_final_mean']:>9.3f} +/- {arm['verified_final_std']:<5.3f}"
              f" {arm['verified_gen0_mean']:>11.3f} {arm['verified_gain_mean']:>+10.3f} +/- {arm['verified_gain_std']:<5.3f}"
              f" {arm['best_mean'][-1]:>13.3f}")
    print("* in-run best is selection-biased (it keeps lucky draws); judge Creators by the verified gain")
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
    cmp_.add_argument("--creators", default="clone,heuristic,random")
    cmp_.add_argument("--seeds", default="0,1,2")
    cmp_.add_argument("--out", default=".")
    cmp_.add_argument("--name", default="comparison")
    _sizing(cmp_)
    cmp_.set_defaults(func=cmd_compare)

    from .creators.training import TrainConfig
    t = TrainConfig()
    tr = sub.add_parser("train-creator", help="train the Creator network with reinforcement learning")
    tr.add_argument("--iterations", type=int, default=t.iterations, help="iterations to run (more, when resuming)")
    tr.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    tr.add_argument("--runs-per-iteration", type=int, default=t.runs_per_iteration)
    tr.add_argument("--checkpoint-every", type=int, default=t.checkpoint_every)
    tr.add_argument("--generations", type=int, default=t.generations)
    tr.add_argument("--population", type=int, default=t.population)
    tr.add_argument("--elite", type=int, default=t.elite)
    tr.add_argument("--episodes-per-lesson", type=int, default=t.episodes_per_lesson)
    tr.add_argument("--eval-worlds", type=int, default=t.eval_worlds)
    tr.add_argument("--replicates", type=int, default=t.replicates)
    tr.add_argument("--lr", type=float, default=t.lr)
    tr.add_argument("--workers", type=int, default=t.workers)
    tr.add_argument("--seed", type=int, default=t.seed)
    tr.add_argument("--checkpoints", default=None, help="checkpoint directory (default: <repo>/checkpoints)")
    tr.set_defaults(func=cmd_train_creator)

    pg = sub.add_parser("creator-progress", help="evaluate Creator checkpoints on held-out universes")
    pg.add_argument("--versions", default="all", help="'all' or a list like v000,v010")
    pg.add_argument("--seeds", default="0,1,2,3,4")
    pg.add_argument("--out", default=".")
    pg.add_argument("--checkpoints", default=None)
    _sizing(pg)
    pg.set_defaults(func=cmd_creator_progress)
    return parser


def cmd_train_creator(args) -> None:
    from .creators.checkpoints import default_checkpoint_dir
    from .creators.training import TrainConfig, train
    ckpt = Path(args.checkpoints) if args.checkpoints else default_checkpoint_dir()
    cfg = TrainConfig(iterations=args.iterations, runs_per_iteration=args.runs_per_iteration,
                      checkpoint_every=args.checkpoint_every, generations=args.generations,
                      population=args.population, elite=args.elite, episodes_per_lesson=args.episodes_per_lesson,
                      eval_worlds=args.eval_worlds, replicates=args.replicates, lr=args.lr, workers=args.workers,
                      seed=args.seed)
    print(f"{'iter':>5} {'reward':>8} {'gen0 mean':>9} {'final mean':>10} {'improve':>8} {'pred MAE':>8} "
          f"{'entropy':>8} {'kl':>7} {'secs':>6} ckpt", flush=True)

    def log(row):
        ck = f"v{row['version']:03d}" if row.get("version") else ""
        print(f"{row['iteration']:>5} {row['mean_reward']:>+8.3f} {row['gen0_mean_fitness']:>9.3f} "
              f"{row['final_mean_fitness']:>10.3f} {row['improvement']:>+8.3f} {row['prediction_mae']:>8.3f} "
              f"{row['entropy']:>8.2f} {row['approx_kl']:>7.4f} {row['seconds']:>6.1f} {ck}", flush=True)

    train(cfg, ckpt, ckpt / "training_history.json", resume=args.resume, log=log)
    print(f"checkpoints in {ckpt}")


def cmd_creator_progress(args) -> None:
    from .creators.checkpoints import default_checkpoint_dir
    from .progress import creator_progress, resolve_versions
    ckpt = Path(args.checkpoints) if args.checkpoints else default_checkpoint_dir()
    try:
        versions = resolve_versions(args.versions, ckpt)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = creator_progress(versions, seeds, _config(args, "neural", seeds[0]), Path(args.out), ckpt,
                            progress=lambda v, s, r: print(f"  {v} seed {s}: verified champion "
                                                           f"{r.verification['final_best']['mean']:.3f}", flush=True))
    print(f"\n{'version':<8} {'iteration':>9} {'verified champion':>20} {'verified gain':>18} {'pred MAE':>9}")
    for r in rows:
        mae = f"{r['prediction_mae']:.3f}" if r["prediction_mae"] is not None else "n/a"
        print(f"{r['version']:<8} {r['iteration']:>9} {r['verified_final_mean']:>11.3f} +/- {r['verified_final_std']:<5.3f}"
              f" {r['verified_gain_mean']:>+9.3f} +/- {r['verified_gain_std']:<5.3f} {mae:>9}")
    print(f"wrote {Path(args.out) / 'creator' / 'progress.json'}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
