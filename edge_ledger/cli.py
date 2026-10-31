"""edge-ledger command line.

  edge-ledger ic          JOURNAL --factor A B --returns fwd_20
  edge-ledger permute     JOURNAL --factor A --returns fwd_20
  edge-ledger walkforward JOURNAL --factor A --returns fwd_20
  edge-ledger walkforward JOURNAL --select A B C --returns fwd_20
  edge-ledger ledger      add NAME | update NAME | list | bar
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from . import ledger as L
from . import permutation as P
from . import walkforward as W
from .data import JournalError, load_csv, pairs_by_date
from .ic import MIN_N_FOR_T, MIN_NAMES, date_ics, ic_stats, thinned

DEFAULT_LEDGER = "edge_ledger.json"


def _f(x: float | None, places: int = 3) -> str:
    if x is None:
        return "—"
    if math.isinf(x):
        return "inf"
    return f"{x:+.{places}f}"


def _horizon(args: argparse.Namespace) -> int:
    if args.horizon_days:
        return args.horizon_days
    m = re.search(r"(\d+)\s*$", args.returns)
    if not m:
        raise SystemExit(f"cannot read a horizon from column {args.returns!r}; "
                         "pass --horizon-days")
    return int(m.group(1))


def _journal(args: argparse.Namespace, factors: list[str]) -> list[dict]:
    try:
        rows, columns = load_csv(args.journal)
    except (OSError, JournalError) as exc:
        raise SystemExit(str(exc)) from None
    missing = [c for c in [*factors, args.returns] if c not in columns]
    if missing:
        raise SystemExit(f"{args.journal}: no column(s) {missing}")
    return rows


def _n_tests(args: argparse.Namespace, floor: int) -> tuple[int, str]:
    """N for the multiplicity bar: the ledger's count, or `floor` with a warning."""
    if args.ledger and Path(args.ledger).exists():
        n = L.count(L.load(args.ledger))
        return max(n, floor), f"ledger {args.ledger}"
    return floor, "no --ledger: only this command's tests counted, your real N is higher"


def _clear_call(t: float | None, n_obs: int, n_tests: int) -> tuple[str, float | None]:
    if t is None or n_obs < MIN_N_FOR_T:
        return "too few dates to test", None
    bar = L.bonferroni_bar(n_tests, n_obs=n_obs)
    if abs(t) < bar:
        return "does not clear", bar
    return ("CLEARS" if t > 0 else "significantly NEGATIVE"), bar


# ── ic ──────────────────────────────────────────────────────────────────────
def cmd_ic(args: argparse.Namespace) -> int:
    rows = _journal(args, args.factor)
    h = _horizon(args)
    n_tests, basis = _n_tests(args, len(args.factor))
    print(f"\n  Per-date IC · returns {args.returns} (H = {h}) · N = {n_tests} ({basis})")
    print(f"\n  {'factor':<18}{'dates':>6}{'mean IC':>9}{'IR':>7}{'t(all)':>8}  |"
          f"{'thin n':>7}{'t(thin)':>9}{'bar':>7}  verdict")
    for factor in args.factor:
        ics = date_ics(pairs_by_date(rows, factor, args.returns), args.min_names)
        full, thin = ic_stats(ics.values()), ic_stats(thinned(ics, h))
        call, bar = _clear_call(thin["t"], thin["n"], n_tests)
        print(f"  {factor:<18}{full['n']:>6}{_f(full['mean']):>9}{_f(full['ir'], 2):>7}"
              f"{_f(full['t'], 2):>8}  |{thin['n']:>7}{_f(thin['t'], 2):>9}"
              f"{(f'{bar:.2f}' if bar else '—'):>7}  {call}")
    print(f"\n  t(all) uses every date, and overlapping {h}-day returns inflate it.")
    print(f"  The verdict uses every {h}th date against a Bonferroni t(n-1) bar.\n")
    return 0


# ── permute ─────────────────────────────────────────────────────────────────
def cmd_permute(args: argparse.Namespace) -> int:
    rows = _journal(args, [args.factor])
    h = _horizon(args)
    every = 1 if args.all_dates else h
    by_date = pairs_by_date(rows, args.factor, args.returns)
    dates = P.select_dates(by_date, args.min_names, every)
    res = P.within_date_permutation(by_date, dates, args.iters, args.seed)
    n_tests, basis = _n_tests(args, 1)
    print(f"\n  Permutation test · {args.factor} vs {args.returns}"
          f" · {'all dates' if every == 1 else f'every {every}th date'}"
          f" · N = {n_tests} ({basis})")
    if res["n"] == 0:
        print("  no usable dates\n")
        return 0
    if every == 1 and h > 1:
        print(f"  ! all dates with overlapping {h}-day returns: p will be too small")
    boot = P.date_bootstrap(date_ics({d: by_date[d] for d in dates}, args.min_names).values(),
                            seed=args.seed)
    print(f"  dates {res['n']} · real mean IC {_f(res['real_mean'], 4)}")
    print(f"  luck: mean {_f(res['null_mean'], 4)}, sd {res['null_sd']:.4f}"
          f" · real IC sits at the {res['percentile']:.1f}th percentile of luck")
    print(f"  p = {res['p']:.4f} ({'positive' if res['real_mean'] >= 0 else 'negative'}"
          f" direction) → {P.verdict(res['p'], n_tests)}")
    if boot["lo"] is not None:
        straddle = boot["lo"] <= 0 <= boot["hi"]
        print(f"  date bootstrap 90% CI [{_f(boot['lo'], 4)}, {_f(boot['hi'], 4)}]"
              f" → {'straddles 0' if straddle else 'excludes 0'}\n")
    return 0


# ── walkforward ─────────────────────────────────────────────────────────────
def _print_folds(res: dict, select: bool) -> None:
    print(f"\n  {'fold':>4}  {'in-sample':<22}{'n':>4}{'purged':>7}  {'out-of-sample':<22}"
          f"{'n':>4}{'IS IC':>8}{'OOS IC':>8}  {'pick (OOS rank)' if select else 'sign'}")
    for p in res["per_fold"]:
        f = p["fold"]
        if select:
            tail = f"{p['pick']} ({p['oos_rank']})"
        else:
            tail = "held" if (p["is"]["mean"] or 0) * (p["oos"]["mean"] or 0) > 0 else "flipped"
        print(f"  {f.fold_id:>4}  {f.is_dates[0]}..{f.is_dates[-1]}{len(f.is_dates):>4}"
              f"{len(f.purged):>7}  {f.oos_dates[0]}..{f.oos_dates[-1]}{len(f.oos_dates):>4}"
              f"{_f(p['is']['mean']):>8}{_f(p['oos']['mean']):>8}  {tail}")


def _print_wf_summary(res: dict, h: int, n_tests: int, select: bool) -> None:
    print("\n  Rules of thumb (WF03 heuristics, not tests)")
    metrics = {"overfit_ratio": res["overfit_ratio"], "efficiency": res["efficiency"]}
    for name, value, sig, desc in W.traffic_lights(metrics, res["pooled"]["mean"],
                                                   res["n_folds"]):
        print(f"    {name:<14}{_f(value, 2):>7}  [{sig}] {desc}")
    held, n = res["sign_hold"]
    print(f"    {'decay/fold':<14}{_f(res['decay']):>7}  slope of OOS IC across folds")
    print(f"    {'sign held':<14}{f'{held}/{n}':>7}  folds where OOS IC kept the IS sign")
    full, thin = res["pooled"], res["pooled_thin"]
    call, bar = _clear_call(thin["t"], thin["n"], n_tests)
    print(f"\n  Pooled out-of-sample, {full['n']} dates: mean IC {_f(full['mean'])},"
          f" t(all) {_f(full['t'], 2)} (inflated by overlap)")
    print(f"  every {h}th date: n {thin['n']}, t {_f(thin['t'], 2)}"
          f"{f', bar {bar:.2f}' if bar else ''} → {call}")
    if select:
        print(f"  average candidate OOS IC {_f(res['baseline']['mean'])}"
              f" → selection skill {_f(res['skill'])}")
    print()


def cmd_walkforward(args: argparse.Namespace) -> int:
    factors = args.select or args.factor
    rows = _journal(args, factors)
    h = _horizon(args)
    n_tests, basis = _n_tests(args, len(factors))
    ics = {f: date_ics(pairs_by_date(rows, f, args.returns), args.min_names) for f in factors}
    common = sorted(set.intersection(*(set(v) for v in ics.values())))
    purge_days = max(args.purge_cal_days, W.purge_cal_days(h))
    try:
        folds, skipped = W.anchored_folds(
            common, train_weeks=args.train_weeks, test_weeks=args.test_weeks,
            step_weeks=args.step_weeks, purge_days=purge_days,
            extra_pct=args.extra_purge_pct)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    label = f"select from {', '.join(factors)}" if args.select else factors[0]
    print(f"\n  Purged anchored walk-forward · {label} · {args.returns} (H = {h})")
    print(f"  {len(common)} dates · purge {purge_days} calendar days"
          f" + {args.extra_purge_pct:.0%} of IS · {len(folds)} folds, {skipped} skipped"
          f" · N = {n_tests} ({basis})")
    if not folds:
        print("  no fold survives the purge: needs more dates or a shorter horizon\n")
        return 0
    res = (W.evaluate_select(ics, folds, h) if args.select
           else W.evaluate_fixed(ics[factors[0]], folds, h))
    _print_folds(res, bool(args.select))
    _print_wf_summary(res, h, n_tests, bool(args.select))
    return 0


# ── ledger ──────────────────────────────────────────────────────────────────
def _tests(n: int) -> str:
    return f"{n} test" if n == 1 else f"{n} tests"


def cmd_ledger(args: argparse.Namespace) -> int:
    path = args.ledger or DEFAULT_LEDGER
    led = L.load(path)
    try:
        if args.action == "add":
            led = L.add(led, args.name, verdict=args.verdict, status=args.status,
                        source=args.source)
            L.save(path, led)
            print(f"registered {args.name!r}; ledger now holds {_tests(L.count(led))}")
        elif args.action == "update":
            led = L.update(led, args.name, status=args.status, verdict=args.verdict)
            L.save(path, led)
            print(f"updated {args.name!r}")
        elif args.action == "list":
            for t in led["tests"]:
                print(f"  {t['name']:<24}{t['registered']:<12}{t['status']:<15}{t['verdict']}")
    except L.LedgerError as exc:
        raise SystemExit(str(exc)) from None
    n = L.count(led)
    obs = f", t({args.n_obs - 1}) for {args.n_obs} dates" if getattr(args, "n_obs", None) else ""
    bar = L.bonferroni_bar(n, n_obs=getattr(args, "n_obs", None))
    print(f"\n  {_tests(n)} registered → adoption bar |t| >= {bar:.2f}"
          f" (two-sided Bonferroni, FWER 5%{obs})\n")
    return 0


# ── parser ──────────────────────────────────────────────────────────────────
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("journal", help="CSV with date, symbol, factor and return columns")
    p.add_argument("--returns", required=True, help="forward-return column, e.g. fwd_20")
    p.add_argument("--horizon-days", type=int, default=0,
                   help="return horizon in trading days (default: digits in --returns)")
    p.add_argument("--min-names", type=int, default=MIN_NAMES)
    p.add_argument("--ledger", default=None, help="ledger JSON; its size sets N")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edge-ledger",
                                     description="Is this stock-selection signal real?")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ic", help="per-date IC with an overlap-aware t and multiplicity bar")
    _common(p)
    p.add_argument("--factor", nargs="+", required=True)
    p.set_defaults(func=cmd_ic)

    p = sub.add_parser("permute", help="within-date permutation test + date bootstrap")
    _common(p)
    p.add_argument("--factor", required=True)
    p.add_argument("--iters", type=int, default=P.DEFAULT_ITERS)
    p.add_argument("--seed", type=int, default=P.SEED)
    p.add_argument("--all-dates", action="store_true",
                   help="use every date (only sound for 1-day returns)")
    p.set_defaults(func=cmd_permute)

    p = sub.add_parser("walkforward", help="purged anchored walk-forward of the IC")
    _common(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--factor", nargs=1)
    g.add_argument("--select", nargs="+", help="pick the best in-sample candidate per fold")
    p.add_argument("--train-weeks", type=int, default=W.DEFAULT_TRAIN_WEEKS)
    p.add_argument("--test-weeks", type=int, default=W.DEFAULT_TEST_WEEKS)
    p.add_argument("--step-weeks", type=int, default=W.DEFAULT_STEP_WEEKS)
    p.add_argument("--purge-cal-days", type=int, default=0,
                   help="raise the purge; the horizon's minimum always applies")
    p.add_argument("--extra-purge-pct", type=float, default=W.DEFAULT_EXTRA_PURGE_PCT)
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("ledger", help="register tests; print the adoption bar")
    p.add_argument("--ledger", default=None, help=f"ledger JSON (default {DEFAULT_LEDGER})")
    acts = p.add_subparsers(dest="action", required=True)
    a = acts.add_parser("add")
    a.add_argument("name")
    a.add_argument("--verdict", default="registered, result pending")
    a.add_argument("--status", default="queued", choices=L.VALID_STATUS)
    a.add_argument("--source", default="")
    u = acts.add_parser("update")
    u.add_argument("name")
    u.add_argument("--verdict", default=None)
    u.add_argument("--status", default=None, choices=L.VALID_STATUS)
    acts.add_parser("list")
    b = acts.add_parser("bar")
    b.add_argument("--n-obs", type=int, default=None, help="dates behind the t-stat")
    p.set_defaults(func=cmd_ledger)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "min_names", MIN_NAMES) < 3:
        raise SystemExit("--min-names must be >= 3 (Spearman needs 3 names)")
    if hasattr(args, "extra_purge_pct") and not 0 <= args.extra_purge_pct < 1:
        raise SystemExit("--extra-purge-pct must be in [0, 1)")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
