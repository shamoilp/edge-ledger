"""Anchored walk-forward of per-date IC, with a pre-out-of-sample purge.

Ported and modified from QuantJourney Backtester (Apache License 2.0,
Copyright (c) 2026 QuantJourney, https://github.com/QuantJourneyOrg/quantjourney-bt):
example WF03 "Anchored Walk-Forward With Pre-OOS Purging" and the engine pieces
it calls (walkforward/folds/anchored.py, folds/purge.py, statistics/overfit.py,
statistics/interpretation.py). Changes: rewritten in stdlib + numpy, and
retargeted from a strategy's Sharpe/CAGR to a factor's per-date IC.

WHY THE PURGE
-------------
An in-sample row dated d carries a label (its H-day forward return) that runs
past d. If that label reaches into the out-of-sample window, the two halves
share returns and the comparison leaks. So every IS date whose label could
reach the first OOS date is dropped. At H = 20 trading days that is ~5
calendar weeks of IS per fold, which on a short journal is most of it.

TWO MODES
---------
evaluate_fixed   One factor, nothing fitted: does the IC hold on later dates?
                 A stability check, not proof against overfitting, if the
                 factor was designed on the same data.
evaluate_select  Per fold, pick the candidate with the best IS mean IC and
                 score only that pick out of sample, against the average
                 candidate. This tests the selection procedure itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .ic import ic_stats, thinned

HOLIDAY_MARGIN_D = 6          # calendar slack for exchange-holiday clusters
DEFAULT_TRAIN_WEEKS = 4
DEFAULT_TEST_WEEKS = 2
DEFAULT_STEP_WEEKS = 2        # == test: OOS windows tile without overlap
DEFAULT_EXTRA_PURGE_PCT = 0.02
MIN_IS_DATES = 5
MIN_OOS_DATES = 3
MIN_FOLDS_FOR_VERDICT = 6


@dataclass(frozen=True)
class Fold:
    fold_id: int
    is_dates: tuple[str, ...]     # effective in-sample dates, after the purge
    purged: tuple[str, ...]       # raw in-sample dates removed before OOS
    oos_dates: tuple[str, ...]


def purge_cal_days(horizon_days: int) -> int:
    """Calendar days an H-trading-day label can reach forward.

    Over-purges on purpose: a calendar bound can drop extra IS dates but can
    never keep a leaking one. H = 20 gives 34.
    """
    return math.ceil(horizon_days * 7 / 5) + HOLIDAY_MARGIN_D


def purge(raw_is: list[str], oos_start: str, purge_days: int,
          extra_pct: float = 0.0) -> tuple[list[str], list[str]]:
    """Split sorted IS dates into (kept, purged).

    Drops dates within `purge_days` of the first OOS date, then
    int(extra_pct * len(raw_is)) more from the tail (WF03's percentage
    extension; like theirs, a pre-OOS purge and not a post-test embargo).
    """
    o = date.fromisoformat(oos_start)
    n_keep = sum(1 for d in raw_is if (o - date.fromisoformat(d)).days > purge_days)
    n_keep = max(0, n_keep - max(0, int(extra_pct * len(raw_is))))
    return list(raw_is[:n_keep]), list(raw_is[n_keep:])


def anchored_folds(dates, *, train_weeks: int = DEFAULT_TRAIN_WEEKS,
                   test_weeks: int = DEFAULT_TEST_WEEKS,
                   step_weeks: int = DEFAULT_STEP_WEEKS, purge_days: int,
                   extra_pct: float = DEFAULT_EXTRA_PURGE_PCT,
                   min_is: int = MIN_IS_DATES,
                   min_oos: int = MIN_OOS_DATES) -> tuple[list[Fold], int]:
    """Folds whose IS always starts at the first date and grows by step_weeks.

    Windows left with too few IS dates after the purge, or too few OOS dates,
    are skipped. Returns (folds, n_skipped).
    """
    if step_weeks < test_weeks:
        raise ValueError("step_weeks < test_weeks would make OOS windows overlap")
    ds = sorted(set(dates))
    if not ds:
        return [], 0
    end = date.fromisoformat(ds[-1])
    cursor = date.fromisoformat(ds[0]) + timedelta(weeks=train_weeks)
    folds: list[Fold] = []
    skipped = 0
    while cursor <= end:
        lo = cursor.isoformat()
        hi = min(cursor + timedelta(weeks=test_weeks, days=-1), end).isoformat()
        cursor += timedelta(weeks=step_weeks)
        raw_is = [d for d in ds if d < lo]
        oos = [d for d in ds if lo <= d <= hi]
        if len(oos) < min_oos or not raw_is:
            skipped += 1
            continue
        kept, purged = purge(raw_is, oos[0], purge_days, extra_pct)
        if len(kept) < min_is:
            skipped += 1
            continue
        folds.append(Fold(len(folds), tuple(kept), tuple(purged), tuple(oos)))
    return folds, skipped


# ── WF03 overfit metrics, on IC ─────────────────────────────────────────────
def overfit_ratio(is_ir: float | None, oos_ir: float | None) -> float | None:
    """IS IR / OOS IR. inf when OOS <= 0 < IS; 0.0 when both <= 0."""
    if is_ir is None or oos_ir is None:
        return None
    if oos_ir <= 0:
        return math.inf if is_ir > 0 else 0.0
    return is_ir / oos_ir


def efficiency(is_ic: float | None, oos_ic: float | None) -> float | None:
    """OOS mean IC / IS mean IC. None (not WF03's 0.0) when IS IC <= 0: there
    is no in-sample edge to transfer."""
    if is_ic is None or oos_ic is None or is_ic <= 0:
        return None
    return oos_ic / is_ic


def decay_slope(values: list[float]) -> float:
    """OLS slope of a per-fold series (WF03 sharpe_decay). 0.0 under 2 folds."""
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float),
                            np.asarray(values, dtype=float), 1)[0])


_LIGHTS = {
    "overfit_ratio": ((lambda v: v < 1.5, "< 1.5: robust"),
                      (lambda v: v <= 2.5, "1.5-2.5: caution"),
                      "> 2.5: likely overfit"),
    "efficiency": ((lambda v: v > 0.7, "> 0.7: robust transfer"),
                   (lambda v: v >= 0.4, "0.4-0.7: moderate degradation"),
                   "< 0.4: poor transfer"),
}


def traffic_lights(metrics: dict, pooled_oos_ic: float | None,
                   n_folds: int) -> list[tuple[str, float | None, str, str]]:
    """WF03's rules of thumb for the two scale-free ratios. Same gates: never
    green on a losing OOS, a negative ratio is not interpretable, and under 6
    folds every light is tagged indicative. Heuristics, not tests."""
    losing = pooled_oos_ic is not None and pooled_oos_ic <= 0
    out = []
    for name, value in metrics.items():
        if name not in _LIGHTS:
            continue
        if value is None:
            out.append((name, None, "n/a", "not computable"))
            continue
        (green, g_desc), (yellow, y_desc), r_desc = _LIGHTS[name]
        if name == "overfit_ratio" and losing:
            sig, desc = "red", "OOS IC <= 0: loses out of sample"
        elif name == "overfit_ratio" and value < 0:
            sig, desc = "yellow", "ratio < 0: IS IR negative, not interpretable"
        elif green(value):
            sig, desc = "green", g_desc
        elif yellow(value):
            sig, desc = "yellow", y_desc
        else:
            sig, desc = "red", r_desc
        if n_folds < MIN_FOLDS_FOR_VERDICT:
            desc += f" (only {n_folds} fold{'' if n_folds == 1 else 's'}: indicative)"
        out.append((name, value, sig, desc))
    return out


# ── evaluation ──────────────────────────────────────────────────────────────
def _mean(xs) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _summarize(per_fold: list[dict], pooled: dict[str, float], horizon_days: int) -> dict:
    is_means = [p["is"]["mean"] for p in per_fold]
    oos_means = [p["oos"]["mean"] for p in per_fold]
    held = sum(1 for i, o in zip(is_means, oos_means)
               if i is not None and o is not None and i * o > 0)
    return {
        "per_fold": per_fold,
        "n_folds": len(per_fold),
        "efficiency": efficiency(_mean(is_means), _mean(oos_means)),
        "overfit_ratio": overfit_ratio(_mean(p["is"]["ir"] for p in per_fold),
                                       _mean(p["oos"]["ir"] for p in per_fold)),
        "decay": decay_slope([o for o in oos_means if o is not None]),
        "sign_hold": (held, len(per_fold)),
        "pooled": ic_stats([pooled[d] for d in sorted(pooled)]),
        "pooled_thin": ic_stats(thinned(pooled, horizon_days)),
    }


def evaluate_fixed(ic_by_date: dict[str, float], folds: list[Fold],
                   horizon_days: int) -> dict:
    per_fold = [{"fold": f,
                 "is": ic_stats(ic_by_date[d] for d in f.is_dates),
                 "oos": ic_stats(ic_by_date[d] for d in f.oos_dates)} for f in folds]
    pooled = {d: ic_by_date[d] for f in folds for d in f.oos_dates}
    return _summarize(per_fold, pooled, horizon_days)


def evaluate_select(ics_by_factor: dict[str, dict[str, float]], folds: list[Fold],
                    horizon_days: int) -> dict:
    """Pick the best IS mean IC per fold; score only the pick out of sample.
    Folds must be built on dates every candidate covers."""
    per_fold, pooled, baseline = [], {}, {}
    for f in folds:
        is_s = {k: ic_stats(v[d] for d in f.is_dates) for k, v in ics_by_factor.items()}
        oos_s = {k: ic_stats(v[d] for d in f.oos_dates) for k, v in ics_by_factor.items()}
        pick = max(is_s, key=lambda k: is_s[k]["mean"])
        rank = 1 + sum(1 for k in oos_s if oos_s[k]["mean"] > oos_s[pick]["mean"])
        per_fold.append({"fold": f, "pick": pick, "is": is_s[pick],
                         "oos": oos_s[pick], "oos_rank": rank})
        for d in f.oos_dates:
            pooled[d] = ics_by_factor[pick][d]
            baseline[d] = sum(v[d] for v in ics_by_factor.values()) / len(ics_by_factor)
    res = _summarize(per_fold, pooled, horizon_days)
    res["baseline"] = ic_stats(baseline[d] for d in sorted(baseline))
    sel, base = res["pooled"]["mean"], res["baseline"]["mean"]
    res["skill"] = sel - base if sel is not None and base is not None else None
    return res
