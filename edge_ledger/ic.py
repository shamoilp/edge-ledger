"""Per-date cross-sectional information coefficient (IC).

The IC on a date is the Spearman rank correlation between a factor and the
forward return, across the names logged that day. Averaging per-date ICs,
rather than pooling every row into one correlation, stops a few unusual dates
from carrying the whole result (the Alphalens convention).
"""
from __future__ import annotations

import math

import numpy as np

from .data import Pairs

MIN_NAMES = 5      # fewest names on a date for its IC to count
MIN_N_FOR_T = 3    # fewest dates an IR / t-stat is reported on


def average_ranks(values) -> np.ndarray:
    """Ranks with ties sharing their average rank, as Spearman requires.

    Plain argsort ranking breaks ties by input order. On a discrete factor (a
    0-10 score, a stage label) most names are tied, so the IC would then depend
    on how the file happened to be sorted.
    """
    a = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2.0)[inverse]


def spearman(xs, ys) -> float | None:
    """Spearman rank correlation, or None when undefined (n < 3 or a constant side)."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return None
    c = float(np.corrcoef(rx, ry)[0, 1])
    return None if math.isnan(c) else c


def date_ics(by_date: Pairs, min_names: int = MIN_NAMES) -> dict[str, float]:
    """IC for every date with at least `min_names` usable names."""
    out: dict[str, float] = {}
    for d, pairs in by_date.items():
        if len(pairs) < min_names:
            continue
        ic = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            out[d] = ic
    return out


def ic_stats(ics) -> dict:
    """n / mean / IR / t / % positive over a list of per-date ICs.

    IR = mean / sd, t = IR * sqrt(n). IR and t need MIN_N_FOR_T dates: a t from
    two points prints as evidence when it is noise.
    """
    vals = [float(x) for x in ics]
    if not vals:
        return {"n": 0, "mean": None, "ir": None, "t": None, "pos": None}
    a = np.asarray(vals)
    std = float(a.std(ddof=1)) if len(a) >= MIN_N_FOR_T else 0.0
    ir = float(a.mean()) / std if std > 0 else None
    return {"n": len(a), "mean": float(a.mean()), "ir": ir,
            "t": ir * math.sqrt(len(a)) if ir is not None else None,
            "pos": float((a > 0).mean() * 100)}


def thinned(ic_by_date: dict[str, float], every: int, dates=None) -> list[float]:
    """ICs of every `every`-th date, in DATE order.

    With an H-day forward return, consecutive dates share H-1 of their H days,
    so their ICs are close to copies of each other and a t over all dates is
    inflated. Keeping every H-th date leaves windows that do not overlap.
    Sorting matters: thinning in dict order (rows grouped by symbol, say)
    keeps an arbitrary subset and the overlap it was meant to remove.
    """
    pool = ic_by_date if dates is None else [d for d in dates if d in ic_by_date]
    ordered = sorted(pool)
    return [ic_by_date[d] for d in ordered[::max(1, every)]]
