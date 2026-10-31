"""Is the IC distinguishable from luck? Two checks.

1. Within-date permutation (Chan, *Algorithmic Trading*, Ch.1). Inside each
   date, shuffle the factor against the returns and recompute the mean IC,
   many times. That keeps every date's factor and return distributions and
   breaks only the link between them. p = how often luck matches the real IC.
2. Date bootstrap. Resample the per-date ICs with replacement. If the 90%
   interval of the mean straddles zero, the average depends on which dates you
   happened to get.

Overlap caveat: permuting each date independently treats dates as
independent. With an H-day return and a persistent factor, neighbouring dates
are near-copies, so the null is too narrow and p comes out too small. The
default therefore runs on every H-th date (see ic.thinned). Use all dates only
for one-day returns or a factor that is not persistent.
"""
from __future__ import annotations

import numpy as np

from .data import Pairs
from .ic import MIN_NAMES, average_ranks
from .ledger import FWER

DEFAULT_ITERS = 5000
BOOTSTRAP_ITERS = 10_000
SEED = 42


def _centred_ranks(pairs: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray] | None:
    f = average_ranks([p[0] for p in pairs])
    r = average_ranks([p[1] for p in pairs])
    f, r = f - f.mean(), r - r.mean()
    denom = float(np.sqrt((f * f).sum() * (r * r).sum()))
    return None if denom == 0 else (f / denom, r)


def select_dates(by_date: Pairs, min_names: int = MIN_NAMES, every: int = 1) -> list[str]:
    """Dates with enough names, sorted, keeping every `every`-th one."""
    usable = sorted(d for d, p in by_date.items() if len(p) >= min_names)
    return usable[::max(1, every)]


def within_date_permutation(by_date: Pairs, dates: list[str], iters: int = DEFAULT_ITERS,
                            seed: int = SEED) -> dict:
    """Permutation p-values for the mean per-date IC, in both directions."""
    rng = np.random.default_rng(seed)
    real, null_sum, used = [], np.zeros(iters), 0
    for d in dates:
        cr = _centred_ranks(by_date[d])
        if cr is None:
            continue
        f, r = cr
        real.append(float(f @ r))
        shuffled = rng.permuted(np.tile(f, (iters, 1)), axis=1)
        null_sum += shuffled @ r
        used += 1
    if used == 0:
        return {"n": 0}
    null = null_sum / used
    mean = float(np.mean(real))
    p_pos = float((np.sum(null >= mean) + 1) / (iters + 1))
    p_neg = float((np.sum(null <= mean) + 1) / (iters + 1))
    return {"n": used, "real_mean": mean, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)) if iters > 1 else 0.0,
            "percentile": float((null < mean).mean() * 100),
            "p_pos": p_pos, "p_neg": p_neg, "p": p_pos if mean >= 0 else p_neg}


def date_bootstrap(ics, iters: int = BOOTSTRAP_ITERS, seed: int = SEED) -> dict:
    """90% bootstrap interval of the mean IC, resampling dates."""
    a = np.asarray(list(ics), dtype=float)
    if len(a) < 2:
        return {"n": len(a), "lo": None, "hi": None, "p_pos": None}
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(iters, len(a)), replace=True).mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(means, [5, 95]))
    return {"n": len(a), "lo": lo, "hi": hi, "p_pos": float((means > 0).mean())}


def p_bar(n_tests: int = 1, fwer: float = FWER) -> float:
    """The p a directional result must beat. The direction is chosen after
    seeing the sign, so this is two-sided, then Bonferroni-divided by N."""
    return fwer / (2 * max(1, n_tests))


def verdict(p: float, n_tests: int = 1, fwer: float = FWER) -> str:
    bar = p_bar(n_tests, fwer)
    if p <= bar:
        return f"unlikely to be luck (p <= {bar:.4f} for N = {n_tests})"
    if p <= fwer / 2:
        return f"would pass alone, but not after N = {n_tests} tests (need p <= {bar:.4f})"
    return "not distinguishable from luck"
