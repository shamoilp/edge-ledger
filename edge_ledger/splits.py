"""Split-aware forward returns.

Free daily price data is not consistently split-adjusted. Some symbols come
back raw (a 1-for-125 reverse split shows as a 125x jump), some come back
back-adjusted, and the same symbol can change state between two downloads.
Two failure modes follow, and a fix for one breaks the other:

1. Raw series. Forward bars after a split sit on a different scale from the
   entry bar, so the "return" is the split ratio. A real case: a +1.4% move
   logged as +11,724%.
2. Back-adjusted series, stored entry. The series is internally consistent,
   but an entry price saved on an earlier download is on the old scale. A real
   case: a 4-for-1 forward split turned a +15.9% gain into a logged -71.0%.

Rules that handle both:
- Locate the split bar from the price PATH, using the labelled ratio only for
  its magnitude. Vendor labels can sit a bar off the real re-scale.
- If there is no discontinuity in the path, report no split. The series is
  already adjusted, and adjusting it again invents an error of the same size
  in the other direction.
- Derive entry and exit prices from ONE download. Never anchor a return on a
  stored price (consistent_entry() checks one if you must).
- Do not filter on magnitude. Large drops without a split are real crashes;
  only the split calendar tells them apart.

Ratios follow the common vendor convention: a 2-for-1 forward split is 2.0,
a 1-for-125 reverse split is 0.008.
"""
from __future__ import annotations

import numpy as np

SPLIT_JUMP_MIN = 1.8      # a reverse split lifts price at least this much bar-to-bar
SPLIT_DROP_MAX = 0.55     # a forward split cuts it at least this much
SPLIT_WINDOW = 3          # bars either side of the label to search for the re-scale
SPLIT_MATCH_FRAC = 0.35   # jump must be at least this fraction of the labelled ratio
ENTRY_SCALE_LO = 0.67     # stored vs fresh entry outside this band = a scale break
ENTRY_SCALE_HI = 1.5


def _prev_finite(close: np.ndarray, i: int) -> float | None:
    """Last finite close before bar i (a halted bar must not poison the comparison)."""
    fin = np.nonzero(np.isfinite(close[:i]))[0]
    return float(close[fin[-1]]) if len(fin) else None


def _best_rescale_bar(open_: np.ndarray, close: np.ndarray, label_i: int,
                      ratio: float, window: int) -> int | None:
    want_up = ratio < 1.0                     # reverse split: price jumps up
    expect = (1.0 / ratio) if want_up else ratio
    need = max(SPLIT_JUMP_MIN, SPLIT_MATCH_FRAC * expect)
    best_score, best_i = None, None
    for i in range(max(1, label_i - window), min(len(open_) - 1, label_i + window) + 1):
        prev = _prev_finite(close, i)
        if prev is None or prev <= 0 or not np.isfinite(open_[i]) or open_[i] <= 0:
            continue
        jump = open_[i] / prev
        if want_up:
            if jump < need:
                continue
            score = jump
        else:
            if not 0 < jump <= min(SPLIT_DROP_MAX, 1.0 / need):
                continue
            score = 1.0 / jump
        if best_score is None or score > best_score:
            best_score, best_i = score, i
    return best_i


def split_breaks(open_, close, split_ratio, window: int = SPLIT_WINDOW) -> list[tuple[int, float]]:
    """[(bar_index, ratio)] for bars where the series ACTUALLY changes scale.

    `split_ratio` is the vendor's per-bar split column (0 or NaN on ordinary
    bars). A labelled split with no matching discontinuity yields nothing.
    """
    op = np.asarray(open_, dtype=float)
    cl = np.asarray(close, dtype=float)
    ratios = np.nan_to_num(np.asarray(split_ratio, dtype=float))
    out: list[tuple[int, float]] = []
    for j in np.nonzero(ratios)[0]:
        r = float(ratios[j])
        if r <= 0 or r == 1.0:
            continue
        i = _best_rescale_bar(op, cl, int(j), r, window)
        if i is not None:
            out.append((i, r))
    return out


def split_factors(breaks: list[tuple[int, float]], n_bars: int, entry_i: int) -> np.ndarray:
    """Multipliers putting every bar on the ENTRY bar's scale: adjusted = raw * f.
    For return math. Splits at or before the entry need no adjustment."""
    f = np.ones(n_bars)
    for i, r in breaks:
        if i > entry_i:
            f[i:] *= r
    return f


def split_factors_to_last(breaks: list[tuple[int, float]], n_bars: int) -> np.ndarray:
    """Multipliers putting every bar on the LAST bar's scale: adjusted = raw * f.
    For screens ("how far above its 52-week low?"). Divide volume by f."""
    f = np.ones(n_bars)
    for i, r in breaks:
        if 0 < i < n_bars and np.isfinite(r) and r > 0:
            f[:i] /= r
    return f


def forward_return(open_, close, split_ratio, entry_i: int, horizon: int) -> float | None:
    """Percent return from close[entry_i] to close[entry_i + horizon], split-aware,
    with both prices taken from the same series."""
    cl = np.asarray(close, dtype=float)
    j = entry_i + horizon
    if entry_i < 0 or j >= len(cl) or not (np.isfinite(cl[entry_i]) and np.isfinite(cl[j])):
        return None
    if cl[entry_i] <= 0:
        return None
    f = split_factors(split_breaks(open_, close, split_ratio), len(cl), entry_i)
    return float((cl[j] * f[j] / cl[entry_i] - 1) * 100)


def consistent_entry(stored: float | None, fresh: float | None) -> tuple[float | None, bool]:
    """Return (entry, rebased): the stored entry, unless it disagrees with a freshly
    derived one by more than ordinary movement, in which case the fresh one."""
    if stored and stored > 0 and fresh and fresh > 0:
        if not ENTRY_SCALE_LO <= fresh / stored <= ENTRY_SCALE_HI:
            return fresh, True
        return stored, False
    return (stored or fresh), False
