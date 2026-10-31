"""Load a factor journal: one row per (date, symbol).

The CSV needs a `date` column (ISO YYYY-MM-DD) and a `symbol` column. Every
other column is either a factor (known on the scan date) or a forward return
in percent (known only after the horizon has passed). A blank forward return
means that horizon has not matured yet, and the row is skipped for it.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

REQUIRED_COLUMNS = ("date", "symbol")

Pairs = dict[str, list[tuple[float, float]]]


class JournalError(ValueError):
    """The journal file cannot be used as given."""


def load_csv(path: str | Path) -> tuple[list[dict], list[str]]:
    """Read and validate a journal. Returns (rows, column names).

    Fails fast on a missing date/symbol column, a non-ISO date, or a duplicate
    (date, symbol) row, since each would silently distort a per-date IC.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise JournalError(f"{path}: missing column(s) {missing}. Need date, symbol, "
                               "then factor and forward-return columns.")
        rows = [dict(r) for r in reader]
    _validate(rows, path)
    return rows, columns


def _validate(rows: list[dict], path: str | Path) -> None:
    seen: set[tuple[str, str]] = set()
    for line, r in enumerate(rows, start=2):
        d, s = (r.get("date") or "").strip(), (r.get("symbol") or "").strip()
        try:
            date.fromisoformat(d)
        except ValueError:
            raise JournalError(f"{path}:{line}: date {d!r} is not YYYY-MM-DD") from None
        if not s:
            raise JournalError(f"{path}:{line}: empty symbol")
        if (d, s) in seen:
            raise JournalError(f"{path}:{line}: duplicate row for {s} on {d}")
        seen.add((d, s))


def to_number(value: object) -> float | None:
    """Parse a cell as a finite float, or None for blank / non-numeric / inf / nan."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            x = float(text)
        except ValueError:
            return None
    return x if math.isfinite(x) else None


def pairs_by_date(rows: list[dict], factor: str, returns: str) -> Pairs:
    """Per-date (factor value, forward return) pairs, skipping rows missing either."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        f, y = to_number(r.get(factor)), to_number(r.get(returns))
        if f is not None and y is not None:
            out[r["date"].strip()].append((f, y))
    return dict(out)
