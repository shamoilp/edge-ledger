"""The multiplicity ledger: a registry of every test you have run.

Run 20 tests on noise and one of them will look significant at the usual 5%
level. The fix (Harvey, Liu & Zhu, "...and the Cross-Section of Expected
Returns", RFS 2016) is to count every test, failures included, and raise the
bar with the count. Here that is a two-sided Bonferroni bar:

    t* = quantile(1 - fwer / (2 * N))

The bar is only honest if N is complete, so register a test BEFORE looking at
its result, and never delete an entry. Deleting lowers the bar.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

VALID_STATUS = ("queued", "observational", "watch", "adopted", "dead")
FWER = 0.05


class LedgerError(ValueError):
    """The ledger operation would make N dishonest or is malformed."""


def bonferroni_bar(n_tests: int, n_obs: int | None = None, fwer: float = FWER) -> float:
    """Two-sided Bonferroni bar for |t|.

    With n_obs, the quantile is Student-t with n_obs - 1 degrees of freedom
    rather than normal. That matters at small n: three near-identical ICs can
    give t = 37, which clears a normal bar of ~3.4 but not the t(2) bar of ~37.
    """
    from scipy.stats import norm, t
    q = 1 - fwer / (2 * max(1, n_tests))
    if n_obs is None:
        return float(norm.ppf(q))
    if n_obs < 2:
        raise LedgerError("need at least 2 observations for a t bar")
    return float(t.ppf(q, df=n_obs - 1))


def empty() -> dict:
    return {"tests": []}


def load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return empty()
    ledger = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict) or not isinstance(ledger.get("tests"), list):
        raise LedgerError(f"{path}: expected an object with a 'tests' list")
    return ledger


def save(path: str | Path, ledger: dict) -> None:
    """Write atomically, so a crash mid-write cannot truncate the registry."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def add(ledger: dict, name: str, *, verdict: str = "registered, result pending",
        status: str = "queued", source: str = "", today: date | None = None) -> dict:
    """Return a NEW ledger with `name` registered. Refuses duplicates."""
    if status not in VALID_STATUS:
        raise LedgerError(f"status must be one of {VALID_STATUS}")
    if any(t["name"] == name for t in ledger["tests"]):
        raise LedgerError(f"{name!r} is already registered. Update its entry instead; "
                          "a duplicate would inflate N.")
    entry = {"name": name, "registered": (today or date.today()).isoformat(),
             "status": status, "verdict": verdict, "source": source}
    return {**ledger, "tests": [*ledger["tests"], entry]}


def update(ledger: dict, name: str, *, status: str | None = None,
           verdict: str | None = None, today: date | None = None) -> dict:
    """Return a NEW ledger with `name`'s status/verdict changed and a retest stamp."""
    if status is not None and status not in VALID_STATUS:
        raise LedgerError(f"status must be one of {VALID_STATUS}")
    if not any(t["name"] == name for t in ledger["tests"]):
        raise LedgerError(f"{name!r} is not registered")
    stamp = (today or date.today()).isoformat()

    def changed(t: dict) -> dict:
        if t["name"] != name:
            return t
        return {**t, "status": status or t["status"],
                "verdict": verdict if verdict is not None else t["verdict"],
                "retested": stamp}
    return {**ledger, "tests": [changed(t) for t in ledger["tests"]]}


def count(ledger: dict) -> int:
    return len(ledger["tests"])
