import math
from datetime import date, timedelta

import numpy as np
import pytest

from edge_ledger import walkforward as W
from edge_ledger.data import pairs_by_date
from edge_ledger.ic import date_ics


def weekdays(start: str, n_weeks: int, skip=frozenset()) -> list[str]:
    d0 = date.fromisoformat(start)
    days = [(d0 + timedelta(days=i)) for i in range(n_weeks * 7)]
    return [d.isoformat() for d in days if d.weekday() < 5 and d.isoformat() not in skip]


def journal_ics(dates, sign_of, seed=0, names=20, noise=0.3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for i in range(names):
            x = float(rng.normal())
            rows.append({"date": d, "f": x, "r": sign_of(d) * x + noise * float(rng.normal())})
    return date_ics(pairs_by_date(rows, "f", "r"))


# ── purge ───────────────────────────────────────────────────────────────────
def test_purge_window_sizes():
    assert (W.purge_cal_days(20), W.purge_cal_days(5), W.purge_cal_days(1)) == (34, 13, 8)


JAN = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(31)]


def test_purge_keeps_only_dates_beyond_the_label_reach():
    kept, purged = W.purge(JAN, "2026-02-10", 34)
    assert kept == JAN[:6]                 # Jan 7 is exactly 34 days out: purged
    assert kept + purged == JAN


def test_extra_purge_pct_trims_the_tail():
    ten = JAN[:10]
    assert W.purge(ten, "2026-03-31", 1, 0.2) == (ten[:8], ten[8:])
    assert W.purge(ten, "2026-01-12", 5, 0.7) == ([], ten)
    assert W.purge(ten, "2026-03-31", 1, 0.0) == (ten, [])


# ── folds ───────────────────────────────────────────────────────────────────
DATES = weekdays("2026-01-05", 20, skip=frozenset({"2026-02-16", "2026-03-04"}))


@pytest.fixture
def folds():
    f, skipped = W.anchored_folds(DATES, purge_days=34)
    assert skipped >= 1                     # a 4-week IS cannot survive a 34-day purge
    return f


def test_folds_are_anchored_and_growing(folds):
    assert len(folds) >= 4
    assert all(f.is_dates[0] == DATES[0] for f in folds)
    assert all(len(b.is_dates) > len(a.is_dates) for a, b in zip(folds, folds[1:]))
    assert [f.fold_id for f in folds] == list(range(len(folds)))


def test_oos_windows_are_ordered_and_disjoint(folds):
    assert all(a.oos_dates[-1] < b.oos_dates[0] for a, b in zip(folds, folds[1:]))
    assert all(max(f.is_dates + f.purged) < f.oos_dates[0] for f in folds)


def test_no_kept_in_sample_date_can_reach_the_oos_window(folds):
    for f in folds:
        o = date.fromisoformat(f.oos_dates[0])
        assert all((o - date.fromisoformat(d)).days > 34 for d in f.is_dates)


def test_fold_edge_cases():
    assert W.anchored_folds([], purge_days=34) == ([], 0)
    with pytest.raises(ValueError, match="overlap"):
        W.anchored_folds(DATES, step_weeks=1, test_weeks=2, purge_days=34)


# ── WF03 metrics ────────────────────────────────────────────────────────────
def test_overfit_ratio():
    assert W.overfit_ratio(1.0, 0.5) == 2.0
    assert W.overfit_ratio(1.0, -0.1) == math.inf
    assert W.overfit_ratio(-1.0, -1.0) == 0.0
    assert W.overfit_ratio(None, 0.5) is None


def test_efficiency_and_decay():
    assert W.efficiency(0.10, 0.05) == 0.5
    assert W.efficiency(-0.02, 0.05) is None
    assert W.decay_slope([1, 2, 3]) == pytest.approx(1.0)
    assert W.decay_slope([0.4]) == 0.0


def lights(metrics, pooled, n):
    return {name: (sig, desc) for name, _v, sig, desc in W.traffic_lights(metrics, pooled, n)}


def test_lights_never_green_on_a_losing_oos():
    assert lights({"overfit_ratio": 1.1}, -0.01, 8)["overfit_ratio"][0] == "red"


def test_lights_other_gates():
    got = lights({"overfit_ratio": -0.5, "efficiency": 0.9}, 0.05, 8)
    assert got["overfit_ratio"][0] == "yellow" and got["efficiency"][0] == "green"
    got = lights({"overfit_ratio": math.inf, "efficiency": None}, 0.05, 3)
    assert got["overfit_ratio"][0] == "red" and got["efficiency"][0] == "n/a"
    assert "indicative" in got["overfit_ratio"][1]


# ── evaluation ──────────────────────────────────────────────────────────────
LONG = weekdays("2026-01-05", 24)


def run_fixed(ics):
    folds, _ = W.anchored_folds(sorted(ics), purge_days=34)
    return folds, W.evaluate_fixed(ics, folds, horizon_days=20)


def test_a_stable_signal_holds_out_of_sample():
    folds, res = run_fixed(journal_ics(LONG, lambda d: 1.0))
    assert res["sign_hold"] == (len(folds), len(folds))
    assert res["pooled"]["mean"] > 0.7 and 0.8 < res["efficiency"] < 1.2
    assert res["pooled"]["n"] == sum(len(f.oos_dates) for f in folds)
    assert res["pooled_thin"]["n"] == len(range(0, res["pooled"]["n"], 20))


def test_a_regime_flip_shows_up_late():
    half = LONG[len(LONG) // 2]
    _, res = run_fixed(journal_ics(LONG, lambda d: 1.0 if d < half else -1.0))
    late = [p for p in res["per_fold"] if p["fold"].oos_dates[0] >= half]
    assert late and all(p["oos"]["mean"] < 0 for p in late)
    assert res["sign_hold"][0] < res["sign_hold"][1] and res["decay"] < 0


def test_select_picks_on_in_sample_only():
    cut = "2026-03-02"
    by_f = {"a": journal_ics(LONG, lambda d: 1.0 if d < cut else -1.0, seed=1),
            "b": journal_ics(LONG, lambda d: -0.2 if d < cut else 1.0, seed=2)}
    common = sorted(set(by_f["a"]) & set(by_f["b"]))
    folds, _ = W.anchored_folds(common, train_weeks=10, purge_days=34)
    first = W.evaluate_select(by_f, folds, horizon_days=20)["per_fold"][0]
    assert first["pick"] == "a" and first["oos_rank"] == 2


def test_select_skill_against_the_average_candidate():
    folds, _ = W.anchored_folds(LONG, purge_days=34)
    res = W.evaluate_select({"a": {d: 0.5 for d in LONG}, "b": {d: 0.1 for d in LONG}},
                            folds, horizon_days=20)
    assert all(p["pick"] == "a" for p in res["per_fold"])
    assert res["skill"] == pytest.approx(0.2)
