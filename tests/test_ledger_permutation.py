from datetime import date

import numpy as np
import pytest

from edge_ledger import ledger as L
from edge_ledger import permutation as P
from edge_ledger.ic import spearman

TODAY = date(2026, 1, 1)


# ── ledger ──────────────────────────────────────────────────────────────────
def test_normal_bar_at_70_tests():
    assert L.bonferroni_bar(70) == pytest.approx(3.38, abs=0.01)


def test_small_sample_needs_the_t_bar():
    # three near-identical ICs gave t = 36.9 in real data; it must not clear
    assert L.bonferroni_bar(70, n_obs=3) == pytest.approx(37.4, abs=0.1)
    assert L.bonferroni_bar(70, n_obs=1000) == pytest.approx(3.39, abs=0.02)


def test_bar_rises_with_the_number_of_tests():
    assert L.bonferroni_bar(1) < L.bonferroni_bar(10) < L.bonferroni_bar(100)


def test_add_returns_a_new_ledger_and_leaves_the_old_one():
    old = L.empty()
    new = L.add(old, "momentum", today=TODAY)
    assert L.count(old) == 0 and L.count(new) == 1
    assert new["tests"][0] == {"name": "momentum", "registered": "2026-01-01",
                               "status": "queued", "verdict": "registered, result pending",
                               "source": ""}


def test_duplicates_and_bad_status_are_refused():
    led = L.add(L.empty(), "momentum", today=TODAY)
    with pytest.raises(L.LedgerError, match="already registered"):
        L.add(led, "momentum")
    with pytest.raises(L.LedgerError, match="status"):
        L.add(led, "other", status="maybe")


def test_update_stamps_a_retest():
    led = L.add(L.empty(), "momentum", today=TODAY)
    led = L.update(led, "momentum", status="dead", verdict="no", today=date(2026, 2, 1))
    t = led["tests"][0]
    assert (t["status"], t["verdict"], t["retested"]) == ("dead", "no", "2026-02-01")
    with pytest.raises(L.LedgerError, match="not registered"):
        L.update(led, "unknown", status="dead")


def test_save_load_round_trip_leaves_no_temp_file(tmp_path):
    path = tmp_path / "ledger.json"
    led = L.add(L.empty(), "momentum", today=TODAY)
    L.save(path, led)
    assert L.load(path) == led
    assert [p.name for p in tmp_path.iterdir()] == ["ledger.json"]
    assert L.load(tmp_path / "missing.json") == L.empty()


# ── permutation ─────────────────────────────────────────────────────────────
def journal(signal: float, n_dates: int = 30, n_names: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(n_dates):
        x = rng.normal(size=n_names)
        y = signal * x + rng.normal(size=n_names)
        out[f"2025-01-{k + 1:02d}"] = list(zip(x.tolist(), y.tolist()))
    return out


def test_a_planted_signal_is_not_luck():
    by_date = journal(signal=1.0)
    res = P.within_date_permutation(by_date, sorted(by_date), iters=2000)
    assert res["p"] == pytest.approx(1 / 2001)
    assert "unlikely to be luck" in P.verdict(res["p"], n_tests=50)


def test_noise_is_luck():
    by_date = journal(signal=0.0, seed=5)
    res = P.within_date_permutation(by_date, sorted(by_date), iters=2000)
    assert res["p"] > 0.05


def test_real_mean_is_the_mean_spearman():
    by_date = journal(signal=0.3)
    res = P.within_date_permutation(by_date, sorted(by_date), iters=200)
    expected = np.mean([spearman(*zip(*by_date[d])) for d in sorted(by_date)])
    assert res["real_mean"] == pytest.approx(expected)


def test_flipping_the_factor_flips_the_direction():
    by_date = journal(signal=0.3)
    flipped = {d: [(-x, y) for x, y in p] for d, p in by_date.items()}
    a = P.within_date_permutation(by_date, sorted(by_date), iters=500)
    b = P.within_date_permutation(flipped, sorted(flipped), iters=500)
    assert a["real_mean"] == pytest.approx(-b["real_mean"])
    assert a["p"] == pytest.approx(b["p"])


def test_select_dates_thins_in_date_order():
    by_date = {f"2025-01-{d:02d}": [(1, 1)] * 5 for d in (9, 3, 5, 1, 7)}
    assert P.select_dates(by_date, min_names=5, every=2) == ["2025-01-01", "2025-01-05",
                                                             "2025-01-09"]


def test_bootstrap():
    assert P.date_bootstrap([0.1])["lo"] is None
    b = P.date_bootstrap([0.08, 0.10, 0.12, 0.09])
    assert 0 < b["lo"] < b["hi"] and b["p_pos"] == 1.0


@pytest.mark.parametrize("p, n, word", [
    (0.0162, 1, "unlikely to be luck"),
    (0.0162, 12, "not after N = 12"),
    (0.031, 1, "not distinguishable"),
])
def test_verdict_is_two_sided_and_multiplicity_aware(p, n, word):
    assert word in P.verdict(p, n)
