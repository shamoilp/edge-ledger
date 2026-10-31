import random

import numpy as np
import pytest
from scipy.stats import spearmanr

from edge_ledger.data import JournalError, load_csv, pairs_by_date, to_number
from edge_ledger.ic import average_ranks, date_ics, ic_stats, spearman, thinned


def write(tmp_path, text):
    p = tmp_path / "j.csv"
    p.write_text(text, encoding="utf-8")
    return p


# ── data ────────────────────────────────────────────────────────────────────
def test_load_valid_journal(tmp_path):
    rows, cols = load_csv(write(tmp_path, "date,symbol,f,fwd_5\n2025-01-06,A,1,2\n"))
    assert cols == ["date", "symbol", "f", "fwd_5"]
    assert rows[0]["symbol"] == "A"


@pytest.mark.parametrize("text, msg", [
    ("day,symbol,f\n2025-01-06,A,1\n", "missing column"),
    ("date,symbol,f\n06/01/2025,A,1\n", "not YYYY-MM-DD"),
    ("date,symbol,f\n2025-01-06,,1\n", "empty symbol"),
    ("date,symbol,f\n2025-01-06,A,1\n2025-01-06,A,2\n", "duplicate"),
])
def test_load_rejects_bad_journals(tmp_path, text, msg):
    with pytest.raises(JournalError, match=msg):
        load_csv(write(tmp_path, text))


@pytest.mark.parametrize("raw, expected", [
    ("1.5", 1.5), (" 2 ", 2.0), ("", None), ("n/a", None),
    ("inf", None), ("nan", None), (None, None), (3, 3.0), (float("nan"), None),
])
def test_to_number(raw, expected):
    assert to_number(raw) == expected


def test_pairs_skip_unmatured_returns():
    rows = [{"date": "2025-01-06", "f": "1", "r": "2"},
            {"date": "2025-01-06", "f": "2", "r": ""},
            {"date": "2025-01-07", "f": "", "r": "1"}]
    assert pairs_by_date(rows, "f", "r") == {"2025-01-06": [(1.0, 2.0)]}


# ── ranks and IC ────────────────────────────────────────────────────────────
def test_ties_share_their_average_rank():
    assert average_ranks([10, 20, 20, 30]).tolist() == [0.0, 1.5, 1.5, 3.0]


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 4, size=40).astype(float)      # heavily tied, like a score
    y = x + rng.normal(size=40)
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)


def test_spearman_does_not_depend_on_row_order_when_tied():
    rng = np.random.default_rng(2)
    pairs = [(float(rng.integers(0, 3)), float(rng.normal())) for _ in range(30)]
    first = spearman(*zip(*pairs))
    random.Random(3).shuffle(pairs)
    assert spearman(*zip(*pairs)) == pytest.approx(first)


def test_spearman_undefined_cases():
    assert spearman([1, 2], [1, 2]) is None
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_date_ics_respects_min_names():
    by_date = {"a": [(1, 1), (2, 2), (3, 3)], "b": [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]}
    assert set(date_ics(by_date, min_names=5)) == {"b"}


def test_no_ir_or_t_from_two_dates():
    s = ic_stats([0.1, 0.3])
    assert s["ir"] is None and s["t"] is None and s["mean"] == pytest.approx(0.2)
    assert ic_stats([0.1, 0.2, 0.3])["t"] is not None


def test_thinning_follows_date_order_not_insertion_order():
    scrambled = {"2025-01-09": 9.0, "2025-01-05": 5.0, "2025-01-07": 7.0,
                 "2025-01-06": 6.0, "2025-01-08": 8.0}
    assert thinned(scrambled, 2) == [5.0, 7.0, 9.0]
    assert len(thinned(scrambled, 0)) == 5
