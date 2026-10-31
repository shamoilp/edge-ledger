import numpy as np
import pytest

from edge_ledger import splits as S

N = 20
SPLIT_BAR = 10


def series(raw: bool, ratio: float = 0.008, label_at: int = SPLIT_BAR, drift: float = 0.01):
    """A price path rising 1% a bar with a split at SPLIT_BAR.

    raw=True: post-split bars sit on the new scale (a 1:125 reverse split
    multiplies price by 125). raw=False: back-adjusted, no discontinuity.
    """
    close = 0.04 * (1 + drift) ** np.arange(N)
    if raw:
        close[SPLIT_BAR:] /= ratio
    open_ = close.copy()
    labels = np.zeros(N)
    labels[label_at] = ratio
    return open_, close, labels


def true_return(entry: int, horizon: int, drift: float = 0.01) -> float:
    return ((1 + drift) ** horizon - 1) * 100


def test_raw_reverse_split_is_found_and_repaired():
    op, cl, lab = series(raw=True)
    assert S.split_breaks(op, cl, lab) == [(SPLIT_BAR, 0.008)]
    naive = (cl[15] / cl[5] - 1) * 100
    assert naive > 10_000                                   # the artifact
    assert S.forward_return(op, cl, lab, 5, 10) == pytest.approx(true_return(5, 10))


def test_already_adjusted_series_is_left_alone():
    op, cl, lab = series(raw=False)
    assert S.split_breaks(op, cl, lab) == []
    assert S.forward_return(op, cl, lab, 5, 10) == pytest.approx(true_return(5, 10))


def test_label_one_bar_off_still_finds_the_real_bar():
    op, cl, lab = series(raw=True, label_at=SPLIT_BAR + 1)
    assert S.split_breaks(op, cl, lab) == [(SPLIT_BAR, 0.008)]


def test_forward_split():
    op, cl, lab = series(raw=True, ratio=2.0)
    assert S.split_breaks(op, cl, lab) == [(SPLIT_BAR, 2.0)]
    assert S.forward_return(op, cl, lab, 5, 10) == pytest.approx(true_return(5, 10))


def test_halted_bar_before_the_split_does_not_hide_it():
    op, cl, lab = series(raw=True)
    cl[SPLIT_BAR - 1] = np.nan
    op[SPLIT_BAR - 1] = np.nan
    assert S.split_breaks(op, cl, lab) == [(SPLIT_BAR, 0.008)]


def test_ordinary_volatility_near_a_label_is_not_a_split():
    op, cl, lab = series(raw=False, ratio=0.1)
    cl[SPLIT_BAR:] *= 1.3                                   # a big day, not a 10x re-scale
    op[SPLIT_BAR:] *= 1.3
    assert S.split_breaks(op, cl, lab) == []


def test_split_before_entry_needs_no_adjustment():
    op, cl, lab = series(raw=True)
    assert S.forward_return(op, cl, lab, 12, 5) == pytest.approx(true_return(12, 5))


def test_restating_history_onto_todays_scale():
    f = S.split_factors_to_last([(SPLIT_BAR, 0.008)], N)
    assert f[-1] == 1.0 and f[0] == pytest.approx(125.0)


def test_forward_return_out_of_range():
    op, cl, lab = series(raw=False)
    assert S.forward_return(op, cl, lab, 15, 10) is None


@pytest.mark.parametrize("stored, fresh, expected", [
    (100.0, 25.0, (25.0, True)),       # 4:1 split since the entry was stored
    (100.0, 110.0, (100.0, False)),    # ordinary movement / rounding
    (None, 50.0, (50.0, False)),
    (40.0, None, (40.0, False)),
])
def test_consistent_entry(stored, fresh, expected):
    assert S.consistent_entry(stored, fresh) == expected
