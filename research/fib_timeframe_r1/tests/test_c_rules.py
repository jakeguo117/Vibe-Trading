"""Minimal correctness tests for the C-line engine (plan §11 items 5,7,8,9).

Run: pytest -q research/fib_timeframe_r1/tests/test_c_rules.py
"""
import numpy as np
import pandas as pd
import pytest

import fib_r1.ceng as ceng


def mk_panel(O, H, L, C, syms=("AAA",), dates=None, dv=5e7, sigma=0.15):
    O, H, L, C = (np.asarray(x, float).reshape(len(x), -1) for x in (O, H, L, C))
    T, N = C.shape
    if dates is None:
        dates = pd.bdate_range("2020-01-01", periods=T)
    return ceng.Panel(dates=pd.DatetimeIndex(dates), syms=list(syms), O=O, H=H, L=L, C=C,
                      V=np.full((T, N), 1e6), DV=np.full((T, N), float(dv)),
                      SIGMA=np.full((T, N), float(sigma)), ELIG=np.ones((T, N), bool),
                      last_daily_idx=np.arange(T), tf="D")


def test_wilder_atr_init_and_recursion():
    H = np.array([[10.0], [11.0], [12.0]])
    L = np.array([[9.0], [9.5], [10.0]])
    C = np.array([[9.5], [10.5], [11.5]])
    a = ceng.wilder_atr(H, L, C, n=2)
    tr = [10.0 - 9.0, max(11 - 9.5, abs(11 - 9.5), abs(9.5 - 9.5)), max(12 - 10, abs(12 - 10.5), abs(10 - 10.5))]
    assert a[1, 0] == pytest.approx((tr[0] + tr[1]) / 2)
    assert a[2, 0] == pytest.approx((1 * a[1, 0] + tr[2]) / 2)


def test_breakout_window_excludes_current_bar():
    T = 30
    H = np.full((T, 1), 100.0)
    C = np.full((T, 1), 99.0)
    H[25, 0] = 200.0          # the bar itself spikes
    C[25, 0] = 150.0
    B = ceng.breakout_line(H, 20)
    assert B[25, 0] == pytest.approx(100.0)      # t itself excluded
    assert C[25, 0] > B[25, 0]                   # so this is a valid breakout
    assert B[29, 0] == pytest.approx(200.0)      # but it enters the window afterwards


def test_squeeze_uses_previous_bar_only_and_inclusive_quantile():
    T, N = 200, 1
    C = np.full((T, N), 100.0)
    atr = np.full((T, N), 3.0)
    # reference v_(t-1) is ATR/C = 0.02; the 120-value window is t-120..t-1 (includes t-1 itself)
    atr[T - 2, 0] = 2.0
    atr[T - 121:T - 98, 0] = 2.0          # 23 more low-vol bars -> 24 of 120 with equality
    ok = ceng.squeeze_flags(atr, C)
    assert ok[T - 1, 0]                    # 24/120 = 0.20 exactly, equality counted
    atr2 = atr.copy()
    atr2[T - 121:T - 97, 0] = 2.0          # 24 more + the reference = 25 of 120
    ok2 = ceng.squeeze_flags(atr2, C)
    assert not ok2[T - 1, 0]               # 25/120 = 0.2083 > 0.20


def test_no_same_bar_fill_and_single_position():
    T = 60
    O = np.full((T, 1), 100.0)
    H = np.full((T, 1), 101.0)
    L = np.full((T, 1), 99.0)
    C = np.full((T, 1), 100.0)
    H[25:, 0] = 130.0     # breakout at bar 25
    C[25:, 0] = 120.0
    O[26:, 0] = 121.0
    P = mk_panel(O, H, L, C)
    r = ceng.run_breakout(P, squeeze=False)
    od = r["orders"]
    ent = od[od.kind == "entry"]
    assert len(ent) >= 1
    assert int(ent.iloc[0].bar) == 26                 # filled one bar after the signal
    assert int(ent.iloc[0].bar) > 25


def test_stop_is_close_based_not_intrabar():
    T = 60
    O = np.full((T, 1), 100.0)
    H = np.full((T, 1), 101.0)
    L = np.full((T, 1), 99.0)
    C = np.full((T, 1), 100.0)
    H[25:, 0] = 130.0
    C[25:, 0] = 120.0
    O[26:, 0] = 121.0
    L[27, 0] = 80.0            # deep intrabar dip, close recovers
    C[27, 0] = 121.0
    P = mk_panel(O, H, L, C)
    r = ceng.run_breakout(P, squeeze=False)
    ex = r["orders"][r["orders"].kind == "exit"]
    assert len(ex) == 0 or int(ex.iloc[0].bar) > 27


def test_time_exit_at_40_bars():
    T = 80
    O = np.full((T, 1), 100.0)
    H = np.full((T, 1), 101.0)
    L = np.full((T, 1), 99.0)
    C = np.full((T, 1), 100.0)
    H[25:, 0] = 130.0
    C[25:, 0] = 120.0
    O[26:, 0] = 121.0
    C[26:, 0] = 123.0
    P = mk_panel(O, H, L, C)
    r = ceng.run_breakout(P, squeeze=False)
    tr = r["trades"]
    assert len(tr) == 1
    assert int(tr.iloc[0].bars_held) == 40
    assert tr.iloc[0].exit_reason == "time_40"
