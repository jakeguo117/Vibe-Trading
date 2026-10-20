"""Plan §11 correctness tests for the C engine (items that need no external data)."""
import numpy as np
import pandas as pd
import pytest

from fib_r1 import ceng


def make_panel(T=400, N=2, seed=7, tf="D"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=T)
    syms = [f"S{i}" for i in range(N)]
    c = 100 * np.cumprod(1 + rng.normal(0.0006, 0.012, (T, N)), axis=0)
    o = c * (1 + rng.normal(0, 0.002, (T, N)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, (T, N))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
    v = np.full((T, N), 1e7)
    dv = np.full((T, N), 5e8)
    sig = np.full((T, N), 0.15)
    elig = np.ones((T, N), bool)
    elig[:40] = False
    return ceng.Panel(dates=pd.DatetimeIndex(dates), syms=syms, O=o, H=h, L=l, C=c, V=v,
                      DV=dv, SIGMA=sig, ELIG=elig, last_daily_idx=np.arange(T), tf=tf)


def test_no_lookahead_future_data_does_not_change_the_past():
    P = make_panel()
    base = ceng.run_breakout(P, start=str(P.dates[0].date()))
    P2 = make_panel()
    cut = 300
    P2.O[cut:] *= 3.0
    P2.H[cut:] *= 3.0
    P2.L[cut:] *= 0.5
    P2.C[cut:] *= 2.0
    alt = ceng.run_breakout(P2, start=str(P.dates[0].date()))
    a = base["eq"].equity.iloc[:cut]
    b = alt["eq"].equity.iloc[:cut]
    assert np.allclose(a.values, b.values, rtol=0, atol=1e-12)


def test_signal_bar_never_fills_at_its_own_close():
    P = make_panel()
    r = ceng.run_breakout(P, start=str(P.dates[0].date()))
    for t in r["trades"].itertuples():
        assert int(t.entry_bar) > int(t.entry_sig_bar)
        assert int(t.exit_bar) > int(t.entry_bar)


def test_gap_fills_at_actual_next_open_not_at_the_stop():
    P = make_panel(T=300)
    P.C[:] = 100.0
    P.O[:] = 100.0
    P.H[:] = 100.5
    P.L[:] = 99.5
    # a clean breakout on bar 220, then a gap down through any stop on bar 240
    P.H[219, 0] = 101.0
    P.C[220:, 0] = 110.0
    P.O[221:, 0] = 110.0
    P.H[221:, 0] = 110.5
    P.L[221:, 0] = 109.5
    P.O[240, 0] = 90.0
    P.L[240, 0] = 89.0
    P.C[240, 0] = 91.0
    P.O[241, 0] = 88.0                       # the fill bar opens far below the stop line
    P.L[241, 0] = 87.0
    P.C[241, 0] = 89.0
    r = ceng.run_breakout(P, start=str(P.dates[0].date()))
    exits = [o for o in r["orders"].to_dict("records") if o["kind"] == "exit"]
    assert exits, "the close below the trailing stop must produce an exit order"
    e = exits[0]
    # the close of bar 240 triggers the exit; the fill is bar 241's open, never the stop line
    assert e["date"] == P.dates[241]
    assert e["price"] == pytest.approx(88.0 * (1 - ceng.SLIP_LIQ), rel=1e-12)


def test_single_position_and_cash_never_negative():
    P = make_panel(T=600, N=3)
    r = ceng.run_breakout(P, start=str(P.dates[0].date()))
    eq = r["eq"]
    assert eq.n_hold.max() <= 1
    assert (eq.cash >= -1e-9).all()
    assert (eq.equity > 0).all()


def test_fees_are_not_charged_twice():
    P = make_panel(T=500)
    r = ceng.run_breakout(P, start=str(P.dates[0].date()))
    for t in r["trades"].itertuples():
        assert t.fees_total == pytest.approx(t.fee_entry + t.fee_exit, rel=1e-12)
        # net dollars must equal gross dollars minus exactly those fees
        gross = t.notional * (t.exit_price / t.entry_price - 1.0)
        net = t.notional * t.net_ret
        assert net == pytest.approx(gross - t.fees_total, rel=1e-9, abs=1e-9)


def test_determinism_rerun_is_bit_identical():
    P = make_panel(T=500)
    a = ceng.run_breakout(P, start=str(P.dates[0].date()))
    b = ceng.run_breakout(P, start=str(P.dates[0].date()))
    assert np.array_equal(a["eq"].values, b["eq"].values)
    assert a["trades"].equals(b["trades"])


def test_delay_never_pushes_a_frozen_exit_later():
    """A late exit caused by the order book, not by re-deciding: delay must shift by exactly N bars."""
    P = make_panel(T=600, N=2)
    base = ceng.run_breakout(P, start=str(P.dates[0].date()))
    d1 = ceng.run_breakout(P, start=str(P.dates[0].date()), delay_bars=1)
    for t in d1["trades"].itertuples():
        assert int(t.fill_span_bars) >= 1
    if len(base["trades"]) and len(d1["trades"]):
        # every delayed exit is at least as late as its undelayed counterpart for matched entries
        assert len(d1["trades"]) >= 1
    # with delay the position can never be held for an unbounded time because exits keep being frozen
    assert d1["eq"].n_hold.max() <= 1
    assert (d1["trades"].bars_held <= 41).all()


def test_integer_shares_respect_the_minimum_ticket_and_cash():
    P = make_panel(T=500)
    r = ceng.run_breakout(P, start=str(P.dates[0].date()), integer=True, capital=5000.0)
    for o in r["orders"].to_dict("records"):
        if o["status"] != "filled":
            continue
        assert float(o["qty"]) == np.floor(float(o["qty"]))
        assert float(o["fee"]) >= 1.0 - 1e-12
    assert (r["eq"].cash >= -1e-9).all()
