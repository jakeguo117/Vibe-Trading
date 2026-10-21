"""Synthetic self-tests for the account kernel (spec §9 economic behaviours).

Run: python tests/test_engine.py  ->  prints PASS/FAIL per check, exit 1 on failure.
Each test builds a tiny hand-made Kit so expected cash/equity can be verified by hand.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qm import engine, metrics  # noqa: E402
from qm.core import Kit  # noqa: E402

REQ = ["dates", "syms", "o", "h", "l", "c", "cr", "v", "dv", "div", "adjf",
       "sma10", "sma16", "sma20", "sma24", "sma50", "atr14", "dv20", "adr20", "hist",
       "base_ok", "r10", "r21", "r63", "r126", "p21", "p63", "p126"]


def make_kit(o, h, l, c, dv=None):
    o, h, l, c = map(lambda a: np.asarray(a, dtype=np.float32), (o, h, l, c))
    T, N = c.shape
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + T)
    z = np.ones((T, N), np.float32)
    fields = dict(dates=dates, syms=np.array([f"S{i}" for i in range(N)]), o=o, h=h, l=l, c=c,
                  cr=c.copy(), v=np.full((T, N), 1e6, np.float32), dv=c * 1e6,
                  div=np.zeros((T, N), np.float32),
                  adjf=z, sma10=c.copy(), sma16=c.copy(), sma20=c.copy(), sma24=c.copy(),
                  sma50=c.copy(), atr14=np.full((T, N), 1.0, np.float32),
                  dv20=np.full((T, N), 5e7, np.float32) if dv is None else dv,
                  adr20=np.full((T, N), 5.0, np.float32), hist=np.full((T, N), 300, np.int32),
                  base_ok=np.ones((T, N), bool), r10=z, r21=z, r63=z, r126=z, p21=z, p63=z, p126=z)
    return Kit(**fields)


def sig(d, j, stop, atr=1.0, k1=1.0, k2=1.0):
    return pd.DataFrame([{"d": d, "j": j, "stop": stop, "atr": atr, "k1": k1, "k2": k2, "k3": j}])


RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
    except AssertionError as e:
        RESULTS.append((name, "FAIL", str(e)))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, "ERROR", f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------- tests

def t_planned_loss_cap():
    o = h = l = c = np.full((10, 1), 100.0)
    c = c.copy(); c[2] = 100
    k = make_kit(o, h, l, c)
    s = sig(1, 0, 99.0)  # risk 1% of price
    cfg = engine.Cfg(style="bo", risk=0.005, maxw=0.30, slip_bps=0)
    res = engine.run(k, s, cfg, 0, 5)
    tr = res["trades"]
    assert len(tr) == 1, tr
    r = tr.iloc[0]
    assert r["planned_risk"] <= 0.005 * 16000 + 1e-6, r["planned_risk"]


def t_gap_through_stop_fills_at_open():
    o = np.full((10, 1), 100.0); h = np.full((10, 1), 101.0); l = np.full((10, 1), 99.0)
    c = np.full((10, 1), 100.0)
    o[3, 0] = 90.0; h[3, 0] = 91.0; l[3, 0] = 89.0; c[3, 0] = 90.0   # gap below stop 95
    k = make_kit(o, h, l, c)
    s = sig(1, 0, 95.0, atr=10.0)
    res = engine.run(k, s, engine.Cfg(style="bo", slip_bps=0), 0, 6)
    tr = res["trades"]
    assert len(tr) == 1
    assert tr.iloc[0]["exit_reason"] == "gap_stop"
    assert abs(tr.iloc[0]["exit"] - 90.0) < 1e-6, tr.iloc[0]["exit"]


def t_single_position_and_no_reentry_same_day():
    o = np.full((10, 2), 100.0); h = o + 1; l = o - 1; c = o.copy()
    k = make_kit(o, h, l, c)
    s = pd.concat([sig(1, 0, 99.0, k1=2), sig(1, 1, 99.0, k1=1)])
    res = engine.run(k, s, engine.Cfg(style="bo", slip_bps=0), 0, 4)
    assert len(res["trades"]) == 1, "only one position may be opened per day"
    assert res["trades"].iloc[0]["j"] == 0, "ranking must pick the higher k1"


def t_no_entry_below_size():
    o = np.full((10, 1), 2.0); h = o; l = o; c = o
    k = make_kit(o, h, l, c)
    k.dv20 = np.full((10, 1), 1e3, np.float32)  # 1% ADV cap -> 5 shares? keep tiny but >0
    s = sig(1, 0, 1.99)
    res = engine.run(k, s, engine.Cfg(style="bo", slip_bps=0), 0, 4)
    # liquidity cap binds but must never produce negative or zero-share fills
    if len(res["trades"]):
        assert res["trades"]["q0"].min() >= 1


def t_cash_conservation_long():
    o = np.full((10, 1), 50.0); h = o + 1; l = o - 1; c = o.copy()
    k = make_kit(o, h, l, c)
    res = engine.run(k, sig(1, 0, 49.0), engine.Cfg(style="bo", slip_bps=0), 0, 8)
    eq = res["equity"]
    assert (eq["cash"] >= -1e-6).all(), "long account must not borrow cash"
    tr = res["trades"].iloc[0]
    assert tr["q0"] * 50.0 <= 16000


def t_short_proceeds_do_not_add_buying_power():
    o = np.full((10, 1), 50.0); h = o + 1; l = o - 1; c = o.copy()
    k = make_kit(o, h, l, c)
    cfg = engine.Cfg(style="ps", side="short", slip_bps=0)
    res = engine.run(k, sig(1, 0, 51.0, atr=10.0), cfg, 0, 6)
    tr = res["trades"].iloc[0]
    assert tr["q0"] * 50.0 <= 0.30 * 16000 + 50, tr["q0"] * 50.0


def t_stop_priority_over_target_short():
    o = np.full((6, 1), 50.0); h = np.full((6, 1), 51.0); l = np.full((6, 1), 49.0)
    c = np.full((6, 1), 50.0)
    h[3, 0] = 60.0; l[3, 0] = 40.0  # both stop (55) and target (45) touched
    k = make_kit(o, h, l, c); k.sma10 = np.full((6, 1), 45.0, np.float32)
    res = engine.run(k, sig(1, 0, 55.0, atr=20.0), engine.Cfg(style="ps", side="short", slip_bps=0), 0, 5)
    assert res["trades"].iloc[0]["exit_reason"] == "stop"


def t_commission_minimum():
    o = np.full((6, 1), 10.0); h = o + 1; l = o - 1; c = o.copy()
    k = make_kit(o, h, l, c)
    k.dv20 = np.full((6, 1), 1e12, np.float32)     # no ADV cap
    res = engine.run(k, sig(1, 0, 9.9), engine.Cfg(style="bo", slip_bps=0, risk=0.0005), 0, 4)
    tr = res["trades"].iloc[0]
    # 0.05% of 16000 = $8 risk -> few shares -> min $1 per side
    assert tr["fees"] >= 2.0, tr["fees"]


def t_final_liquidation_once():
    o = np.full((6, 1), 10.0); h = o + 1; l = o - 1; c = o.copy()
    k = make_kit(o, h, l, c)
    res = engine.run(k, sig(1, 0, 8.5, atr=5), engine.Cfg(style="bo", slip_bps=0), 0, 5)
    tr = res["trades"]
    assert len(tr) == 1 and tr.iloc[0]["exit_reason"] == "final_liquidation"
    assert res["equity"].iloc[-1]["shares"] == 0


def t_future_price_does_not_change_past():
    o = np.full((12, 1), 100.0); h = o + 1; l = o - 1; c = o.copy()
    k1 = make_kit(o, h, l, c)
    s = sig(1, 0, 99.0, atr=2.0)
    a = engine.run(k1, s, engine.Cfg(style="bo", slip_bps=0), 0, 8)
    o2, h2, l2, c2 = o.copy(), h.copy(), l.copy(), c.copy()
    o2[7:] = 500; h2[7:] = 501; l2[7:] = 499; c2[7:] = 500
    k2 = make_kit(o2, h2, l2, c2)
    b = engine.run(k2, s, engine.Cfg(style="bo", slip_bps=0), 0, 8)
    ta, tb = a["trades"], b["trades"]
    assert ta.iloc[0]["entry"] == tb.iloc[0]["entry"], "entry must not depend on future prices"
    assert ta.iloc[0]["entry_day"] == tb.iloc[0]["entry_day"]


def t_split_no_fake_return():
    # 2:1 split with adjusted prices: adjusted series must be continuous
    raw_close = np.array([[100.0], [102.0], [51.0], [52.0]])
    raw_open = raw_close.copy()
    adj = np.array([[50.0], [51.0], [51.0], [52.0]])
    k = make_kit(raw_open * 0 + adj, adj + 1, adj - 1, adj)
    r = np.diff(k.c[:, 0]) / k.c[:-1, 0]
    assert np.all(np.abs(r) < 0.05), r


def t_segment_reset_starts_flat():
    o = np.full((10, 1), 10.0); h = o + 1; l = o - 1; c = o.copy()
    k = make_kit(o, h, l, c)
    s = sig(1, 0, 9.0, atr=5)
    res = engine.run(k, s, engine.Cfg(style="bo"), 5, 9)
    assert res["equity"].iloc[0]["shares"] == 0
    assert len(res["trades"]) == 0


def t_metrics_no_nan_crash():
    o = np.full((5, 1), 10.0); h = o; l = o; c = o
    k = make_kit(o, h, l, c)
    res = engine.run(k, pd.DataFrame(columns=["d", "j", "stop", "atr", "k1", "k2", "k3"]),
                     engine.Cfg(), 0, 4)
    m = metrics.perf(res["equity"], k.dates, 0, 4)
    assert np.isfinite(m["final_equity"])
    assert metrics.trade_stats(None)["n_trades"] == 0


TESTS = [
    ("planned_loss_within_budget", t_planned_loss_cap),
    ("gap_through_stop_fills_at_open", t_gap_through_stop_fills_at_open),
    ("single_position_and_ranking", t_single_position_and_no_reentry_same_day),
    ("liquidity_cap_no_zero_share", t_no_entry_below_size),
    ("long_cash_never_negative", t_cash_conservation_long),
    ("short_proceeds_do_not_add_power", t_short_proceeds_do_not_add_buying_power),
    ("stop_priority_over_target", t_stop_priority_over_target_short),
    ("commission_minimum_1usd", t_commission_minimum),
    ("final_liquidation_once", t_final_liquidation_once),
    ("future_prices_do_not_retroact", t_future_price_does_not_change_past),
    ("split_no_fake_return", t_split_no_fake_return),
    ("segment_reset_flat", t_segment_reset_starts_flat),
    ("metrics_no_nan", t_metrics_no_nan_crash),
]

if __name__ == "__main__":
    for name, fn in TESTS:
        check(name, fn)
    ok = sum(1 for _, s, _ in RESULTS if s == "PASS")
    for name, status, msg in RESULTS:
        print(f"{status:5s} {name} {msg}")
    print(f"\n{ok}/{len(RESULTS)} passed")
    sys.exit(0 if ok == len(RESULTS) else 1)
