"""Plan §11 invariants for the hourly batch: no look-ahead, no same-bar fill, one position,
cash non-negative, deterministic reruns, and the D/H setup frozen before the session.

These run against the real Databento file (the only data this batch uses) on a truncated slice so
the suite stays usable; the truncation itself is one of the tests.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/jake/projects/trading/trading-system/backtest")

from fib_r1 import ceng, hourly as H  # noqa: E402
from fib_r1.data import load_etf_daily  # noqa: E402

SNAP = Path("/Users/jake/projects/trading/data/ts_v111")
CUT = "2021-06-30"
START = "2019-02-04"


def _daily():
    return load_etf_daily(SNAP / "raw")


def _long(daily):
    rows = []
    for s, d in daily.items():
        x = d.reset_index()
        x["sym"] = s
        rows.append(x[["date", "sym", "open", "high", "low", "close", "volume", "dv60"]]
                    .rename(columns={"dv60": "dv"}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["close"])


@pytest.fixture(scope="module")
def fixture():
    daily = _daily()
    Ld = _long(daily)
    csv = H.HOURLY_ROOT / "csv" / H.CSV_NAME
    cut = pd.Timestamp(CUT, tz="UTC") + pd.Timedelta(days=1)
    raw = pd.read_csv(csv)
    ts = pd.to_datetime(raw.ts_event, utc=True, format="mixed")
    tmp = Path("/Users/jake/projects/hermes-tasks/runs/backtest-fib-hourly-20261020-1900/cache")
    tmp.mkdir(parents=True, exist_ok=True)
    tr_path = tmp / f"truncated_{CUT}.csv"
    raw[ts < cut].to_csv(tr_path, index=False)

    full = H.load_hourly(daily, csv)
    trunc = H.load_hourly(daily, tr_path)
    P = H.hourly_panel(full, Ld)
    Pt = H.hourly_panel(trunc, Ld)
    S = H.setup_table(Ld, full["sessions"])
    return dict(daily=daily, Ld=Ld, full=full, trunc=trunc, P=P, Pt=Pt, S=S, tr_path=tr_path)


def test_grid_rule(fixture):
    bars = fixture["full"]["bars"]
    for d, g in bars.groupby("date"):
        exp = H.expected_hours(d)
        assert sorted(g.hour.unique()) == exp
        assert not g.short_bar.any()
    assert (bars.hour >= 10).all() and (bars.hour <= 15).all()


def test_early_close_detection(fixture):
    # known NYSE 13:00 closes inside the window, and full sessions that must NOT be flagged
    for d in ("2019-11-29", "2020-12-24", "2021-11-26", "2025-07-03", "2025-12-24"):
        assert H.early_close(pd.Timestamp(d)), d
    for d in ("2019-12-27", "2021-12-23", "2022-12-23", "2024-07-05"):
        assert not H.early_close(pd.Timestamp(d)), d
        assert len(H.expected_hours(pd.Timestamp(d))) == 6


def test_prior_day_inputs_only(fixture):
    """Daily-derived inputs at a session's bars must not move when that session's daily bar moves."""
    daily = {s: d.copy() for s, d in fixture["daily"].items()}
    Ld = fixture["Ld"].copy()
    d0 = pd.Timestamp("2020-06-15")
    for s in daily:
        if d0 in daily[s].index:
            daily[s].loc[d0, ["open", "high", "low", "close", "volume"]] *= 3.0
    Ld2 = _long(daily)
    P2 = H.hourly_panel(fixture["full"], Ld2)
    P1 = fixture["P"]
    i = int(np.flatnonzero(P1.dates.normalize() == d0)[0])
    j = int(np.flatnonzero(P1.dates.normalize() == d0)[-1])
    assert np.allclose(P1.SIGMA[i:j + 1], P2.SIGMA[i:j + 1], equal_nan=True)
    assert np.allclose(P1.DV[i:j + 1], P2.DV[i:j + 1], equal_nan=True)
    assert (P1.ELIG[i:j + 1] == P2.ELIG[i:j + 1]).all()
    # prove the perturbation was actually effective on the daily panel of d0 itself
    D1, D2 = ceng.build_panel(fixture["Ld"], "D"), ceng.build_panel(Ld2, "D")
    pos = int(np.searchsorted(D1.dates.values, np.datetime64(d0)))
    assert not np.allclose(D1.SIGMA[pos], D2.SIGMA[pos], equal_nan=True)


def test_dh_setup_frozen(fixture):
    Ld = fixture["Ld"]
    S = fixture["S"]
    d0 = pd.Timestamp("2020-06-15")
    d = int(np.flatnonzero(S["dates"] == d0)[0])
    daily = {s: x.copy() for s, x in fixture["daily"].items()}
    for s in daily:
        if d0 in daily[s].index:
            daily[s].loc[d0, ["high", "low", "close"]] *= 2.0
    S2 = H.setup_table(_long(daily), fixture["full"]["sessions"])
    assert np.allclose(S["brk"][d], S2["brk"][d], equal_nan=True)
    assert np.allclose(S["atr_prev"][d], S2["atr_prev"][d], equal_nan=True)
    assert (S["sq"][d] == S2["sq"][d]).all()
    assert np.allclose(S["sigma_prev"][d], S2["sigma_prev"][d], equal_nan=True)
    # atr_prev is the previous session's ATR by construction
    assert np.allclose(S["atr_prev"][d], S["atr"][d - 1], equal_nan=True)


def test_no_lookahead_truncation(fixture):
    """Signals before the cut cannot depend on bars after it."""
    P, Pt = fixture["P"], fixture["Pt"]
    a = ceng.run_breakout(P, squeeze=False, start=START, end=CUT)
    b = ceng.run_breakout(Pt, squeeze=False, start=START, end=CUT)
    ta, tb = a["trades"], b["trades"]
    ca = ta[pd.to_datetime(ta.exit_date) < pd.Timestamp(CUT)]
    cb = tb[pd.to_datetime(tb.exit_date) < pd.Timestamp(CUT)]
    assert len(ca) == len(cb) > 10
    assert list(ca.sym) == list(cb.sym)
    assert np.allclose(ca.net_ret.values, cb.net_ret.values, atol=1e-12)
    n = int(np.flatnonzero(P.dates <= P.dates[P.dates <= pd.Timestamp(CUT)][-1])[-1])
    assert np.allclose(a["eq"].equity.values[:len(b["eq"])], b["eq"].equity.values, atol=1e-9)


def test_no_same_bar_fill(fixture):
    P = fixture["P"]
    for squeeze in (False, True):
        r = ceng.run_breakout(P, squeeze=squeeze, start=START, end=CUT)
        t = r["trades"]
        assert len(t) > 5
        assert (t.entry_bar > t.entry_sig_bar).all()
        assert (t.exit_bar > t.entry_bar).all()


def test_single_position_cash_and_determinism(fixture):
    P = fixture["P"]
    r1 = ceng.run_breakout(P, squeeze=False, start=START, end=CUT)
    r2 = ceng.run_breakout(P, squeeze=False, start=START, end=CUT)
    assert np.array_equal(r1["eq"].equity.values, r2["eq"].equity.values)
    assert (r1["eq"].cash >= -1e-9).all()
    assert (r1["eq"].n_hold <= 1).all()
    t = r1["trades"]
    assert (t.entry_bar.values[1:] >= t.exit_bar.values[:-1]).all()


def test_dh_no_same_bar_and_daily_exit(fixture):
    P, S = fixture["P"], fixture["S"]
    r = H.run_dh(P, S, squeeze=False, start=START, end=CUT)
    t = r["trades"]
    assert len(t) > 5
    assert (t.entry_bar > t.entry_sig_bar).all()
    assert (r["eq"].cash >= -1e-9).all()
    assert (r["eq"].n_hold <= 1).all()
    # every exit lands on the first bar of a session (daily-close trigger -> next open)
    off = P.off
    assert (off[t.exit_bar.values] == 0).all()
    # the entry signal itself must be an intraday bar of the setup session, never the last-bar state
    assert ((t.entry_bar - t.entry_sig_bar) <= 7).all()
