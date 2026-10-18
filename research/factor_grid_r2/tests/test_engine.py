"""Deterministic checks that do not download prices."""

import numpy as np
import pandas as pd

from factor_grid_r2.engine import Pack, cfg_control, cfg_momentum, run_backtest
from factor_grid_r2.options import call_delta, strike_for_delta
from factor_grid_r2.signals import fib_stage_codes, ma_leg


def _pack(n_days=420, seed=20261013):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    names = ["SPY", "QQQ", "GLD", "AAA", "BBB"]
    n = len(names)
    close = np.zeros((n_days, n))
    close[0] = [100, 80, 50, 40, 30]
    rets = rng.normal(0.0008, 0.01, size=(n_days, n))
    rets[:, 2] += 0.0004
    for t in range(1, n_days):
        close[t] = close[t - 1] * (1.0 + rets[t])
    # Force a crash on AAA so the anchor stop has something to do.
    close[300:, 3] *= np.linspace(1.0, 0.5, n_days - 300)
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1.0 + rng.normal(0, 0.001, size=close.shape))
    volu = np.full_like(close, 1e6)
    volu[:, 0] = 5e6
    ones = np.ones((n_days, n), dtype=bool)
    stage = np.zeros((n_days, n), dtype=np.int8)
    stage[:, 0] = 2
    up = {30: ones.copy(), 24: ones.copy(), 36: ones.copy()}
    # AAA leaves the up-leg late; before that it is confirmed.
    up[30][320:, 3] = False
    confirm = {30: ones.copy()}
    anchor = {30: np.minimum(low, close) }
    # Anchor above the later crash closes.
    anchor[30][:, 3] = close[280, 3]
    dn = {30: np.zeros((n_days, n), dtype=bool)}
    dn[30][340:, 3] = True
    mom = {20: np.full((n_days, n), 0.05)}
    mom[20][:, 4] = 0.02
    kind = np.array([1, 1, 2, 0, 0], dtype=np.int8)
    return Pack(
        names=names,
        ix={s: i for i, s in enumerate(names)},
        n=n,
        dates=pd.DatetimeIndex(dates),
        O=open_,
        H=high,
        L=low,
        C=close,
        V=volu,
        up=up,
        dn=dn,
        confirm=confirm,
        anchor=anchor,
        stage=stage,
        route=np.zeros((n_days, n), dtype=np.int8),
        vol=np.full((n_days, n), 0.12),
        realized=np.full((n_days, n), 0.20),
        mom=mom,
        dv60=volu * close,
        spy_trend=np.ones(n_days, dtype=bool),
        kind=kind,
        is_crypto=np.zeros(n, dtype=bool),
        tradable_control=np.array([True, True, True, True, True]),
        tradable_stock=np.array([True, True, False, True, True]),
        tradable_sc=np.array([True, True, False, True, True]),
        side=np.full(n, 0.0005),
        lev_index=np.array([-1, -1, -1, -1, -1]),
    )


def test_two_runs_match():
    pack = _pack()
    a = run_backtest(pack, cfg_control(), "2018-06-01", "2019-08-01")
    b = run_backtest(pack, cfg_control(), "2018-06-01", "2019-08-01")
    assert np.allclose(a["curve"].dropna().to_numpy(), b["curve"].dropna().to_numpy())
    assert a["max_exposure"] <= 0.701


def test_cash_floor_and_anchor():
    pack = _pack()
    s = run_backtest(pack, cfg_momentum("S", "S"), "2018-06-01", "2019-08-01")
    assert np.isfinite(s["cagr"])
    assert s["max_exposure"] <= 1.001
    assert s["n_trades"] >= 1


def test_ma_leg_needs_two_weeks():
    close = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4] + [4] * 40, dtype=float)
    on, ma, dn = ma_leg(close, 5)
    assert on.dtype == bool
    assert on.sum() > 0


def test_fib_codes_in_range():
    rng = np.random.default_rng(1)
    x = np.cumprod(1 + rng.normal(0, 0.01, 80))
    codes = fib_stage_codes(x * 1.01, x * 0.99, x)
    assert set(np.unique(codes)).issubset({0, 1, 2, 3})


def test_strike_delta():
    k = strike_for_delta(100, 1.0, 0.25, 0.70, 0.02, 0.0)
    delta = call_delta(100, k, 1.0, 0.25, 0.02, 0.0)
    assert abs(delta - 0.70) < 0.03
