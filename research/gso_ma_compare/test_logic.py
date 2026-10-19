"""Deterministic checks for the up-leg rules. No market data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gso.config import CONFIRM_WEEKS, DAILY_BARS_PER_WEEK, SLOPE_LAG_WEEKS
from gso.indicators import (
    fib_stage_frame,
    following_week_execution,
    portfolio_vol,
    sma_upleg,
    stage_from_level,
    target_weights,
    to_weekly,
    upleg_from_flags,
    wilder_atr,
)
from gso.simulate import apply_targets
from gso.study import best_jul_aug, detect_rallies


def test_entry_needs_two_weeks_and_halve_is_not_latched():
    # Enter on bar 1, halve on the second close below, restore when price reclaims,
    # halve again, then clear on the bar where the falling-MA streak also reaches 2.
    above = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    below = np.array([0, 0, 1, 1, 0, 1, 1, 1], dtype=bool)
    slope = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    inn, halve = upleg_from_flags(above, below, slope, confirm=2)
    assert list(map(int, inn)) == [0, 1, 1, 1, 1, 1, 1, 0]
    assert list(map(int, halve)) == [0, 0, 0, 1, 0, 0, 1, 0]


def test_slope_alone_does_not_clear():
    above = np.array([1, 1, 1, 1], dtype=bool)
    below = np.zeros(4, dtype=bool)
    slope = np.array([0, 0, 1, 1], dtype=bool)
    inn, halve = upleg_from_flags(above, below, slope, confirm=2)
    assert inn[-1] and not halve.any()


def test_one_week_does_not_enter():
    above = np.array([1, 0, 1, 0], dtype=bool)
    below = np.zeros(4, dtype=bool)
    slope = np.zeros(4, dtype=bool)
    inn, _ = upleg_from_flags(above, below, slope, confirm=2)
    assert not inn.any()


def test_sma_upleg_is_causal_and_uses_strict_ma_slope():
    close = pd.Series(np.arange(1, 81, dtype=float))
    full = sma_upleg(close, length=10, confirm=2, slope_lag=4)
    prefix = sma_upleg(close.iloc[:-6], length=10, confirm=2, slope_lag=4)
    pd.testing.assert_series_equal(full["in_upleg"].iloc[:-6], prefix["in_upleg"])
    assert full["in_upleg"].iloc[-1]
    flat = sma_upleg(pd.Series(np.ones(40)), length=5, confirm=2, slope_lag=4)
    assert not flat["in_upleg"].any()


def test_stage_bands_and_boundary_goes_up():
    origin, amp = 50.0, 50.0
    assert stage_from_level(80.89, origin, amp) == "磨底"
    assert stage_from_level(50 + 0.618 * 50, origin, amp) == "震荡"
    assert stage_from_level(99.0, origin, amp) == "震荡"
    assert stage_from_level(100.0, origin, amp) == "主升"
    assert stage_from_level(120.0, origin, amp) == "主升"
    assert stage_from_level(50 + 1.618 * 50, origin, amp) == "鱼尾"


def test_fib_stage_is_causal_and_reaches_extension():
    # Rise, drop, then extend beyond 1.618 of that drop so 鱼尾 can confirm.
    prices = []
    for value in range(80, 121, 2):
        prices.append(float(value))
    for value in range(118, 59, -2):
        prices.append(float(value))
    for value in range(62, 181, 2):
        prices.append(float(value))
    for value in range(178, 99, -2):
        prices.append(float(value))
    close = pd.Series(prices)
    high = close + 0.4
    low = close - 0.4
    atr = wilder_atr(high, low, close, n=14)
    full = fib_stage_frame(high, low, close, atr)
    prefix = fib_stage_frame(high.iloc[:-8], low.iloc[:-8], close.iloc[:-8], atr.iloc[:-8])
    pd.testing.assert_series_equal(full["fib_stage"].iloc[:-8], prefix["fib_stage"])
    assert set(full["fib_stage"].unique()) >= {"磨底", "主升", "鱼尾"}
    assert full["fib_stage"].iloc[0] == "震荡"


def test_weights_respect_cash_floor_vol_cap_and_name_limit():
    names = ["GLD", "SLV", "USO"]
    quiet = {name: 0.01 for name in names}
    full = target_weights(
        names,
        {name: True for name in names},
        {name: False for name in names},
        {name: "主升" for name in names},
        quiet,
    )
    assert abs(sum(full.values()) - 0.70) < 1e-9
    assert abs(full["GLD"] - 0.70 / 3) < 1e-12

    hot = target_weights(
        names,
        {name: True for name in names},
        {name: False for name in names},
        {name: "主升" for name in names},
        {name: 1.0 for name in names},
    )
    assert sum(hot.values()) < 0.70 - 1e-4

    halved = target_weights(
        names,
        {"GLD": True, "SLV": False, "USO": False},
        {"GLD": True, "SLV": False, "USO": False},
        {"GLD": "磨底", "SLV": "主升", "USO": "主升"},
        quiet,
    )
    assert abs(halved["GLD"] - (0.70 / 3) * 0.3 * 0.5) < 1e-12
    assert halved["SLV"] == 0.0 and halved["USO"] == 0.0

    four = target_weights(
        ["A", "B", "C", "D"],
        {name: True for name in "ABCD"},
        {name: False for name in "ABCD"},
        {name: "主升" for name in "ABCD"},
        {name: 0.01 for name in "ABCD"},
    )
    assert four["D"] == 0.0
    assert abs(sum(four.values()) - 0.70) < 1e-9


def test_portfolio_vol_matches_constant_correlation():
    weights = np.array([0.2, 0.2, 0.2])
    vols = np.array([0.2, 0.2, 0.2])
    # 3*w^2*v^2 + 6*0.35*w*w*v*v
    expected = np.sqrt(3 * (0.2 ** 2) * (0.2 ** 2) + 6 * 0.35 * (0.2 ** 2) * (0.2 ** 2))
    assert abs(portfolio_vol(weights, vols) - expected) < 1e-12


def test_execution_is_the_second_session_of_the_next_week():
    days = pd.bdate_range("2019-01-07", "2019-01-11")
    assert following_week_execution(pd.Timestamp("2019-01-04"), days) == pd.Timestamp("2019-01-08")
    holiday = pd.bdate_range("2019-01-22", "2019-01-25")
    assert following_week_execution(pd.Timestamp("2019-01-18"), holiday) == pd.Timestamp("2019-01-23")
    one_day = pd.DatetimeIndex(["2019-01-22"])
    assert following_week_execution(pd.Timestamp("2019-01-18"), one_day) == pd.Timestamp("2019-01-22")


def test_incomplete_week_is_dropped():
    idx = list(pd.bdate_range("2019-01-07", "2019-01-11")) + [pd.Timestamp("2019-01-14")]
    frame = pd.DataFrame(
        {"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": np.linspace(10, 12, len(idx)), "Volume": 1.0},
        index=pd.DatetimeIndex(idx),
    )
    weekly = to_weekly(frame)
    assert pd.Timestamp("2019-01-11") in weekly.index
    assert pd.Timestamp("2019-01-18") not in weekly.index
    assert weekly.loc[pd.Timestamp("2019-01-11"), "session"] == pd.Timestamp("2019-01-11")
    assert weekly.loc[pd.Timestamp("2019-01-11"), "Open"] == 10.0


def test_cost_is_ten_bps_of_traded_notional():
    cash, shares = apply_targets(1.0, {"GLD": 0.0}, {"GLD": 100.0}, {"GLD": 0.5}, ("GLD",), 0.001)
    assert abs(shares["GLD"] - 0.005) < 1e-12
    assert abs(cash - 0.4995) < 1e-12


def test_rally_window_and_gold_month():
    idx = pd.date_range("2019-01-04", periods=20, freq="W-FRI")
    slow = 100 + np.arange(14) * (10 / 13)
    slow_frame = pd.DataFrame({"Close": slow, "session": pd.date_range("2019-01-03", periods=14, freq="W-FRI")}, index=pd.date_range("2019-01-04", periods=14, freq="W-FRI"))
    assert detect_rallies(slow_frame, "GLD") == []

    px = np.full(len(idx), 100.0)
    px[8:14] = [100, 103, 106, 109, 112, 116]
    frame = pd.DataFrame({"Close": px, "session": idx}, index=idx)
    found = detect_rallies(frame, "SLV")
    assert len(found) == 1
    assert found[0]["周数"] <= 12
    assert found[0]["涨幅"] >= 0.10

    # Peak on 2025-08-01 (Friday) after a 12% rise.
    gold_idx = pd.date_range("2025-05-02", periods=20, freq="W-FRI")
    gold_px = np.full(len(gold_idx), 100.0)
    peak_at = list(gold_idx).index(pd.Timestamp("2025-08-01"))
    gold_px[peak_at - 4 : peak_at + 1] = [100, 103, 106, 110, 114]
    gold = pd.DataFrame({"Close": gold_px, "session": gold_idx}, index=gold_idx)
    season = best_jul_aug(gold, "GLD", 2025)
    assert season is not None
    assert season["见顶日"] == "2025-08-01"
    assert season["涨幅"] >= 0.10
    missing = best_jul_aug(gold, "GLD", 2026)
    assert missing is None


def test_daily_variant_scales_weeks_by_five_sessions():
    assert CONFIRM_WEEKS * DAILY_BARS_PER_WEEK == 10
    assert SLOPE_LAG_WEEKS * DAILY_BARS_PER_WEEK == 20
    assert 10 * DAILY_BARS_PER_WEEK == 50
    assert 30 * DAILY_BARS_PER_WEEK == 150
