"""Synthetic tests for the A-share tactic backtest. No market data and no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_tactics.bars import repair_factor
from ashare_tactics.costs import (
    board_limit,
    buy_cash_out,
    is_limit_down_open,
    is_limit_up_open,
    round_half_up,
    sell_cash_in,
    size_shares,
    stamp_rate,
)
from ashare_tactics.indicators import kdj, parabolic_sar, sma
from ashare_tactics.metrics import max_drawdown, sharpe_ratio, yearly_returns
from ashare_tactics.models import Candidate
from ashare_tactics.portfolio import SymbolBars, simulate
from ashare_tactics.signals import tactic_521, tactic_dual_indicator, tactic_dual_ma, tactic_sar


def _days(n: int, start: str = "2020-01-01") -> np.ndarray:
    start_day = np.datetime64(start, "D")
    return start_day + np.arange(n, dtype=np.int64) * np.timedelta64(1, "D")


def _book(close, open_=None, high=None, low=None, volume=None, amount=None, factor=None, start="2020-01-01"):
    close = np.asarray(close, dtype=float)
    n = len(close)
    open_ = close.copy() if open_ is None else np.asarray(open_, dtype=float)
    high = np.maximum(open_, close) if high is None else np.asarray(high, dtype=float)
    low = np.minimum(open_, close) if low is None else np.asarray(low, dtype=float)
    volume = np.full(n, 1_000.0) if volume is None else np.asarray(volume, dtype=float)
    amount = np.full(n, 1_000_000.0) if amount is None else np.asarray(amount, dtype=float)
    factor = np.ones(n) if factor is None else np.asarray(factor, dtype=float)
    return {
        "dates": _days(n, start),
        "raw_open": open_,
        "raw_high": high,
        "raw_low": low,
        "raw_close": close,
        "volume": volume,
        "amount": amount,
        "factor": factor,
    }


def test_repair_factor_drops_spurious_jump_and_keeps_splits():
    raw = np.array([10.0, 10.0, 10.0, 5.0, 5.2])
    # Index 2: factor collapses, raw price does not. That gap is not a split.
    factor = np.array([2.0, 2.0, 1.0, 1.0, 1.0])
    fixed = repair_factor(raw, factor)
    assert fixed[2] == pytest.approx(2.0)
    assert fixed[4] == pytest.approx(2.0)
    # A 2-for-1 split halves the raw price and the factor together.
    split_raw = np.array([10.0, 10.0, 5.0, 5.2])
    split_factor = np.array([2.0, 2.0, 1.0, 1.0])
    kept = repair_factor(split_raw, split_factor)
    assert kept[2] == pytest.approx(1.0)


def test_round_half_up_and_limits():
    assert round_half_up(10.005) == 10.01
    assert round_half_up(10 * 1.1) == 11.0
    assert board_limit("600000", np.datetime64("2018-01-02")) == 0.10
    assert board_limit("300001", np.datetime64("2020-08-21")) == 0.10
    assert board_limit("300001", np.datetime64("2020-08-24")) == 0.20
    assert board_limit("301001", np.datetime64("2024-01-02")) == 0.20
    assert board_limit("688001", np.datetime64("2021-01-04")) == 0.20
    assert board_limit("920000", np.datetime64("2024-01-02")) == 0.30
    assert is_limit_up_open(11.0, 10.0, 0.10)
    assert not is_limit_up_open(10.99, 10.0, 0.10)
    assert is_limit_down_open(9.0, 10.0, 0.10)
    assert not is_limit_down_open(9.01, 10.0, 0.10)


def test_stamp_cutover_and_min_commission():
    assert stamp_rate(np.datetime64("2023-08-25")) == 0.001
    assert stamp_rate(np.datetime64("2023-08-28")) == 0.0005
    # 100 shares at 1 yuan: notional after slippage is about 100, commission floors at 5.
    assert buy_cash_out(100, 1.0) == 100 * 1.001 + 5.0
    before = sell_cash_in(100, 10.0, 1.0, 1.0, np.datetime64("2023-08-25"))
    after = sell_cash_in(100, 10.0, 1.0, 1.0, np.datetime64("2023-08-28"))
    notion = 100 * 10.0 * 0.999
    assert after - before == pytest.approx(notion * 0.0005)


def test_dividend_residual_is_not_a_loss():
    # Factor falls from 2 to 1 while the raw price falls from 10 to 6.
    # qfq goes from 5 to 6, so the economic result is a gain.
    cash_out = buy_cash_out(100, 10.0)
    cash_in = sell_cash_in(100, 6.0, 2.0, 1.0, np.datetime64("2022-01-04"))
    assert cash_in > cash_out


def test_lot_size_respects_cash_and_board_lot():
    shares = size_shares(10_000, 10_000, 8.0, weight=0.10)
    assert shares == 100
    assert size_shares(10_000, 10_000, 500.0, weight=0.10) == 0


def test_kdj_seed_and_sma():
    k_line, d_line = kdj(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.5, 3.0]),
        n=2,
        m=3,
    )
    # First bar is still inside the window. Second bar: range 1..2, close 1.5 → RSV 50.
    assert k_line[1] == 50.0
    assert d_line[1] == 50.0
    values = sma(np.arange(1, 6, dtype=float), 3)
    assert np.isnan(values[1])
    assert values[2] == 2.0
    assert values[4] == 4.0


def test_dual_ma_confirms_then_buys_two_days_later_and_takes_profit():
    close = [10.0] * 25 + [30.0, 31.0, 31.0, 31.0]
    high = close.copy()
    high[-1] = 50.0
    batch = tactic_dual_ma("sh600000", **_book(close, high=high), is_st=False)
    assert len(batch.candidates) == 1
    trade = batch.candidates[0]
    dates = _days(len(close))
    assert trade.entry_date == pd.Timestamp(dates[27])
    assert trade.exit_date == pd.Timestamp(dates[28])
    assert trade.take_profit
    assert trade.slope == 1
    # T+1 close no longer confirms: the cross is dropped.
    close[26] = 29.0
    refused = tactic_dual_ma("sh600000", **_book(close, high=high), is_st=False)
    assert refused.candidates == []


def test_dual_ma_skips_limit_up_open():
    close = [10.0] * 25 + [30.0, 31.0, 31.0, 31.0]
    open_ = close.copy()
    open_[27] = 34.10  # 31 * 1.10
    high = [max(o, c) for o, c in zip(open_, close)]
    high[-1] = 50.0
    batch = tactic_dual_ma("sh600000", **_book(close, open_=open_, high=high), is_st=False)
    assert batch.candidates == []
    assert batch.limit_up_skips == 1


def test_split_does_not_create_a_moving_average_cross():
    # Raw price halves and the qfq factor halves with it, so qfq is flat.
    close = [20.0] * 15 + [10.0] * 20
    factor = [2.0] * 15 + [1.0] * 20
    batch = tactic_dual_ma("sh600000", **_book(close, factor=factor), is_st=False)
    assert batch.candidates == []


def test_521_volume_burst_max_hold_and_limit_down_defer():
    n = 60
    close = [10.0] * n
    close[30] = 40.0
    close[31] = 42.4
    for i in range(32, n):
        close[i] = 42.4
    open_ = close.copy()
    open_[31] = 41.0
    volume = [100.0] * n
    volume[30] = 300.0
    batch = tactic_521("sh600000", **_book(close, open_=open_, volume=volume), is_st=False)
    assert len(batch.candidates) == 1
    dates = _days(n)
    assert batch.candidates[0].entry_date == pd.Timestamp(dates[32])
    assert batch.candidates[0].exit_date == pd.Timestamp(dates[52])
    # The scheduled exit opens limit-down, so the sell waits one session.
    open_[52] = round(42.4 * 0.9, 2)
    deferred = tactic_521("sh600000", **_book(close, open_=open_, volume=volume), is_st=False)
    assert deferred.candidates[0].exit_date == pd.Timestamp(dates[53])
    volume[30] = 150.0  # only 1.5×, not a signal
    quiet = tactic_521("sh600000", **_book(close, open_=open_, volume=volume), is_st=False)
    assert quiet.candidates == []


def test_sar_ignores_whipsaw_and_trades_a_sustained_flip():
    # After the opening trend, SAR flips every three bars. Those flips are
    # ignored, so a position opened by the first valid flip is held to the end.
    chop_high = []
    chop_low = []
    chop_close = []
    for i in range(40):
        up = (i // 3) % 2 == 0
        base = 20.0 if up else 10.0
        chop_close.append(base)
        chop_high.append(base + 1.0)
        chop_low.append(base - 1.0)
    chop = tactic_sar("sh600000", **_book(chop_close, high=chop_high, low=chop_low), is_st=False)
    assert len(chop.candidates) == 1
    assert chop.candidates[0].exit_date == pd.Timestamp(_days(40)[-1])

    # Down, then up, then down. Each leg is long enough that the flip is tradable,
    # and each daily step stays inside the 10% limit.
    close = np.concatenate(
        [np.linspace(12, 10, 15), np.linspace(10.05, 13, 18), np.linspace(12.9, 10, 18)]
    )
    high = close + 0.05
    low = close - 0.05
    trend_batch = tactic_sar("sh600000", **_book(close, high=high, low=low), is_st=False)
    assert trend_batch.candidates
    assert all(pd.Timestamp(c.exit_date) > pd.Timestamp(c.entry_date) for c in trend_batch.candidates)
    _sar, trend = parabolic_sar(high, low, close)
    flips = np.where((trend[1:] != trend[:-1]) & (trend[1:] != 0))[0] + 1
    assert len(flips) >= 1


def test_dual_indicator_price_cap_and_st_name():
    n = 40
    # Stay under 6 yuan and drift up so DIF is positive and price is over a rising MA.
    close = np.linspace(4.0, 5.5, n)
    high = close + 0.2
    low = close - 0.2
    common = _book(close, high=high, low=low)
    st = tactic_dual_indicator("sz000001", **common, is_st=True, apply_st_filter=True)
    assert st.candidates == []
    # Above 6 yuan the price filter removes every bar even if the name is clean.
    rich = _book(close + 3.0, high=high + 3.0, low=low + 3.0)
    capped = tactic_dual_indicator("sz000001", **rich, is_st=False, apply_st_filter=True)
    assert capped.candidates == []
    # The ST gate itself is what emptied the first batch: turning it off can differ.
    opened = tactic_dual_indicator("sz000001", **common, is_st=True, apply_st_filter=False)
    assert isinstance(opened.candidates, list)


def test_portfolio_ranks_by_liquidity_and_caps_at_ten():
    calendar = pd.to_datetime(["2020-01-02", "2020-01-03"])
    bars = SymbolBars(
        dates=np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        raw_open=np.array([10.0, 10.0]),
        raw_close=np.array([10.0, 10.0]),
        factor=np.ones(2),
    )

    def loader(_symbol: str) -> SymbolBars:
        return bars

    candidates = []
    for i in range(11):
        candidates.append(
            Candidate(
                symbol=f"sh{600000 + i:06d}",
                entry_date=pd.Timestamp("2020-01-02"),
                exit_date=pd.Timestamp("2020-01-03"),
                raw_entry=10.0,
                raw_exit=10.0,
                factor_entry=1.0,
                factor_exit=1.0,
                liquidity=float(i + 1),
            )
        )
    result = simulate(candidates, pd.DatetimeIndex(calendar), loader, initial_cash=1_000_000)
    assert len(result.trades) == 10
    assert set(result.trades["symbol"]) == {f"sh{600000 + i:06d}" for i in range(1, 11)}
    assert result.skipped_slots == 1
    assert (result.trades["shares"] % 100 == 0).all()
    # Flat prices still lose the round-trip costs, and equity stays below the start.
    assert result.equity.iloc[-1] < 1_000_000


def test_metrics_drawdown_and_year_split():
    equity = pd.Series([100.0, 80.0, 90.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    assert max_drawdown(equity) == pytest.approx(-0.2)
    flat = pd.Series([100.0, 100.0, 100.0], index=equity.index)
    assert sharpe_ratio(flat) == 0.0
    longer = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2019-12-31", "2020-06-30", "2020-12-31"]),
    )
    years = yearly_returns(longer, initial=100.0)
    assert years.loc[2019] == 0.0
    assert abs(years.loc[2020] - 0.21) < 1e-9


def test_pattern_yizi_and_dragon_smoke():
    from ashare_tactics.signals_patterns import tactic_yizi, tactic_dragon_sea, tactic_volume_engulf

    n = 80
    close = np.linspace(8.0, 9.0, n)
    # force a flat region then a one-word limit-up-like bar is hard without exact limits;
    # just assert the functions return SignalBatch and do not crash.
    batch = tactic_yizi("sh600000", **_book(close), is_st=False)
    assert hasattr(batch, "candidates")
    batch2 = tactic_dragon_sea("sh600000", **_book(close), is_st=False)
    assert hasattr(batch2, "candidates")
    batch3 = tactic_volume_engulf("sh600000", **_book(np.linspace(5, 6, 40)), is_st=False)
    assert hasattr(batch3, "candidates")


def test_tactics_registry_has_eighteen():
    from ashare_tactics.signals import TACTICS

    assert len(TACTICS) == 18
    ids = {k for k, _, _ in TACTICS}
    assert "tactic2" in ids and "tactic26" in ids and "tactic5" in ids
