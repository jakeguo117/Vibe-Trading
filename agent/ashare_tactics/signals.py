"""Standalone entry and exit rules for the four tactics.

Each function walks one symbol's own trading days (suspended sessions already
removed). A signal known at the close is filled at the next open. Take-profit
at +8% is an intraday sell on a later session, and it is checked before a
same-day close-based exit. One position per symbol at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_tactics.constants import (
    LIQUIDITY_WINDOW,
    MAX_HOLD_521,
    PRICE_CAP,
    SAR_MIN_TREND,
    SLOPE_FLAT,
    TAKE_PROFIT,
)
from ashare_tactics.costs import (
    code6,
    ipo_free_bars,
    is_limit_down_open,
    is_limit_up_open,
    limit_band,
)
from ashare_tactics.indicators import kdj, macd_dif, parabolic_sar, slope_bucket, sma
from ashare_tactics.models import Candidate, SignalBatch


def _strictly_below(price: float, line: float) -> bool:
    """True when price is below the line by more than float noise."""
    if not np.isfinite(price) or not np.isfinite(line):
        return False
    return bool(price < line and not np.isclose(price, line, rtol=1e-9, atol=1e-6))


def _as_day(value: np.datetime64) -> pd.Timestamp:
    return pd.Timestamp(np.datetime64(value, "D"))


def _liquidity(amount: np.ndarray, index: int) -> float:
    lo = max(0, index - LIQUIDITY_WINDOW + 1)
    window = amount[lo : index + 1]
    window = window[np.isfinite(window)]
    if len(window) == 0:
        return 0.0
    return float(window.mean())


class _Book:
    """Shared per-symbol state while a tactic walks bars."""

    def __init__(
        self,
        symbol: str,
        dates: np.ndarray,
        raw_open: np.ndarray,
        raw_high: np.ndarray,
        raw_low: np.ndarray,
        raw_close: np.ndarray,
        volume: np.ndarray,
        amount: np.ndarray,
        factor: np.ndarray,
        is_st: bool,
    ) -> None:
        self.symbol = symbol
        self.code = code6(symbol)
        self.dates = np.asarray(dates, dtype="datetime64[D]")
        self.raw_open = np.asarray(raw_open, dtype=float)
        self.raw_high = np.asarray(raw_high, dtype=float)
        self.raw_low = np.asarray(raw_low, dtype=float)
        self.raw_close = np.asarray(raw_close, dtype=float)
        self.volume = np.asarray(volume, dtype=float)
        self.amount = np.asarray(amount, dtype=float)
        self.factor = np.asarray(factor, dtype=float)
        self.factor = np.where(self.factor > 0, self.factor, np.nan)
        self.is_st = bool(is_st)
        self.n = len(self.dates)
        self.ipo_n = ipo_free_bars(self.code, self.dates[0]) if self.n else 0
        with np.errstate(divide="ignore", invalid="ignore"):
            self.q_open = self.raw_open / self.factor
            self.q_high = self.raw_high / self.factor
            self.q_low = self.raw_low / self.factor
            self.q_close = self.raw_close / self.factor
        self.batch = SignalBatch([])

    def band_at(self, index: int) -> float | None:
        return limit_band(self.code, self.dates[index], self.is_st, index < self.ipo_n)

    def can_buy(self, entry: int) -> bool:
        if entry <= 0 or entry >= self.n:
            return False
        raw_open = self.raw_open[entry]
        prev = self.raw_close[entry - 1]
        if not np.isfinite(raw_open) or raw_open <= 0 or not np.isfinite(self.factor[entry]):
            return False
        if is_limit_up_open(raw_open, prev, self.band_at(entry)):
            self.batch.limit_up_skips += 1
            return False
        return True

    def resolve_open_sell(self, start: int) -> tuple[int, float] | None:
        """First open at or after ``start`` that is not locked limit-down."""
        j = start
        while j < self.n:
            prev = self.raw_close[j - 1] if j > 0 else np.nan
            if not is_limit_down_open(self.raw_open[j], prev, self.band_at(j)):
                if np.isfinite(self.raw_open[j]) and self.raw_open[j] > 0 and np.isfinite(self.factor[j]):
                    return j, float(self.raw_open[j])
            j += 1
        if self.n == 0:
            return None
        last = self.n - 1
        if np.isfinite(self.raw_close[last]) and self.raw_close[last] > 0:
            return last, float(self.raw_close[last])
        return None

    def emit(
        self,
        entry: int,
        exit_i: int,
        raw_exit: float,
        *,
        slope: int | None,
        take_profit: bool,
    ) -> None:
        if exit_i <= entry or exit_i >= self.n:
            return
        if not np.isfinite(self.factor[entry]) or not np.isfinite(self.factor[exit_i]):
            return
        self.batch.candidates.append(
            Candidate(
                symbol=self.symbol,
                entry_date=_as_day(self.dates[entry]),
                exit_date=_as_day(self.dates[exit_i]),
                raw_entry=float(self.raw_open[entry]),
                raw_exit=float(raw_exit),
                factor_entry=float(self.factor[entry]),
                factor_exit=float(self.factor[exit_i]),
                liquidity=_liquidity(self.amount, entry - 1),
                slope=slope,
                take_profit=take_profit,
            )
        )

    def take_profit_hit(self, entry: int, j: int) -> float | None:
        """Unadjusted execution price if +8% is touched, else None.

        The 8% is measured on the forward-adjusted open of the entry session,
        so a dividend gap does not create or cancel the target. A gap through
        the target sells at that session's open; otherwise at the target.
        """
        if j <= entry:
            return None
        base = self.q_open[entry]
        if not np.isfinite(base) or base <= 0 or not np.isfinite(self.q_high[j]):
            return None
        target = base * (1.0 + TAKE_PROFIT)
        if self.q_high[j] < target:
            return None
        if np.isfinite(self.q_open[j]) and self.q_open[j] >= target:
            return float(self.raw_open[j])
        return float(target * self.factor[j])


def _prepare(
    symbol: str,
    dates: np.ndarray,
    raw_open: np.ndarray,
    raw_high: np.ndarray,
    raw_low: np.ndarray,
    raw_close: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    factor: np.ndarray,
    is_st: bool,
) -> _Book | None:
    if len(dates) < 22:
        return None
    return _Book(
        symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st
    )


def tactic_dual_ma(
    symbol: str,
    dates: np.ndarray,
    raw_open: np.ndarray,
    raw_high: np.ndarray,
    raw_low: np.ndarray,
    raw_close: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    factor: np.ndarray,
    is_st: bool = False,
) -> SignalBatch:
    """Tactic 5. MA5 crosses above MA20, next close confirms, buy the open after.

    Exit at +8% (intraday) or the next open after a close back under MA5.
    ``slope`` is the MA20 5-session change on the last close before the buy:
    +1 up, 0 flat (within ±0.5%), -1 down.
    """
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    ma20 = sma(book.q_close, 20)
    i = 20
    while i < book.n - 2:
        if not (
            np.isfinite(ma5[i])
            and np.isfinite(ma20[i])
            and np.isfinite(ma5[i - 1])
            and np.isfinite(ma20[i - 1])
        ):
            i += 1
            continue
        crossed = ma5[i] > ma20[i] and ma5[i - 1] <= ma20[i - 1]
        confirmed = np.isfinite(book.q_close[i + 1]) and book.q_close[i + 1] > book.q_close[i]
        if not (crossed and confirmed):
            i += 1
            continue
        book.batch.signals += 1
        entry = i + 2
        if not book.can_buy(entry):
            i += 1
            continue
        exit_i, raw_exit, take_profit = _exit_dual_ma(book, entry, ma5)
        if exit_i is None:
            i = entry
            continue
        book.emit(
            entry,
            exit_i,
            raw_exit,
            slope=slope_bucket(ma20, entry - 1, SLOPE_FLAT),
            take_profit=take_profit,
        )
        i = exit_i
    return book.batch


def _exit_dual_ma(book: _Book, entry: int, ma5: np.ndarray) -> tuple[int | None, float, bool]:
    j = entry
    while j < book.n:
        tp = book.take_profit_hit(entry, j)
        if tp is not None:
            return j, tp, True
        if _strictly_below(book.q_close[j], ma5[j]):
            sold = book.resolve_open_sell(j + 1)
            if sold is None or sold[0] <= entry:
                return None, 0.0, False
            return sold[0], sold[1], False
        j += 1
    sold = book.resolve_open_sell(book.n)
    if sold is None or sold[0] <= entry:
        return None, 0.0, False
    return sold[0], sold[1], False


def tactic_521(
    symbol: str,
    dates: np.ndarray,
    raw_open: np.ndarray,
    raw_high: np.ndarray,
    raw_low: np.ndarray,
    raw_close: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    factor: np.ndarray,
    is_st: bool = False,
) -> SignalBatch:
    """Tactic 13. MA5 crosses above MA21 with volume ≥ 2×, next bar up > 5%.

    Buy the following open. Exit the next open after a close under MA5, or at
    the open after 20 trading days, whichever comes first.
    """
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    ma21 = sma(book.q_close, 21)
    i = 21
    while i < book.n - 2:
        if not (
            np.isfinite(ma5[i])
            and np.isfinite(ma21[i])
            and np.isfinite(ma5[i - 1])
            and np.isfinite(ma21[i - 1])
        ):
            i += 1
            continue
        crossed = ma5[i] > ma21[i] and ma5[i - 1] <= ma21[i - 1]
        vol_ok = book.volume[i - 1] > 0 and book.volume[i] >= 2.0 * book.volume[i - 1]
        prev = book.q_close[i]
        nxt = book.q_close[i + 1]
        up = np.isfinite(prev) and prev > 0 and np.isfinite(nxt) and nxt / prev - 1.0 > 0.05
        bull = np.isfinite(book.q_open[i + 1]) and nxt > book.q_open[i + 1]
        if not (crossed and vol_ok and up and bull):
            i += 1
            continue
        book.batch.signals += 1
        entry = i + 2
        if not book.can_buy(entry):
            i += 1
            continue
        exit_i, raw_exit = _exit_521(book, entry, ma5)
        if exit_i is None:
            i = entry
            continue
        book.emit(entry, exit_i, raw_exit, slope=None, take_profit=False)
        i = exit_i
    return book.batch


def _exit_521(book: _Book, entry: int, ma5: np.ndarray) -> tuple[int | None, float]:
    j = entry
    while j < book.n:
        ma_break = _strictly_below(book.q_close[j], ma5[j])
        # Entry day is day 1. After 20 sessions, sell the next open.
        timed = j >= entry + MAX_HOLD_521 - 1
        if ma_break or timed:
            sold = book.resolve_open_sell(j + 1)
            if sold is None or sold[0] <= entry:
                return None, 0.0
            return sold[0], sold[1]
        j += 1
    sold = book.resolve_open_sell(book.n)
    if sold is None or sold[0] <= entry:
        return None, 0.0
    return sold[0], sold[1]


def tactic_sar(
    symbol: str,
    dates: np.ndarray,
    raw_open: np.ndarray,
    raw_high: np.ndarray,
    raw_low: np.ndarray,
    raw_close: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    factor: np.ndarray,
    is_st: bool = False,
) -> SignalBatch:
    """Tactic 19. Buy the next open when price flips above SAR.

    Sell the next open on a flip back below SAR. Flips whose prior trend
    lasted fewer than 5 bars are ignored (no entry and no exit).
    """
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    _sar, trend = parabolic_sar(book.q_high, book.q_low, book.q_close)
    in_pos = False
    entry = -1
    trend_start = 0
    # The seed writes a constant trend across the first window; the first
    # comparable bar is index 1 so a flip is a real change.
    for i in range(1, book.n):
        if trend[i] == 0 or trend[i] == trend[i - 1]:
            continue
        length = i - trend_start
        trend_start = i
        if length < SAR_MIN_TREND:
            continue
        book.batch.signals += 1
        if trend[i] == 1 and not in_pos:
            buy_at = i + 1
            if buy_at >= book.n or not book.can_buy(buy_at):
                continue
            in_pos = True
            entry = buy_at
            continue
        if trend[i] == -1 and in_pos:
            sold = book.resolve_open_sell(i + 1)
            if sold is None or sold[0] <= entry:
                in_pos = False
                continue
            book.emit(entry, sold[0], sold[1], slope=None, take_profit=False)
            in_pos = False
    if in_pos:
        sold = book.resolve_open_sell(book.n)
        if sold is not None and sold[0] > entry:
            book.emit(entry, sold[0], sold[1], slope=None, take_profit=False)
    return book.batch


def tactic_dual_indicator(
    symbol: str,
    dates: np.ndarray,
    raw_open: np.ndarray,
    raw_high: np.ndarray,
    raw_low: np.ndarray,
    raw_close: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    factor: np.ndarray,
    is_st: bool = False,
    apply_st_filter: bool = True,
) -> SignalBatch:
    """Tactic 25. DIF > 0 and rising, K crosses above D, close > MA20, price < 6.

    ST / *ST names are dropped when ``apply_st_filter`` is set. The feed only
    has the latest abbreviated name, so this is not a point-in-time ST flag.
    Exit the next open after a KDJ dead cross or a close back under MA20.
    """
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    if apply_st_filter and is_st:
        return book.batch
    dif = macd_dif(book.q_close)
    k_line, d_line = kdj(book.q_high, book.q_low, book.q_close)
    ma20 = sma(book.q_close, 20)
    in_pos = False
    entry = -1
    i = 26
    while i < book.n - 1:
        if in_pos:
            dead = (
                np.isfinite(k_line[i])
                and np.isfinite(d_line[i])
                and np.isfinite(k_line[i - 1])
                and np.isfinite(d_line[i - 1])
                and k_line[i] < d_line[i]
                and k_line[i - 1] >= d_line[i - 1]
            )
            under = _strictly_below(book.q_close[i], ma20[i])
            if dead or under:
                sold = book.resolve_open_sell(i + 1)
                if sold is not None and sold[0] > entry:
                    book.emit(entry, sold[0], sold[1], slope=None, take_profit=False)
                    in_pos = False
                    i = sold[0]
                    continue
            i += 1
            continue
        golden = (
            np.isfinite(k_line[i])
            and np.isfinite(d_line[i])
            and np.isfinite(k_line[i - 1])
            and np.isfinite(d_line[i - 1])
            and k_line[i] > d_line[i]
            and k_line[i - 1] <= d_line[i - 1]
        )
        dif_ok = np.isfinite(dif[i]) and np.isfinite(dif[i - 1]) and dif[i] > 0 and dif[i] > dif[i - 1]
        above = np.isfinite(ma20[i]) and book.q_close[i] > ma20[i]
        cheap = np.isfinite(book.raw_close[i]) and book.raw_close[i] < PRICE_CAP
        if not (golden and dif_ok and above and cheap):
            i += 1
            continue
        book.batch.signals += 1
        buy_at = i + 1
        if not book.can_buy(buy_at):
            i += 1
            continue
        in_pos = True
        entry = buy_at
        i = buy_at
    if in_pos:
        sold = book.resolve_open_sell(book.n)
        if sold is not None and sold[0] > entry:
            book.emit(entry, sold[0], sold[1], slope=None, take_profit=False)
    return book.batch


TACTICS = (
    ("tactic5", "战术5 双均线", tactic_dual_ma),
    ("tactic13", "战术13 主力521", tactic_521),
    ("tactic19", "战术19 抛物转向", tactic_sar),
    ("tactic25", "战术25 双指标", tactic_dual_indicator),
)
