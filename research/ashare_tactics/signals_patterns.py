"""Daily-bar encodings for the remaining NotebookLM A-share tactics.

Vague exits default to: close below MA5 → next open sell, or max-hold
``MAX_HOLD_DEFAULT`` sessions, or ``TAKE_PROFIT_WIDE`` (+15%). Documented in
``report.py``. One position per symbol at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_tactics.constants import (
    CONSOLIDATION_DAYS,
    GIANT_VOL,
    MAX_HOLD_DEFAULT,
    PRICE_CAP_17,
    SMALL_BODY,
    TAKE_PROFIT_WIDE,
)
from ashare_tactics.costs import is_limit_up_close
from ashare_tactics.indicators import kdj, sma
from ashare_tactics.models import SignalBatch
from ashare_tactics.signals import _Book, _prepare, _strictly_below


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    tp = (np.asarray(high, float) + np.asarray(low, float) + np.asarray(close, float)) / 3.0
    n = len(tp)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_tp = tp[i - window + 1 : i + 1]
        if not np.isfinite(window_tp).all():
            continue
        mean = float(window_tp.mean())
        mad = float(np.mean(np.abs(window_tp - mean)))
        if mad < 1e-12:
            out[i] = 0.0
        else:
            out[i] = (tp[i] - mean) / (0.015 * mad)
    return out


def _default_exit(book: _Book, entry: int, ma5: np.ndarray) -> tuple[int | None, float, bool]:
    """+15% TP, MA5 break next open, or max-hold next open."""
    j = entry
    while j < book.n:
        # take profit on forward-adjusted high
        base = book.q_open[entry]
        if np.isfinite(base) and base > 0 and np.isfinite(book.q_high[j]):
            target = base * (1.0 + TAKE_PROFIT_WIDE)
            if book.q_high[j] >= target:
                if np.isfinite(book.q_open[j]) and book.q_open[j] >= target:
                    return j, float(book.raw_open[j]), True
                return j, float(target * book.factor[j]), True
        ma_break = _strictly_below(book.q_close[j], ma5[j]) if np.isfinite(ma5[j]) else False
        timed = j >= entry + MAX_HOLD_DEFAULT - 1
        if ma_break or timed:
            sold = book.resolve_open_sell(j + 1)
            if sold is None or sold[0] <= entry:
                return None, 0.0, False
            return sold[0], sold[1], False
        j += 1
    sold = book.resolve_open_sell(book.n)
    if sold is None or sold[0] <= entry:
        return None, 0.0, False
    return sold[0], sold[1], False


def _exit_ma(book: _Book, entry: int, ma: np.ndarray, max_hold: int = MAX_HOLD_DEFAULT) -> tuple[int | None, float]:
    j = entry
    while j < book.n:
        if _strictly_below(book.q_close[j], ma[j]) or j >= entry + max_hold - 1:
            sold = book.resolve_open_sell(j + 1)
            if sold is None or sold[0] <= entry:
                return None, 0.0
            return sold[0], sold[1]
        j += 1
    sold = book.resolve_open_sell(book.n)
    if sold is None or sold[0] <= entry:
        return None, 0.0
    return sold[0], sold[1]


def _is_limit_up_bar(book: _Book, i: int) -> bool:
    if i <= 0:
        return False
    return is_limit_up_close(book.raw_close[i], book.raw_close[i - 1], book.band_at(i))


def _is_yizi(book: _Book, i: int) -> bool:
    """一字板: open≈high≈low≈close and limit-up."""
    if not _is_limit_up_bar(book, i):
        return False
    o, h, l, c = book.raw_open[i], book.raw_high[i], book.raw_low[i], book.raw_close[i]
    if min(o, h, l, c) <= 0:
        return False
    return abs(h - l) / c < 0.002


def _is_tzi(book: _Book, i: int) -> bool:
    """T字板: open≈close≈high=limit-up, with a lower wick."""
    if not _is_limit_up_bar(book, i):
        return False
    o, h, l, c = book.raw_open[i], book.raw_high[i], book.raw_low[i], book.raw_close[i]
    if c <= 0:
        return False
    return abs(o - c) / c < 0.003 and abs(h - c) / c < 0.003 and (o - l) / c > 0.01


def _bullish_ma_stack(ma_fast, ma_mid, ma_slow, i: int) -> bool:
    vals = (ma_fast[i], ma_mid[i], ma_slow[i])
    if not all(np.isfinite(v) for v in vals):
        return False
    return vals[0] > vals[1] > vals[2]


def _range_flat(close: np.ndarray, start: int, end: int, band: float = 0.15) -> bool:
    """True if the window's high/low span is within ``band`` of the mid (横盘)."""
    if end <= start:
        return False
    window = close[start:end]
    window = window[np.isfinite(window)]
    if len(window) < max(5, (end - start) // 2):
        return False
    hi, lo = float(window.max()), float(window.min())
    mid = 0.5 * (hi + lo)
    if mid <= 0:
        return False
    return (hi - lo) / mid <= band


def _emit_trade(book: _Book, entry: int, ma5: np.ndarray) -> int:
    """Try to buy at entry and walk default exit. Returns next scan index."""
    if not book.can_buy(entry):
        return entry
    exit_i, raw_exit, tp = _default_exit(book, entry, ma5)
    if exit_i is None:
        return entry
    book.emit(entry, exit_i, raw_exit, slope=None, take_profit=tp)
    return exit_i


# ----- Tactic 2: 涨停板回调介入 -----
def tactic_limitup_pullback(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Limit-up after ≥20d consolidation, pullback ~½ of gain, then +3% volume yang."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    i = CONSOLIDATION_DAYS + 5
    while i < book.n - 3:
        if not _is_limit_up_bar(book, i):
            i += 1
            continue
        if not _range_flat(book.q_close, i - CONSOLIDATION_DAYS, i, band=0.18):
            i += 1
            continue
        # count consecutive limit-ups ending at i
        boards = 1
        k = i - 1
        while k > 0 and _is_limit_up_bar(book, k) and boards < 3:
            boards += 1
            k -= 1
        launch = book.q_close[i - boards] if i - boards >= 0 else book.q_close[i]
        peak = book.q_close[i]
        if not np.isfinite(launch) or launch <= 0 or not np.isfinite(peak):
            i += 1
            continue
        gain = peak / launch - 1.0
        # required pullback depth ≈ half prior gain (notes: 1板回前低附近, 2板~10%, 3板~20%)
        need = max(0.04, 0.5 * gain) if boards == 1 else (0.10 if boards == 2 else 0.20)
        # search pullback within next 15 sessions
        found = None
        for j in range(i + 1, min(book.n - 1, i + 16)):
            if not np.isfinite(book.q_close[j]):
                continue
            dd = peak / book.q_close[j] - 1.0
            if dd < need * 0.7:
                continue
            # stop-ish: low of j near recent low, then next bar +3% volume yang
            if j + 1 >= book.n:
                break
            nxt = j + 1
            ret = book.q_close[nxt] / book.q_close[j] - 1.0 if book.q_close[j] > 0 else 0
            vol_up = book.volume[nxt] > book.volume[j]
            yang = book.q_close[nxt] > book.q_open[nxt]
            if ret >= 0.03 and vol_up and yang:
                found = nxt
                break
        if found is None:
            i += 1
            continue
        book.batch.signals += 1
        entry = found + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, found + 1)
    return book.batch


# ----- Tactic 3: 天衣无缝 -----
def tactic_seamless(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Bottom yi-zi/T-zi breakout; next day open +2–5%, volume not giant."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    i = CONSOLIDATION_DAYS
    while i < book.n - 2:
        yizi = _is_yizi(book, i)
        tzi = _is_tzi(book, i)
        if not (yizi or tzi):
            i += 1
            continue
        if not _range_flat(book.q_close, max(0, i - CONSOLIDATION_DAYS), i, band=0.20):
            i += 1
            continue
        # volume: yi-zi shrink vs prior; T not giant
        if yizi and i > 0 and book.volume[i - 1] > 0 and book.volume[i] > book.volume[i - 1]:
            i += 1
            continue
        if tzi and i > 0 and book.volume[i - 1] > 0 and book.volume[i] >= GIANT_VOL * book.volume[i - 1]:
            i += 1
            continue
        nxt = i + 1
        if nxt >= book.n:
            break
        if nxt > 0 and book.volume[i] > 0 and book.volume[nxt] >= GIANT_VOL * book.volume[i]:
            i += 1
            continue
        if not np.isfinite(book.q_open[nxt]) or book.q_close[i] <= 0:
            i += 1
            continue
        gap = book.q_open[nxt] / book.q_close[i] - 1.0
        if not (0.02 <= gap <= 0.05):
            i += 1
            continue
        book.batch.signals += 1
        # notes: ideal open day — buy that open if not limit-up
        entry = nxt
        i = _emit_trade(book, entry, ma5)
        i = max(i, nxt + 1)
    return book.batch


# ----- Tactic 4: 反包上影 -----
def tactic_engulf_upper(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Inverted hammer then later bar engulfs its upper shadow; MA bullish."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    ma10 = sma(book.q_close, 10)
    ma20 = sma(book.q_close, 20)
    i = 25
    while i < book.n - 2:
        o, h, l, c = book.q_open[i], book.q_high[i], book.q_low[i], book.q_close[i]
        if not all(np.isfinite(x) for x in (o, h, l, c)) or c <= 0:
            i += 1
            continue
        body = abs(c - o)
        upper = h - max(o, c)
        lower = min(o, c) - l
        # inverted hammer: long upper, short body, little lower
        if body <= 0 or upper < 2.0 * body or lower > 0.5 * body:
            i += 1
            continue
        if not _bullish_ma_stack(ma5, ma10, ma20, i):
            i += 1
            continue
        # engulf upper shadow within 5 sessions
        tip = h
        engulf = None
        for j in range(i + 1, min(book.n - 1, i + 6)):
            if np.isfinite(book.q_close[j]) and book.q_close[j] > tip and book.q_close[j] > book.q_open[j]:
                engulf = j
                break
        if engulf is None:
            i += 1
            continue
        book.batch.signals += 1
        entry = engulf + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, engulf + 1)
    return book.batch


# ----- Tactic 7: 蛟龙出海 -----
def tactic_dragon_sea(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """地量整理后大阳穿过粘合的 MA5/10/30，收盘站上三条均线。"""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    ma10 = sma(book.q_close, 10)
    ma30 = sma(book.q_close, 30)
    i = 35
    while i < book.n - 1:
        if not all(np.isfinite(x) for x in (ma5[i], ma10[i], ma30[i], book.q_close[i])):
            i += 1
            continue
        # MAs sticky: pairwise relative gap < 3%
        vals = sorted([ma5[i], ma10[i], ma30[i]])
        if vals[0] <= 0 or (vals[2] / vals[0] - 1.0) > 0.03:
            i += 1
            continue
        # prior 10d volume near low (地量): today volume >= 1.5× median of prior 10
        if i < 10:
            i += 1
            continue
        prior = book.volume[i - 10 : i]
        if not np.isfinite(prior).all() or np.median(prior) <= 0:
            i += 1
            continue
        if book.volume[i] < 1.5 * float(np.median(prior)):
            i += 1
            continue
        # big yang through all three, close above all
        yang = book.q_close[i] > book.q_open[i]
        through = book.q_low[i] <= vals[2] and book.q_close[i] > vals[2]
        above = book.q_close[i] > ma5[i] and book.q_close[i] > ma10[i] and book.q_close[i] > ma30[i]
        ret = book.q_close[i] / book.q_open[i] - 1.0 if book.q_open[i] > 0 else 0
        if not (yang and through and above and ret >= 0.05):
            i += 1
            continue
        book.batch.signals += 1
        entry = i + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, entry)
    return book.batch


# ----- Tactic 8: 一字形态涨停 -----
def tactic_yizi(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Low-region yi-zi after consolidation; buy next open if not locked."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    i = CONSOLIDATION_DAYS + 20
    while i < book.n - 1:
        if not _is_yizi(book, i):
            i += 1
            continue
        # low region: close near 60d low (within 15%)
        lo = float(np.nanmin(book.q_close[max(0, i - 60) : i]))
        if not np.isfinite(lo) or lo <= 0 or book.q_close[i] > lo * 1.15:
            i += 1
            continue
        if not _range_flat(book.q_close, max(0, i - CONSOLIDATION_DAYS), i, band=0.22):
            i += 1
            continue
        book.batch.signals += 1
        entry = i + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, entry)
    return book.batch


# ----- Tactic 9: 飞雁开路 -----
def tactic_flying_goose(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Limit-up + bullish MA, then ±3% shrink-volume digests, then breakout yang."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    ma10 = sma(book.q_close, 10)
    ma20 = sma(book.q_close, 20)
    i = 25
    while i < book.n - 8:
        if not _is_limit_up_bar(book, i):
            i += 1
            continue
        if not _bullish_ma_stack(ma5, ma10, ma20, i):
            i += 1
            continue
        # consolidation 3–8 days
        found = None
        for end in range(i + 3, min(book.n - 1, i + 9)):
            window = book.q_close[i + 1 : end]
            if len(window) == 0:
                continue
            rets = np.diff(np.concatenate([[book.q_close[i]], window])) / book.q_close[i]
            if np.any(np.abs(rets) > 0.03):
                continue
            vols = book.volume[i + 1 : end]
            if len(vols) and book.volume[i] > 0 and float(np.nanmean(vols)) > 0.8 * book.volume[i]:
                continue  # need shrinkage
            # breakout bar at end
            br = end
            vol_up = book.volume[br] > float(np.nanmean(vols)) if len(vols) else True
            big = (book.q_close[br] / book.q_close[br - 1] - 1.0) >= 0.05 if book.q_close[br - 1] > 0 else False
            if vol_up and (big or _is_limit_up_bar(book, br)):
                found = br
                break
        if found is None:
            i += 1
            continue
        book.batch.signals += 1
        entry = found + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, found + 1)
    return book.batch


def cci_signal_months(book: _Book) -> set:
    """Months where monthly CCI went -100→+100 then made two bottoms above 0."""
    # build monthly OHLC from daily
    stamps = pd.to_datetime(book.dates)
    frame = pd.DataFrame(
        {
            "date": stamps,
            "high": book.q_high,
            "low": book.q_low,
            "close": book.q_close,
        }
    ).dropna()
    if frame.empty:
        return set()
    frame["ym"] = frame["date"].dt.to_period("M")
    monthly = frame.groupby("ym", sort=True).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    if len(monthly) < 20:
        return set()
    m_cci = _cci(monthly["high"].to_numpy(), monthly["low"].to_numpy(), monthly["close"].to_numpy(), 14)
    # find months where CCI crossed from below -100 through +100, then two bottoms > 0
    signal_months: set[pd.Period] = set()
    crossed_at = None
    bottoms = 0
    last_was_low = False
    for mi in range(1, len(m_cci)):
        if crossed_at is None:
            if np.isfinite(m_cci[mi]) and np.isfinite(m_cci[mi - 1]):
                if m_cci[mi - 1] < -100 and m_cci[mi] > 100:
                    crossed_at = mi
                    bottoms = 0
                    last_was_low = False
            continue
        # after cross: track bottoms above 0 (local min of CCI while CCI>0)
        if not np.isfinite(m_cci[mi]):
            continue
        if m_cci[mi] > 0:
            if mi >= 2 and m_cci[mi] < m_cci[mi - 1] and m_cci[mi - 1] <= m_cci[mi - 2]:
                # descending into a trough
                last_was_low = True
            if last_was_low and m_cci[mi] > m_cci[mi - 1]:
                bottoms += 1
                last_was_low = False
                if bottoms >= 2:
                    signal_months.add(monthly.index[mi])
                    # reset to look for next cycle
                    crossed_at = None
                    bottoms = 0
        else:
            # fell back below 0 — cancel
            crossed_at = None
            bottoms = 0
            last_was_low = False

    return signal_months


# ----- Tactic 12: CCI 月线翻倍 -----
def tactic_cci_monthly(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Monthly CCI rises -100→+100, two bottoms above 0, then daily close > MA5."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None or book.n < 80:
        return SignalBatch([])
    signal_months = cci_signal_months(book)
    if not signal_months:
        return SignalBatch([])
    stamps = pd.to_datetime(book.dates)
    ma5 = sma(book.q_close, 5)
    # map period -> last daily index in that month
    periods = pd.Series(stamps).dt.to_period("M").to_numpy()
    i = 30
    while i < book.n - 1:
        if periods[i] not in signal_months:
            i += 1
            continue
        # within this month or next, buy when daily closes above MA5 after being below
        if (
            np.isfinite(ma5[i])
            and np.isfinite(ma5[i - 1])
            and book.q_close[i] > ma5[i]
            and book.q_close[i - 1] <= ma5[i - 1]
        ):
            book.batch.signals += 1
            entry = i + 1
            exit_i, raw_exit = _exit_ma(book, entry, sma(book.q_close, 20), max_hold=40)
            if exit_i is not None:
                book.emit(entry, exit_i, raw_exit, slope=None, take_profit=False)
                i = exit_i
                continue
        i += 1
    return book.batch


# ----- Tactic 16: 蚂蚁上树 -----
def tactic_ants(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """5 small yangs at bottom, then volume yang breaks MA10; buy pullback yin."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma10 = sma(book.q_close, 10)
    ma5 = sma(book.q_close, 5)
    i = 30
    while i < book.n - 4:
        # five consecutive small yangs ending at i-1
        ok = True
        for k in range(i - 5, i):
            if k < 1:
                ok = False
                break
            o, c = book.q_open[k], book.q_close[k]
            if not (np.isfinite(o) and np.isfinite(c) and c > o):
                ok = False
                break
            if o > 0 and (c / o - 1.0) > SMALL_BODY:
                ok = False
                break
        if not ok:
            i += 1
            continue
        # day i: volume yang breaking MA10
        if not (
            np.isfinite(ma10[i])
            and book.q_close[i] > book.q_open[i]
            and book.q_close[i] > ma10[i]
            and book.volume[i] > book.volume[i - 1]
        ):
            i += 1
            continue
        # low region
        lo = float(np.nanmin(book.q_close[max(0, i - 60) : i]))
        if book.q_close[i] > lo * 1.25:
            i += 1
            continue
        # buy first/second yin pullback or yang-eats-yin within 5 days
        entry = None
        for j in range(i + 1, min(book.n - 1, i + 6)):
            yin = book.q_close[j] < book.q_open[j]
            if yin and entry is None:
                entry = j + 1  # next open after first yin
                break
            # yang eats prior yin
            if j >= 1 and book.q_close[j] > book.q_open[j] and book.q_close[j] > book.q_open[j - 1]:
                if book.q_close[j - 1] < book.q_open[j - 1]:
                    entry = j + 1
                    break
        if entry is None:
            i += 1
            continue
        book.batch.signals += 1
        if not book.can_buy(entry):
            i += 1
            continue
        exit_i, raw_exit = _exit_ma(book, entry, ma10, max_hold=20)
        if exit_i is None:
            i = entry
            continue
        book.emit(entry, exit_i, raw_exit, slope=None, take_profit=False)
        i = exit_i
    return book.batch


# ----- Tactic 17: 擒龙捉妖 -----
def tactic_catch_dragon(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Price <10, limit-up, close > MA5, MA5>10>20, volume up."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    ma10 = sma(book.q_close, 10)
    ma20 = sma(book.q_close, 20)
    i = 25
    while i < book.n - 1:
        if not _is_limit_up_bar(book, i):
            i += 1
            continue
        if not (np.isfinite(book.raw_close[i]) and book.raw_close[i] < PRICE_CAP_17):
            i += 1
            continue
        if not _bullish_ma_stack(ma5, ma10, ma20, i):
            i += 1
            continue
        if not (np.isfinite(ma5[i]) and book.q_close[i] > ma5[i]):
            i += 1
            continue
        if i > 0 and book.volume[i] <= book.volume[i - 1]:
            i += 1
            continue
        book.batch.signals += 1
        entry = i + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, entry)
    return book.batch


# ----- Tactic 21: 量价反包 -----
def tactic_volume_engulf(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Prior J<20; today engulfs prior bar and volume; close > MA5."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    k_line, d_line = kdj(book.q_high, book.q_low, book.q_close)
    j_line = 3 * k_line - 2 * d_line
    ma5 = sma(book.q_close, 5)
    i = 20
    while i < book.n - 1:
        if not (np.isfinite(j_line[i - 1]) and j_line[i - 1] < 20):
            i += 1
            continue
        # engulf prior bar
        if not (
            book.q_high[i] >= book.q_high[i - 1]
            and book.q_low[i] <= book.q_low[i - 1]
            and book.q_close[i] > book.q_open[i]
            and book.volume[i] > book.volume[i - 1]
            and np.isfinite(ma5[i])
            and book.q_close[i] > ma5[i]
        ):
            i += 1
            continue
        book.batch.signals += 1
        entry = i + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, entry)
    return book.batch


# ----- Tactic 22: 涨停三连阴低吸 -----
def tactic_limitup_three_yin(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Limit-up (not yi-zi), next day high-open, then 3 small yins above limit-up open, shrink vol."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    i = 5
    while i < book.n - 5:
        if not _is_limit_up_bar(book, i) or _is_yizi(book, i):
            i += 1
            continue
        # day i+1 must high-open
        if i + 1 >= book.n or not (book.q_open[i + 1] > book.q_close[i]):
            i += 1
            continue
        # if day i+1 is small yang, skip (notes)
        if book.q_close[i + 1] > book.q_open[i + 1]:
            i += 1
            continue
        floor = book.q_open[i]
        ok = True
        for k in range(i + 1, i + 4):
            if k >= book.n:
                ok = False
                break
            if book.q_close[k] >= book.q_open[k]:  # must be yin
                ok = False
                break
            body = abs(book.q_close[k] / book.q_open[k] - 1.0) if book.q_open[k] > 0 else 1
            if body > SMALL_BODY:
                ok = False
                break
            if book.q_low[k] < floor:
                ok = False
                break
        if not ok:
            i += 1
            continue
        # staircase shrink volume
        v1, v2, v3 = book.volume[i + 1], book.volume[i + 2], book.volume[i + 3]
        if not (v1 >= v2 >= v3 and v3 > 0):
            i += 1
            continue
        book.batch.signals += 1
        # notes: enter last 5 min of 3rd yin → model as next open after 3rd yin
        entry = i + 4
        i = _emit_trade(book, entry, ma5)
        i = max(i, entry)
    return book.batch


# ----- Tactic 23: 双阳定乾坤 -----
def tactic_double_yang(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """2× volume limit-up, ~5d shrink wash, second higher limit-up."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    i = 5
    while i < book.n - 8:
        if not _is_limit_up_bar(book, i):
            i += 1
            continue
        if i == 0 or book.volume[i - 1] <= 0 or book.volume[i] < 2.0 * book.volume[i - 1]:
            i += 1
            continue
        first_body_high = max(book.q_open[i], book.q_close[i])
        # wash 3–7 days
        found = None
        for gap in range(3, 8):
            j = i + gap
            if j >= book.n:
                break
            wash = book.volume[i + 1 : j]
            if len(wash) < 2:
                continue
            if not (float(np.nanmean(wash)) < book.volume[i] * 0.7):
                continue
            if not _is_limit_up_bar(book, j):
                continue
            if book.q_close[j] <= first_body_high:
                continue
            found = j
            break
        if found is None:
            i += 1
            continue
        book.batch.signals += 1
        entry = found + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, found + 1)
    return book.batch


# ----- Tactic 24: 三连阴涨停 -----
def tactic_three_yin_limitup(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Non yi/T limit-up, next day new high, shrink real yin(s), then volume yang engulf."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma5 = sma(book.q_close, 5)
    i = 5
    while i < book.n - 5:
        if not _is_limit_up_bar(book, i) or _is_yizi(book, i) or _is_tzi(book, i):
            i += 1
            continue
        # next day new high
        if i + 1 >= book.n or not (book.q_high[i + 1] > book.q_high[i]):
            i += 1
            continue
        # one or more shrink yins after that, then engulf yang
        found = None
        for j in range(i + 2, min(book.n - 1, i + 6)):
            # require at least one yin with shrink vol before j
            yins = [k for k in range(i + 2, j) if book.q_close[k] < book.q_open[k]]
            if not yins:
                continue
            if book.volume[yins[-1]] >= book.volume[i]:
                continue
            # j is volume yang engulfing prior yin
            prev = j - 1
            if not (
                book.q_close[j] > book.q_open[j]
                and book.q_close[j] > book.q_open[prev]
                and book.q_close[j] > book.q_close[prev]
                and book.volume[j] > book.volume[prev]
            ):
                continue
            found = j
            break
        if found is None:
            i += 1
            continue
        book.batch.signals += 1
        entry = found + 1
        i = _emit_trade(book, entry, ma5)
        i = max(i, found + 1)
    return book.batch


# ----- Tactic 26: 双阳单阴 -----
def tactic_two_yang_one_yin(
    symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st=False
) -> SignalBatch:
    """Big yang above MA60 near platform; dual-yang (≥2× vol, ≥5%); single yin half vol; buy yin."""
    book = _prepare(symbol, dates, raw_open, raw_high, raw_low, raw_close, volume, amount, factor, is_st)
    if book is None:
        return SignalBatch([])
    ma60 = sma(book.q_close, 60)
    ma5 = sma(book.q_close, 5)
    i = 65
    while i < book.n - 2:
        # look for dual yang at i-1 (second yang), yin at i
        y1, y2, yin = i - 2, i - 1, i
        if y1 < 1:
            i += 1
            continue
        if not all(np.isfinite(x) for x in (ma60[y2], book.q_close[y2], book.q_close[y1], book.q_close[yin])):
            i += 1
            continue
        # second yang stands on MA60
        if not (book.q_close[y2] > book.q_open[y2] and book.q_close[y2] > ma60[y2]):
            i += 1
            continue
        # dual-yang: volume of y2 >= 2× prior day, gain >= 5%
        if book.volume[y1] <= 0 or book.volume[y2] < 2.0 * book.volume[y1]:
            i += 1
            continue
        ret = book.q_close[y2] / book.q_close[y1] - 1.0 if book.q_close[y1] > 0 else 0
        # notes: 当日双阳涨幅 — interpret as the yang day's own gain >= 5%
        day_ret = book.q_close[y2] / book.q_open[y2] - 1.0 if book.q_open[y2] > 0 else 0
        if day_ret < 0.05 and ret < 0.05:
            i += 1
            continue
        # near prior platform high (60d high within 5%)
        plat = float(np.nanmax(book.q_high[max(0, y2 - 60) : y2]))
        if not np.isfinite(plat) or plat <= 0 or book.q_close[y2] < plat * 0.95:
            i += 1
            continue
        # single yin half volume
        if not (book.q_close[yin] < book.q_open[yin] and book.volume[yin] <= 0.5 * book.volume[y2]):
            i += 1
            continue
        # yin does not break MA60 / platform
        if book.q_low[yin] < ma60[yin] and book.q_low[yin] < plat * 0.98:
            i += 1
            continue
        book.batch.signals += 1
        entry = yin + 1  # buy next open after yin signal (T+1 style)
        if not book.can_buy(entry):
            i += 1
            continue
        exit_i, raw_exit = _exit_ma(book, entry, ma60, max_hold=20)
        if exit_i is None:
            i = entry
            continue
        book.emit(entry, exit_i, raw_exit, slope=None, take_profit=False)
        i = exit_i
    return book.batch


PATTERN_TACTICS = (
    ("tactic2", "战术2 涨停回调", tactic_limitup_pullback),
    ("tactic3", "战术3 天衣无缝", tactic_seamless),
    ("tactic4", "战术4 反包上影", tactic_engulf_upper),
    ("tactic7", "战术7 蛟龙出海", tactic_dragon_sea),
    ("tactic8", "战术8 一字涨停", tactic_yizi),
    ("tactic9", "战术9 飞雁开路", tactic_flying_goose),
    ("tactic12", "战术12 CCI月线", tactic_cci_monthly),
    ("tactic16", "战术16 蚂蚁上树", tactic_ants),
    ("tactic17", "战术17 擒龙捉妖", tactic_catch_dragon),
    ("tactic21", "战术21 量价反包", tactic_volume_engulf),
    ("tactic22", "战术22 涨停三连阴", tactic_limitup_three_yin),
    ("tactic23", "战术23 双阳定乾坤", tactic_double_yang),
    ("tactic24", "战术24 三连阴涨停", tactic_three_yin_limitup),
    ("tactic26", "战术26 双阳单阴", tactic_two_yang_one_yin),
)
