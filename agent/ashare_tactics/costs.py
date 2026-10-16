"""A-share fees, lot size, and price-limit bands.

Signals are computed on forward-adjusted prices. Fills, lot size, and the
limit-up / limit-down gate use unadjusted prices.
"""

from __future__ import annotations

import numpy as np

from ashare_tactics.constants import (
    CHINEXT_REFORM,
    COMMISSION_MIN,
    COMMISSION_RATE,
    LOT,
    SLIPPAGE,
    STAMP_AFTER,
    STAMP_BEFORE,
    STAMP_CUTOVER,
)


def code6(symbol: str) -> str:
    """Return the 6-digit code from ``sh600519`` / ``sz000001`` / ``bj920000``."""
    digits = "".join(ch for ch in symbol if ch.isdigit())
    return digits[-6:].zfill(6)


def round_half_up(price: float) -> float:
    """Round a positive price to 0.01 CNY, half away from zero."""
    if not np.isfinite(price):
        return float("nan")
    return float(np.floor(price * 100.0 + 0.5 + 1e-8) / 100.0)


def board_limit(code: str, day: np.datetime64) -> float:
    """Daily limit half-width before the ST overlay.

    Main board 10%. ChiNext 10% before 2020-08-24 and 20% after. STAR 20%.
    Beijing exchange 30%.
    """
    day = np.datetime64(day, "D")
    if code.startswith(("300", "301")):
        return 0.20 if day >= CHINEXT_REFORM else 0.10
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("8", "4", "92")):
        return 0.30
    return 0.10


def limit_band(code: str, day: np.datetime64, is_st: bool, ipo_free: bool) -> float | None:
    """Return the limit fraction, or None when the session has no band.

    The first five listed sessions of STAR, post-reform ChiNext, and Beijing
    names have no band. A latest-name ST flag forces 5% (see the report: the
    free feed has no dated ST intervals).
    """
    if ipo_free and not is_st:
        return None
    if is_st:
        return 0.05
    return board_limit(code, day)


def ipo_free_bars(code: str, first_day: np.datetime64) -> int:
    """How many leading bars have no price limit.

    The cache starts in 2014, so a first bar on the warmup open is a truncated
    history, not an IPO.
    """
    first_day = np.datetime64(first_day, "D")
    if first_day <= np.datetime64("2014-01-15"):
        return 0
    if code.startswith(("688", "689")):
        return 5
    if code.startswith(("300", "301")) and first_day >= CHINEXT_REFORM:
        return 5
    if code.startswith(("8", "4", "92")):
        return 5
    return 0


def stamp_rate(day: np.datetime64) -> float:
    """Sell-side stamp duty. Halved on 2023-08-28."""
    day = np.datetime64(day, "D")
    if day >= STAMP_CUTOVER:
        return STAMP_AFTER
    return STAMP_BEFORE


def buy_fill(raw_open: float) -> float:
    return float(raw_open) * (1.0 + SLIPPAGE)


def sell_fill(raw_px: float) -> float:
    return float(raw_px) * (1.0 - SLIPPAGE)


def commission(notional: float) -> float:
    return float(max(notional * COMMISSION_RATE, COMMISSION_MIN))


def buy_cash_out(shares: int, raw_open: float) -> float:
    """Cash spent to buy ``shares``, including slippage and commission."""
    notion = shares * buy_fill(raw_open)
    return float(notion + commission(notion))


def sell_cash_in(
    shares: int,
    raw_px: float,
    factor_entry: float,
    factor_exit: float,
    day: np.datetime64,
) -> float:
    """Cash received on a sell, after slippage, commission, stamp, and the
    corporate-action residual that forward adjustment implies.

    Sina's qfq factor divides raw prices (``qfq = raw / factor``, factor ends
    at 1). The economic value of a share is ``raw * factor_entry / factor_now``.
    The gap versus the raw sale is credited here so a split or dividend is not
    booked as a price loss. Fees stay on the raw notional.
    """
    fill = sell_fill(raw_px)
    notion = shares * fill
    fee = commission(notion) + notion * stamp_rate(day)
    ratio = 1.0
    if factor_exit > 0 and np.isfinite(factor_entry) and np.isfinite(factor_exit):
        ratio = float(factor_entry) / float(factor_exit)
    dividend = shares * float(raw_px) * (ratio - 1.0)
    return float(notion - fee + dividend)


def size_shares(equity: float, cash: float, raw_open: float, weight: float = 0.10) -> int:
    """Equal-weight lot size capped by cash. Returns a multiple of 100, or 0."""
    if not np.isfinite(raw_open) or raw_open <= 0 or equity <= 0 or cash <= 0:
        return 0
    budget = min(weight * equity, cash)
    fill = buy_fill(raw_open)
    shares = int(budget / fill / LOT) * LOT
    while shares >= LOT:
        if buy_cash_out(shares, raw_open) <= cash + 1e-6:
            return shares
        shares -= LOT
    return 0


def is_limit_up_open(raw_open: float, prev_close: float, limit: float | None) -> bool:
    if limit is None or not np.isfinite(raw_open) or not np.isfinite(prev_close) or prev_close <= 0:
        return False
    cap = round_half_up(prev_close * (1.0 + limit))
    return raw_open >= cap - 1e-6


def is_limit_down_open(raw_open: float, prev_close: float, limit: float | None) -> bool:
    if limit is None or not np.isfinite(raw_open) or not np.isfinite(prev_close) or prev_close <= 0:
        return False
    floor = round_half_up(prev_close * (1.0 - limit))
    return raw_open <= floor + 1e-6
