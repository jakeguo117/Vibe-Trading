"""Close-only indicators on one name's own bars. No calendar fill."""

from __future__ import annotations

from bakeoff.fixed20.universe import DOWN_CONFIRM_DAYS, RET_N, SLOPE_LOOKBACK, SMA_N


def sma_last(closes: tuple[float, ...], n: int = SMA_N) -> float | None:
    if len(closes) < n:
        return None
    window = closes[-n:]
    return sum(window) / n


def ret_last(closes: tuple[float, ...], n: int = RET_N) -> float | None:
    """``close[t] / close[t-n] - 1``. Fewer than ``n`` prior bars → unqualified."""
    if len(closes) <= n:
        return None
    base = closes[-1 - n]
    if base == 0:
        return None
    return closes[-1] / base - 1.0


def down_confirmed(
    closes: tuple[float, ...],
    n: int = SMA_N,
    slope: int = SLOPE_LOOKBACK,
    days: int = DOWN_CONFIRM_DAYS,
) -> bool:
    """Three signal days of close < SMA and SMA < SMA ``slope`` bars ago.

    Production confirms a down-leg when close is under the MA and the MA
    itself is under its lag. Freeze keeps that shape and sets the window to
    three signal days instead of two weekly bars.
    """
    if days < 1 or len(closes) < n + slope + days - 1:
        return False
    for back in range(days):
        end = len(closes) - back
        start = end - n
        lag_end = end - slope
        lag_start = lag_end - n
        if start < 0 or lag_start < 0:
            return False
        sma_now = sum(closes[start:end]) / n
        sma_lag = sum(closes[lag_start:lag_end]) / n
        if not (closes[end - 1] < sma_now and sma_now < sma_lag):
            return False
    return True


def hard_stop_hit(last_close: float, avg_cost: float | None, stop: float) -> bool:
    """True when the close is at or through ``stop`` versus this wave's average cost."""
    if avg_cost is None or avg_cost <= 0:
        return False
    return last_close / avg_cost - 1.0 <= stop
