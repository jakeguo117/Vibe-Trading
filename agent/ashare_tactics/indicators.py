"""Vectorised indicators used by the four tactics.

Moving averages and MACD use the standard recursive definitions. KDJ uses the
Chinese SMA, SMA(X, N, 1) = (X + (N-1)*prev) / N, seeded at 50. Parabolic SAR
is Wilder with a 4-bar seed (the '4, 2, 2, 6' note: period 4, step 2%,
increment 2%, max 6%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_tactics.constants import SAR_INCREMENT, SAR_MAX, SAR_PERIOD, SAR_STEP


def sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average. Leading values are NaN until ``window`` bars exist."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if window <= 0 or len(x) < window:
        return out
    c = np.cumsum(np.where(np.isfinite(x), x, 0.0))
    c = np.insert(c, 0, 0.0)
    finite = np.cumsum(np.isfinite(x).astype(np.int32))
    finite = np.insert(finite, 0, 0)
    total = c[window:] - c[:-window]
    count = finite[window:] - finite[:-window]
    good = count == window
    out[window - 1 :] = np.where(good, total / window, np.nan)
    return out


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Recursive EMA with ``adjust=False`` (seed = first observation)."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if len(x) == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    started = False
    prev = 0.0
    for i, value in enumerate(x):
        if not np.isfinite(value):
            out[i] = np.nan if not started else prev
            continue
        prev = value if not started else alpha * value + (1.0 - alpha) * prev
        started = True
        out[i] = prev
    return out


def macd_dif(close: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    """DIF = EMA(fast) - EMA(slow). The signal line is not required."""
    return ema(close, fast) - ema(close, slow)


def kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 9, m: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return K and D for KDJ(n, m, m).

    RSV is 50 when the n-day range is zero. K and D are seeded at 50 before
    the first finite RSV, matching the usual domestic terminal initialisation.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    length = len(close)
    k_out = np.full(length, np.nan)
    d_out = np.full(length, np.nan)
    if length == 0:
        return k_out, d_out
    hh = _rolling_extreme(high, n, "max")
    ll = _rolling_extreme(low, n, "min")
    span = hh - ll
    rsv = np.full(length, np.nan)
    valid = np.isfinite(span) & np.isfinite(close)
    nonzero = valid & (span > 0)
    rsv[nonzero] = (close[nonzero] - ll[nonzero]) / span[nonzero] * 100.0
    flat = valid & (span <= 0)
    rsv[flat] = 50.0
    k_prev = 50.0
    d_prev = 50.0
    for i in range(length):
        if not np.isfinite(rsv[i]):
            continue
        k_prev = (rsv[i] + (m - 1) * k_prev) / m
        d_prev = (k_prev + (m - 1) * d_prev) / m
        k_out[i] = k_prev
        d_out[i] = d_prev
    return k_out, d_out


def _rolling_extreme(x: np.ndarray, window: int, how: str) -> np.ndarray:
    series = pd.Series(np.asarray(x, dtype=float))
    rolled = series.rolling(window, min_periods=window).max() if how == "max" else series.rolling(window, min_periods=window).min()
    return rolled.to_numpy(dtype=float)


def parabolic_sar(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = SAR_PERIOD,
    step: float = SAR_STEP,
    increment: float = SAR_INCREMENT,
    max_af: float = SAR_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilder parabolic SAR.

    Returns ``(sar, trend)`` where trend is +1 (long), -1 (short), or 0
    before the seed window. The seed uses the first ``period`` bars: an up
    seed when the window's close is not down, SAR at the window low, extreme
    point at the window high.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int8)
    if n <= period:
        return sar, trend
    up = bool(close[period - 1] >= close[0])
    if up:
        sar_prev = float(np.min(low[:period]))
        ep = float(np.max(high[:period]))
        direction = 1
    else:
        sar_prev = float(np.max(high[:period]))
        ep = float(np.min(low[:period]))
        direction = -1
    af = step
    for i in range(period):
        trend[i] = direction
        sar[i] = sar_prev
    for i in range(period, n):
        sar_i = sar_prev + af * (ep - sar_prev)
        if direction == 1:
            sar_i = min(sar_i, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar_i:
                direction = -1
                sar_i = ep
                ep = float(low[i])
                af = step
            else:
                if high[i] > ep:
                    ep = float(high[i])
                    af = min(af + increment, max_af)
        else:
            sar_i = max(sar_i, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar_i:
                direction = 1
                sar_i = ep
                ep = float(high[i])
                af = step
            else:
                if low[i] < ep:
                    ep = float(low[i])
                    af = min(af + increment, max_af)
        sar[i] = sar_i
        sar_prev = sar_i
        trend[i] = direction
    return sar, trend


def slope_bucket(ma: np.ndarray, index: int, band: float) -> int:
    """+1 up, -1 down, 0 flat, from a 5-bar change in ``ma`` ending at ``index``."""
    if index < 5 or not np.isfinite(ma[index]) or not np.isfinite(ma[index - 5]) or ma[index - 5] == 0:
        return 0
    change = float(ma[index] / ma[index - 5] - 1.0)
    if change > band:
        return 1
    if change < -band:
        return -1
    return 0
