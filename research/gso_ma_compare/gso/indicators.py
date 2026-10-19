"""Causal weekly/daily indicators: SMA up-leg, Fibonacci stage, target weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gso.config import (
    ATR_SWING_MULT,
    ATR_WEEKS,
    CONST_CORR,
    DEFAULT_STAGE,
    FIB_BANDS,
    FIB_CONFIRM_WEEKS,
    MAX_HOLDINGS,
    PIVOT_WEEKS,
    SLOT,
    STAGE_WEIGHT,
    TARGET_VOL,
)


def normalize_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Session-date index, sorted, one row per date. Timezone-aware stamps use New York dates."""
    df = frame.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York")
        idx = pd.DatetimeIndex(np.array([ts.date() for ts in idx], dtype="datetime64[ns]"))
    else:
        idx = idx.normalize()
    df.index = idx
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for col in ("Open", "High", "Low", "Close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]
    return df[["Open", "High", "Low", "Close", "Volume"]]


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Friday-labeled weekly bars. The in-progress week (Friday still ahead of the last session) is dropped.

    ``session`` is the last trading day inside that Monday–Friday bin. Signals use that session's close.
    Execution scheduling still keys off the Friday label so the next week is unambiguous.
    """
    d = normalize_ohlc(daily)
    weekly = d.resample("W-FRI").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    weekly = weekly.dropna(subset=["Open", "High", "Low", "Close"])
    last = d.index.max()
    weekly = weekly.loc[weekly.index <= last]
    sessions = []
    for friday in weekly.index:
        monday = friday - pd.offsets.Day(4)
        days = d.index[(d.index >= monday) & (d.index <= friday)]
        sessions.append(days.max() if len(days) else pd.NaT)
    weekly["session"] = pd.to_datetime(sessions)
    weekly = weekly.dropna(subset=["session"])
    return weekly


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = ATR_WEEKS) -> pd.Series:
    """Wilder ATR. The seed is the simple average of the first ``n`` true ranges after the first bar."""
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    values = tr.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) <= n:
        return pd.Series(out, index=close.index)
    seed = values[1 : n + 1]
    if np.isnan(seed).any():
        return pd.Series(out, index=close.index)
    out[n] = float(seed.mean())
    for i in range(n + 1, len(values)):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return pd.Series(out, index=close.index)


def upleg_from_flags(
    above: np.ndarray,
    below: np.ndarray,
    slope_down: np.ndarray,
    confirm: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Live streak state machine.

    Enter after ``confirm`` consecutive bars with price above the MA and the MA above its lag.
    While in an up-leg, ``confirm`` consecutive closes below the MA halve the position.
    Clear when those closes-below bars and a falling MA (MA below its lag) are both true
    for ``confirm`` consecutive bars. Halving is not latched: a close back at or above the MA
    restores the full up-leg budget. Clearing requires a fresh entry streak.
    """
    n = len(above)
    in_upleg = np.zeros(n, dtype=bool)
    halve = np.zeros(n, dtype=bool)
    entry_streak = 0
    below_streak = 0
    slope_streak = 0
    flag = False
    for i in range(n):
        entry_streak = entry_streak + 1 if bool(above[i]) else 0
        below_streak = below_streak + 1 if bool(below[i]) else 0
        slope_streak = slope_streak + 1 if bool(slope_down[i]) else 0
        if not flag:
            if entry_streak >= confirm:
                flag = True
        elif below_streak >= confirm and slope_streak >= confirm:
            flag = False
        in_upleg[i] = flag
        halve[i] = bool(flag and below_streak >= confirm)
    return in_upleg, halve


def sma_upleg(close: pd.Series, length: int, confirm: int, slope_lag: int) -> pd.DataFrame:
    """Simple MA up-leg. Comparisons are strict. Warmup bars stay flat."""
    ma = close.rolling(length, min_periods=length).mean()
    ma_lag = ma.shift(slope_lag)
    above = ((close > ma) & (ma > ma_lag)).fillna(False).to_numpy()
    below = (close < ma).fillna(False).to_numpy()
    slope_down = (ma < ma_lag).fillna(False).to_numpy()
    in_upleg, halve = upleg_from_flags(above, below, slope_down, confirm)
    return pd.DataFrame(
        {"ma": ma.to_numpy(), "in_upleg": in_upleg, "halve": halve},
        index=close.index,
    )


def stage_from_level(close: float, origin: float, amplitude: float, bands: tuple[float, ...] = FIB_BANDS) -> str:
    """Map a close onto 磨底 / 震荡 / 主升 / 鱼尾 using upside projections from a swing low.

    Levels are origin + band * amplitude. A print exactly on a band belongs to the higher stage.
    """
    b618, b100, b1618 = (origin + band * amplitude for band in bands)
    if close < b618:
        return "磨底"
    if close < b100:
        return "震荡"
    if close < b1618:
        return "主升"
    return "鱼尾"


def _pivot_offset(values: np.ndarray, i: int, window: int, kind: str) -> bool:
    if i < window or i + window >= len(values):
        return False
    seg = values[i - window : i + window + 1]
    if np.isnan(seg).any():
        return False
    if kind == "H":
        return int(np.argmax(seg)) == window
    return int(np.argmin(seg)) == window


def projection_from_pivots(pivots: list[dict]) -> tuple[float, float] | None:
    """Upside fib anchor: the latest completed down-swing (high then low).

    If the latest pivot is still that low, project from it.
    If a later high has confirmed, keep projecting from the low that started the up-swing,
    using the amplitude of the down-swing that made that low.
    """
    if len(pivots) < 2:
        return None
    last = pivots[-1]
    prev = pivots[-2]
    if last["kind"] == "L" and prev["kind"] == "H":
        amp = prev["price"] - last["price"]
        if amp > 0:
            return last["price"], amp
        return None
    if last["kind"] == "H" and len(pivots) >= 3:
        low = pivots[-2]
        high = pivots[-3]
        if low["kind"] == "L" and high["kind"] == "H":
            amp = high["price"] - low["price"]
            if amp > 0:
                return low["price"], amp
    return None


def fib_stage_frame(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    window: int = PIVOT_WEEKS,
    atr_mult: float = ATR_SWING_MULT,
    confirm: int = FIB_CONFIRM_WEEKS,
) -> pd.DataFrame:
    """Causal zigzag plus a 2-bar confirmed fib stage.

    A pivot at i is eligible only once bar i+window exists. Same-direction pivots replace
    the previous extreme when they are more extreme. An opposite pivot must travel at least
    ``atr_mult`` times ATR(at the pivot bar). Until a stage prints the same value for
    ``confirm`` bars, the effective stage stays at the previous one, starting from 震荡.
    """
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    a = atr.to_numpy(dtype=float)
    n = len(c)
    raw = np.array([None] * n, dtype=object)
    origin = np.full(n, np.nan)
    amplitude = np.full(n, np.nan)
    pivots: list[dict] = []

    def consider(i: int, kind: str, price: float) -> None:
        if not np.isfinite(a[i]) or a[i] <= 0 or not np.isfinite(price):
            return
        if not pivots:
            pivots.append({"idx": i, "kind": kind, "price": price})
            return
        last = pivots[-1]
        if kind == last["kind"]:
            if kind == "H" and price > last["price"]:
                pivots[-1] = {"idx": i, "kind": kind, "price": price}
            elif kind == "L" and price < last["price"]:
                pivots[-1] = {"idx": i, "kind": kind, "price": price}
            return
        if abs(price - last["price"]) >= atr_mult * a[i]:
            pivots.append({"idx": i, "kind": kind, "price": price})

    for t in range(n):
        i = t - window
        if i >= window:
            is_high = _pivot_offset(h, i, window, "H")
            is_low = _pivot_offset(lo, i, window, "L")
            if is_high and is_low:
                up = h[i] - c[i]
                down = c[i] - lo[i]
                if up >= down:
                    is_low = False
                else:
                    is_high = False
            if is_high:
                consider(i, "H", float(h[i]))
            elif is_low:
                consider(i, "L", float(lo[i]))
        proj = projection_from_pivots(pivots)
        if proj is None or not np.isfinite(c[t]):
            raw[t] = None
        else:
            origin[t] = proj[0]
            amplitude[t] = proj[1]
            raw[t] = stage_from_level(float(c[t]), proj[0], proj[1])

    effective: list[str] = []
    current = DEFAULT_STAGE
    streak = 0
    prev = None
    for value in raw:
        if value is None:
            prev = None
            streak = 0
        elif value == prev:
            streak += 1
        else:
            prev = value
            streak = 1
        if value is not None and streak >= confirm:
            current = str(value)
        effective.append(current)

    return pd.DataFrame(
        {
            "fib_raw": raw,
            "fib_stage": effective,
            "fib_origin": origin,
            "fib_amp": amplitude,
        },
        index=close.index,
    )


def ewma_ann_vol(close: pd.Series, span: int, periods_per_year: int) -> pd.Series:
    """Annualized EWMA volatility of simple close-to-close returns. ``adjust=False``."""
    ret = close.pct_change()
    var = ret.pow(2).ewm(span=span, min_periods=span, adjust=False).mean()
    return np.sqrt(var * periods_per_year)


def portfolio_vol(weights: np.ndarray, vols: np.ndarray, corr: float = CONST_CORR) -> float:
    """Constant-correlation portfolio vol. ``vols`` are annualized."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(vols, dtype=float)
    cov = np.outer(v, v) * corr
    np.fill_diagonal(cov, v ** 2)
    variance = float(w @ cov @ w)
    return float(np.sqrt(max(variance, 0.0)))


def target_weights(
    names: list[str],
    in_upleg: dict[str, bool],
    halve: dict[str, bool],
    stage: dict[str, str],
    vol: dict[str, float],
) -> dict[str, float]:
    """Per-name budget = 70%/3 × stage weight × 0.5 if halved. Then scale down to 16% vol, never up.

    At most ``MAX_HOLDINGS`` positive weights are kept (highest budget, then name).
    Vol scaling waits until every held name has a finite positive vol; otherwise the scale is 1.
    """
    raw = {}
    for name in names:
        if not in_upleg.get(name, False):
            raw[name] = 0.0
            continue
        weight = STAGE_WEIGHT.get(stage.get(name, DEFAULT_STAGE), STAGE_WEIGHT[DEFAULT_STAGE])
        if halve.get(name, False):
            weight *= 0.5
        raw[name] = SLOT * weight
    ranked = sorted((n for n in names if raw[n] > 0), key=lambda n: (-raw[n], n))
    keep = set(ranked[:MAX_HOLDINGS])
    raw = {n: (raw[n] if n in keep else 0.0) for n in names}
    held = [n for n in names if raw[n] > 0]
    scale = 1.0
    if held and all(np.isfinite(vol.get(n, np.nan)) and vol[n] > 0 for n in held):
        w = np.array([raw[n] for n in names], dtype=float)
        v = np.array([float(vol[n]) if n in held else 0.0 for n in names], dtype=float)
        pvol = portfolio_vol(w, v)
        if pvol > 1e-12:
            scale = min(1.0, TARGET_VOL / pvol)
    out = {n: raw[n] * scale for n in names}
    if sum(out.values()) > MAX_HOLDINGS * SLOT + 1e-9:
        raise RuntimeError("target weights broke the 70% investment cap")
    return out


def participation(in_upleg: bool, halve: bool) -> float:
    """Move-capture participation: 1 in a full up-leg, 0.5 while halved, 0 when flat."""
    if not in_upleg:
        return 0.0
    return 0.5 if halve else 1.0


def following_week_execution(friday: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Second trading day of the next Monday–Friday week.

    A one-session week uses that session. A week with no sessions looks ahead up to four weeks.
    """
    friday = pd.Timestamp(friday).normalize()
    if friday.weekday() != 4:
        raise ValueError(f"decision label must be a Friday, got {friday.date()} weekday {friday.weekday()}")
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize().sort_values().unique()
    monday = friday + pd.offsets.Day(3)
    for step in range(4):
        start = monday + pd.offsets.Day(7 * step)
        end = start + pd.offsets.Day(4)
        week = days[(days >= start) & (days <= end)]
        if len(week) >= 2:
            return pd.Timestamp(week[1])
        if len(week) == 1:
            return pd.Timestamp(week[0])
    return None


def next_trading_day(day: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    day = pd.Timestamp(day).normalize()
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize().sort_values().unique()
    later = days[days > day]
    if len(later) == 0:
        return None
    return pd.Timestamp(later[0])
