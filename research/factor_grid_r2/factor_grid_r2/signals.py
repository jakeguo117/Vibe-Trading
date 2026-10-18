"""Weekly up-leg, Fib stage, and lagged daily features.

Up-leg matches the v1.1.1 engine: enter when weekly close is above a rising
MA for two weeks; leave when weekly close is below the MA for two weeks.
A week is usable only on later sessions (the signal week itself is excluded).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

G_CODE = np.array([0.30, 0.50, 1.00, 0.50], dtype=np.float64)


def ma_leg(close: np.ndarray, n: int, slope_lb: int = 4, confirm: int = 2):
    """Return (in_upleg, ma, down_confirmed) on the same index as ``close``."""
    s = pd.Series(close, dtype="float64")
    ma = s.rolling(n, min_periods=n).mean()
    ma_lag = ma.shift(slope_lb)
    c = s.to_numpy()
    m = ma.to_numpy()
    ml = ma_lag.to_numpy()
    on = np.zeros(len(c), dtype=bool)
    st = False
    cnt = 0
    for i in range(len(c)):
        if not (np.isfinite(m[i]) and np.isfinite(ml[i]) and np.isfinite(c[i])):
            on[i] = st
            continue
        cond = (c[i] > m[i] and m[i] > ml[i]) if not st else (c[i] < m[i])
        cnt = cnt + 1 if cond else 0
        if cnt >= confirm:
            st = not st
            cnt = 0
        on[i] = st
    below = (s < ma) & (ma < ma_lag)
    dn = (below & below.shift(1, fill_value=False)).to_numpy()
    return on, m, dn


def fib_stage_codes(high: np.ndarray, low: np.ndarray, close: np.ndarray, n_pivot: int = 3) -> np.ndarray:
    """Locked weekly Fib stage codes: 0 grind, 1 chop, 2 trend, 3 fish. Unknown stays 0."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    stage = np.zeros(n, dtype=np.int8)
    if n < 20:
        return stage
    prev = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.column_stack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=1)
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    width = 2 * n_pivot + 1
    is_ph = np.zeros(n, dtype=bool)
    is_pl = np.zeros(n, dtype=bool)
    if n >= width:
        from numpy.lib.stride_tricks import sliding_window_view

        mh = sliding_window_view(h, width).max(axis=1)
        ml = sliding_window_view(l, width).min(axis=1)
        idx = np.arange(n_pivot, n - n_pivot)
        is_ph[idx] = h[idx] >= mh
        is_pl[idx] = l[idx] <= ml
    cnt = np.cumsum(is_ph.astype(np.int32) + is_pl.astype(np.int32))
    ph_idx = np.flatnonzero(is_ph)
    pl_idx = np.flatnonzero(is_pl)
    fib = np.full(n, np.nan)
    cand = np.full(n, -1, dtype=np.int8)
    for t in range(n):
        k = t - n_pivot
        if k < 0 or cnt[k] < 2 or not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        j = int(np.searchsorted(ph_idx, k, side="right") - 1)
        if j < 0:
            continue
        ih = int(ph_idx[j])
        m = int(np.searchsorted(pl_idx, ih, side="left") - 1)
        if m < 0:
            continue
        il = int(pl_idx[m])
        hh, ll = h[ih], l[il]
        if hh - ll < 1.5 * atr[t] or hh <= ll:
            continue
        fp = (c[t] - ll) / (hh - ll)
        fib[t] = fp
        cand[t] = 0 if fp < 0.618 else (1 if fp < 1.0 else (2 if fp <= 1.618 else 3))
    cur = -1
    pend = -2
    pc = 0
    bands = (0.618, 1.0, 1.618)
    for t in range(n):
        fp = fib[t]
        cd = int(cand[t])
        if not np.isfinite(fp) or cd < 0:
            stage[t] = cur if cur >= 0 else 0
            continue
        near = any(abs(fp - b) <= 0.02 for b in bands)
        if near and cur >= 0:
            pend, pc = -2, 0
            stage[t] = cur
            continue
        if cd == cur:
            pend, pc = -2, 0
            stage[t] = cur
            continue
        if cur < 0:
            if pend == cd:
                pc += 1
            else:
                pend, pc = cd, 1
            if pc >= 2:
                cur, pend, pc = cd, -2, 0
            stage[t] = cur if cur >= 0 else 0
            continue
        if pend == cd:
            pc += 1
        else:
            pend, pc = cd, 1
        if pc >= 2:
            cur, pend, pc = cd, -2, 0
        stage[t] = cur if cur >= 0 else 0
    return stage


def ewma_ann_vol(close: np.ndarray, span: int = 32) -> np.ndarray:
    """Annualized EWMA vol. Row t uses returns through close t (not lagged)."""
    c = np.asarray(close, dtype=np.float64)
    t_len, n = c.shape
    alpha = 2.0 / (span + 1.0)
    filled = pd.DataFrame(c).ffill().to_numpy()
    out = np.full((t_len, n), np.nan, dtype=np.float64)
    var = np.full(n, np.nan)
    for t in range(1, t_len):
        x = filled[t] / filled[t - 1] - 1.0
        ok = np.isfinite(x) & np.isfinite(filled[t]) & np.isfinite(filled[t - 1]) & (filled[t - 1] > 0)
        prev_ok = np.isfinite(var)
        var = np.where(ok, np.where(prev_ok, alpha * x * x + (1.0 - alpha) * var, x * x), var)
        out[t] = np.sqrt(np.where(np.isfinite(var), var, np.nan)) * math.sqrt(252.0)
    return out


def week_ends_and_frames(dates: pd.DatetimeIndex, high, low, close):
    """Group sessions into W-FRI weeks. Index of each frame is the last real session."""
    df = pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)
    df["week"] = df.index.to_period("W-FRI")
    df["end"] = df.index
    g = df.groupby("week", sort=True)
    weekly = pd.DataFrame(
        {
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "end": g["end"].max(),
        }
    )
    weekly = weekly.dropna(subset=["close"])
    weekly.index = pd.DatetimeIndex(weekly["end"])
    weekly = weekly[~weekly.index.duplicated(keep="last")].sort_index()
    return weekly


def map_weeks_to_days(daily_dates: np.ndarray, week_ends: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Last week that ended strictly before each day."""
    pos = np.searchsorted(week_ends, daily_dates, side="left") - 1
    out = np.full(daily_dates.shape[0], np.nan, dtype=np.float64)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


def three_candle(weekly_close: np.ndarray, weekly_low: np.ndarray, ma: np.ndarray, up: np.ndarray):
    """Rising three-week closes, each above the MA, and the up-leg flag on."""
    c = weekly_close
    n = len(c)
    confirm = np.zeros(n, dtype=bool)
    anchor = np.full(n, np.nan)
    for i in range(2, n):
        if not (np.isfinite(c[i]) and np.isfinite(c[i - 1]) and np.isfinite(c[i - 2])):
            continue
        rising = c[i] > c[i - 1] > c[i - 2]
        above = (
            np.isfinite(ma[i])
            and np.isfinite(ma[i - 1])
            and np.isfinite(ma[i - 2])
            and c[i] > ma[i]
            and c[i - 1] > ma[i - 1]
            and c[i - 2] > ma[i - 2]
        )
        if bool(up[i]) and rising and above:
            confirm[i] = True
            anchor[i] = np.nanmin(weekly_low[i - 2 : i + 1])
    anchor = pd.Series(anchor).ffill().to_numpy()
    return confirm, anchor


def build_features(daily: pd.DataFrame, names: list[str], mas: tuple[int, ...] = (24, 30, 36), moms: tuple[int, ...] = (16, 20, 24)):
    """daily columns: MultiIndex (name, field) with open/high/low/close/volume."""
    dates = pd.DatetimeIndex(daily.index)
    t_len = len(dates)
    n = len(names)
    fields = {}
    for field in ("open", "high", "low", "close", "volume"):
        arr = np.full((t_len, n), np.nan, dtype=np.float64)
        for j, name in enumerate(names):
            if (name, field) not in daily.columns:
                continue
            arr[:, j] = daily[(name, field)].to_numpy(dtype=np.float64)
        fields[field] = arr
    day_np = dates.to_numpy()
    up = {m: np.zeros((t_len, n), dtype=bool) for m in mas}
    dn = {m: np.zeros((t_len, n), dtype=bool) for m in mas}
    confirm = {m: np.zeros((t_len, n), dtype=bool) for m in mas}
    anchor = {m: np.full((t_len, n), np.nan) for m in mas}
    stage = np.zeros((t_len, n), dtype=np.int8)
    route = np.zeros((t_len, n), dtype=np.int8)
    for j, name in enumerate(names):
        w = week_ends_and_frames(dates, fields["high"][:, j], fields["low"][:, j], fields["close"][:, j])
        if w.empty:
            continue
        w_end = w.index.to_numpy()
        stage_w = fib_stage_codes(w["high"].to_numpy(), w["low"].to_numpy(), w["close"].to_numpy())
        route_w = np.zeros(len(stage_w), dtype=np.int8)
        last = 0
        for i, s in enumerate(stage_w):
            if int(s) != 2:
                last = int(s)
            route_w[i] = 2 if last == 3 else last
        stage[:, j] = np.nan_to_num(map_weeks_to_days(day_np, w_end, stage_w.astype(np.float64)), nan=0.0).astype(np.int8)
        route[:, j] = np.nan_to_num(map_weeks_to_days(day_np, w_end, route_w.astype(np.float64)), nan=0.0).astype(np.int8)
        for m in mas:
            on, ma, down = ma_leg(w["close"].to_numpy(), m)
            up[m][:, j] = map_weeks_to_days(day_np, w_end, on.astype(np.float64)) > 0.5
            dn[m][:, j] = map_weeks_to_days(day_np, w_end, down.astype(np.float64)) > 0.5
            conf, anc = three_candle(w["close"].to_numpy(), w["low"].to_numpy(), ma, on)
            confirm[m][:, j] = map_weeks_to_days(day_np, w_end, conf.astype(np.float64)) > 0.5
            anchor[m][:, j] = map_weeks_to_days(day_np, w_end, anc)
    close = fields["close"]
    vol_raw = ewma_ann_vol(close)
    vol = np.vstack([np.full((1, n), np.nan), vol_raw[:-1]])
    ret_vol = pd.DataFrame(close).pct_change()
    rv = ret_vol.rolling(63, min_periods=30).std() * math.sqrt(252.0)
    realized = np.vstack([np.full((1, n), np.nan), rv.to_numpy()[:-1]])
    mom = {}
    for k in moms:
        prev = np.vstack([np.full((1, n), np.nan), close[:-1]])
        base = np.vstack([np.full((k + 1, n), np.nan), close[: t_len - (k + 1)]])
        with np.errstate(divide="ignore", invalid="ignore"):
            mom[k] = prev / base - 1.0
    dv = close * fields["volume"]
    dv60 = pd.DataFrame(dv).rolling(60, min_periods=20).mean().to_numpy()
    dv60 = np.vstack([np.full((1, n), np.nan), dv60[:-1]])
    spy_trend = np.zeros(t_len, dtype=bool)
    return {
        "dates": dates,
        "open": fields["open"],
        "high": fields["high"],
        "low": fields["low"],
        "close": close,
        "volume": fields["volume"],
        "up": up,
        "dn": dn,
        "confirm": confirm,
        "anchor": anchor,
        "stage": stage,
        "route": route,
        "vol": vol,
        "realized": realized,
        "mom": mom,
        "dv60": dv60,
        "spy_trend": spy_trend,
    }
