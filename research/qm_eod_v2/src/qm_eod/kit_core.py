"""QM-RESEARCH-001 core: panels, screens (daily-bar layer).

These screens implement the *end-of-day diagnostic* variants documented in
DATA_READINESS.md.  The intraday OR5 + next-minute execution of the frozen
spec v1.0 cannot be run here: no historical minute data is available on this
machine or through any already-authorised free source (see data audit).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Kit:
    dates: np.ndarray
    syms: np.ndarray
    o: np.ndarray; h: np.ndarray; l: np.ndarray; c: np.ndarray   # split-adjusted
    cr: np.ndarray             # raw close
    v: np.ndarray              # raw volume
    dv: np.ndarray             # raw dollar volume
    div: np.ndarray            # raw dividend per share
    adjf: np.ndarray           # adj_close / close
    sma10: np.ndarray; sma16: np.ndarray; sma20: np.ndarray; sma24: np.ndarray; sma50: np.ndarray
    atr14: np.ndarray
    dv20: np.ndarray
    adr20: np.ndarray
    hist: np.ndarray
    base_ok: np.ndarray
    r10: np.ndarray; r21: np.ndarray; r63: np.ndarray; r126: np.ndarray
    p21: np.ndarray; p63: np.ndarray; p126: np.ndarray

    def didx(self, date_str):
        return int(np.searchsorted(self.dates, np.datetime64(date_str)))

    @property
    def n(self):
        return self.c.shape[1]


def _p(df, w, how):
    return getattr(df.rolling(w, min_periods=w), how)().to_numpy(np.float32)


def wilder_atr(h, l, c, n=14):
    """Wilder ATR; seed = mean of the first n true ranges *of each symbol*.

    Symbols that list after the panel start must be seeded at their own first
    bar, otherwise the recursion stays NaN for the whole column.
    """
    prev_c = np.vstack([np.full((1, c.shape[1]), np.nan, np.float32), c[:-1]])
    tr = h - l
    tr = np.fmax(tr, np.abs(h - prev_c))
    tr = np.fmax(tr, np.abs(l - prev_c))
    T, N = tr.shape
    valid = np.isfinite(tr)
    atr = np.full((T, N), np.nan, np.float32)
    if T <= n or N == 0:
        return atr
    fv = np.argmax(valid, axis=0)
    has = valid.any(axis=0)
    seed_idx = np.where(has, fv + n - 1, T)
    rows = np.minimum(fv[None, :] + np.arange(n)[:, None], T - 1)
    vals = tr[rows, np.arange(N)[None, :]]
    with np.errstate(invalid="ignore"):
        seed = np.nanmean(vals, axis=0)
    seed = np.where(has & np.isfinite(seed), seed, np.nan).astype(np.float32)
    a = 1.0 / n
    prev = np.full(N, np.nan, np.float32)
    for t in range(int(seed_idx.min()), T):
        new = seed_idx == t
        if new.any():
            prev[new] = seed[new]
        upd = np.isfinite(prev) & (~new) & np.isfinite(tr[t])
        if upd.any():
            prev[upd] = prev[upd] * (1 - a) + tr[t][upd] * a
        atr[t] = prev
    return atr


def load_kit(path, cut_start="2009-06-01"):
    z = np.load(path, allow_pickle=False)
    dates = z["dates"].astype("datetime64[D]")
    syms = z["syms"].astype(str)
    o = z["open"].astype(np.float32); h = z["high"].astype(np.float32)
    l = z["low"].astype(np.float32); c = z["close"].astype(np.float32)
    ac = z["adjclose"].astype(np.float32)
    v = z["volume"].astype(np.float32); dv = z["dollarvol"].astype(np.float64)
    div = z["dividend"].astype(np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        adjf = np.where((c > 0) & np.isfinite(ac), ac / c, 1.0).astype(np.float32)
    oa = (o * adjf).astype(np.float32); ha = (h * adjf).astype(np.float32)
    la = (l * adjf).astype(np.float32); ca = (c * adjf).astype(np.float32)

    idx = pd.DatetimeIndex(dates)
    dv20 = _p(pd.DataFrame(dv, index=idx, columns=syms), 20, "mean")
    with np.errstate(invalid="ignore", divide="ignore"):
        rng = np.where(la > 0, ha / la - 1.0, np.nan).astype(np.float32)
    adr20 = (_p(pd.DataFrame(rng, index=idx, columns=syms), 20, "mean") * 100.0).astype(np.float32)
    hist = np.cumsum(np.isfinite(ca), axis=0).astype(np.int32)

    base_ok = ((c >= 5.0) & np.isfinite(dv20) & (dv20 >= 2.0e7)
               & np.isfinite(adr20) & (adr20 >= 3.0) & (adr20 <= 15.0) & (hist >= 126))
    i0 = int(np.searchsorted(dates, np.datetime64(cut_start)))
    keep = base_ok[i0:].any(axis=0)
    for b in ('QQQ', 'SPY'):
        keep |= (syms == b)
    print(f"[kit] base-eligible symbols kept: {int(keep.sum())} / {len(syms)}")

    def sub(a):
        return np.ascontiguousarray(a[:, keep])

    k = Kit(dates=dates, syms=syms[keep], o=sub(oa), h=sub(ha), l=sub(la), c=sub(ca),
            cr=sub(c), v=sub(v), dv=sub(dv), div=sub(div), adjf=sub(adjf),
            sma10=None, sma16=None, sma20=None, sma24=None, sma50=None, atr14=None,
            dv20=sub(dv20), adr20=sub(adr20), hist=sub(hist), base_ok=sub(base_ok),
            r10=None, r21=None, r63=None, r126=None, p21=None, p63=None, p126=None)

    cdf = pd.DataFrame(k.c, index=idx, columns=k.syms)
    k.sma10 = _p(cdf, 10, "mean"); k.sma16 = _p(cdf, 16, "mean"); k.sma20 = _p(cdf, 20, "mean")
    k.sma24 = _p(cdf, 24, "mean"); k.sma50 = _p(cdf, 50, "mean")
    k.atr14 = wilder_atr(k.h, k.l, k.c)
    k.r10 = (cdf / cdf.shift(10) - 1).to_numpy(np.float32)
    k.r21 = (cdf / cdf.shift(21) - 1).to_numpy(np.float32)
    k.r63 = (cdf / cdf.shift(63) - 1).to_numpy(np.float32)
    k.r126 = (cdf / cdf.shift(126) - 1).to_numpy(np.float32)
    for n in (21, 63, 126):
        r = pd.DataFrame(getattr(k, f"r{n}"), index=idx, columns=k.syms).where(k.base_ok)
        setattr(k, f"p{n}", (r.rank(axis=1, pct=True) * 100.0).to_numpy(np.float32))
    return k


# ------------------------------------------------------------- BO screens

def bo_conditions(k: Kit, leader=True, prior_up=True, consolidation=True,
                  prior_up_min=0.30, ratio_max=0.60, tight_max=0.18, pct_min=98):
    idx = pd.DatetimeIndex(k.dates); cols = k.syms
    hdf = pd.DataFrame(k.h, index=idx, columns=cols)
    ldf = pd.DataFrame(k.l, index=idx, columns=cols)
    cdf = pd.DataFrame(k.c, index=idx, columns=cols)
    s20 = pd.DataFrame(k.sma20, index=idx, columns=cols)
    s50 = pd.DataFrame(k.sma50, index=idx, columns=cols)

    m = pd.DataFrame(k.base_ok, index=idx, columns=cols)
    if leader:
        p = (pd.DataFrame(k.p21, index=idx, columns=cols) >= pct_min) | \
            (pd.DataFrame(k.p63, index=idx, columns=cols) >= pct_min) | \
            (pd.DataFrame(k.p126, index=idx, columns=cols) >= pct_min)
        m &= p
    if prior_up:
        m &= (cdf.shift(15) / cdf.shift(63) - 1) >= prior_up_min
    if consolidation:
        maxh = hdf.rolling(15, min_periods=15).max()
        minl = ldf.rolling(15, min_periods=15).min()
        m &= (maxh / minl - 1) <= tight_max
        r5 = hdf.rolling(5, min_periods=5).max() - ldf.rolling(5, min_periods=5).min()
        r10p = hdf.shift(5).rolling(10, min_periods=10).max() - ldf.shift(5).rolling(10, min_periods=10).min()
        with np.errstate(invalid="ignore", divide="ignore"):
            m &= (r5 / r10p) <= ratio_max
        m &= ldf.rolling(5, min_periods=5).min() > ldf.shift(5).rolling(10, min_periods=10).min()
        m &= (cdf >= s20) & (s20 >= s50) & (s20 > s20.shift(5))
    return m


def bo_signals(k: Kit, mask: pd.DataFrame, tight_ratio=None):
    """Breakout confirmed at the close of day s; execution next open (proxy)."""
    idx = pd.DatetimeIndex(k.dates); cols = k.syms
    hdf = pd.DataFrame(k.h, index=idx, columns=cols)
    base_high = hdf.rolling(15, min_periods=15).max()
    trig = mask.shift(1, fill_value=False).to_numpy() & (k.c > base_high.shift(1).to_numpy())
    return trig


# ------------------------------------------------------------- PS screens

def ps_conditions(k: Kit, r10_min=0.50, ext_min=0.25):
    idx = pd.DatetimeIndex(k.dates); cols = k.syms
    cdf = pd.DataFrame(k.c, index=idx, columns=cols)
    s10 = pd.DataFrame(k.sma10, index=idx, columns=cols)
    r10 = cdf / cdf.shift(10) - 1
    up = (cdf > cdf.shift(1)) & (cdf.shift(1) > cdf.shift(2)) & (cdf.shift(2) > cdf.shift(3))
    m = (r10 >= r10_min) & up & ((cdf / s10 - 1) >= ext_min)
    return m & pd.DataFrame(k.base_ok, index=idx, columns=cols)


def ps_signals(k: Kit, mask: pd.DataFrame, wait_for_weakness=True):
    """Daily proxy: close breaks the day's opening range and closes weak."""
    idx = pd.DatetimeIndex(k.dates); cols = k.syms
    cdf = pd.DataFrame(k.c, index=idx, columns=cols)
    weak = (k.c < k.o) & (k.c < (k.h + k.l + k.c) / 3.0) & (k.c < np.vstack([cdf.shift(1).to_numpy()[0:1], k.c[:-1]]))
    if wait_for_weakness:
        # candidate screen is evaluated at p = s-1; weakness confirms on day s
        trig = mask.shift(1, fill_value=False).to_numpy() & weak
    else:
        trig = mask.to_numpy()
    return trig
