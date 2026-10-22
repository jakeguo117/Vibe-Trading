"""BO + EP signal generation for QM-EOD-v2 (spec 2026-10-22)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def qqq_rising_filter(kit) -> np.ndarray:
    """QQQ 10SMA > 20SMA and 20SMA > 20SMA[t-5]. Length T bool."""
    j = int(np.where(kit.syms == "QQQ")[0][0])
    s10 = kit.sma10[:, j]
    s20 = kit.sma20[:, j]
    s20_lag = np.roll(s20, 5)
    s20_lag[:5] = np.nan
    ok = (s10 > s20) & (s20 > s20_lag) & np.isfinite(s10) & np.isfinite(s20) & np.isfinite(s20_lag)
    return ok


def bo_signals_v2(kit, adr_min=4.0, pct_min=98.0, prior_up_min=0.30,
                  consol_min=10, consol_max=40, range_max=0.25):
    """Breakout signals matching QM-EOD-v2 §4 (vectorized approximation).

    Returns DataFrame with columns for engine.
    """
    T, N = kit.c.shape
    idx = pd.DatetimeIndex(kit.dates)
    cols = kit.syms
    hdf = pd.DataFrame(kit.h, index=idx, columns=cols)
    ldf = pd.DataFrame(kit.l, index=idx, columns=cols)
    cdf = pd.DataFrame(kit.c, index=idx, columns=cols)
    s10 = pd.DataFrame(kit.sma10, index=idx, columns=cols)
    s20 = pd.DataFrame(kit.sma20, index=idx, columns=cols)

    # Universe + ADR filter (ADR20 >= 4% for BO; base_ok already has [3,15])
    base = pd.DataFrame(kit.base_ok, index=idx, columns=cols)
    adr = pd.DataFrame(kit.adr20, index=idx, columns=cols)
    m = base & (adr >= adr_min)

    # Leader: 21/63/126 rank top 2%
    p = ((pd.DataFrame(kit.p21, index=idx, columns=cols) >= pct_min) |
         (pd.DataFrame(kit.p63, index=idx, columns=cols) >= pct_min) |
         (pd.DataFrame(kit.p126, index=idx, columns=cols) >= pct_min))
    m &= p

    # Trend: close > 10SMA > 20SMA, 20SMA rising
    m &= (cdf > s10) & (s10 > s20) & (s20 > s20.shift(5))

    # Prior impulse in past 63d: low->high >= 30%
    # Use rolling max/min over 63 ending  before consolidation window.
    # Approximate: (max high of [t-63,t-consol] / min low of same) - 1 >= 30%,
    # with consolidation length searched in [10,40].
    # Practical vectorized approx used in research_001 style + range check:
    # For each day, consolidation = last 15 bars (within 10-40), range <= 25%,
    # and impulse before that in the 63 window.
    # We scan a few consolidation lengths for better fidelity.

    # Daily range for contraction test
    day_rng = (hdf - ldf)

    best_trig = np.zeros((T, N), dtype=bool)
    best_bh = np.full((T, N), np.nan, np.float32)
    # Try consol lengths 10,15,20,25,30,35,40
    for clen in (10, 15, 20, 25, 30, 35, 40):
        if clen < consol_min or clen > consol_max:
            continue
        maxh = hdf.rolling(clen, min_periods=clen).max()
        minl = ldf.rolling(clen, min_periods=clen).min()
        rng = maxh / minl - 1.0
        consol_ok = rng <= range_max

        # Impulse in window before consolidation: look back from t-clen over up to 63-clen days
        look = max(5, 63 - clen)
        # high of impulse window / low of impulse window
        # impulse window ends at t-clen (shift clen), length `look`
        imp_h = hdf.shift(clen).rolling(look, min_periods=max(5, look // 2)).max()
        imp_l = ldf.shift(clen).rolling(look, min_periods=max(5, look // 2)).min()
        impulse_ok = (imp_h / imp_l - 1.0) >= prior_up_min

        # Contraction: avg day-range last 5 < avg day-range over consolidation
        avg5 = day_rng.rolling(5, min_periods=5).mean()
        avg_c = day_rng.rolling(clen, min_periods=clen).mean()
        tight = avg5 < avg_c

        setup = m & consol_ok & impulse_ok & tight
        # Breakout: close > base_high of consolidation (prior days' high, exclude today)
        bh = maxh.shift(1)  # base high as of prior close
        trig = setup.shift(1, fill_value=False).to_numpy() & (kit.c > bh.to_numpy())
        # Keep first/best: prefer shorter consol when both fire; OR them and keep bh
        new = trig & (~best_trig)
        best_trig |= trig
        bh_np = bh.to_numpy(np.float32)
        best_bh = np.where(new, bh_np, best_bh)
        # if already triggered, keep existing bh

    # Build signal table: trigger day = breakout close day; entry next open
    dd, jj = np.nonzero(best_trig)
    if len(dd) == 0:
        return pd.DataFrame(columns=[
            "d", "j", "style", "entry_mode", "stop", "atr", "adr_frac",
            "max_stop_mult", "base_high", "k1", "k2", "k3", "earnings_tagged"
        ])

    # ranking: breakout-day return (close/prev_close - 1), then r63
    prev_c = np.vstack([np.full((1, N), np.nan, np.float32), kit.c[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        day_ret = kit.c[dd, jj] / prev_c[dd, jj] - 1.0
    k1 = day_ret.astype(float)
    k2 = kit.r63[dd - 1, jj].astype(float)  # prior momentum
    adr_frac = (kit.adr20[dd, jj] / 100.0).astype(float)
    atr = (adr_frac * kit.c[dd, jj]).astype(float)  # dollar ADR proxy

    out = pd.DataFrame({
        "d": dd.astype(int),
        "j": jj.astype(int),
        "style": "bo",
        "entry_mode": "next_open",
        "stop": np.nan,  # set from entry-day LOD
        "atr": atr,
        "adr_frac": adr_frac,
        "max_stop_mult": 1.0,
        "base_high": best_bh[dd, jj].astype(float),
        "k1": k1,
        "k2": k2,
        "k3": jj.astype(int),
        "earnings_tagged": False,
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["k1", "base_high", "adr_frac"])
    return out


def ep_signals_v2(kit, edgar_csv=None, gap_min=0.10, vol_mult=3.0, r126_max=0.30):
    """Earnings-gap (price-action) signals; optional 8-K tag. Spec §5.

    Main result does NOT require 8-K; tags earnings_tagged when aligned.
    """
    T, N = kit.c.shape
    prev_c = np.vstack([np.full((1, N), np.nan, np.float32), kit.c[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = kit.o / prev_c - 1.0
    # 20d avg volume as of prior day
    v20 = pd.DataFrame(kit.v).rolling(20, min_periods=20).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        volr = kit.v / v20

    # no other >=10% up-gap in past 63 sessions (exclude today)
    gap_up = (gap >= gap_min) & np.isfinite(gap)
    # rolling count of prior gaps in 63
    prior_gap_cnt = pd.DataFrame(gap_up.astype(np.float32)).rolling(63, min_periods=1).sum().shift(1).to_numpy()

    close_up = (kit.c > kit.o) & np.isfinite(kit.c) & np.isfinite(kit.o)
    r126 = kit.r126  # as of same day uses close; use prior close return
    r126_p = np.vstack([np.full((1, N), np.nan, np.float32), kit.r126[:-1]])

    trig = (
        gap_up
        & (volr >= vol_mult)
        & (r126_p < r126_max)
        & (prior_gap_cnt < 1)  # no other gap in past 63
        & close_up
        & kit.base_ok
        & np.isfinite(volr)
    )

    # Optional EDGAR tag
    earnings = np.zeros((T, N), dtype=bool)
    if edgar_csv:
        earnings = _edgar_mask(kit, edgar_csv)

    dd, jj = np.nonzero(trig)
    if len(dd) == 0:
        return pd.DataFrame(columns=[
            "d", "j", "style", "entry_mode", "stop", "atr", "adr_frac",
            "max_stop_mult", "base_high", "k1", "k2", "k3", "earnings_tagged"
        ])

    stop = kit.l[dd, jj].astype(float)
    adr_frac = (kit.adr20[dd, jj] / 100.0).astype(float)
    atr = (adr_frac * kit.o[dd, jj]).astype(float)
    k1 = gap[dd, jj].astype(float)          # rank by gap size
    k2 = volr[dd, jj].astype(float)

    out = pd.DataFrame({
        "d": dd.astype(int),
        "j": jj.astype(int),
        "style": "ep",
        "entry_mode": "same_close",
        "stop": stop,
        "atr": atr,
        "adr_frac": adr_frac,
        "max_stop_mult": 1.5,
        "base_high": np.nan,
        "k1": k1,
        "k2": k2,
        "k3": jj.astype(int),
        "earnings_tagged": earnings[dd, jj],
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["k1", "stop", "adr_frac"])
    return out


def _edgar_mask(kit, edgar_csv):
    from zoneinfo import ZoneInfo
    from datetime import datetime, time as dtime
    NY = ZoneInfo("America/New_York")
    T, N = kit.c.shape
    mask = np.zeros((T, N), dtype=bool)
    ev = pd.read_csv(edgar_csv)
    ev = ev[ev["acceptance"].notna() & (~ev["accn"].isin(["FETCH_FAIL", "NO_CIK"]))]
    ev["acc_dt"] = pd.to_datetime(ev["acceptance"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["acc_dt"])
    j_of = {s: i for i, s in enumerate(kit.syms)}
    dates = pd.DatetimeIndex(kit.dates)
    for s, g in ev.groupby("sym"):
        j = j_of.get(s)
        if j is None:
            continue
        for acc in g["acc_dt"]:
            day = acc.tz_convert(NY).date()
            for off in range(0, 2):  # T or T+1
                cand = pd.Timestamp(day) + pd.Timedelta(days=off)
                if cand not in dates:
                    continue
                ti = int(dates.get_loc(cand))
                open_dt = datetime.combine(cand.date(), dtime(9, 30), tzinfo=NY).astimezone(ZoneInfo("UTC"))
                # visible by this session if acceptance before open, or same/prior calendar for T+1
                if off == 0 and acc >= open_dt:
                    continue
                if 0 <= ti < T:
                    mask[ti, j] = True
                break
    return mask


def combine_signals(bo: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    if len(bo) == 0 and len(ep) == 0:
        return bo
    if len(bo) == 0:
        return ep.copy()
    if len(ep) == 0:
        return bo.copy()
    return pd.concat([bo, ep], ignore_index=True)
