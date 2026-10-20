"""D_COMMON / U_ETF loader: the fixed 9-ETF pool from the same vendor snapshot as A.

Bars are built exactly like the formal pipeline builds its matrices
(trading-system@2ac825b backtest/upleg_bt/data/build.py `_prep`):
    adjusted OHLC = raw OHLC x (Adj Close / Close)      <- dividend+split adjusted, used for signals
    raw_close (split-continuous, dividend-unadjusted) = Close x product of splits after t
    dv60 = 60-day rolling median of Close x Volume      <- actual traded notional at the time
so A and C share one vendor, one adjustment convention and one dollar-volume definition.
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import pandas as pd

SYMS = ["SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "TLT", "EFA", "EEM"]


def load_etf_daily(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    want = set(SYMS)
    frames = []
    for fn in sorted(glob.glob(str(raw_dir / "b*.parquet"))) + [str(raw_dir / "core.parquet")]:
        d = pd.read_parquet(fn)
        d = d[d.sym.isin(want)]
        if len(d):
            frames.append(d)
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["sym", "date"]).sort_values(["sym", "date"])
    out = {}
    for s, g in d.groupby("sym"):
        g = g.copy()
        g["date"] = pd.to_datetime(g.date)
        g = g[(g.Close > 0) & (g["Adj Close"] > 0) & (g.Open > 0) & (g.High > 0) & (g.Low > 0)]
        g = g.set_index("date").sort_index()
        f = (g["Adj Close"] / g["Close"]).values
        o, h, l, c = g.Open.values * f, g.High.values * f, g.Low.values * f, g["Adj Close"].values
        spl = g["Stock Splits"].fillna(0).values
        spl = np.where(spl > 0, spl, 1.0)
        fut = np.r_[np.cumprod(spl[::-1])[::-1][1:], 1.0]
        raw = g.Close.values * fut
        dv = pd.Series(g.Close.values * g.Volume.values).rolling(60, min_periods=60).median().values
        sel = np.asarray(spl, dtype=float)
        # SWITCH (hourly batch): keep the raw split table and the per-day adjustment ratio so the
        # intraday feed can be mapped onto exactly this scale (hourly.split_factor). No existing
        # column changes, so A/B/C results are untouched.
        out[s] = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "raw_close": raw,
                               "volume": g.Volume.values, "dv60": dv,
                               "dividend": g.Dividends.fillna(0).values,
                               "splits": sel, "adj_ratio": (g["Adj Close"] / g.Close).values,
                               "raw_close_col": g.Close.values}, index=g.index)
    missing = [s for s in SYMS if s not in out]
    if missing:
        raise RuntimeError(f"missing ETF bars in snapshot: {missing}")
    return out


def coverage_report(daily: dict[str, pd.DataFrame], start="2019-01-02", end="2026-09-30") -> pd.DataFrame:
    rows = []
    cal = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in daily.values()])))
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    for s, d in daily.items():
        hit = d.reindex(cal)
        rows.append({"sym": s, "first": str(d.index[0].date()), "last": str(d.index[-1].date()),
                     "n_bars_total": int(len(d)), "n_bars_window": int(len(hit.dropna(subset=["close"]))),
                     "n_missing_window": int(hit["close"].isna().sum()),
                     "coverage_window": float(hit["close"].notna().mean()),
                     "dv60_median_musd": float(hit["dv60"].median() / 1e6)})
    return pd.DataFrame(rows)


def cross_source_check(daily, comm_dir) -> pd.DataFrame:
    """GLD/SLV/USO: our raw-parquet adjusted close vs the commodities CSV used by A."""
    rows = []
    for s in ("GLD", "SLV", "USO"):
        p = Path(comm_dir) / f"{s}_yahoo_adj.csv"
        if not p.exists():
            continue
        c = pd.read_csv(p, parse_dates=["date"]).set_index("date")["close"].astype(float)
        a = daily[s]["close"].astype(float)
        j = pd.concat([a.rename("raw_parquet_adj"), c.rename("commodities_csv")], axis=1, join="inner").dropna()
        rel = (j.raw_parquet_adj / j.commodities_csv - 1).abs()
        rows.append({"sym": s, "overlap_days": int(len(j)), "max_abs_rel_diff": float(rel.max()),
                     "median_abs_rel_diff": float(rel.median()),
                     "worst_date": str(rel.idxmax().date()) if len(rel) else None,
                     "diff_last_60d": float(rel.iloc[-60:].max()) if len(rel) >= 60 else None})
    return pd.DataFrame(rows)
