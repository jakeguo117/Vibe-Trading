"""B-line entry experience (plan §9.3): price-based MAE/MFE per completed round trip.

B has no initial stop, so only the price and percent measures apply (no R multiple); 1-day and
5-day windows are counted in trading sessions from the entry session.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import beng
from .data import load_etf_daily

START, END = "2019-01-02", "2026-09-30"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--backtest-dir", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.backtest_dir)
    out = Path(a.out)
    daily = load_etf_daily(a.raw_dir)
    rows = []
    for cid in ("B-W-ON", "B-W-OFF", "B-D-ON", "B-D-OFF"):
        f = out / f"trades_{cid}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, parse_dates=["entry_date", "exit_date"])
        for r in t.itertuples():
            d = daily[r.sym]
            win = d.loc[(d.index >= r.entry_date) & (d.index <= r.exit_date)]
            if len(win) < 2:
                continue
            E = float(win.open.iloc[0])
            after = win.iloc[1:]                       # the entry session's own low is not post-fill
            if len(after) == 0:
                continue
            mae = float(after.low.min() / E - 1.0)
            mfe = float(after.high.max() / E - 1.0)
            w1 = after.iloc[:1]
            w5 = after.iloc[:5]
            rows.append(dict(config=cid, sym=r.sym, entry_date=r.entry_date.date(),
                             exit_date=r.exit_date.date(), bars_held=len(after),
                             mae=mae, mfe=mfe, mae_1d=float(w1.low.min() / E - 1.0),
                             mae_1d_pct=float(w1.low.min() / E - 1.0),
                             net_1d=float(w1.close.iloc[-1] / E - 1.0),
                             net_5d=float(w5.close.iloc[-1] / E - 1.0) if len(w5) else np.nan,
                             right_censored=bool(len(after) < 5), net_ret=r.net_ret))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out / "b_entry_experience.csv", index=False)
        print(df.groupby("config").agg(n=("mae", "size"), mae_med=("mae", "median"),
                                       mae_p90_loss=("mae", lambda s: float(np.quantile(np.maximum(0, -s), .9))),
                                       mfe_med=("mfe", "median"),
                                       adv=("mae_1d", lambda s: float((s <= -0.005).mean()))).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
