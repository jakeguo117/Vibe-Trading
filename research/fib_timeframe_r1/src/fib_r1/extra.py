"""Background and robustness tables that are not primary comparisons.

1. Buy-and-hold of the nine ETFs (plan §12 批次2 background table).
2. Per-symbol contribution of each C config, and the §10.2(7) drop-the-best-symbol rerun.
3. §8.4 thinning diagnostic: records whether its trigger condition was even met.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import ceng, cdriver
from .data import load_etf_daily
from .metrics import perf

START, END = "2019-01-02", "2026-09-30"


def buy_hold(daily: dict) -> pd.DataFrame:
    rows = []
    for s, d in daily.items():
        c = d.close.loc[(d.index >= pd.Timestamp(START)) & (d.index <= pd.Timestamp(END))].dropna()
        if len(c) < 2:
            continue
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        r = c.pct_change().dropna()
        rows.append({"sym": s, "first": str(c.index[0].date()), "last": str(c.index[-1].date()),
                     "total_return": float(c.iloc[-1] / c.iloc[0] - 1),
                     "cagr": float((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1),
                     "ann_vol": float(r.std(ddof=1) * np.sqrt(252)),
                     "maxdd": float((c / c.cummax() - 1).min()),
                     "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252))})
    return pd.DataFrame(rows).sort_values("cagr", ascending=False)


def run_variant(L, tf, squeeze, drop=None, capital=10000.0, cost_mult=1.0):
    Lf = L if drop is None else L[L.sym != drop]
    P = ceng.build_panel(Lf, tf)
    return ceng.run_breakout(P, squeeze=squeeze, cost_mult=cost_mult, capital=capital)


def symbol_contribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    g = trades.groupby("sym").agg(n=("net_ret", "size"), net=("net_ret", "sum"),
                                  avg=("net_ret", "mean"), dollars=("notional", "sum"))
    return g.sort_values("net", ascending=False).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--run-root", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    daily = load_etf_daily(a.raw_dir)
    L = cdriver.to_long(daily)

    bh = buy_hold(daily)
    bh.to_csv(out / "buy_and_hold.csv", index=False, float_format="%.8g")
    print(bh.to_string(index=False), flush=True)

    contrib, drop_res = {}, {}
    for cid, (tf, mode) in cdriver.CONFIGS.items():
        sq = (mode == "S")
        r = run_variant(L, tf, sq)
        tr = r["trades"]
        c = symbol_contribution(tr)
        if len(c):
            c.insert(0, "config", cid)
            contrib[cid] = c.to_dict("records")
            top = c.iloc[0].sym
            base = perf(r["eq"]["equity"], r["eq"]["exposure"])
            r2 = run_variant(L, tf, sq, drop=top)
            m2 = perf(r2["eq"]["equity"], r2["eq"]["exposure"])
            drop_res[cid] = {"dropped": top, "cagr_full": base["cagr"], "cagr_without": m2["cagr"],
                             "maxdd_full": base["maxdd"], "maxdd_without": m2["maxdd"],
                             "n_trades_full": int(len(tr)), "n_trades_without": int(len(r2["trades"]))}
            print(f"{cid}: top={top} cagr {base['cagr']:.4f} -> {m2['cagr']:.4f}", flush=True)

    # §8.4 trigger check: only C-D-P vs C-W-P or C-H-P vs C-D-P, and only if the fast leg had both
    # higher holdout net return AND more completed trades. C-H-P cannot run (no hourly data).
    cmp = pd.read_csv(Path(a.run_root) / "comparisons.csv")
    def row(cid):
        x = cmp[cmp.comparison == cid]
        return None if not len(x) else x.iloc[0]
    trig = {}
    for cid in ("C-D-P-C-W-P", "C-H-P-C-D-P"):
        r = row(cid)
        if r is None or pd.isna(r.get("cagr_diff_pp")):
            trig[cid] = {"triggered": False, "why": "comparison not runnable (BLOCKED)"}
        else:
            trig[cid] = {"triggered": bool(r.cagr_diff_pp > 0 and (r.trades_left or 0) > (r.trades_right or 0)),
                         "cagr_diff_pp": float(r.cagr_diff_pp),
                         "trades_left": None if pd.isna(r.trades_left) else int(r.trades_left),
                         "trades_right": None if pd.isna(r.trades_right) else int(r.trades_right),
                         "why": "fast leg did not show both higher net return and more completed trades"
                         if not (r.cagr_diff_pp > 0) else "see values"}
    res = {"buy_and_hold": bh.to_dict("records"), "symbol_contribution": contrib,
           "drop_best_symbol": drop_res, "thinning_trigger": trig}
    (out / "extra.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"thinning": trig}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
