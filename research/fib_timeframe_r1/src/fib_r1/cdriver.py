"""C-line driver: the four runnable configs, cost stress, latency stress, integer-share stress.

Writes per-config equity / orders+trades / entry-experience files and one c_line.json.
All 8 configs are registered: the 4 hourly ones stay BLOCKED_NO_HOURLY_DATA (plan §2).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import ceng
from .data import load_etf_daily, coverage_report, cross_source_check
from .metrics import perf, yearly, trade_stats, mae_mfe_stats

CONFIGS = {
    "C-W-P": ("W", "P"), "C-W-S": ("W", "S"),
    "C-D-P": ("D", "P"), "C-D-S": ("D", "S"),
}
BLOCKED = {"C-H-P", "C-H-S", "C-DH-P", "C-DH-S"}


def to_long(daily: dict) -> pd.DataFrame:
    rows = []
    for s, d in daily.items():
        x = d.reset_index()
        x["sym"] = s
        rows.append(x[["date", "sym", "open", "high", "low", "close", "volume", "dv60"]].rename(columns={"dv60": "dv"}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["close"])


def run_config(P, squeeze, cost_mult=1.0, delay=0, integer=False, capital=10000.0,
               episodes=None, above=None) -> dict:
    r = ceng.run_breakout(P, squeeze=squeeze, cost_mult=cost_mult, delay_bars=delay,
                          integer=integer, capital=capital, episodes=episodes, above=above)
    m = perf(r["eq"]["equity"], r["eq"]["exposure"])
    tr = r["trades"]
    closed = tr.copy() if len(tr) else pd.DataFrame()
    m["trades"] = trade_stats(closed) if len(closed) else {"n_trades": 0}
    ex = ceng.entry_experience(P, closed, fee_r=ceng.FEE * cost_mult)
    m["entry"] = mae_mfe_stats(ex) if len(ex) else {"n": 0}
    if len(ex):
        m["entry"]["failed_breakout_rate"] = float(ex.failed_breakout_5bars.mean())
        m["entry"]["n_failed"] = int(ex.failed_breakout_5bars.sum())
    m["yearly"] = yearly(r["eq"]["equity"])
    m["n_rebalances"] = int((r["eq"]["n_hold"].diff().fillna(0) != 0).sum())
    m["open_position"] = None if r["_pos_open"] is None else {
        k: (str(v) if isinstance(v, (pd.Timestamp,)) else v) for k, v in r["_pos_open"].items()}
    m["stale_days"] = int(r["eq"]["stale"].sum())
    return {"res": r, "metrics": m, "trades": closed, "exp": ex}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--comm-dir", required=True)
    ap.add_argument("--stress", action="store_true", help="also run cost x2/x3, delay, integer shares")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    daily = load_etf_daily(a.raw_dir)
    cov = coverage_report(daily); cov.to_csv(out / "etf_coverage.csv", index=False)
    xs = cross_source_check(daily, a.comm_dir); xs.to_csv(out / "cross_source_commodities.csv", index=False)
    print(cov.to_string(index=False)); print(xs.to_string(index=False), flush=True)

    L = to_long(daily)
    panels = {"D": ceng.build_panel(L, "D"), "W": ceng.build_panel(L, "W")}
    summary, exp_all, eqs, trades_all = {}, [], [], []
    # signal episodes are a property of the plain-breakout condition, per structure period
    epix = {tf: ceng.signal_episodes(P) for tf, P in panels.items()}
    for cid, (tf, mode) in CONFIGS.items():
        P = panels[tf]
        sq = (mode == "S")
        r = run_config(P, sq, episodes=epix[tf][0], above=epix[tf][1])
        m = r["metrics"]; m["config"] = cid; m["tf"] = tf; m["mode"] = mode
        summary[cid] = {"base": m}
        r["res"]["eq"].to_csv(out / f"equity_daily_{cid}.csv", float_format="%.12g")
        r["res"]["eq"].rename(columns={"equity": "nav"}).assign(config=cid).to_csv(
            out / f"eq_long_{cid}.csv", float_format="%.12g")
        if len(r["trades"]):
            t = r["trades"].copy(); t.insert(0, "config", cid)
            t.to_csv(out / f"orders_trades_{cid}.csv", index=False)
            trades_all.append(t)
        if len(r["exp"]):
            e = r["exp"].copy(); e.insert(0, "config", cid); exp_all.append(e)
        # order log always (cancellations/rejections matter)
        if len(r["res"]["orders"]):
            o = r["res"]["orders"].copy(); o.insert(0, "config", cid)
            o.to_csv(out / f"orders_log_{cid}.csv", index=False)
        eqs.append(r["res"]["eq"].equity.rename(cid))
        b = m
        print(f"{cid}: cagr={b['cagr']:.4f} maxdd={b['maxdd']:.4f} sharpe={b['sharpe']:.3f} "
              f"trades={b['trades'].get('n_trades')} expo={b['avg_exposure']:.3f} "
              f"mae_p90_loss={b['entry'].get('mae_p90_loss')}", flush=True)
        if not a.stress:
            continue
        for cm in (2.0, 3.0):
            rr = run_config(P, sq, cost_mult=cm)
            if cm == 2.0:
                rr["res"]["eq"].equity.rename("equity").to_csv(out / f"equity_daily_{cid}_costx2.csv",
                                                               float_format="%.12g")
            summary[cid][f"cost_x{int(cm)}"] = {**rr["metrics"], "config": cid, "cost_mult": cm}
        rr = run_config(P, sq, delay=1)
        rr["res"]["eq"].equity.rename("equity").to_csv(out / f"equity_daily_{cid}_delay1.csv",
                                                       float_format="%.12g")
        summary[cid]["delay_1bar"] = {**rr["metrics"], "config": cid, "delay": 1}
        for cap in (5000, 10000, 20000):
            summary[cid][f"int_{cap}"] = {**run_config(P, sq, integer=True, capital=cap)["metrics"],
                                          "config": cid, "capital": cap, "integer": True}

    pd.concat(eqs, axis=1).to_csv(out / "equity_c_all.csv", float_format="%.12g")
    if exp_all:
        pd.concat(exp_all, ignore_index=True).to_csv(out / "entry_experience.csv", index=False)
    if trades_all:
        pd.concat(trades_all, ignore_index=True).to_csv(out / "orders_trades.csv", index=False)
    summary["_blocked"] = {k: "BLOCKED_NO_HOURLY_DATA" for k in sorted(BLOCKED)}
    (out / "c_line.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    flat = {}
    for cid, d in summary.items():
        if cid.startswith("_"):
            continue
        for vk, vm in d.items():
            code = cid if vk == "base" else f"{cid}|{vk}"
            flat[code] = {k: vm.get(k) for k in ("cagr", "maxdd", "sharpe", "avg_exposure")}
    print(json.dumps(flat, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
