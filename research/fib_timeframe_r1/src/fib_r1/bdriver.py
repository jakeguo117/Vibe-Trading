"""B-line driver: B-W-ON/OFF, B-D-ON/OFF (hourly pair stays BLOCKED) + cost/latency stress."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import beng
from .data import load_etf_daily
from .metrics import perf, yearly, trade_stats

CONFIGS = {"B-W-ON": ("W", "official"), "B-W-OFF": ("W", "ones"),
           "B-D-ON": ("D", "official"), "B-D-OFF": ("D", "ones")}
BLOCKED = {"B-H-ON", "B-H-OFF"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--backtest-dir", required=True)
    ap.add_argument("--stress", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.backtest_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    daily = load_etf_daily(a.raw_dir)
    panels = {tf: beng.build_b(daily, tf) for tf in ("W", "D")}
    for tf, P in panels.items():
        cov = pd.DataFrame({"sym": P["syms"], "n_bars": [len(P["dates"])] * len(P["syms"]),
                            "eligible_last": P["ELIG"][-1].tolist(),
                            "dv60_median_musd": [float(np.nanmedian(P["DV"][:, j]) / 1e6) for j in range(len(P["syms"]))]})
        cov.to_csv(out / f"b_coverage_{tf}.csv", index=False)
        print(f"{tf}: {len(P['dates'])} bars {P['dates'][0].date()}..{P['dates'][-1].date()}", flush=True)

    summ, eqs = {}, []
    for cid, (tf, gm) in CONFIGS.items():
        P = panels[tf]
        r = beng.run_b(P, g_mode=gm)
        rt = beng.round_trips(r["orders"], P["syms"])
        m = perf(r["eq"].equity, r["eq"].exposure)
        m["trades"] = trade_stats(rt)
        m["yearly"] = yearly(r["eq"].equity)
        m["n_rebal_days"] = int(len(r["rebal"]))
        m["rebal_notional"] = float(r["rebal"].notional.sum()) if len(r["rebal"]) else 0.0
        m["n_event_enter"] = int((r["events"].what == "enter").sum()) if len(r["events"]) else 0
        m["n_event_exit"] = int(r["events"].what.str.startswith("exit").sum()) if len(r["events"]) else 0
        m["n_event_halve"] = int((r["events"].what == "halve").sum()) if len(r["events"]) else 0
        summ[cid] = {"base": m}
        r["eq"].to_csv(out / f"equity_daily_{cid}.csv", float_format="%.12g")
        if len(r["orders"]):
            o = r["orders"].copy(); o.insert(0, "config", cid)
            o.to_csv(out / f"orders_{cid}.csv", index=False)
        if len(rt):
            t = rt.copy(); t.insert(0, "config", cid); t.to_csv(out / f"trades_{cid}.csv", index=False)
        if len(r["targets"]):
            r["targets"].to_csv(out / f"target_weights_{cid}.csv", float_format="%.10g")
        eqs.append(r["eq"].equity.rename(cid))
        print(f"{cid}: cagr={m['cagr']:.4f} maxdd={m['maxdd']:.4f} sharpe={m['sharpe']:.3f} "
              f"trades={m['trades'].get('n_trades')} expo={m['avg_exposure']:.3f} "
              f"enter={m['n_event_enter']} rebal_days={m['n_rebal_days']}", flush=True)
        if a.stress:
            for cm in (2.0, 3.0):
                rr = beng.run_b(P, g_mode=gm, cost_mult=cm)
                if cm == 2.0:
                    rr["eq"].equity.rename("equity").to_csv(out / f"equity_daily_{cid}_costx2.csv",
                                                            float_format="%.12g")
                summ[cid][f"cost_x{int(cm)}"] = scene(rr, P, cost_mult=cm)
            rr = beng.run_b(P, g_mode=gm, delay_bars=1)
            rr["eq"].equity.rename("equity").to_csv(out / f"equity_daily_{cid}_delay1.csv",
                                                    float_format="%.12g")
            summ[cid]["delay_1bar"] = scene(rr, P, delay_bars=1)

    pd.concat(eqs, axis=1).to_csv(out / "equity_b_all.csv", float_format="%.12g")
    summ["_blocked"] = {k: "BLOCKED_NO_HOURLY_DATA" for k in sorted(BLOCKED)}
    (out / "b_line.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({f"{k}|{vk}": {kk: vv for kk, vv in vm.items() if kk in ("cagr", "maxdd", "sharpe")}
                      for k, v in summ.items() if not k.startswith("_")
                      for vk, vm in v.items()}, indent=1, ensure_ascii=False, default=str))
    return 0


def scene(r, P, **kw):
    rt = beng.round_trips(r["orders"], P["syms"])
    m = perf(r["eq"].equity, r["eq"].exposure)
    m["trades"] = trade_stats(rt)
    m["yearly"] = yearly(r["eq"].equity)
    m.update(kw)
    return m


def run_stress(P, gm, **kw):
    tag = kw.pop("tag")
    r = beng.run_b(P, g_mode=gm, **kw)
    rt = beng.round_trips(r["orders"], P["syms"])
    m = perf(r["eq"].equity, r["eq"].exposure)
    m["trades"] = trade_stats(rt)
    m["yearly"] = yearly(r["eq"].equity)
    m.update(kw)
    return m


if __name__ == "__main__":
    sys.exit(main())
