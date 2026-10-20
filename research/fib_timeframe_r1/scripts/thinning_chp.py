"""§8.4 pre-registered opportunity-thinning diagnostic for C-H-P vs C-D-P.

The plan triggers it only when the fast side shows a higher net return AND more completed trades
in the validation segment; both hold here (C-H-P +6.93pp on the holdout with 136 vs 27 trades).

Frozen procedure:
  * count candidate signals in the DESIGN window (2019-02-04..2023-12-29), one per decision time
    (at most one name can be chosen, so the count is per decision time, not per name);
  * p = min(1, N_slow/N_fast) is frozen before any thinned path is run;
  * seeds 0..99, every path kept, no best-seed selection;
  * a refusal never falls through to the second-ranked name.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/jake/projects/trading/Vibe-Trading/research/fib_timeframe_r1/src")
sys.path.insert(0, "/Users/jake/projects/trading/trading-system/backtest")

from fib_r1 import ceng, configs, hourly as H  # noqa: E402
from fib_r1.data import load_etf_daily  # noqa: E402
from fib_r1.metrics import perf  # noqa: E402

RUN = Path("/Users/jake/projects/hermes-tasks/runs/backtest-fib-hourly-20261020-1900/runs")
SNAP = Path("/Users/jake/projects/trading/data/ts_v111")
START = "2019-02-04"
DESIGN_END = configs.WINDOWS["design"][1]
END = configs.WINDOWS["full"][1]


def candidates(P: ceng.Panel, start, end) -> int:
    """Number of decision times with at least one eligible plain-breakout candidate."""
    B = ceng.breakout_line(P.H)
    i0 = int(P.dates.searchsorted(pd.Timestamp(start)))
    i1 = int(P.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    n = 0
    for t in range(i0, i1 + 1):
        ok = P.ELIG[t] & np.isfinite(P.C[t]) & np.isfinite(B[t]) & (P.C[t] > B[t])
        n += int(ok.any())
    return n


def main():
    daily = load_etf_daily(SNAP / "raw")
    Ld = pd.concat([d.reset_index().assign(sym=s)[["date", "sym", "open", "high", "low",
                                                   "close", "volume", "dv60"]].rename(columns={"dv60": "dv"})
                    for s, d in daily.items()], ignore_index=True).dropna(subset=["close"])
    bundle = H.load_hourly(daily)
    P_h = H.hourly_panel(bundle, Ld)
    P_d = ceng.build_panel(Ld, "D")

    n_slow = candidates(P_d, START, DESIGN_END)
    n_fast = candidates(P_h, START, DESIGN_END)
    p = float(min(1.0, n_slow / n_fast)) if n_fast else 1.0
    print(f"design-window candidate decision times: D={n_slow} H={n_fast} -> p={p:.6f}")

    rows = []
    base_fast = ceng.run_breakout(P_h, squeeze=False, start=START, end=END)
    base_d = ceng.run_breakout(P_d, squeeze=False, start=START, end=END)
    for label, r in (("C-H-P", base_fast), ("C-D-P", base_d)):
        eq = H.daily_marks(r["eq"], P_h.dates, P_h.is_last, START, END) if label == "C-H-P" else r["eq"]
        m = perf(eq.equity, eq.exposure)
        rows.append(dict(path=label, seed=-1, p=p, cagr=m["cagr"], maxdd=m["maxdd"], sharpe=m["sharpe"],
                         n_trades=len(r["trades"]), avg_exposure=m["avg_exposure"],
                         total_fees=float(r["trades"].fees_total.sum()) if len(r["trades"]) else 0.0))
    for seed in range(100):
        r = ceng.run_breakout(P_h, squeeze=False, start=START, end=END, thin_p=p, thin_seed=seed)
        eq = H.daily_marks(r["eq"], P_h.dates, P_h.is_last, START, END)
        m = perf(eq.equity, eq.exposure)
        rows.append(dict(path=f"thinned_s{seed}", seed=seed, p=p, cagr=m["cagr"], maxdd=m["maxdd"],
                         sharpe=m["sharpe"], n_trades=len(r["trades"]), avg_exposure=m["avg_exposure"],
                         total_fees=float(r["trades"].fees_total.sum()) if len(r["trades"]) else 0.0))
    df = pd.DataFrame(rows)
    out = RUN / "combined" / "thinning_paths.csv"
    df.to_csv(out, index=False, float_format="%.10g")
    th = df[df.seed >= 0]
    q = th.cagr.quantile([0.05, 0.5, 0.95]).values
    print(f"thinned (100 seeds): cagr p5={q[0]:.4f} median={q[1]:.4f} p95={q[2]:.4f} "
          f"| base H={df[df.path=='C-H-P'].cagr.iloc[0]:.4f} base D={df[df.path=='C-D-P'].cagr.iloc[0]:.4f}")
    print(f"trades: thinned median={th.n_trades.median():.0f} | H={df[df.path=='C-H-P'].n_trades.iloc[0]} "
          f"D={df[df.path=='C-D-P'].n_trades.iloc[0]}")
    print(f"share of seeds beating D: {float((th.cagr > df[df.path=='C-D-P'].cagr.iloc[0]).mean()):.3f}")
    (RUN / "combined" / "thinning_summary.json").write_text(json.dumps({
        "trigger": "C-H-P beat C-D-P on the holdout with more completed trades (§8.4)",
        "n_slow": n_slow, "n_fast": n_fast, "p": p, "seeds": 100,
        "base": {k: float(df[df.path == k].cagr.iloc[0]) for k in ("C-H-P", "C-D-P")},
        "thinned_cagr_p5": float(q[0]), "thinned_cagr_median": float(q[1]), "thinned_cagr_p95": float(q[2]),
        "thinned_n_trades_median": float(th.n_trades.median()),
        "share_seeds_beating_daily": float((th.cagr > df[df.path == "C-D-P"].cagr.iloc[0]).mean()),
        "note": "100 random paths, all kept, no best-seed selection; p only matches candidate "
                "frequency, not completed trades",
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
