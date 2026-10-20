"""A-line cost stress: the four variants re-run with cost_mult=2 (plan §7.2 'A').

Full re-run (the ranking cost term moves too), plus a fixed-trade cash recost of A00's own
recorded turnover for attribution. Writes a_cost.json next to the base A outputs.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

from . import aport
from .metrics import perf
from .adriver import A_VARIANTS, _load_D
from .compare import cagr_window, HOLDOUT


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--backtest-dir", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.backtest_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = _load_D(a.data_dir)
    res = {}
    for vid, (gm, rank) in A_VARIANTS.items():
        r = aport.run_portfolio(D, g_mode=gm, rank=rank, cost_mult=2.0)
        eq = r["eq"]["equity"]
        eq.to_csv(out / f"equity_daily_{vid}_costx2.csv", float_format="%.12g")
        m = perf(eq, r["eq"]["exposure"])
        res[vid] = {"cagr_full": m["cagr"], "maxdd_full": m["maxdd"], "sharpe_full": m["sharpe"],
                    "cagr_holdout": cagr_window(eq, *HOLDOUT),
                    "fee_total": float(r["eq"]["fee"].sum())}
    # fixed-trade cash recost of A00: the same turnover, charged at the x2 rate
    r1 = aport.run_portfolio(D, g_mode="official", rank="hist", cost_mult=1.0)
    base_fee = float(r1["eq"]["fee"].sum())
    res["_fixed_recost_A00"] = {"fee_at_x1": base_fee, "fee_at_x2_equivalent": base_fee * 2.0,
                               "note": "multiplicative recost of the recorded turnover; "
                                       "identical to a full re-run only if the path is unchanged"}
    (out / "a_cost.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: {kk: round(vv, 6) for kk, vv in v.items() if kk.startswith("cagr")}
                      for k, v in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
