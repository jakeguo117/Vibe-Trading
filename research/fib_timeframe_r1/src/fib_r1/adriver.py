"""A-line driver: A00 / A10 / A01 / A11 on the formal locked snapshot (D_LOCKED/U_FULL).

Run:  python -m fib_r1.adriver --out runs/<ts> --data-dir .../data/ts_v111
Requires UPLEG_BT_BACKTEST on sys.path (the pinned trading-system checkout), read-only.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from . import aport
from .metrics import perf, yearly, slice_window, trade_stats

A_VARIANTS = {
    # id: (g_mode, rank)
    "A00": ("official", "hist"),
    "A10": ("ones", "hist"),
    "A01": ("official", "dv"),
    "A11": ("ones", "dv"),
}


def _load_D(data_dir: str):
    from upleg_bt.engine.fm import FM
    from upleg_bt.paths import data_paths
    return FM(data_paths(data_dir), processes=6)


def run_all(D, out: Path, official_mod=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for vid, (g_mode, rank) in A_VARIANTS.items():
        t0 = time.time()
        r = aport.run_portfolio(D, g_mode=g_mode, rank=rank, trace=True)
        eq = r["eq"]["equity"]
        m = perf(eq, r["eq"]["exposure"])
        r["eq"].to_csv(out / f"equity_daily_{vid}.csv", float_format="%.12g")
        r["target_trace"].to_csv(out / f"target_trace_{vid}.csv", index=False, float_format="%.10g")
        r["weights"].to_csv(out / f"weights_{vid}.csv", index=False, float_format="%.10g")
        r["events"].to_csv(out / f"events_{vid}.csv", index=False)
        res[vid] = {"metrics": m, "yearly": yearly(eq), "elapsed_s": round(time.time() - t0, 1),
                    "n_events": int(len(r["events"])), "n_trade_days": int((r["eq"].turnover > 0).sum())}
        res[vid]["_eq"] = eq; res[vid]["_r"] = r
    if official_mod is not None:
        # PROOF: the research copy with official switches == the untouched engine.
        off = official_mod.run_portfolio(D)
        d = float(np.abs(off["eq"]["equity"].values - res["A00"]["_eq"].values).max())
        res["_identity"] = {"a00_vs_official_maxabs": d, "pass": d < 1e-12,
                            "index_match": bool((off["eq"].index == res["A00"]["_eq"].index).all())}
    return res


def cancellation_table(D, trace: pd.DataFrame, capv=0.70, vt=0.16) -> tuple[pd.DataFrame, dict]:
    """§4.3 cancellation audit, computed on A00's own frozen decisions.

    For each decision day t with held set H, recompute the target with g=1 while holding
    H, sigma and the 16% scaling exactly as A00 had them. If the resulting weight vector
    equals A00's, the coefficient was cancelled at that decision.
    """
    from .aport import pvol
    rows = []
    for t, g in trace.groupby("t", sort=True):
        n = len(g)
        stage, half, sig, raw = g.stage_code.values, g.half.values, g.sigma.values, g.raw.values
        if (half == False).all():
            raw_off = np.full(n, 0.70 / 3.0)
        else:
            raw_off = (0.70 / 3.0) * np.where(half, 0.5, 1.0)
            if half.any() and (~half).any():
                freed = (0.70 / 3.0 * 0.5) * half.sum()
                raw_off[~half] += freed * raw_off[~half] / raw_off[~half].sum()
        # A00 scale: use the recorded portfolio vol / scale factor (frozen)
        scale = float(g.scale.iloc[0])
        tg_on = raw * scale; tg_off = raw_off * scale
        if tg_on.sum() > capv + 1e-12: tg_on = tg_on * capv / tg_on.sum()
        if tg_off.sum() > capv + 1e-12: tg_off = tg_off * capv / tg_off.sum()
        l1 = float(np.abs(tg_on - tg_off).sum())
        rows.append({"t": int(t), "n_hold": n, "cap_binding": bool(g.cap_binding.iloc[0]),
                     "g_distinct": int(len(np.unique(np.round(g.g.values, 12)))),
                     "target_l1_delta": l1,
                     "cancelled": bool(l1 <= 1e-12)})
    df = pd.DataFrame(rows)
    summ = {
        "n_decisions": int(len(df)),
        "cancel_rate_all": float(df.cancelled.mean()) if len(df) else float("nan"),
        "cancel_rate_by_nhold": {int(k): float(v.cancelled.mean()) for k, v in df.groupby("n_hold")},
        "n_by_nhold": {int(k): int(len(v)) for k, v in df.groupby("n_hold")},
        "l1_quantiles": {q: float(df.target_l1_delta.quantile(q)) for q in (0.5, 0.9, 0.95, 1.0)} if len(df) else {},
        "cancel_rate_single_position": float(df[df.n_hold == 1].cancelled.mean()) if (df.n_hold == 1).any() else None,
    }
    return df, summ


def describe_decision_divergence(res: dict) -> dict:
    """§4.2: A00 / A10 (and A01 / A11) must share entry, rank and state sequences."""
    out = {}
    for a, b in (("A00", "A10"), ("A01", "A11")):
        ea, eb = res[a]["_r"]["events"], res[b]["_r"]["events"]
        ka = list(zip(ea.date.astype(str), ea.name, ea.what))
        kb = list(zip(eb.date.astype(str), eb.name, eb.what))
        out[f"{a}_vs_{b}"] = {"events_equal": ka == kb, "n_a": len(ka), "n_b": len(kb),
                              "first_diff": next((i for i, (x, y) in enumerate(zip(ka, kb)) if x != y), None)}
    wa, wb = res["A00"]["_r"]["weights"], res["A10"]["_r"]["weights"]
    M = wa.merge(wb, on=["date", "name"], how="outer", suffixes=("_a00", "_a10"), indicator=True)
    out["A00_vs_A10_weights"] = {
        "rows": int(len(M)), "unmatched": int((M._merge != "both").sum()),
        "target_maxabs": float(np.nanmax(np.abs(M.w_target_a00 - M.w_target_a10))) if len(M) else None,
        "holding_maxabs": float(np.nanmax(np.abs(M.w_close_a00 - M.w_close_a10))) if len(M) else None,
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--backtest-dir", required=True, help="pinned trading-system/backtest (read-only)")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.backtest_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = _load_D(a.data_dir)
    import upleg_bt.engine.portfolio as official_mod
    res = run_all(D, out, official_mod=official_mod)
    canc_df, canc = cancellation_table(D, res["A00"]["_r"]["target_trace"])
    canc_df.to_csv(out / "cancellation_by_decision.csv", index=False)
    div = describe_decision_divergence(res)
    summary = {"variants": {k: v["metrics"] for k, v in res.items() if not k.startswith("_")},
               "yearly": {k: v["yearly"] for k, v in res.items() if not k.startswith("_")},
               "identity": res.get("_identity"), "cancellation": canc, "divergence": div,
               "elapsed_s": {k: v["elapsed_s"] for k, v in res.items() if not k.startswith("_")}}
    (out / "a_line.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: summary[k] for k in ("identity", "cancellation")}, indent=1, ensure_ascii=False, default=str))
    for k in A_VARIANTS:
        m = res[k]["metrics"]
        print(f"{k}: cagr={m['cagr']:.6f} maxdd={m['maxdd']:.6f} sharpe={m['sharpe']:.4f} "
              f"sortino={m['sortino']:.4f} expo={m['avg_exposure']:.4f} days={m['n_days']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
