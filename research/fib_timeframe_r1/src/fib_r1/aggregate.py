"""Aggregates every run directory into the plan §13 deliverables that are pure data.

Writes: result.json, trials.csv, equity_daily.csv, target_trace.csv, orders_trades.csv,
entry_experience.csv, acceptance.json. Every number is read from a file a run produced.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import configs
from .compare import HOLDOUT, cagr_window

BLOCKED = {"B-H-ON": "BLOCKED_NO_HOURLY_DATA", "B-H-OFF": "BLOCKED_NO_HOURLY_DATA",
           "C-H-P": "BLOCKED_NO_HOURLY_DATA", "C-H-S": "BLOCKED_NO_HOURLY_DATA",
           "C-DH-P": "BLOCKED_NO_HOURLY_DATA", "C-DH-S": "BLOCKED_NO_HOURLY_DATA"}
LINE_OF = {c[0]: c[1] for c in configs.CONFIGS}


def _j(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def load_all(run_root: Path):
    a = _j(run_root / "a_line" / "a_line.json")
    b = _j(run_root / "b_line" / "b_line.json")
    c = _j(run_root / "c_line" / "c_line.json")
    c2 = _j(run_root / "a_line" / "a_cost.json")
    return a, b, c, c2


def variant_table(a, b, c, run_root: Path | None = None):
    rows = {}
    epj = {}
    epm = {}
    if run_root is not None and (run_root / "a_line" / "a_episodes.json").exists():
        epj = _j(run_root / "a_line" / "a_episodes.json")
    if run_root is not None and (run_root / "a_line" / "a_episodes.csv").exists():
        e = pd.read_csv(run_root / "a_line" / "a_episodes.csv")
        for v, g in e.groupby("variant"):
            L = np.maximum(0.0, -g.mae.astype(float).values)
            epm[v] = {"mae_p90_loss": float(np.quantile(L, 0.90)), "mae_median": float(np.median(g.mae)),
                      "adverse_rate_1d": float((g.ma1.astype(float) <= -0.005).mean()),
                      "n_mae_1d": int(g.ma1.notna().sum())}
    for k, v in a.get("variants", {}).items():
        m = dict(v)
        if k in epj:
            e = epj[k]
            m["trades"] = {"n_trades": e.get("n_trades"), "win_rate": e.get("win_rate"),
                           "avg_net": e.get("avg_net"), "median_net": e.get("median_net"),
                           "avg_win": e.get("avg_win"), "avg_loss": e.get("avg_loss"),
                           "profit_factor": e.get("profit_factor"), "worst": e.get("worst"),
                           "best": e.get("best"), "expectancy": e.get("expectancy")}
            m["entry"] = {"n": e.get("n_episodes"), "mean_days": e.get("mean_days"),
                          "median_days": e.get("median_days"),
                          "mean_price_ret": e.get("mean_price_ret"),
                          "mean_nav_contribution": e.get("mean_nav_contribution"),
                          **(epm.get(k) or {}), "basis": "close_only"}
        rows[k] = {"metrics": m, "yearly": a.get("yearly", {}).get(k, {}), "line": "A"}
    for k, v in b.items():
        if k.startswith("_"):
            continue
        rows[k] = {"metrics": v.get("base", v), "variants": {kk: vv for kk, vv in v.items() if kk != "base"},
                   "line": "B"}
    for k, v in c.items():
        if k.startswith("_"):
            continue
        rows[k] = {"metrics": v.get("base", v), "variants": {kk: vv for kk, vv in v.items() if kk != "base"},
                   "line": "C"}
    return rows


def equity_long(run_root: Path) -> pd.DataFrame:
    frames = []
    for d, suffix in ((run_root / "a_line", "A"), (run_root / "b_line", "B"), (run_root / "c_line", "C")):
        if not d.exists():
            continue
        for f in d.glob("equity_daily_*.csv"):
            vid = f.stem.replace("equity_daily_", "")
            if vid.endswith(("_costx2", "_delay1")):
                continue
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df.rename(columns={"equity": "nav"})
            df["variant"] = vid
            df["line"] = suffix
            df["net_return"] = df.nav.pct_change()
            df["gross_return"] = df.net_return + (df["fee"] if "fee" in df else 0.0)
            for col in ("cash", "exposure", "fee", "stale", "turnover"):
                if col not in df:
                    df[col] = np.nan
            frames.append(df.reset_index()[["date", "variant", "line", "nav", "gross_return", "net_return",
                                            "cash", "exposure", "fee", "stale", "turnover", "n_hold"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def trace_long(run_root: Path) -> pd.DataFrame:
    frames = []
    for f in (run_root / "a_line").glob("target_trace_*.csv"):
        df = pd.read_csv(f)
        df["variant"] = f.stem.replace("target_trace_", "")
        # join the realised weight for the same decision day/name
        wf = run_root / "a_line" / f"weights_{df.variant.iloc[0]}.csv"
        if wf.exists():
            w = pd.read_csv(wf)
            w["t"] = w["t"].astype(int)
            df = df.merge(w[["t", "j", "w_close", "w_target", "traded_today"]], on=["t", "j"], how="left")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def trades_long(run_root: Path) -> pd.DataFrame:
    frames = []
    for d, glob in ((run_root / "c_line", "orders_trades_*.csv"), (run_root / "b_line", "trades_*.csv")):
        for f in d.glob(glob):
            df = pd.read_csv(f)
            df["variant"] = f.stem.replace("orders_trades_", "").replace("trades_", "")
            df["line"] = "C" if d.name.startswith("c") else "B"
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def entry_long(run_root: Path) -> pd.DataFrame:
    frames = []
    f = run_root / "c_line" / "entry_experience.csv"
    if f.exists():
        d = pd.read_csv(f); d["line"] = "C"; frames.append(d)
    f = run_root / "b_line" / "b_entry_experience.csv"
    if f.exists():
        d = pd.read_csv(f).rename(columns={"config": "config"}); d["line"] = "B"; frames.append(d)
    f = run_root / "a_line" / "a_episodes.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d.rename(columns={"variant": "config", "mae_1d": "mae_1d_pct", "ma1": "mae_1d"})
        d["line"] = "A"; frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def trials(vt: dict, a, b, c) -> pd.DataFrame:
    """Every registered attempt, including the ones that never ran."""
    rows = []
    for cid in configs.CONFIG_IDS:
        line = LINE_OF[cid]
        if cid in BLOCKED:
            rows.append({"variant": cid, "line": line, "pool": configs.POOL_A if line == "A" else configs.POOL_BC,
                         "window": "full", "cost": "base", "seed": np.nan, "status": BLOCKED[cid],
                         "reason": "no 60-minute history on this machine; not synthesised"})
            continue
        v = vt.get(cid)
        if not v:
            rows.append({"variant": cid, "line": line, "pool": "", "window": "full", "cost": "base",
                         "seed": np.nan, "status": "MISSING", "reason": "no output file"})
            continue
        m = v["metrics"]
        rows.append({"variant": cid, "line": line, "pool": configs.POOL_A if line == "A" else configs.POOL_BC,
                     "window": "2019-01-02..2026-09-30", "cost": "base", "seed": np.nan, "status": "DONE",
                     "reason": "", "cagr": m.get("cagr"), "maxdd": m.get("maxdd"), "sharpe": m.get("sharpe")})
        for sk, sm in (v.get("variants") or {}).items():
            if isinstance(sm, dict) and "cagr" in sm:
                rows.append({"variant": cid, "line": line, "pool": rows[-1]["pool"],
                             "window": "2019-01-02..2026-09-30",
                             "cost": sk, "seed": np.nan, "status": "DONE", "reason": "",
                             "cagr": sm.get("cagr"), "maxdd": sm.get("maxdd"), "sharpe": sm.get("sharpe")})
    for k in ("A00", "A10", "A01", "A11"):
        v = vt.get(k)
        if v:
            rows.append({"variant": k, "line": "A", "pool": configs.POOL_A, "window": "2019-01-02..2026-09-30",
                         "cost": "cost_x2", "seed": np.nan, "status": "DONE", "reason": ""})
    df = pd.DataFrame(rows)
    return df


def acceptance(vt, cmp_df, a, c2) -> dict:
    """§10.2 gate-by-gate verdicts, each with the evidence it came from."""
    g = configs.GATE
    out = {"gate0": {}, "configs": {}, "comparisons": {}, "entry_experience": {}, "cost": {},
           "small_capital": {}, "multiline": {}}
    gate0 = _j(Path("/Users/jake/projects/hermes-tasks/runs/backtest-fib-breakout-20261020-1655/runs/gate0.json"))
    out["gate0"] = {
        "baseline_reproduced": {"value": gate0.get("reproduce_v1_1_1", {}).get("daily_equity_maxabs"),
                                "threshold": 1e-9, "status": "PASS" if gate0.get("reproduce_v1_1_1", {}).get("PASS") else "FAIL",
                                "evidence": "runs/gate0.json"},
        "verify_data_exact": {"value": gate0.get("verify_data", {}).get("inputs_exact"),
                              "status": "PASS" if gate0.get("verify_data", {}).get("exit_code") == 0 else "FAIL"},
        "pytest": {"value": gate0.get("pytest", {}).get("result_line"),
                   "status": "PASS" if gate0.get("pytest", {}).get("n_failed") == 0 else "FAIL"},
        "a00_identity_vs_official": {"value": a.get("identity", {}).get("a00_vs_official_maxabs"),
                                     "threshold": 1e-12,
                                     "status": "PASS" if a.get("identity", {}).get("pass") else "FAIL"},
    }
    for cid in configs.CONFIG_IDS:
        if cid in BLOCKED:
            out["configs"][cid] = {"status": BLOCKED[cid]}
            continue
        v = vt.get(cid)
        if not v:
            out["configs"][cid] = {"status": "MISSING"}
            continue
        m = v["metrics"]
        n = (m.get("trades") or {}).get("n_trades")
        out["configs"][cid] = {
            "status": "DONE", "cagr": m.get("cagr"), "maxdd": m.get("maxdd"), "sharpe": m.get("sharpe"),
            "n_trades": n,
            "enough_trades_for_primary_test": bool(n is not None and n >= g["min_holdout_trades"]),
        }
    for _, r in cmp_df.iterrows():
        out["comparisons"][r.comparison] = {
            "status": r.status, "cagr_diff_pp": None if pd.isna(r.get("cagr_diff_pp")) else float(r.cagr_diff_pp),
            "p_raw": None if pd.isna(r.get("p_raw")) else float(r.p_raw),
            "holm_reject": bool(r.get("holm_reject", False)) if pd.notna(r.get("holm_reject", np.nan)) else False,
            "fails": r.get("fails") if isinstance(r.get("fails"), list) else [],
            "trades_holdout_left": None if pd.isna(r.get("trades_left", np.nan)) else int(r.trades_left),
            "trades_holdout_right": None if pd.isna(r.get("trades_right", np.nan)) else int(r.trades_right),
        }
    return out


def main(run_root, repo_dir=None):
    run_root = Path(run_root)
    a, b, c, c2 = load_all(run_root)
    vt = variant_table(a, b, c, run_root)
    cmp_df = pd.read_csv(run_root / "comparisons.csv")
    result = {
        "research_id": "JAKE-FIB-TIMEFRAME-20261020-v0.1",
        "generated_from": str(run_root),
        "plan_sha256": configs.PLAN_SHA256,
        "windows": {k: list(v) for k, v in configs.WINDOWS.items()},
        "holdout_tag": configs.HOLDOUT_TAG,
        "configs": {},
        "blocked": BLOCKED,
        "cancellation": a.get("cancellation"),
        "identity": a.get("identity"),
        "divergence": a.get("divergence"),
        "cost_a_line": c2,
        "comparisons": {r.comparison: {k: (None if isinstance(v, float) and pd.isna(v) else v)
                                       for k, v in r.to_dict().items() if k != "fails"} |
                        {"fails": r.get("fails") if isinstance(r.get("fails"), list) else []}
                        for _, r in cmp_df.iterrows()},
    }
    for cid, v in vt.items():
        m = dict(v["metrics"])
        result["configs"][cid] = {
            "line": v["line"], "status": "DONE",
            "cagr": m.get("cagr"), "total_return": m.get("total_return"), "maxdd": m.get("maxdd"),
            "sharpe": m.get("sharpe"), "sortino": m.get("sortino"), "ann_vol": m.get("ann_vol"),
            "avg_exposure": m.get("avg_exposure"), "time_in_market": m.get("time_in_market"),
            "n_days": m.get("n_days"), "years": m.get("years"),
            "trades": m.get("trades"), "yearly": v.get("yearly"),
            "entry": m.get("entry"), "stress": v.get("variants"),
            "n_rebalances": m.get("n_rebalances") or m.get("n_rebal_days"),
            "rebal_notional": m.get("rebal_notional"),
        }
    for cid in BLOCKED:
        result["configs"][cid] = {"line": LINE_OF[cid], "status": BLOCKED[cid]}
    (run_root / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))

    equity_long(run_root).to_csv(run_root / "equity_daily.csv", index=False, float_format="%.10g")
    tl = trace_long(run_root)
    if len(tl):
        tl.to_csv(run_root / "target_trace.csv", index=False, float_format="%.10g")
    tr = trades_long(run_root)
    if len(tr):
        tr.to_csv(run_root / "orders_trades.csv", index=False, float_format="%.10g")
    el = entry_long(run_root)
    if len(el):
        el.to_csv(run_root / "entry_experience_all.csv", index=False, float_format="%.10g")
    trials(vt, a, b, c).to_csv(run_root / "trials.csv", index=False, float_format="%.10g")
    acc = acceptance(vt, cmp_df, a, c2)
    (run_root / "acceptance.json").write_text(json.dumps(acc, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"configs": list(result["configs"]), "n_comparisons": len(cmp_df),
                      "files": [p.name for p in sorted(run_root.glob("*.json")) + sorted(run_root.glob("*.csv"))]},
                     indent=1, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
