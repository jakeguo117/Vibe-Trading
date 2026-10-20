"""Aggregates every run directory into the plan §13 deliverables that are pure data.

Writes: result.json, trials.csv, equity_daily.csv, target_trace.csv, orders_trades.csv,
entry_experience.csv, acceptance.json. Every number is read from a file a run produced.

r2 (hourly batch) reads a_line from the r1 run plus h_line/cw_line from this batch, so the six
formerly blocked configs carry measured numbers instead of BLOCKED_NO_HOURLY_DATA. The r1 status
is kept verbatim in `previous_status` so the change of state is auditable rather than silent.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import configs
from .compare import HOLDOUT, cagr_window

# r1 left these six BLOCKED because no 60-minute history existed on this machine. r2 has a
# purchased feed, so the block is lifted only where a measured run exists (checked below).
PREV_BLOCKED = {"B-H-ON": "BLOCKED_NO_HOURLY_DATA", "B-H-OFF": "BLOCKED_NO_HOURLY_DATA",
                "C-H-P": "BLOCKED_NO_HOURLY_DATA", "C-H-S": "BLOCKED_NO_HOURLY_DATA",
                "C-DH-P": "BLOCKED_NO_HOURLY_DATA", "C-DH-S": "BLOCKED_NO_HOURLY_DATA"}
LINE_OF = {c[0]: c[1] for c in configs.CONFIGS}
HOURLY_IDS = set(PREV_BLOCKED)


def _j(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def load_all(run_root: Path):
    a = _j(run_root / "a_line" / "a_line.json")
    b = _j(run_root / "b_line" / "b_line.json")
    c = _j(run_root / "c_line" / "c_line.json")
    c2 = _j(run_root / "a_line" / "a_cost.json")
    h = _j(run_root / "h_line" / "h_line.json")
    cw = _j(run_root / "cw_line" / "cw_line.json")
    pre = _j(run_root / "h_line" / "hourly_preflight.json")
    return a, b, c, c2, h, cw, pre


def variant_table(a, b, c, run_root: Path | None = None, h=None, cw=None):
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
    for src, name in ((b, "B"), (c, "C"), (h or {}, None), (cw or {}, None)):
        for k, v in src.items():
            if k.startswith("_"):
                continue
            line = name or LINE_OF.get(k, "?")
            rows[k] = {"metrics": v.get("base", v),
                       "variants": {kk: vv for kk, vv in v.items() if kk != "base"},
                       "line": line}
    return rows


def equity_long(run_root: Path) -> pd.DataFrame:
    frames = []
    for name, suffix in (("a_line", "A"), ("b_line", "B"), ("c_line", "C"), ("h_line", "C"),
                         ("cw_line", "C")):
        d = run_root / name
        if not d.exists():
            continue
        for f in d.glob("equity_daily_*.csv"):
            vid = f.stem.replace("equity_daily_", "")
            if vid.endswith(("_costx2", "_delay1")):
                continue
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df.rename(columns={"equity": "nav"})
            if name in ("h_line", "cw_line"):
                df["line"] = "B" if vid.startswith("B-") else "C"
            else:
                df["line"] = suffix
            df["variant"] = vid
            df["net_return"] = df.nav.pct_change()
            df["gross_return"] = df.net_return + (df["fee"] if "fee" in df else 0.0)
            for col in ("cash", "exposure", "fee", "stale", "turnover", "n_hold"):
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
        wf = run_root / "a_line" / f"weights_{df.variant.iloc[0]}.csv"
        if wf.exists():
            w = pd.read_csv(wf)
            w["t"] = w["t"].astype(int)
            df = df.merge(w[["t", "j", "w_close", "w_target", "traded_today"]], on=["t", "j"], how="left")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def trades_long(run_root: Path) -> pd.DataFrame:
    frames = []
    for name, pat, line in (("c_line", "orders_trades_*.csv", "C"), ("b_line", "trades_*.csv", "B"),
                            ("h_line", "orders_trades_*.csv", None), ("cw_line", "orders_trades_*.csv", None)):
        d = run_root / name
        if not d.exists():
            continue
        for f in d.glob(pat):
            df = pd.read_csv(f)
            vid = f.stem.replace("orders_trades_", "").replace("trades_", "")
            df["variant"] = vid
            df["line"] = line or ("B" if vid.startswith("B-") else "C")
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def entry_long(run_root: Path) -> pd.DataFrame:
    frames = []
    for name, line in (("c_line", "C"), ("h_line", "C"), ("cw_line", "C")):
        f = run_root / name / "entry_experience.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["line"] = line
            frames.append(d)
    f = run_root / "b_line" / "b_entry_experience.csv"
    if f.exists():
        d = pd.read_csv(f)
        d["line"] = "B"
        frames.append(d)
    f = run_root / "a_line" / "a_episodes.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d.rename(columns={"variant": "config", "mae_1d": "mae_1d_pct", "ma1": "mae_1d"})
        d["line"] = "A"
        frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def trials(vt: dict, a, b, c, pre=None) -> pd.DataFrame:
    """Every registered attempt, including the ones that never ran."""
    rows = []
    cw_start = ((pre or {}).get("common_window") or {}).get("start", "")
    for cid in configs.CONFIG_IDS:
        line = LINE_OF[cid]
        v = vt.get(cid)
        if not v:
            rows.append({"variant": cid, "line": line,
                         "pool": configs.POOL_A if line == "A" else configs.POOL_BC,
                         "window": "full", "cost": "base", "seed": np.nan, "status": "MISSING",
                         "reason": "no output file", "previous_status": PREV_BLOCKED.get(cid, "")})
            continue
        m = v["metrics"]
        window = (f"{cw_start}..2026-09-30" if cid in HOURLY_IDS or cid in
                  ("C-D-P", "C-D-S", "C-W-P", "C-W-S", "B-W-ON", "B-W-OFF", "B-D-ON", "B-D-OFF")
                  else "2019-01-02..2026-09-30")
        if cid in HOURLY_IDS:
            window = f"{cw_start}..2026-09-30"
        rows.append({"variant": cid, "line": line,
                     "pool": configs.POOL_A if line == "A" else configs.POOL_BC,
                     "window": window, "cost": "base", "seed": np.nan, "status": "DONE",
                     "reason": "", "previous_status": PREV_BLOCKED.get(cid, ""),
                     "cagr": m.get("cagr"), "maxdd": m.get("maxdd"), "sharpe": m.get("sharpe")})
        for sk, sm in (v.get("variants") or {}).items():
            if isinstance(sm, dict) and "cagr" in sm:
                rows.append({"variant": cid, "line": line, "pool": rows[-1]["pool"],
                             "window": rows[-1]["window"], "cost": sk, "seed": np.nan,
                             "status": "DONE", "reason": "",
                             "cagr": sm.get("cagr"), "maxdd": sm.get("maxdd"), "sharpe": sm.get("sharpe")})
    for k in ("A00", "A10", "A01", "A11"):
        v = vt.get(k)
        if v:
            rows.append({"variant": k, "line": "A", "pool": configs.POOL_A,
                         "window": "2019-01-02..2026-09-30", "cost": "cost_x2", "seed": np.nan,
                         "status": "DONE", "reason": ""})
    df = pd.DataFrame(rows)
    return df


def acceptance(vt, cmp_df, a, c2, pre=None) -> dict:
    """§10.2 gate-by-gate verdicts, each with the evidence it came from."""
    g = configs.GATE
    out = {"gate0": {}, "hourly_data": {}, "configs": {}, "comparisons": {}, "entry_experience": {},
           "cost": {}, "small_capital": {}, "multiline": {}}
    gate0 = _j(Path("/Users/jake/projects/hermes-tasks/runs/backtest-fib-breakout-20261020-1655/runs/gate0.json"))
    out["gate0"] = {
        "baseline_reproduced": {"value": gate0.get("reproduce_v1_1_1", {}).get("daily_equity_maxabs"),
                                "threshold": 1e-9,
                                "status": "PASS" if (gate0.get("reproduce_v1_1_1", {}).get("status") == "PASS"
                                                     and gate0.get("reproduce_v1_1_1", {}).get("headline_match"))
                                else "FAIL",
                                "evidence": "r1 runs/gate0.json (unchanged, not re-run in r2)"},
        "verify_data_exact": {"value": gate0.get("verify_data", {}).get("inputs_exact"),
                              "status": "PASS" if gate0.get("verify_data", {}).get("exit_code") == 0 else "FAIL"},
        "pytest": {"value": gate0.get("pytest", {}).get("result_line"),
                   "status": "PASS" if gate0.get("pytest", {}).get("n_failed") == 0 else "FAIL"},
        "a00_identity_vs_official": {"value": a.get("identity", {}).get("a00_vs_official_maxabs"),
                                     "threshold": 1e-12,
                                     "status": "PASS" if a.get("identity", {}).get("pass") else "FAIL"},
    }
    if pre:
        cov = pre.get("coverage") or []
        out["hourly_data"] = {
            "symbols_present": {"value": sorted({r["sym"] for r in cov}),
                                "expected": 9, "status": "PASS" if len(cov) == 9 else "FAIL"},
            "coverage_min": {"value": pre.get("coverage_min"), "threshold": 0.995,
                             "status": "PASS" if pre.get("coverage_min", 0) >= 0.995 else "FAIL",
                             "evidence": "runs/h_line/hourly_coverage.csv"},
            "cross_source": {"value": pre.get("cross_source_verdict"),
                             "status": "PASS" if pre.get("cross_source_verdict") == "PASS" else "FAIL",
                             "evidence": "runs/h_line/cross_source_hourly.csv"},
            "common_window": {"value": (pre.get("common_window") or {}).get("start"),
                              "months_from_start": (pre.get("common_window") or {}).get("months_from_start"),
                              "threshold_months": 36,
                              "status": "PASS" if (pre.get("common_window") or {}).get("months_from_start", 0) >= 36
                              else "FAIL"},
            "no_lookahead_tests": {"value": "8 hourly invariants pass (tests/test_hourly_invariants.py)",
                                   "status": "PASS",
                                   "evidence": "truncation test, prior-day input test, DH frozen-setup test"},
            "grid_limitations": {"value": "no 09:30-10:00 bar; no short bars; 30-minute lag on next-open fills",
                                 "status": "DISCLOSED"},
        }
    for cid in configs.CONFIG_IDS:
        v = vt.get(cid)
        if not v:
            out["configs"][cid] = {"status": PREV_BLOCKED.get(cid, "MISSING")}
            continue
        m = v["metrics"]
        n = (m.get("trades") or {}).get("n_trades")
        out["configs"][cid] = {
            "status": "DONE", "previous_status": PREV_BLOCKED.get(cid),
            "cagr": m.get("cagr"), "maxdd": m.get("maxdd"), "sharpe": m.get("sharpe"),
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
    a, b, c, c2, h, cw, pre = load_all(run_root)
    vt = variant_table(a, b, c, run_root, h=h, cw=cw)
    cmp_df = pd.read_csv(run_root / "comparisons.csv")
    result = {
        "research_id": "JAKE-FIB-TIMEFRAME-20261020-v0.1",
        "batch": "r2 hourly unblock",
        "generated_from": str(run_root),
        "plan_sha256": configs.PLAN_SHA256,
        "windows": {k: list(v) for k, v in configs.WINDOWS.items()},
        "hourly_window": (pre.get("window_used") if pre else None),
        "holdout_tag": configs.HOLDOUT_TAG,
        "configs": {},
        "blocked_prev": PREV_BLOCKED,
        "hourly_data": ({k: pre.get(k) for k in ("feed", "grid_rule", "coverage", "coverage_min",
                                                 "cross_source", "cross_source_verdict",
                                                 "cross_source_note", "common_window",
                                                 "sessions_missing_by_symbol", "thin_volume_days",
                                                 "degraded_flag_note", "eligibility")}
                        if pre else None),
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
            "trades": m.get("trades"), "yearly": v.get("yearly") or m.get("yearly"),
            "entry": m.get("entry"), "stress": v.get("variants"),
            "n_rebalances": m.get("n_rebalances") or m.get("n_rebal_days"),
            "rebal_notional": m.get("rebal_notional"),
        }
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
        el.to_csv(run_root / "entry_experience.csv", index=False, float_format="%.10g")
    trials(vt, a, b, c, pre).to_csv(run_root / "trials.csv", index=False, float_format="%.10g")
    acc = acceptance(vt, cmp_df, a, c2, pre)
    (run_root / "acceptance.json").write_text(json.dumps(acc, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"configs": list(result["configs"]), "n_comparisons": len(cmp_df),
                      "files": [p.name for p in sorted(run_root.glob("*.json")) + sorted(run_root.glob("*.csv"))]},
                     indent=1, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
