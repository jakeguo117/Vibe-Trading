"""The 15 pre-registered comparisons: paired block bootstrap, Holm correction, §10.2 gates.

Reads the equity files produced by adriver / bdriver / cdriver, writes comparisons.csv and
returns the rows for result.json. Nothing here re-ranks configs: the comparison family is the
one frozen in configs.COMPARISONS, and unrun members keep p=1.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

from . import configs
from .metrics import perf

HOLDOUT = configs.WINDOWS["holdout"]


STRESS_SUFFIX = ("_costx2", "_delay1")

# SWITCH (hourly batch): h_line holds the six formerly-blocked configs, cw_line the W/D controls
# re-run on the hourly common window. Later directories win, so the B/C family in the r2
# comparison table is uniformly measured from one common start date (§3.1).
DIRS = ("a_line", "b_line", "c_line", "h_line", "cw_line")


def load_equity(run_root: Path, stress: str | None = None) -> dict:
    """variant -> daily equity Series.

    stress=None reads the base runs; stress="costx2" / "delay1" reads that stress variant
    (A-line files carry the suffix, B/C/H-line files are written by their drivers the same way).
    """
    eq = {}
    for name in DIRS:
        d = run_root / name
        if not d.exists():
            continue
        for f in d.glob("equity_daily_*.csv"):
            vid = f.stem.replace("equity_daily_", "")
            has = next((s for s in STRESS_SUFFIX if vid.endswith(s)), None)
            if (stress is None and has) or (stress is not None and has != f"_{stress}"):
                continue
            if has:
                vid = vid[: -len(has)]
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            col = "equity" if "equity" in df else df.columns[0]
            eq[vid] = df[col].astype(float)
    return eq


def holdout_round_trips(run_root: Path) -> dict:
    """§8.1 completed trades inside the holdout window, per variant."""
    out = {}
    h0, h1 = pd.Timestamp(HOLDOUT[0]), pd.Timestamp(HOLDOUT[1])
    for name in DIRS:
        d = run_root / name
        if not d.exists():
            continue
        for f in d.glob("trades_*.csv"):
            vid = f.stem.replace("trades_", "")
            t = pd.read_csv(f)
            if "entry_date" not in t:
                continue
            e = pd.to_datetime(t.entry_date)
            out[vid] = int(((e >= h0) & (e <= h1)).sum())
        for f in d.glob("orders_trades_*.csv"):
            vid = f.stem.replace("orders_trades_", "")
            t = pd.read_csv(f)
            e = pd.to_datetime(t.entry_date)
            out[vid] = int(((e >= h0) & (e <= h1)).sum())
    # A line: a completed round trip is one "enter" event for that code (§8.1 episode counting)
    for f in (run_root / "a_line").glob("events_*.csv"):
        vid = f.stem.replace("events_", "")
        e = pd.read_csv(f, parse_dates=["date"])
        e = e[(e.date >= h0) & (e.date <= h1)]
        out[vid] = int((e.what == "enter").sum())
    return out


def load_metrics(run_root: Path) -> dict:
    out = {}
    for name, fn in ((run_root / "a_line", "a_line.json"), (run_root / "b_line", "b_line.json"),
                     (run_root / "c_line", "c_line.json"), (run_root / "h_line", "h_line.json"),
                     (run_root / "cw_line", "cw_line.json")):
        if not (name / fn).exists():
            continue
        j = json.loads((name / fn).read_text())
        for k, v in j.items():
            if k.startswith("_"):
                continue
            if fn == "a_line.json":
                out[k] = v if "metrics" in v else v
            else:
                out.setdefault(k, v)
    return out


def rets(eq: pd.Series, start, end) -> pd.Series:
    s = eq.loc[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
    return s.pct_change().dropna()


def cagr_window(eq: pd.Series, start, end) -> float:
    s = eq.loc[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
    if len(s) < 3:
        return float("nan")
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1.0)


def block_bootstrap(d: np.ndarray, block=20, n=10000, seed=20261020):
    T = len(d)
    if T < 5:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p": 1.0}
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(T / block))
    starts = rng.integers(0, T, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    idx = idx.reshape(n, -1)[:, :T]
    obs = float(d.mean())
    boot = d[idx].mean(axis=1)
    dm = d - obs
    null = dm[idx].mean(axis=1)
    p = float((1 + np.sum(null >= obs)) / (n + 1))
    return {"mean": obs, "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)), "p": p,
            "block": block, "n": n, "seed": seed}


def holm(pvals: dict, alpha=0.05) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, reject_so_far = {}, True
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        rej = reject_so_far and (p <= thr)
        out[k] = {"p": p, "threshold": thr, "reject": bool(rej)}
        reject_so_far = rej
    return out


def gate_row(cid, left, right, eqs, cfgs, cost2=None, trades=None, segments=3):
    """§10.2 gates for left minus right on the holdout window."""
    has = (left in eqs) and (right in eqs)
    row = {"comparison": cid, "left": left, "right": right,
           "line": next((c[1] for c in configs.COMPARISONS if c[0] == cid), "?")}
    if not has:
        row.update(status="BLOCKED_MISSING_VARIANT", why="one or both variants not runnable "
                   "(hourly data absent or config not in batch)")
        return row
    rl, rr = rets(eqs[left], *HOLDOUT), rets(eqs[right], *HOLDOUT)
    j = rl.index.intersection(rr.index)
    rl, rr = rl.loc[j], rr.loc[j]
    d = np.log1p(rl.values) - np.log1p(rr.values)
    bs = block_bootstrap(d)
    cagr_l, cagr_r = cagr_window(eqs[left], *HOLDOUT), cagr_window(eqs[right], *HOLDOUT)
    ml = perf(eqs[left].loc[(eqs[left].index >= pd.Timestamp(HOLDOUT[0])) & (eqs[left].index <= pd.Timestamp(HOLDOUT[1]))])
    mr = perf(eqs[right].loc[(eqs[right].index >= pd.Timestamp(HOLDOUT[0])) & (eqs[right].index <= pd.Timestamp(HOLDOUT[1]))])
    # three equal time segments of the holdout
    k = len(d) // segments
    seg = [float(d[i * k:(i + 1) * k].sum()) for i in range(segments)]
    row.update({
        "n_days": int(len(d)), "mean_log_diff_daily": bs["mean"],
        "cagr_left": cagr_l, "cagr_right": cagr_r, "cagr_diff_pp": (cagr_l - cagr_r) * 100,
        "sharpe_left": ml["sharpe"], "sharpe_right": mr["sharpe"],
        "maxdd_left": ml["maxdd"], "maxdd_right": mr["maxdd"],
        "maxdd_diff_pp": (ml["maxdd"] - mr["maxdd"]) * 100,
        "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"], "p_raw": bs["p"],
        "segment_cum_logret": seg, "segments_positive": int(sum(1 for x in seg if x > 0)),
        "trades_left": (trades or {}).get(left), "trades_right": (trades or {}).get(right),
    })
    g = configs.GATE
    fails = []
    if not (row["cagr_diff_pp"] >= g["cagr_improvement_pp"] * 100):
        fails.append("cagr_gain_below_2pp")
    if not (row["sharpe_left"] >= row["sharpe_right"]):
        fails.append("sharpe_worse_than_control")
    if row["maxdd_diff_pp"] < -g["maxdd_worsen_pp_max"] * 100:
        fails.append("maxdd_worse_by_more_than_3pp")
    if row["maxdd_left"] < g["maxdd_floor"]:
        fails.append("maxdd_below_-30pct")
    if row["segments_positive"] < g["holdout_segments_min_positive"]:
        fails.append("positive_in_fewer_than_2_of_3_segments")
    ntr = min([x for x in (row["trades_left"], row["trades_right"]) if x is not None], default=None)
    if ntr is not None and ntr < g["min_holdout_trades"]:
        row["insufficient_trades"] = True
    if cost2 is not None:
        row["cagr_diff_pp_cost_x2"] = cost2
        if not (cost2 > 0):
            fails.append("cost_x2_turns_diff_nonpositive")
    row["fails"] = fails
    if row.get("insufficient_trades"):
        row["status"] = "INSUFFICIENT_TRADES_EVIDENCE"
    elif fails:
        row["status"] = "FAIL"
    else:
        row["status"] = "PASS_PENDING_MULTIPLICITY"
    return row


def main(run_root, out_csv):
    run_root = Path(run_root)
    eqs = load_equity(run_root)
    eqs2 = load_equity(run_root, stress="costx2")
    met = load_metrics(run_root)
    trades = holdout_round_trips(run_root)
    rows = []
    for cid, line in configs.COMPARISONS:
        cands = [v for v in eqs if cid.startswith(v + "-")]
        left = max(cands, key=len) if cands else cid.split("-")[0]
        right = cid[len(left) + 1:]
        c2 = None
        if left in eqs2 and right in eqs2:
            c2 = (cagr_window(eqs2[left], *HOLDOUT) - cagr_window(eqs2[right], *HOLDOUT)) * 100
        rows.append(gate_row(cid, left, right, eqs, configs, cost2=c2, trades=trades))
    df = pd.DataFrame(rows)
    hm = holm({r["comparison"]: (r.get("p_raw", 1.0) if pd.notna(r.get("p_raw", np.nan)) else 1.0)
               for r in rows})
    df["p_holm_threshold"] = [hm[c]["threshold"] for c in df.comparison]
    df["holm_reject"] = [hm[c]["reject"] for c in df.comparison]
    for r in rows:
        h = hm[r["comparison"]]
        if r.get("status") == "PASS_PENDING_MULTIPLICITY":
            r["status"] = "PASS" if h["reject"] else "FAIL_NOT_SIGNIFICANT_AFTER_HOLM"
    df["status"] = [r["status"] for r in rows]
    df.to_csv(out_csv, index=False, float_format="%.10g")
    return df, hm


if __name__ == "__main__":
    import sys
    df, hm = main(sys.argv[1], sys.argv[2])
    print(df[["comparison", "status", "cagr_diff_pp", "p_raw", "holm_reject"]].to_string(index=False))
