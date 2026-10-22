"""Metrics, benchmarks and block bootstrap (single implementation, shared)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calendar_days(dates, d0, d1):
    return max(1, int((dates[d1] - dates[d0]) / np.timedelta64(1, "D")))


def perf(eq: pd.DataFrame, dates, d0, d1, rf_daily=None):
    e = eq["equity"].to_numpy(float)
    r = np.diff(e) / e[:-1]
    r = np.concatenate([[0.0], r])
    n = len(r)
    cds = calendar_days(dates, d0, d1)
    yrs = cds / 365.25
    cagr = (e[-1] / e[0]) ** (1 / yrs) - 1 if e[-1] > 0 else -1.0
    peak = np.maximum.accumulate(e)
    dd = e / peak - 1
    mdd = float(dd.min())
    sd = float(np.std(r[1:], ddof=1)) if n > 2 else 0.0
    rf = np.zeros(n) if rf_daily is None else np.asarray(rf_daily, float)
    ex = r[1:] - rf[1:]
    sharpe = math.sqrt(252) * float(np.mean(ex) / np.std(ex, ddof=1)) if np.std(ex, ddof=1) > 0 else float("nan")
    sharpe0 = math.sqrt(252) * float(np.mean(r[1:]) / sd) if sd > 0 else float("nan")
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    sortino = math.sqrt(252) * float(np.mean(ex) / downside) if downside > 0 else float("nan")
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    ann_vol = sd * math.sqrt(252)
    held = (eq["shares"].to_numpy() != 0)
    ytd = float("nan")
    seg_dates = dates[d0:d1 + 1]
    first26 = next((i for i, x in enumerate(seg_dates) if x >= np.datetime64("2026-01-01")), None)
    if first26 is not None and first26 > 0:
        ytd = float(e[-1] / e[first26 - 1] - 1)
    return {
        "final_equity": float(e[-1]),
        "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "sharpe0": sharpe0,
        "sortino": sortino, "calmar": calmar, "ann_vol": ann_vol,
        "exposure_days": float(held.mean()),
        "days": int(n),
        "cal_days": int(cds),
        "ytd_2026": ytd,
    }


def trade_stats(tr: pd.DataFrame):
    if tr is None or len(tr) == 0:
        return {"n_trades": 0}
    R = tr["R"].to_numpy(float)
    wins = R[R > 0]; losses = R[R <= 0]
    gp = tr.loc[tr["pnl"] > 0, "pnl"].sum()
    gl = -tr.loc[tr["pnl"] <= 0, "pnl"].sum()
    order = tr.sort_values("exit_day")
    mx = cur = 0
    for p in order["pnl"]:
        cur = cur + 1 if p <= 0 else 0
        mx = max(mx, cur)
    srt = np.sort(tr["pnl"].to_numpy(float))[::-1]
    top5 = srt[:5].sum() / srt.sum() if srt.sum() > 0 else float("nan")
    return {
        "n_trades": int(len(tr)),
        "win_rate": float((R > 0).mean()),
        "avg_R": float(np.mean(R)), "med_R": float(np.median(R)),
        "pf": float(gp / gl) if gl > 0 else float("nan"),
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan"),
        "avg_hold": float(tr["hold_days"].mean()),
        "sum_R": float(R.sum()),
        "unique_syms": int(tr["j"].nunique()),
        "max_consec_loss": int(mx),
        "top5_share": float(top5),
        "partials": int(tr["partial"].sum()) if "partial" in tr else 0,
        "stop_exits": int((tr["exit_reason"] == "stop").sum()) if "exit_reason" in tr else 0,
        "gap_stop_exits": int((tr["exit_reason"] == "gap_stop").sum()) if "exit_reason" in tr else 0,
    }


def benchmark(kit, sym, weight, d0, d1, e0=16000.0, cfg=None, monthly=False):
    """Buy-and-hold (weight=1) or fixed-weight monthly rebalanced benchmark."""
    j = int(np.where(kit.syms == sym)[0][0])
    o, c, div = kit.o[:, j], kit.c[:, j], kit.div[:, j]
    adjf = kit.adjf[:, j]
    slip = 0.0 if cfg is None else cfg.slip_bps / 1e4
    comm = (lambda q: max(1.0, 0.005 * q)) if cfg is not None else (lambda q: 0.0)
    cash = e0
    q = 0
    rows = []
    last_month = None
    for d in range(d0, d1 + 1):
        if not np.isfinite(o[d]):
            rows.append((d, cash + q * c[d - 1] if q else cash, cash, q))
            continue
        if d == d0:
            q = int((weight * e0) // (o[d] * (1 + slip)))
            if q > 0:
                cash -= q * o[d] * (1 + slip) + comm(q)
        elif monthly:
            m = pd.Timestamp(kit.dates[d]).month
            if m != last_month and d > d0:
                E = cash + q * o[d]
                tq = int((weight * E) // (o[d] * (1 + slip)))
                dq = tq - q
                if dq > 0:
                    cost = dq * o[d] * (1 + slip) + comm(dq)
                    if cost <= cash:
                        cash -= cost; q += dq
                elif dq < 0:
                    cash += (-dq) * o[d] * (1 - slip) - comm(-dq); q += dq
        if np.isfinite(div[d]) and div[d] > 0 and q:
            cash += div[d] * adjf[d] * q
        last_month = pd.Timestamp(kit.dates[d]).month
        rows.append((d, cash + q * c[d], cash, q))
    return pd.DataFrame(rows, columns=["d", "equity", "cash", "shares"])


def block_bootstrap(x, y=None, block=20, reps=2000, seed=20261022):
    """Paired moving-block bootstrap on daily series (x - y if y given)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < block * 2:
        return {"mean": float(np.nanmean(x)), "lo": float("nan"), "hi": float("nan"), "valid": 0}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    out = np.empty(reps)
    for i in range(reps):
        s = rng.choice(starts, nb, replace=True)
        idx = np.concatenate([np.arange(k, k + block) for k in s])[:n]
        if y is None:
            out[i] = np.mean(x[idx])
        else:
            out[i] = np.mean(x[idx] - np.asarray(y)[idx])
    return {"mean": float(np.nanmean(x if y is None else x - y)),
            "lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "valid": reps}
