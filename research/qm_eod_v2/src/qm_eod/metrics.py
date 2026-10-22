"""Metrics wrappers for multi-position QM-EOD-v2."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .metrics_base import calendar_days, block_bootstrap, benchmark  # noqa: F401


def perf(eq: pd.DataFrame, dates, d0, d1, rf_daily=None):
    e = eq["equity"].to_numpy(float)
    r = np.diff(e) / np.maximum(e[:-1], 1e-12)
    r = np.concatenate([[0.0], r])
    n = len(r)
    cds = calendar_days(dates, d0, d1)
    yrs = cds / 365.25
    cagr = (e[-1] / e[0]) ** (1 / yrs) - 1 if e[-1] > 0 and e[0] > 0 else -1.0
    peak = np.maximum.accumulate(e)
    dd = e / peak - 1
    mdd = float(dd.min())
    sd = float(np.std(r[1:], ddof=1)) if n > 2 else 0.0
    rf = np.zeros(n) if rf_daily is None else np.asarray(rf_daily, float)
    ex = r[1:] - rf[1:len(r)]
    sharpe = math.sqrt(252) * float(np.mean(ex) / np.std(ex, ddof=1)) if np.std(ex, ddof=1) > 0 else float("nan")
    sharpe0 = math.sqrt(252) * float(np.mean(r[1:]) / sd) if sd > 0 else float("nan")
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    sortino = math.sqrt(252) * float(np.mean(ex) / downside) if downside > 0 else float("nan")
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    ann_vol = sd * math.sqrt(252)
    if "n_pos" in eq.columns:
        exposure = float((eq["n_pos"].to_numpy() > 0).mean())
        avg_pos = float(eq["n_pos"].mean())
    else:
        exposure = float((eq["shares"].to_numpy() != 0).mean())
        avg_pos = float("nan")
    return {
        "final_equity": float(e[-1]),
        "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "sharpe0": sharpe0,
        "sortino": sortino, "calmar": calmar, "ann_vol": ann_vol,
        "exposure_days": exposure, "avg_positions": avg_pos,
        "days": int(n), "cal_days": int(cds),
    }


def trade_stats(tr: pd.DataFrame):
    if tr is None or len(tr) == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "avg_R": float("nan"),
                "med_R": float("nan"), "pf": float("nan"), "sum_R": 0.0,
                "avg_hold": float("nan"), "unique_syms": 0, "top5_share": float("nan"),
                "top10_share": float("nan")}
    R = tr["R"].to_numpy(float)
    wins = R[np.isfinite(R) & (R > 0)]
    losses = R[np.isfinite(R) & (R <= 0)]
    gp = tr.loc[tr["pnl"] > 0, "pnl"].sum()
    gl = -tr.loc[tr["pnl"] <= 0, "pnl"].sum()
    srt = np.sort(tr["pnl"].to_numpy(float))[::-1]
    tot = srt.sum()
    top5 = srt[:5].sum() / tot if tot > 0 else float("nan")
    top10 = srt[:10].sum() / tot if tot > 0 else float("nan")
    return {
        "n_trades": int(len(tr)),
        "win_rate": float(np.nanmean(R > 0)) if len(R) else float("nan"),
        "avg_R": float(np.nanmean(R)), "med_R": float(np.nanmedian(R)),
        "pf": float(gp / gl) if gl > 0 else float("nan"),
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan"),
        "avg_hold": float(tr["hold_days"].mean()),
        "sum_R": float(np.nansum(R)),
        "unique_syms": int(tr["j"].nunique()),
        "top5_share": float(top5),
        "top10_share": float(top10),
        "partials": int(tr["partial"].sum()) if "partial" in tr else 0,
        "stop_exits": int((tr["exit_reason"] == "stop").sum()) if "exit_reason" in tr else 0,
        "gap_stop_exits": int((tr["exit_reason"] == "gap_stop").sum()) if "exit_reason" in tr else 0,
        "n_bo": int((tr["style"] == "bo").sum()) if "style" in tr else 0,
        "n_ep": int((tr["style"] == "ep").sum()) if "style" in tr else 0,
    }


def signal_layer_stats(tr: pd.DataFrame, n_signals: int, n_fills: int):
    """Signal-layer (per-trade R) separate from account CAGR."""
    ts = trade_stats(tr)
    exec_rate = (n_fills / n_signals) if n_signals > 0 else float("nan")
    return {
        **{f"sig_{k}": v for k, v in ts.items()},
        "n_signals": int(n_signals),
        "n_fills": int(n_fills),
        "execution_rate": float(exec_rate),
    }


def year_segments():
    return {
        "FULL": ("2015-01-02", "2025-12-31"),
        "Y2015_19": ("2015-01-02", "2019-12-31"),
        "Y2020_21": ("2020-01-02", "2021-12-31"),
        "Y2022": ("2022-01-03", "2022-12-30"),
        "Y2023_25": ("2023-01-03", "2025-12-31"),
        "CONT": ("2016-01-04", "2026-10-20"),  # compare to qm_research_001
    }
