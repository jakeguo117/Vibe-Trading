"""Account and trade metrics, implementing plan §9.1/§9.2 formulas.

Deliberately NOT importing upleg_bt.metrics: the plan fixes its own Sortino
(分母含全部交易日, sqrt(mean(min(r,0)^2))) and its own annualisation, and says
old report numbers must not be compared directly when the formula differs.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def perf(eq: pd.Series, exposure: pd.Series | None = None) -> dict:
    """eq: daily equity index (monotone dates), level series. Returns net metrics."""
    eq = eq.astype(float)
    r = eq.pct_change().fillna(0.0).values
    e0, e1 = float(eq.iloc[0]), float(eq.iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (e1 / e0) ** (1 / years) - 1.0 if e0 > 0 and e1 > 0 else float("nan")
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    neg = np.minimum(r, 0.0)
    dsd = math.sqrt(float((neg ** 2).mean()))          # plan §9.1 unified Sortino denominator
    peak = np.maximum.accumulate(eq.values)
    dd = eq.values / peak - 1.0
    maxdd = float(dd.min())
    # longest drawdown recovery (calendar days from peak to recovery, censored at end)
    rec, longest, cur_peak_idx = [], 0.0, 0
    under = dd < 0
    i = 0
    while i < len(dd):
        if under[i]:
            j = i
            while j < len(dd) and under[j]:
                j += 1
            days = (eq.index[j] - eq.index[i - 1]).days if j < len(dd) else (eq.index[-1] - eq.index[i - 1]).days
            rec.append(days); longest = max(longest, days); i = j
        else:
            i += 1
    out = {
        "total_return": float(e1 / e0 - 1.0),
        "cagr": float(cagr),
        "ann_vol": float(sd * math.sqrt(TRADING_DAYS)),
        "sharpe": float(mu / sd * math.sqrt(TRADING_DAYS)) if sd > 1e-12 else float("nan"),
        "sortino": float(mu * math.sqrt(TRADING_DAYS) / dsd) if dsd > 1e-12 else float("nan"),
        "maxdd": maxdd,
        "calmar": float(cagr / abs(maxdd)) if maxdd < 0 else float("nan"),
        "n_days": int(len(eq)),
        "years": float(years),
        "maxdd_recovery_days": float(longest),
        "n_drawdown_episodes": int(len(rec)),
    }
    if exposure is not None:
        out["avg_exposure"] = float(exposure.mean())
        out["time_in_market"] = float((exposure > 0).mean())
    return out


def yearly(eq: pd.Series) -> dict:
    y = eq.groupby(eq.index.year).agg(["first", "last"])
    return {int(k): float(v["last"] / v["first"] - 1.0) for k, v in y.iterrows()}


def slice_window(eq: pd.Series, start, end):
    return eq.loc[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]


def trade_stats(trades: pd.DataFrame) -> dict:
    """trades: one row per completed round trip with column `net_ret` (account-level)."""
    if trades is None or len(trades) == 0:
        return {"n_trades": 0}
    x = trades.net_ret.astype(float).values
    win, loss = x[x > 0], x[x <= 0]
    gp, gl = float(win.sum()), float(-loss.sum())
    return {
        "n_trades": int(len(x)),
        "win_rate": float(len(win) / len(x)),
        "avg_net": float(x.mean()),
        "median_net": float(np.median(x)),
        "avg_win": float(win.mean()) if len(win) else 0.0,
        "avg_loss": float(loss.mean()) if len(loss) else 0.0,
        "profit_factor": float(gp / gl) if gl > 1e-12 else float("inf"),
        "worst": float(x.min()),
        "best": float(x.max()),
        "expectancy": float(x.mean()),
    }


def mae_mfe_stats(rows: pd.DataFrame) -> dict:
    """rows: per-trade entry-experience table with mae (<=0), mfe (>=0), and loss L=max(0,-mae)."""
    if rows is None or len(rows) == 0:
        return {"n": 0}
    L = np.maximum(0.0, -rows.mae.astype(float).values)
    out = {
        "n": int(len(rows)),
        "mae_median": float(np.median(rows.mae)),
        "mae_p90_loss": float(np.quantile(L, 0.90)),
        "mae_p95_loss": float(np.quantile(L, 0.95)),
        "mfe_median": float(np.median(rows.mfe)),
        "mfe_p90": float(np.quantile(rows.mfe.astype(float).values, 0.90)),
    }
    if "mae_1d_atr" in rows:
        v = rows.mae_1d_atr.dropna().astype(float).values
        out["n_mae_1d"] = int(len(v))
        if len(v):
            out["mae_1d_atr_median"] = float(np.median(v))
            out["adverse_rate_1d"] = float((v <= -0.5).mean())
    if "net_1d" in rows:
        out["net_1d_mean"] = float(rows.net_1d.dropna().astype(float).mean())
    if "net_5d" in rows:
        out["net_5d_mean"] = float(rows.net_5d.dropna().astype(float).mean())
    return out
