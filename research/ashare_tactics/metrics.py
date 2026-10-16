"""Performance statistics. Risk-free rate is 0. 252 trading days.

Sharpe uses the sample standard deviation (ddof=1). Sortino uses the
zero-target downside root-mean-square over every return, matching
``backtest.metrics.calc_metrics``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_tactics.constants import INITIAL_CASH, TRADING_DAYS


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def annualized_return(equity: pd.Series, initial: float = INITIAL_CASH) -> float:
    if len(equity) == 0 or initial <= 0:
        return 0.0
    growth = float(equity.iloc[-1] / initial)
    if growth <= 0:
        return -1.0
    return float(growth ** (TRADING_DAYS / len(equity)) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe_ratio(equity: pd.Series) -> float:
    rets = daily_returns(equity)
    if len(rets) < 2:
        return 0.0
    vol = float(rets.std(ddof=1))
    if vol < 1e-12:
        return 0.0
    return float(rets.mean() / vol * np.sqrt(TRADING_DAYS))


def sortino_ratio(equity: pd.Series) -> float:
    rets = daily_returns(equity)
    if len(rets) < 2:
        return 0.0
    downside = rets.to_numpy(dtype=float)
    downside = downside[downside < 0]
    if len(downside) == 0:
        return 0.0
    deviation = float(np.sqrt((downside ** 2).sum() / len(rets)))
    if deviation < 1e-12:
        return 0.0
    return float(rets.mean() / deviation * np.sqrt(TRADING_DAYS))


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades is None or len(trades) == 0:
        return {"win_rate": 0.0, "avg_holding_days": 0.0, "trade_count": 0}
    wins = float((trades["pnl"] > 0).mean())
    hold = float(trades["hold_days"].mean())
    return {"win_rate": wins, "avg_holding_days": hold, "trade_count": int(len(trades))}


def summarize(name: str, equity: pd.Series, trades: pd.DataFrame, initial: float = INITIAL_CASH) -> dict:
    stats = trade_stats(trades)
    final = float(equity.iloc[-1]) if len(equity) else initial
    return {
        "strategy": name,
        "annualized_return": annualized_return(equity, initial),
        "max_drawdown": max_drawdown(equity),
        "sortino": sortino_ratio(equity),
        "sharpe": sharpe_ratio(equity),
        "win_rate": stats["win_rate"],
        "avg_holding_days": stats["avg_holding_days"],
        "trade_count": stats["trade_count"],
        "total_return": final / initial - 1.0 if initial else 0.0,
        "final_equity": final,
    }


def yearly_returns(equity: pd.Series, initial: float = INITIAL_CASH) -> pd.Series:
    """Calendar-year return from the previous year-end equity (or the start)."""
    if len(equity) == 0:
        return pd.Series(dtype=float)
    year_end = equity.groupby(equity.index.year).last()
    base = year_end.shift(1)
    first_year = year_end.index[0]
    base.loc[first_year] = initial
    return year_end / base - 1.0


def slope_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Trade-level split of tactic 5 by MA20 slope at entry."""
    labels = {1: "向上", 0: "走平", -1: "向下"}
    rows = []
    if trades is None or len(trades) == 0 or "slope" not in trades.columns:
        frame = pd.DataFrame(columns=["slope", "trade_count", "win_rate", "avg_return", "avg_holding_days", "total_pnl"])
        return frame
    for key, label in labels.items():
        part = trades[trades["slope"] == key]
        rows.append(
            {
                "slope": label,
                "trade_count": int(len(part)),
                "win_rate": float((part["pnl"] > 0).mean()) if len(part) else 0.0,
                "avg_return": float(part["ret"].mean()) if len(part) else 0.0,
                "avg_holding_days": float(part["hold_days"].mean()) if len(part) else 0.0,
                "total_pnl": float(part["pnl"].sum()) if len(part) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def benchmark_equity(close: pd.Series, calendar: pd.DatetimeIndex, initial: float = INITIAL_CASH) -> pd.Series:
    """Buy-and-hold of a price index, normalised to ``initial`` on the first session."""
    aligned = close.reindex(calendar).ffill().dropna()
    if len(aligned) == 0:
        return pd.Series(initial, index=calendar)
    aligned = aligned.reindex(calendar).ffill()
    return initial * aligned / float(aligned.iloc[0])
