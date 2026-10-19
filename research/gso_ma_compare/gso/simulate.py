"""Friday decision, delayed execution, fractional shares, 10 bp per side, no leverage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gso.config import (
    CONFIRM_WEEKS,
    COST_RATE,
    DAILY_BARS_PER_WEEK,
    EWMA_SPAN,
    SLOPE_LAG_WEEKS,
    SYMBOLS,
    TRADING_DAYS_PER_YEAR,
    WEEKS_PER_YEAR,
)
from gso.indicators import (
    ewma_ann_vol,
    fib_stage_frame,
    following_week_execution,
    participation,
    sma_upleg,
    target_weights,
    to_weekly,
    wilder_atr,
)


def apply_targets(
    cash: float,
    shares: dict[str, float],
    open_px: dict[str, float],
    target_w: dict[str, float],
    symbols: tuple[str, ...],
    cost_rate: float = COST_RATE,
) -> tuple[float, dict[str, float]]:
    """Rebalance at the open to ``target_w`` of pre-trade equity. Costs come out of cash.

    If costs would push cash below zero, scale the target down with a fixed 50-step search.
    """
    equity = cash + sum(shares[s] * open_px[s] for s in symbols)
    if equity <= 0:
        return cash, dict(shares)

    def outcome(scale: float) -> tuple[float, dict[str, float]]:
        new_shares = {}
        invested = 0.0
        cost = 0.0
        for symbol in symbols:
            price = open_px[symbol]
            target_shares = (scale * target_w[symbol] * equity) / price
            new_shares[symbol] = target_shares
            invested += target_shares * price
            cost += abs(target_shares - shares[symbol]) * price * cost_rate
        return equity - invested - cost, new_shares

    new_cash, new_shares = outcome(1.0)
    if new_cash >= -1e-12:
        return max(new_cash, 0.0), new_shares
    best_cash, best_shares = outcome(0.0)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        cash_mid, shares_mid = outcome(mid)
        if cash_mid >= -1e-12:
            lo = mid
            best_cash, best_shares = cash_mid, shares_mid
        else:
            hi = mid
    return max(best_cash, 0.0), best_shares


def simulate(
    calendar: pd.DatetimeIndex,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    order_w: pd.DataFrame,
    order_part: pd.DataFrame,
    symbols: tuple[str, ...] = SYMBOLS,
    cost_rate: float = COST_RATE,
) -> dict:
    """Walk daily closes. Orders execute on their index date at the open."""
    cash = 1.0
    shares = {s: 0.0 for s in symbols}
    part = {s: 0.0 for s in symbols}
    navs = []
    parts = []
    episodes: list[dict] = []
    open_since = {s: None for s in symbols}
    max_gross = 0.0
    order_index = set(pd.Timestamp(x) for x in order_w.index)

    for day in calendar:
        day = pd.Timestamp(day)
        open_px = {s: float(opens.at[day, s]) for s in symbols}
        close_px = {s: float(closes.at[day, s]) for s in symbols}
        if day in order_index:
            weights = {s: float(order_w.at[day, s]) for s in symbols}
            cash, shares = apply_targets(cash, shares, open_px, weights, symbols, cost_rate)
            equity_open = cash + sum(shares[s] * open_px[s] for s in symbols)
            if equity_open > 0:
                gross = sum(shares[s] * open_px[s] for s in symbols) / equity_open
                max_gross = max(max_gross, gross)
            new_part = {s: float(order_part.at[day, s]) for s in symbols}
            for symbol in symbols:
                old = part[symbol]
                new = new_part[symbol]
                if old == 0.0 and new > 0.0:
                    open_since[symbol] = day
                elif old > 0.0 and new == 0.0 and open_since[symbol] is not None:
                    episodes.append(
                        {"symbol": symbol, "entry": open_since[symbol], "exit": day, "open": False}
                    )
                    open_since[symbol] = None
            part = new_part
        nav = cash + sum(shares[s] * close_px[s] for s in symbols)
        navs.append(nav)
        parts.append({s: part[s] for s in symbols})

    last = pd.Timestamp(calendar[-1])
    for symbol in symbols:
        if open_since[symbol] is not None:
            episodes.append(
                {"symbol": symbol, "entry": open_since[symbol], "exit": last, "open": True}
            )
    return {
        "nav": pd.Series(navs, index=pd.DatetimeIndex(calendar), name="nav"),
        "participation": pd.DataFrame(parts, index=pd.DatetimeIndex(calendar)),
        "episodes": episodes,
        "max_gross": max_gross,
    }


def _feature_tables(daily_map: dict[str, pd.DataFrame], freq: str, length: int, confirm: int, slope_lag: int):
    weekly = {}
    for symbol, daily in daily_map.items():
        bars = to_weekly(daily)
        bars["atr"] = wilder_atr(bars["High"], bars["Low"], bars["Close"])
        fib = fib_stage_frame(bars["High"], bars["Low"], bars["Close"], bars["atr"])
        bars = bars.join(fib)
        weekly[symbol] = bars

    if freq == "W":
        signal = {}
        for symbol, bars in weekly.items():
            up = sma_upleg(bars["Close"], length, confirm, slope_lag)
            vol = ewma_ann_vol(bars["Close"], EWMA_SPAN, WEEKS_PER_YEAR)
            frame = pd.DataFrame(
                {
                    "in_upleg": up["in_upleg"].to_numpy(),
                    "halve": up["halve"].to_numpy(),
                    "stage": bars["fib_stage"].to_numpy(),
                    "vol": vol.to_numpy(),
                    "session": bars["session"].to_numpy(),
                },
                index=bars.index,
            )
            signal[symbol] = frame
        return signal, "W"

    signal = {}
    for symbol, daily in daily_map.items():
        bars = weekly[symbol]
        up = sma_upleg(daily["Close"], length, confirm, slope_lag)
        vol = ewma_ann_vol(daily["Close"], EWMA_SPAN, TRADING_DAYS_PER_YEAR)
        stage = (
            pd.Series(bars["fib_stage"].to_numpy(), index=pd.DatetimeIndex(bars["session"]))
            .groupby(level=0)
            .last()
        )
        stage = stage.reindex(daily.index, method="ffill")
        frame = pd.DataFrame(
            {
                "in_upleg": up["in_upleg"].to_numpy(),
                "halve": up["halve"].to_numpy(),
                "stage": stage.fillna("震荡").to_numpy(),
                "vol": vol.to_numpy(),
                "session": daily.index.to_numpy(),
            },
            index=daily.index,
        )
        signal[symbol] = frame
    return signal, "D"


def build_orders(
    daily_map: dict[str, pd.DataFrame],
    freq: str,
    ma_weeks: int,
    symbols: tuple[str, ...] = SYMBOLS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Target weights and participation indexed by execution date."""
    if freq == "W":
        length, confirm, slope_lag = ma_weeks, CONFIRM_WEEKS, SLOPE_LAG_WEEKS
    elif freq == "D":
        length = ma_weeks * DAILY_BARS_PER_WEEK
        confirm = CONFIRM_WEEKS * DAILY_BARS_PER_WEEK
        slope_lag = SLOPE_LAG_WEEKS * DAILY_BARS_PER_WEEK
    else:
        raise ValueError(freq)

    signal, kind = _feature_tables(daily_map, freq, length, confirm, slope_lag)
    calendar = daily_map[symbols[0]].index
    for symbol in symbols[1:]:
        calendar = calendar.union(daily_map[symbol].index)
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()

    if kind == "W":
        fridays = signal[symbols[0]].index
        for symbol in symbols[1:]:
            fridays = fridays.union(signal[symbol].index)
        fridays = pd.DatetimeIndex(fridays).sort_values().unique()
        cal_set = set(pd.Timestamp(x) for x in calendar)
        weight_rows = []
        part_rows = []
        exec_rows = []
        for friday in fridays:
            flags = {}
            vols = {}
            stages = {}
            halves = {}
            for symbol in symbols:
                frame = signal[symbol]
                if friday not in frame.index:
                    flags[symbol] = False
                    halves[symbol] = False
                    stages[symbol] = "震荡"
                    vols[symbol] = np.nan
                    continue
                row = frame.loc[friday]
                flags[symbol] = bool(row["in_upleg"])
                halves[symbol] = bool(row["halve"])
                stages[symbol] = str(row["stage"])
                vols[symbol] = float(row["vol"]) if np.isfinite(row["vol"]) else np.nan
            weights = target_weights(list(symbols), flags, halves, stages, vols)
            parts = {s: participation(flags[s], halves[s]) for s in symbols}
            ex = following_week_execution(pd.Timestamp(friday), calendar)
            if ex is None or pd.Timestamp(ex) not in cal_set:
                continue
            exec_rows.append(ex)
            weight_rows.append(weights)
            part_rows.append(parts)
        order_w = pd.DataFrame(weight_rows, index=pd.DatetimeIndex(exec_rows))
        order_p = pd.DataFrame(part_rows, index=pd.DatetimeIndex(exec_rows))
    else:
        days = signal[symbols[0]].index
        for symbol in symbols[1:]:
            days = days.intersection(signal[symbol].index)
        days = pd.DatetimeIndex(days).sort_values().unique()
        calendar = pd.DatetimeIndex(calendar).sort_values().unique()
        nxt = {pd.Timestamp(calendar[i]): pd.Timestamp(calendar[i + 1]) for i in range(len(calendar) - 1)}
        weight_rows = []
        part_rows = []
        exec_rows = []
        for day in days:
            flags, halves, stages, vols = {}, {}, {}, {}
            for symbol in symbols:
                row = signal[symbol].loc[day]
                flags[symbol] = bool(row["in_upleg"])
                halves[symbol] = bool(row["halve"])
                stages[symbol] = str(row["stage"])
                vols[symbol] = float(row["vol"]) if np.isfinite(row["vol"]) else np.nan
            weights = target_weights(list(symbols), flags, halves, stages, vols)
            parts = {s: participation(flags[s], halves[s]) for s in symbols}
            ex = nxt.get(pd.Timestamp(day))
            if ex is None:
                continue
            exec_rows.append(ex)
            weight_rows.append(weights)
            part_rows.append(parts)
        order_w = pd.DataFrame(weight_rows, index=pd.DatetimeIndex(exec_rows))
        order_p = pd.DataFrame(part_rows, index=pd.DatetimeIndex(exec_rows))

    order_w = order_w.groupby(level=0).last()
    order_p = order_p.groupby(level=0).last()
    return order_w[list(symbols)], order_p[list(symbols)], calendar


def run_strategy(
    daily_map: dict[str, pd.DataFrame],
    freq: str,
    ma_weeks: int,
    start: str,
    symbols: tuple[str, ...] = SYMBOLS,
) -> dict:
    """Simulate from ``start`` (book opens flat). Indicator history before ``start`` is used."""
    order_w, order_p, calendar = build_orders(daily_map, freq, ma_weeks, symbols)
    start_ts = pd.Timestamp(start)
    # Common sessions on/after the book start, requiring a price for every name.
    opens = pd.DataFrame({s: daily_map[s]["Open"] for s in symbols}).sort_index()
    closes = pd.DataFrame({s: daily_map[s]["Close"] for s in symbols}).sort_index()
    live = opens.dropna().index.intersection(closes.dropna().index)
    live = pd.DatetimeIndex(live)
    live = live[live >= start_ts]
    if len(live) < 10:
        raise RuntimeError(f"not enough sessions from {start}")
    order_w = order_w.loc[order_w.index.isin(live)]
    order_p = order_p.loc[order_p.index.isin(live)]
    result = simulate(live, opens, closes, order_w, order_p, symbols)
    result["opens"] = opens.reindex(live)
    result["closes"] = closes.reindex(live)
    result["orders"] = order_w
    return result
