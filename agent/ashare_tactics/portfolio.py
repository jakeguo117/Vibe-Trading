"""Equal-weight book: at most 10 names, 10% of equity each.

Sells are processed at the open before new buys, so the cash and the free
slot from an exit are available the same session. When more signals than
slots arrive, the higher 20-day average amount is filled first.
"""

from __future__ import annotations

from dataclasses import dataclass

from collections import OrderedDict

import numpy as np
import pandas as pd

from ashare_tactics.constants import INITIAL_CASH, MAX_POSITIONS, WEIGHT
from ashare_tactics.costs import buy_cash_out, sell_cash_in, size_shares
from ashare_tactics.models import Candidate


@dataclass
class SymbolBars:
    dates: np.ndarray  # datetime64[D]
    raw_open: np.ndarray
    raw_close: np.ndarray
    factor: np.ndarray

    def closes_through(self, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Last raw close and factor on or before each day in ``days``."""
        idx = np.searchsorted(self.dates, days, side="right") - 1
        if np.any(idx < 0):
            raise ValueError("mark requested before the symbol's first bar")
        return self.raw_close[idx], self.factor[idx]


@dataclass
class _Position:
    candidate: Candidate
    shares: int
    entry_i: int
    exit_i: int
    cash_out: float
    marks: np.ndarray  # economic value per share at each close from entry to exit


@dataclass
class SimResult:
    equity: pd.Series
    trades: pd.DataFrame
    skipped_cash: int = 0
    skipped_slots: int = 0


def simulate(
    candidates: list[Candidate],
    calendar: pd.DatetimeIndex,
    loader,
    initial_cash: float = INITIAL_CASH,
) -> SimResult:
    """Run one tactic.

    ``loader(symbol)`` returns ``SymbolBars``. Marks use
    ``raw_close * factor_entry / factor_today``, which is the forward-adjusted
    value of the shares that were actually bought.
    """
    if len(calendar) == 0:
        empty = pd.Series(dtype=float)
        return SimResult(empty, _empty_trades())

    days = np.asarray(calendar.values, dtype="datetime64[D]")
    index_of = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    by_day: dict[pd.Timestamp, list[Candidate]] = {}
    for cand in candidates:
        entry = pd.Timestamp(cand.entry_date).normalize()
        exit_ = pd.Timestamp(cand.exit_date).normalize()
        if entry not in index_of or exit_ not in index_of:
            continue
        if index_of[exit_] <= index_of[entry]:
            continue
        by_day.setdefault(entry, []).append(cand)
    for entry in by_day:
        by_day[entry].sort(key=lambda c: (-c.liquidity, c.symbol))

    cash = float(initial_cash)
    equity = np.full(len(calendar), np.nan)
    positions: list[_Position] = []
    trade_rows: list[dict] = []
    skipped_cash = 0
    skipped_slots = 0
    cache: OrderedDict[str, SymbolBars] = OrderedDict()

    def bars_for(symbol: str) -> SymbolBars:
        hit = cache.get(symbol)
        if hit is not None:
            cache.move_to_end(symbol)
            return hit
        hit = loader(symbol)
        cache[symbol] = hit
        if len(cache) > 64:
            cache.popitem(last=False)
        return hit

    for i, stamp in enumerate(calendar):
        still: list[_Position] = []
        for pos in positions:
            if pos.exit_i == i:
                cand = pos.candidate
                cash_in = sell_cash_in(
                    pos.shares,
                    cand.raw_exit,
                    cand.factor_entry,
                    cand.factor_exit,
                    np.datetime64(cand.exit_date),
                )
                cash += cash_in
                pnl = cash_in - pos.cash_out
                trade_rows.append(
                    {
                        "symbol": cand.symbol,
                        "entry_date": cand.entry_date,
                        "exit_date": cand.exit_date,
                        "shares": pos.shares,
                        "cash_out": pos.cash_out,
                        "cash_in": cash_in,
                        "pnl": pnl,
                        "ret": pnl / pos.cash_out if pos.cash_out else 0.0,
                        "hold_days": int(pos.exit_i - pos.entry_i),
                        "slope": cand.slope,
                        "take_profit": cand.take_profit,
                        "liquidity": cand.liquidity,
                    }
                )
            else:
                still.append(pos)
        positions = still

        base = cash
        for pos in positions:
            k = i - 1 - pos.entry_i
            # Before the first close, size off cash only. After that, yesterday's close.
            if k >= 0:
                base += pos.shares * float(pos.marks[k])

        todays = by_day.get(pd.Timestamp(stamp).normalize(), [])
        held = {pos.candidate.symbol for pos in positions}
        for cand in todays:
            if len(positions) >= MAX_POSITIONS:
                skipped_slots += 1
                continue
            if cand.symbol in held:
                continue
            shares = size_shares(base, cash, cand.raw_entry, WEIGHT)
            if shares <= 0:
                skipped_cash += 1
                continue
            cost = buy_cash_out(shares, cand.raw_entry)
            if cost > cash + 1e-6:
                skipped_cash += 1
                continue
            entry_i = index_of[pd.Timestamp(cand.entry_date).normalize()]
            exit_i = index_of[pd.Timestamp(cand.exit_date).normalize()]
            window = days[entry_i:exit_i]
            raw_close, factor = bars_for(cand.symbol).closes_through(window)
            with np.errstate(divide="ignore", invalid="ignore"):
                marks = raw_close * (cand.factor_entry / factor)
            if not np.isfinite(marks).all():
                skipped_cash += 1
                continue
            cash -= cost
            positions.append(
                _Position(
                    candidate=cand,
                    shares=shares,
                    entry_i=entry_i,
                    exit_i=exit_i,
                    cash_out=cost,
                    marks=marks,
                )
            )
            held.add(cand.symbol)

        marked = cash
        for pos in positions:
            marked += pos.shares * float(pos.marks[i - pos.entry_i])
        equity[i] = marked

    # Anything still open on the last session is already inside the mark.
    # Force-flat it at the last close so the trade log matches the equity.
    if positions:
        last = len(calendar) - 1
        for pos in positions:
            value = pos.shares * float(pos.marks[-1])
            # Last mark is the close before the calendar ends only if exit is
            # beyond the sample. Treat the marked value as the liquidation
            # cash, without a second round of fees (fees were not modelled
            # on a bar that does not exist). If exit_i is inside the calendar
            # the loop above already closed it.
            cash += value
            pnl = value - pos.cash_out
            trade_rows.append(
                {
                    "symbol": pos.candidate.symbol,
                    "entry_date": pos.candidate.entry_date,
                    "exit_date": calendar[last],
                    "shares": pos.shares,
                    "cash_out": pos.cash_out,
                    "cash_in": value,
                    "pnl": pnl,
                    "ret": pnl / pos.cash_out if pos.cash_out else 0.0,
                    "hold_days": int(last - pos.entry_i),
                    "slope": pos.candidate.slope,
                    "take_profit": False,
                    "liquidity": pos.candidate.liquidity,
                }
            )
        equity[-1] = cash

    eq = pd.Series(equity, index=calendar, name="equity")
    trades = pd.DataFrame(trade_rows) if trade_rows else _empty_trades()
    return SimResult(eq, trades, skipped_cash=skipped_cash, skipped_slots=skipped_slots)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "entry_date",
            "exit_date",
            "shares",
            "cash_out",
            "cash_in",
            "pnl",
            "ret",
            "hold_days",
            "slope",
            "take_profit",
            "liquidity",
        ]
    )
