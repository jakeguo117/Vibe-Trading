"""Delay-1 shell. Signal at the close, fill at the next session's open.

The in-package PIT delay hides bars after the knowledge date. Freeze delay is
the execution lag, so each signal queries the provider with ``delay=0`` on
that session and the fill waits until the next session. The three legs share
that clock. This module returns marks in memory and does not write equity files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from bakeoff.fixed20.costs import cost_meta, side_cost, turnover_fee
from bakeoff.fixed20.models import (
    BakeoffResult,
    BookView,
    Fill,
    LegResult,
    Mark,
    TargetBook,
)
from bakeoff.fixed20.pit import Bar, BarQuery, PitDataProvider
from bakeoff.fixed20.synthetic import DEFAULT_SEED, fixed20_provider, toy_window
from bakeoff.fixed20.targets import bh_targets, f_targets, n1_targets
from bakeoff.fixed20.universe import (
    CASH_FLOOR,
    FIXED20,
    GROSS_CAP,
    INITIAL_NAV,
    META_FB_SEAM,
    SCHEME_ID,
)

_HALT_STREAK = 5
_HALT_POLICY = "mark_last_close_skip_trade; miss_streak>=5 liquidate_at_last_close"


@dataclass
class _Pos:
    cash: float = INITIAL_NAV
    shares: dict[str, float] = field(default_factory=dict)
    avg_cost: dict[str, float] = field(default_factory=dict)
    last_close: dict[str, float] = field(default_factory=dict)
    miss: dict[str, int] = field(default_factory=dict)
    prev_weights: dict[str, float] = field(default_factory=dict)
    halved: frozenset[str] = frozenset()
    has_book: bool = False
    pending: TargetBook | None = None
    rows: list[Mark] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    forced: list[dict[str, str]] = field(default_factory=list)
    fallbacks: int = 0
    hist_calls: int = 0


def _calendar(provider: PitDataProvider, universe: tuple[str, ...], window_end: date) -> tuple[date, ...]:
    catalog = provider.bars(
        BarQuery(as_of=window_end, symbols=universe, delay=0, end=window_end)
    )
    return tuple(sorted({bar.session_date for bar in catalog.bars}))


def _day_bars(
    provider: PitDataProvider,
    universe: tuple[str, ...],
    session: date,
) -> dict[str, Bar]:
    snap = provider.bars(
        BarQuery(
            as_of=session,
            symbols=universe,
            delay=0,
            start=session,
            end=session,
        )
    )
    return {bar.symbol: bar for bar in snap.bars}


def _view(series: dict[str, list[Bar]], session: date, pos: _Pos) -> BookView:
    dates: dict[str, tuple[date, ...]] = {}
    opens: dict[str, tuple[float, ...]] = {}
    highs: dict[str, tuple[float, ...]] = {}
    lows: dict[str, tuple[float, ...]] = {}
    closes: dict[str, tuple[float, ...]] = {}
    volumes: dict[str, tuple[int, ...]] = {}
    for symbol, bars in series.items():
        if not bars:
            continue
        dates[symbol] = tuple(bar.session_date for bar in bars)
        opens[symbol] = tuple(float(bar.open) for bar in bars)
        highs[symbol] = tuple(float(bar.high) for bar in bars)
        lows[symbol] = tuple(float(bar.low) for bar in bars)
        closes[symbol] = tuple(float(bar.close) for bar in bars)
        volumes[symbol] = tuple(bar.volume for bar in bars)
    return BookView(
        session=session,
        dates=dates,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        avg_cost=dict(pos.avg_cost),
        prev_weights=dict(pos.prev_weights),
        halved=pos.halved,
        has_book=pos.has_book,
    )


def _open_price(pos: _Pos, symbol: str, today: dict[str, Bar]) -> float | None:
    bar = today.get(symbol)
    if bar is None:
        return pos.last_close.get(symbol)
    opened = float(bar.open)
    if not math.isfinite(opened) or opened <= 0:
        pos.fallbacks += 1
        return float(bar.close)
    return opened


def _execute(pos: _Pos, session: date, today: dict[str, Bar]) -> None:
    nav = pos.cash
    for symbol, shares in pos.shares.items():
        price = _open_price(pos, symbol, today)
        if price is not None:
            nav += shares * price
    book = pos.pending
    if book is not None and not book.hold and nav > 0:
        current = {}
        for symbol, shares in pos.shares.items():
            price = _open_price(pos, symbol, today)
            if price is None:
                continue
            current[symbol] = shares * price / nav
        target = {symbol: weight for symbol, weight in book.weights.items() if weight > 1e-15}
        for symbol in set(target) | set(current):
            if symbol not in today:
                target[symbol] = current.get(symbol, 0.0)
        side = {}
        for symbol in set(target) | set(current):
            bar = today.get(symbol)
            dollar_volume = float(bar.close) * bar.volume if bar is not None else 0.0
            side[symbol] = side_cost(symbol, dollar_volume)
        fee = turnover_fee(target, current, side)
        nav_after = nav * (1.0 - fee)
        new_shares: dict[str, float] = {}
        for symbol, weight in target.items():
            if symbol not in today:
                if symbol in pos.shares:
                    new_shares[symbol] = pos.shares[symbol]
                continue
            price = _open_price(pos, symbol, today)
            if price is None or price <= 0:
                continue
            new_shares_n = weight * nav_after / price
            old_shares = pos.shares.get(symbol, 0.0)
            if new_shares_n > old_shares + 1e-12:
                added = new_shares_n - old_shares
                old_cost = pos.avg_cost.get(symbol)
                if old_cost is None or old_shares <= 1e-12:
                    pos.avg_cost[symbol] = price
                else:
                    pos.avg_cost[symbol] = (old_shares * old_cost + added * price) / new_shares_n
            if new_shares_n > 1e-12:
                new_shares[symbol] = new_shares_n
            delta = weight - current.get(symbol, 0.0)
            if abs(delta) > 1e-8:
                pos.fills.append(
                    Fill(
                        date=session,
                        symbol=symbol,
                        side="buy" if delta > 0 else "sell",
                        weight_delta=delta,
                        exec_price=price,
                        reason=book.reasons.get(symbol, "rebalance"),
                    )
                )
        for symbol, weight in current.items():
            if symbol in target or symbol not in today or weight <= 1e-8:
                continue
            price = _open_price(pos, symbol, today)
            if price is None:
                continue
            pos.fills.append(
                Fill(
                    date=session,
                    symbol=symbol,
                    side="sell",
                    weight_delta=-weight,
                    exec_price=price,
                    reason=book.reasons.get(symbol, "exit"),
                )
            )
            pos.avg_cost.pop(symbol, None)
        pos.shares = new_shares
        for symbol in list(pos.avg_cost):
            if symbol not in pos.shares:
                pos.avg_cost.pop(symbol, None)
        invested = 0.0
        for symbol, shares in pos.shares.items():
            price = _open_price(pos, symbol, today)
            if price is not None:
                invested += shares * price
        pos.cash = nav_after - invested
        if abs(pos.cash) < 1e-12:
            pos.cash = 0.0

    for symbol, bar in today.items():
        pos.last_close[symbol] = float(bar.close)
        pos.miss[symbol] = 0
    for symbol in list(pos.shares):
        if symbol not in today:
            pos.miss[symbol] = pos.miss.get(symbol, 0) + 1

    position = 0.0
    marks: dict[str, float] = {}
    for symbol, shares in pos.shares.items():
        if symbol in today:
            mark = float(today[symbol].close)
        else:
            mark = pos.last_close.get(symbol, 0.0)
        marks[symbol] = mark
        position += shares * mark
    equity = pos.cash + position
    for symbol in list(pos.shares):
        if pos.miss.get(symbol, 0) < _HALT_STREAK:
            continue
        mark = marks[symbol]
        shares = pos.shares.pop(symbol)
        if equity > 0:
            weight = shares * mark / equity
        else:
            weight = 0.0
        pos.cash += shares * mark
        pos.avg_cost.pop(symbol, None)
        pos.forced.append(
            {
                "symbol": symbol,
                "date": session.isoformat(),
                "reason": "halt_5",
            }
        )
        pos.fills.append(
            Fill(
                date=session,
                symbol=symbol,
                side="sell",
                weight_delta=-weight,
                exec_price=mark,
                reason="forced_halt",
            )
        )
        position -= shares * mark
    if abs(pos.cash) < 1e-12:
        pos.cash = 0.0
    equity = pos.cash + position
    gross = position / equity if equity else 0.0
    pos.rows.append(Mark(date=session, equity=equity, cash=pos.cash, gross=gross))


def _signal(kind: str, view: BookView, universe: tuple[str, ...], pos: _Pos) -> TargetBook:
    if kind == "BH":
        return bh_targets(view, universe=universe, first=not pos.has_book)
    if kind == "F":
        return f_targets(view, universe=universe)
    if kind == "N1":
        return n1_targets(view, universe=universe)
    raise RuntimeError(f"未知腿 {kind}")


def run_bakeoff(
    provider: PitDataProvider,
    *,
    window_start: date,
    window_end: date,
    universe: tuple[str, ...] = FIXED20,
    synthetic: bool = False,
    seed: int | None = None,
) -> BakeoffResult:
    """Run BH, F, and N1 on one provider. Same ``t0``, same cost hook."""
    if type(window_start) is not date or type(window_end) is not date:
        raise TypeError("窗口必须是 datetime.date")
    if window_start > window_end:
        raise ValueError("window_start 晚于 window_end")
    if not universe:
        raise ValueError("universe 为空")
    calendar = _calendar(provider, universe, window_end)
    tradable = [session for session in calendar if window_start <= session <= window_end]
    if len(tradable) < 2:
        raise ValueError("窗口里至少要有两个交易日才能 delay=1")
    t0 = tradable[0]
    t_end = tradable[-1]
    series: dict[str, list[Bar]] = {symbol: [] for symbol in universe}
    books = {leg: _Pos() for leg in ("BH", "F", "N1")}
    missing_at_t0: tuple[str, ...] = ()

    for session in calendar:
        if session > t_end:
            break
        today = _day_bars(provider, universe, session)
        for symbol, bar in today.items():
            series[symbol].append(bar)
        if session == t0:
            missing_at_t0 = tuple(
                symbol for symbol in universe if not today.get(symbol)
            )
        if t0 < session <= t_end:
            for pos in books.values():
                _execute(pos, session, today)
        if t0 <= session < t_end:
            for leg, pos in books.items():
                view = _view(series, session, pos)
                book = _signal(leg, view, universe, pos)
                pos.pending = book
                pos.prev_weights = dict(book.weights)
                pos.halved = book.halved
                pos.has_book = True
                pos.hist_calls += book.hist_calls

    legs = {
        leg: LegResult(leg=leg, rows=tuple(pos.rows), fills=tuple(pos.fills))
        for leg, pos in books.items()
    }
    forced = [
        {"leg": leg, **item}
        for leg, pos in books.items()
        for item in pos.forced
    ]
    meta: dict[str, object] = {
        "scheme_id": SCHEME_ID,
        "synthetic": synthetic,
        "real_window": False,
        "emits_performance_tables": False,
        "universe": list(universe),
        "t0": t0.isoformat(),
        "t_end": t_end.isoformat(),
        "delay": 1,
        "exec_price": "next_session_open",
        "pit_signal_query_delay": 0,
        "pit_note": (
            "Freeze delay=1 是信号日收盘到下一交易日开盘。"
            "本目录 pit 的 delay 按日历日把可知日设为 D-1，会藏起决策日 bar，"
            "所以信号查询用 delay=0，成交仍在下一交易日。"
        ),
        "cost": cost_meta(),
        "cash_floor": CASH_FLOOR,
        "gross_cap": GROSS_CAP,
        "vol_target": None,
        "short": False,
        "meta_fb_splice": True,
        "meta_fb_seam": META_FB_SEAM.isoformat(),
        "meta_fb_rule": (
            "单一主键 META。<=2022-06-08 用当时代码 FB 的价量接到 META；"
            ">=2022-06-09 用 META。合成夹具已经是连续 META，没有 FB 行。"
        ),
        "hist_function": "upleg_bt.engine.fm.FM.score",
        "hist_cells_function": "upleg_bt.engine.fm.FM.cells",
        "stage_function": "upleg_bt.engine.fib_locked.fib_stages_weekly",
        "hist_market": 0,
        "hist_note": (
            "Fixed20 不含 SPY，MKT 输入为 0。合成段样本为空时 FM.score 只剩单边成本惩罚。"
            "真 CA 出数前必须把生产 segments 送进同一个函数，禁止另写胜率。"
        ),
        "f_hist_calls": books["F"].hist_calls,
        "halt_policy": _HALT_POLICY,
        "exec_fallback_count": sum(pos.fallbacks for pos in books.values()),
        "missing_at_t0": list(missing_at_t0),
        "forced_exits": forced,
        "legs": ["BH", "F", "N1"],
        "seed": seed,
    }
    return BakeoffResult(meta=meta, legs=legs)


def run_synthetic(seed: int = DEFAULT_SEED) -> BakeoffResult:
    """Toy window on the in-memory Fixed20 fixture. Not a historical result."""
    start, end = toy_window()
    return run_bakeoff(
        fixed20_provider(seed),
        window_start=start,
        window_end=end,
        synthetic=True,
        seed=seed,
    )
