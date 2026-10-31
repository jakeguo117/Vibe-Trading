"""Deterministic Fixed20 bars. Prices are invented cents, not a market history.

The three leader names follow a rise / pullback / break that the locked weekly
Fib function labels 主升 inside the toy window. The other seventeen drift down
so they stay under the 50-day average. Same seed, same bars.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from bakeoff.fixed20.pit import Bar, InMemoryPitProvider, MembershipInterval
from bakeoff.fixed20.universe import FIXED20, UNIVERSE_ID

DEFAULT_SEED = 20
ORIGIN = date(2018, 1, 2)
N_SESSIONS = 180
# Toy window sits inside the leader 主升 stretch. Warmup before this index
# covers SMA50, ret_20, and the weekly Fib pivot lag.
TOY_START_INDEX = 132
TOY_END_INDEX = 150
_SOURCE = "synthetic"
_VOLUME = 200_000
LEADER_SYMBOLS = ("AAPL", "MSFT", "NVDA")


def _cents(cents: int) -> Decimal:
    return (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))


def _weekdays(origin: date, count: int) -> tuple[date, ...]:
    found: list[date] = []
    cursor = origin
    while len(found) < count:
        if cursor.weekday() < 5:
            found.append(cursor)
        cursor += timedelta(days=1)
    return tuple(found)


def sessions() -> tuple[date, ...]:
    return _weekdays(ORIGIN, N_SESSIONS)


def toy_window() -> tuple[date, date]:
    calendar = sessions()
    return calendar[TOY_START_INDEX], calendar[TOY_END_INDEX]


def _require_seed(seed: int) -> int:
    if type(seed) is not int or seed < 0 or seed > 500:
        raise ValueError("seed 必须是 0 到 500 的整数")
    return seed


def _leader_cents(index: int, tilt: int) -> int:
    if index < 20:
        price = 8000
    elif index < 70:
        price = 8000 + (index - 20) * 150
    elif index < 100:
        price = 15500 - (index - 70) * 80
    elif index < 140:
        price = 13100 + (index - 100) * 120
    else:
        price = 17500 + (index - 140) * 15
    return price + tilt


def _laggard_cents(index: int, base: int) -> int:
    return base - index * 5


def _bar(symbol: str, session: date, close_cents: int) -> Bar:
    open_cents = close_cents - 15
    high_cents = close_cents + 20
    low_cents = close_cents - 25
    return Bar(
        symbol=symbol,
        session_date=session,
        open=_cents(open_cents),
        high=_cents(high_cents),
        low=_cents(low_cents),
        close=_cents(close_cents),
        volume=_VOLUME,
    )


def build_material(
    seed: int = DEFAULT_SEED,
) -> tuple[tuple[Bar, ...], tuple[MembershipInterval, ...]]:
    checked = _require_seed(seed)
    calendar = sessions()
    bars: list[Bar] = []
    intervals: list[MembershipInterval] = []
    for offset, symbol in enumerate(FIXED20):
        intervals.append(MembershipInterval(UNIVERSE_ID, symbol, calendar[0], None))
        if symbol in LEADER_SYMBOLS:
            tilt = checked + LEADER_SYMBOLS.index(symbol) * 40
            closes = [_leader_cents(index, tilt) for index in range(len(calendar))]
        else:
            base = 12000 + offset * 350 + checked
            closes = [_laggard_cents(index, base) for index in range(len(calendar))]
        for session, close_cents in zip(calendar, closes):
            bars.append(_bar(symbol, session, close_cents))
    return tuple(bars), tuple(intervals)


def provider_from_material(
    bars: tuple[Bar, ...],
    intervals: tuple[MembershipInterval, ...],
    *,
    fixture_id: str,
) -> InMemoryPitProvider:
    return InMemoryPitProvider(bars, intervals, source=_SOURCE, fixture_id=fixture_id)


def fixed20_provider(seed: int = DEFAULT_SEED) -> InMemoryPitProvider:
    """In-memory PIT provider. META is already the continuous key (no FB rows)."""
    bars, intervals = build_material(seed)
    return provider_from_material(
        bars,
        intervals,
        fixture_id=f"fixed20-synthetic-{seed}",
    )
