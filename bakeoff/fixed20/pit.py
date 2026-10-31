"""Minimal point-in-time bars for the synthetic shell.

``delay`` is a calendar-day lag. On decision date D, ``delay=1`` means the
knowledge date is D-1. A bar later than that date is never returned.
This is not trading-system ``research.data``. It only serves seeded bars
already in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal


@dataclass(frozen=True)
class Bar:
    symbol: str
    session_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass(frozen=True)
class BarQuery:
    as_of: date
    symbols: tuple[str, ...]
    delay: int = 1
    start: date | None = None
    end: date | None = None


@dataclass(frozen=True)
class MembershipInterval:
    universe_id: str
    symbol: str
    start: date
    end: date | None = None


@dataclass(frozen=True)
class BarSnapshot:
    bars: tuple[Bar, ...]


class PitDataProvider:
    """Bars visible on a knowledge date. Subclasses must not look ahead."""

    def bars(self, query: BarQuery) -> BarSnapshot:
        raise NotImplementedError


class InMemoryPitProvider(PitDataProvider):
    """Seeded bars. Visibility is ``session_date <= as_of - delay calendar days``."""

    def __init__(
        self,
        bars: tuple[Bar, ...] | list[Bar],
        intervals: tuple[MembershipInterval, ...] | list[MembershipInterval],
        *,
        source: str,
        fixture_id: str,
    ) -> None:
        self.source = source
        self.fixture_id = fixture_id
        self._bars = tuple(bars)
        self._intervals = tuple(intervals)

    def bars(self, query: BarQuery) -> BarSnapshot:
        if type(query.as_of) is not date:
            raise TypeError("as_of 必须是 datetime.date")
        if type(query.delay) is not int or query.delay < 0:
            raise ValueError("delay 必须是非负整数")
        knowledge = query.as_of - timedelta(days=query.delay)
        wanted = set(query.symbols)
        visible: list[Bar] = []
        for bar in self._bars:
            if bar.symbol not in wanted:
                continue
            if bar.session_date > knowledge:
                continue
            if query.start is not None and bar.session_date < query.start:
                continue
            if query.end is not None and bar.session_date > query.end:
                continue
            if not self._known_member(bar.symbol, bar.session_date, knowledge):
                continue
            visible.append(bar)
        visible.sort(key=lambda item: (item.symbol, item.session_date))
        return BarSnapshot(bars=tuple(visible))

    def _known_member(self, symbol: str, session: date, knowledge: date) -> bool:
        for interval in self._intervals:
            if interval.symbol != symbol:
                continue
            if interval.start > knowledge:
                continue
            if session < interval.start:
                continue
            if interval.end is not None and session > interval.end:
                continue
            return True
        return False
