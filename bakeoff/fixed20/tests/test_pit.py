"""In-package PIT. Knowledge date is D minus delay calendar days."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from bakeoff.fixed20.pit import Bar, BarQuery, InMemoryPitProvider, MembershipInterval
from bakeoff.fixed20.synthetic import build_material, fixed20_provider, sessions


def _bar(symbol: str, session: date, close: str) -> Bar:
    price = Decimal(close)
    return Bar(symbol, session, price, price, price, price, 100)


def test_delay_one_hides_the_as_of_session_and_the_future() -> None:
    monday = date(2024, 6, 3)
    friday = date(2024, 5, 31)
    tuesday = date(2024, 6, 4)
    bars = (
        _bar("AAPL", friday, "10.00"),
        _bar("AAPL", monday, "11.00"),
        _bar("AAPL", tuesday, "12.00"),
        _bar("MSFT", monday, "20.00"),
    )
    intervals = (
        MembershipInterval("fixed20-v1", "AAPL", friday, None),
        MembershipInterval("fixed20-v1", "MSFT", friday, None),
    )
    provider = InMemoryPitProvider(bars, intervals, source="synthetic", fixture_id="pit-delay")
    same_day = provider.bars(BarQuery(as_of=monday, symbols=("AAPL",), delay=0))
    assert {bar.session_date for bar in same_day.bars} == {friday, monday}
    lagged = provider.bars(BarQuery(as_of=monday, symbols=("AAPL", "MSFT"), delay=1))
    assert {bar.session_date for bar in lagged.bars} == {friday}
    assert all(bar.symbol == "AAPL" for bar in lagged.bars)
    assert all(bar.session_date <= monday - timedelta(days=1) for bar in lagged.bars)


def test_future_membership_is_not_visible() -> None:
    day = date(2024, 6, 3)
    later = date(2024, 6, 10)
    bars = (_bar("AAPL", day, "10.00"),)
    intervals = (MembershipInterval("fixed20-v1", "AAPL", later, None),)
    provider = InMemoryPitProvider(bars, intervals, source="synthetic", fixture_id="pit-future-member")
    snap = provider.bars(BarQuery(as_of=day, symbols=("AAPL",), delay=0))
    assert snap.bars == ()


def test_seeded_series_changes_price_not_calendar() -> None:
    left, _ = build_material(20)
    right, _ = build_material(21)
    again, _ = build_material(20)
    assert left == again

    def closes(bars: tuple[Bar, ...], symbol: str) -> tuple[Decimal, ...]:
        return tuple(bar.close for bar in bars if bar.symbol == symbol)

    assert closes(left, "AAPL") != closes(right, "AAPL")
    assert [bar.session_date for bar in left if bar.symbol == "NVDA"] == [
        bar.session_date for bar in right if bar.symbol == "NVDA"
    ]


def test_provider_never_returns_a_bar_after_the_query() -> None:
    provider = fixed20_provider(20)
    calendar = sessions()
    as_of = calendar[40]
    snap = provider.bars(BarQuery(as_of=as_of, symbols=("META", "GOOGL"), delay=0, end=calendar[-1]))
    assert snap.bars
    assert {bar.symbol for bar in snap.bars} <= {"META", "GOOGL"}
    assert max(bar.session_date for bar in snap.bars) == as_of
    with pytest.raises(ValueError):
        provider.bars(BarQuery(as_of=as_of, symbols=("META",), delay=-1))
