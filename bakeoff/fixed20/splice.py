"""A1. FB prints and META prints become one META series.

``<= 2022-06-08`` keeps the FB bar when both tickers exist that day.
``>= 2022-06-09`` keeps META and drops FB. The signal never sees two keys.
"""

from __future__ import annotations

from datetime import date

from bakeoff.fixed20.pit import Bar
from bakeoff.fixed20.universe import FB_SYMBOL, META_FB_SEAM, META_SYMBOL


def _as_meta(bar: Bar) -> Bar:
    if bar.symbol == META_SYMBOL:
        return bar
    return Bar(
        symbol=META_SYMBOL,
        session_date=bar.session_date,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
    )


def splice_meta_fb(bars: tuple[Bar, ...] | list[Bar]) -> tuple[Bar, ...]:
    """Return bars keyed so META is a single continuous series."""
    other: list[Bar] = []
    pre: dict[date, Bar] = {}
    post: dict[date, Bar] = {}
    for bar in bars:
        if bar.symbol != META_SYMBOL and bar.symbol != FB_SYMBOL:
            other.append(bar)
            continue
        if bar.session_date < META_FB_SEAM:
            current = pre.get(bar.session_date)
            if current is None or bar.symbol == FB_SYMBOL:
                pre[bar.session_date] = _as_meta(bar)
            continue
        if bar.symbol == META_SYMBOL:
            post[bar.session_date] = bar
    merged = [*other, *pre.values(), *post.values()]
    return tuple(sorted(merged, key=lambda item: (item.symbol, item.session_date)))
