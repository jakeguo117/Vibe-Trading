"""In-memory books. Nothing here is written to a performance table."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class BookView:
    """Bars knowable at the close of ``session``. Later sessions are absent."""

    session: date
    dates: dict[str, tuple[date, ...]]
    opens: dict[str, tuple[float, ...]]
    highs: dict[str, tuple[float, ...]]
    lows: dict[str, tuple[float, ...]]
    closes: dict[str, tuple[float, ...]]
    volumes: dict[str, tuple[int, ...]]
    avg_cost: dict[str, float]
    prev_weights: dict[str, float]
    halved: frozenset[str] = frozenset()
    has_book: bool = False


@dataclass(frozen=True)
class TargetBook:
    weights: dict[str, float]
    reasons: dict[str, str] = field(default_factory=dict)
    hold: bool = False
    halved: frozenset[str] = frozenset()
    hist_calls: int = 0


@dataclass(frozen=True)
class Mark:
    date: date
    equity: float
    cash: float
    gross: float


@dataclass(frozen=True)
class Fill:
    date: date
    symbol: str
    side: str
    weight_delta: float
    exec_price: float
    reason: str


@dataclass(frozen=True)
class LegResult:
    leg: str
    rows: tuple[Mark, ...]
    fills: tuple[Fill, ...]


@dataclass(frozen=True)
class BakeoffResult:
    meta: dict[str, object]
    legs: dict[str, LegResult]
