"""Target weights for BH, F, and N1. The engine applies them next open."""

from __future__ import annotations

from bakeoff.fixed20.costs import side_cost
from bakeoff.fixed20.hist import score_symbols
from bakeoff.fixed20.indicators import down_confirmed, hard_stop_hit, ret_last, sma_last
from bakeoff.fixed20.models import BookView, TargetBook
from bakeoff.fixed20.stages import HOLDABLE_STAGES, NEW_STAGES, stage_snapshot
from bakeoff.fixed20.universe import (
    COMMODITY_PRIORITY,
    FIXED20,
    GROSS_CAP,
    HARD_STOP,
    K,
)
from bakeoff.fixed20.upleg_link import require


def _close(weights: dict[str, float], other: dict[str, float]) -> bool:
    keys = set(weights) | set(other)
    return all(abs(weights.get(key, 0.0) - other.get(key, 0.0)) <= 1e-12 for key in keys)


def _cap(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {symbol: weight for symbol, weight in weights.items() if weight > 1e-15}
    for weight in cleaned.values():
        if weight < 0:
            raise RuntimeError("烤炉不允许空头")
    total = sum(cleaned.values())
    if total > GROSS_CAP + 1e-12:
        scale = GROSS_CAP / total
        cleaned = {symbol: weight * scale for symbol, weight in cleaned.items()}
    return cleaned


def _dollar_volume(view: BookView, symbol: str) -> float:
    return float(view.closes[symbol][-1]) * int(view.volumes[symbol][-1])


def bh_targets(view: BookView, *, universe: tuple[str, ...] = FIXED20, first: bool) -> TargetBook:
    """Equal weight on names that have a bar on ``t0``, then hold forever."""
    if not first:
        return TargetBook(dict(view.prev_weights), {}, hold=True)
    tradable = [
        symbol
        for symbol in universe
        if view.dates.get(symbol) and view.dates[symbol][-1] == view.session
    ]
    reasons = {
        symbol: "bh_entry" if symbol in tradable else "missing_at_t0" for symbol in universe
    }
    if not tradable:
        return TargetBook({}, reasons, hold=False)
    weight = 1.0 / len(tradable)
    return TargetBook({symbol: weight for symbol in tradable}, reasons, hold=False)


def rank_pool(pool: list[str], scores: dict[str, float]) -> list[str]:
    """Hist score descending. GLD → SLV → USO first only when at least two are in the pool."""
    commodities = [symbol for symbol in COMMODITY_PRIORITY if symbol in pool]
    use_priority = len(commodities) >= 2

    def key(symbol: str) -> tuple[int, int, float, str]:
        if use_priority and symbol in COMMODITY_PRIORITY:
            return (0, COMMODITY_PRIORITY.index(symbol), 0.0, symbol)
        return (1, 0, -float(scores[symbol]), symbol)

    return sorted(pool, key=key)


def _stopped(view: BookView, symbol: str) -> bool:
    closes = view.closes.get(symbol, ())
    if not closes:
        return False
    return hard_stop_hit(closes[-1], view.avg_cost.get(symbol), HARD_STOP)


def n1_targets(view: BookView, *, universe: tuple[str, ...] = FIXED20) -> TargetBook:
    """ret_20 rank, SMA50 filter, top k, full exit. No Fib and no halve."""
    ranked: list[tuple[float, str]] = []
    stopped: set[str] = set()
    for symbol in universe:
        closes = view.closes.get(symbol, ())
        if _stopped(view, symbol):
            stopped.add(symbol)
            continue
        momentum = ret_last(closes)
        average = sma_last(closes)
        if momentum is None or average is None or closes[-1] < average:
            continue
        ranked.append((momentum, symbol))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    picked = [symbol for _, symbol in ranked[:K]]
    weights: dict[str, float] = {}
    if picked:
        share = 1.0 / len(picked)
        weights = {symbol: share for symbol in picked}
    reasons: dict[str, str] = {}
    for symbol in universe:
        if symbol in stopped:
            reasons[symbol] = "hard_stop"
        elif symbol in picked:
            reasons[symbol] = "n1_select"
        elif view.prev_weights.get(symbol, 0.0) > 0:
            closes = view.closes.get(symbol, ())
            average = sma_last(closes) if closes else None
            if average is not None and closes[-1] < average:
                reasons[symbol] = "break_ma"
            else:
                reasons[symbol] = "slip_topk"
    weights = _cap(weights)
    hold = view.has_book and _close(weights, view.prev_weights)
    return TargetBook(weights, reasons, hold=hold)


def f_targets(view: BookView, *, universe: tuple[str, ...] = FIXED20) -> TargetBook:
    """Shortened Fib sleeve. Stage and hist are the production functions."""
    require()
    from upleg_bt.engine.fib_locked import TICKERS

    symbols = [symbol for symbol in universe if view.closes.get(symbol)]
    stages: dict[str, str] = {}
    routes: list[int] = []
    kinds: list[int] = []
    sides: list[float] = []
    for symbol in symbols:
        name, route = stage_snapshot(view, symbol)
        stages[symbol] = name
        routes.append(route)
        kinds.append(1 if symbol in TICKERS else 0)
        sides.append(side_cost(symbol, _dollar_volume(view, symbol)))
    scores = (
        score_symbols(symbols, route=routes, kind=kinds, side=sides) if symbols else {}
    )
    hist_calls = 1 if symbols else 0

    stopped = {symbol for symbol in symbols if _stopped(view, symbol)}
    confirmed = {symbol for symbol in symbols if down_confirmed(view.closes[symbol])}
    pool: list[str] = []
    for symbol in symbols:
        if symbol in stopped or symbol in confirmed:
            continue
        if stages[symbol] not in HOLDABLE_STAGES:
            continue
        average = sma_last(view.closes[symbol])
        if average is None or view.closes[symbol][-1] < average:
            continue
        pool.append(symbol)
    ordered = rank_pool(pool, scores)
    top = ordered[:K]
    top_set = set(top)

    full: list[str] = []
    halved_weights: dict[str, float] = {}
    halved_next: set[str] = set()
    reasons: dict[str, str] = {}
    for symbol in universe:
        prev = view.prev_weights.get(symbol, 0.0)
        held = prev > 1e-12 or symbol in view.avg_cost
        if symbol in stopped:
            reasons[symbol] = "hard_stop"
            continue
        if symbol in confirmed:
            reasons[symbol] = "down_confirm"
            continue
        if symbol in pool and symbol not in top_set:
            reasons[symbol] = "slip_topk"
            continue
        in_segment = symbol in pool
        if not in_segment:
            if prev > 1e-12:
                if symbol in view.halved:
                    halved_weights[symbol] = prev
                    halved_next.add(symbol)
                    reasons[symbol] = "halved_hold"
                else:
                    halved_weights[symbol] = prev * 0.5
                    halved_next.add(symbol)
                    reasons[symbol] = "halve"
            continue
        stage_name = stages.get(symbol, "grind")
        if stage_name in NEW_STAGES or (stage_name == "fish" and held):
            full.append(symbol)
            reasons[symbol] = "f_select"
            continue
        if stage_name == "fish" and not held:
            reasons[symbol] = "fish_no_new"

    weights = dict(halved_weights)
    if full:
        room = max(0.0, GROSS_CAP - sum(halved_weights.values()))
        share = room / len(full)
        for symbol in full:
            weights[symbol] = share
    weights = _cap(weights)
    halved = frozenset(halved_next)
    hold = (
        view.has_book
        and _close(weights, view.prev_weights)
        and halved == view.halved
    )
    return TargetBook(weights, reasons, hold=hold, halved=halved, hist_calls=hist_calls)
