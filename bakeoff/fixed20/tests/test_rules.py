"""Leg rules and the production hist/cost hooks. No market download."""

from __future__ import annotations

import inspect
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from bakeoff.fixed20.hist import score_symbols
from bakeoff.fixed20.indicators import down_confirmed
from bakeoff.fixed20.models import BookView
from bakeoff.fixed20.splice import splice_meta_fb
from bakeoff.fixed20.targets import bh_targets, f_targets, n1_targets, rank_pool
from bakeoff.fixed20.universe import FIXED20, HARD_STOP, META_FB_SEAM
from bakeoff.fixed20.upleg_link import PIN, UplegUnavailable, probe
from bakeoff.fixed20.pit import Bar


def _need_upleg() -> None:
    reason = probe()
    if reason:
        pytest.skip(reason)


def _view(
    closes: dict[str, tuple[float, ...]],
    *,
    session: date | None = None,
    avg_cost: dict[str, float] | None = None,
    prev: dict[str, float] | None = None,
    halved: frozenset[str] = frozenset(),
    has_book: bool = False,
) -> BookView:
    day = session or date(2024, 6, 3)
    dates = {
        symbol: tuple(day - timedelta(days=len(series) - 1 - index) for index in range(len(series)))
        for symbol, series in closes.items()
    }
    flat = {
        symbol: tuple(series[-1] for _ in series) for symbol, series in closes.items()
    }
    return BookView(
        session=day,
        dates=dates,
        opens=flat,
        highs=flat,
        lows=flat,
        closes=closes,
        volumes={symbol: tuple(200_000 for _ in series) for symbol, series in closes.items()},
        avg_cost=dict(avg_cost or {}),
        prev_weights=dict(prev or {}),
        halved=halved,
        has_book=has_book,
    )


def _cents(dollars: str) -> Decimal:
    return Decimal(dollars)


def test_universe_is_the_frozen_twenty() -> None:
    assert len(FIXED20) == 20
    assert len(set(FIXED20)) == 20
    assert FIXED20[0] == "AAPL"
    assert FIXED20[5] == "META"
    assert "GOOG" not in FIXED20
    assert "GOOGL" in FIXED20
    assert "FB" not in FIXED20
    assert FIXED20[-2:] == ("GLD", "USO")
    assert META_FB_SEAM.isoformat() == "2022-06-09"


def test_upleg_pin_is_recorded() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("requirements-upleg.txt").read_text(encoding="utf-8")
    assert PIN in text
    assert "fib_stages_weekly" in text
    assert "FM.score" in text
    assert "trade_fee" in text


def test_meta_fb_splice_keeps_one_key() -> None:
    fb = Bar("FB", date(2022, 6, 8), _cents("10.00"), _cents("11.00"), _cents("9.00"), _cents("10.50"), 100)
    meta_early = Bar("META", date(2022, 6, 8), _cents("1.00"), _cents("2.00"), _cents("1.00"), _cents("1.50"), 100)
    meta_late = Bar("META", date(2022, 6, 9), _cents("12.00"), _cents("13.00"), _cents("11.00"), _cents("12.50"), 80)
    fb_late = Bar("FB", date(2022, 6, 9), _cents("99.00"), _cents("99.00"), _cents("99.00"), _cents("99.00"), 1)
    other = Bar("AAPL", date(2022, 6, 8), _cents("5.00"), _cents("6.00"), _cents("4.00"), _cents("5.50"), 10)
    merged = splice_meta_fb((meta_early, fb, meta_late, fb_late, other))
    by_key = {(bar.symbol, bar.session_date): bar for bar in merged}
    assert set(by_key) == {
        ("META", date(2022, 6, 8)),
        ("META", date(2022, 6, 9)),
        ("AAPL", date(2022, 6, 8)),
    }
    assert by_key[("META", date(2022, 6, 8))].close == _cents("10.50")
    assert by_key[("META", date(2022, 6, 9))].close == _cents("12.50")


def test_cost_hook_matches_run_portfolio_defaults() -> None:
    _need_upleg()
    from upleg_bt.costs import DV_LIQUID, SIDE_ILLIQUID, SIDE_LIQUID

    from bakeoff.fixed20.costs import cost_meta, side_cost, turnover_fee

    meta = cost_meta()
    assert meta["function"] == "trade_fee"
    assert meta["SIDE_LIQUID"] == float(SIDE_LIQUID)
    assert meta["SIDE_ILLIQUID"] == float(SIDE_ILLIQUID)
    assert meta["DV_LIQUID"] == float(DV_LIQUID)
    assert meta["cost_mult"] == 1.0
    assert side_cost("AAPL", float(DV_LIQUID)) == pytest.approx(float(SIDE_LIQUID))
    assert side_cost("AAPL", 1.0) == pytest.approx(float(SIDE_ILLIQUID))
    assert side_cost("GLD", 1.0) == pytest.approx(float(SIDE_LIQUID))
    assert side_cost("USO", 1.0) == pytest.approx(float(SIDE_LIQUID))
    fee = turnover_fee({"AAPL": 1.0}, {}, {"AAPL": float(SIDE_LIQUID)})
    assert fee == pytest.approx(float(SIDE_LIQUID))
    assert fee > 0


def test_hist_score_is_fm_score() -> None:
    _need_upleg()
    import numpy as np
    from upleg_bt.costs import SIDE_ILLIQUID, SIDE_LIQUID
    from upleg_bt.engine.fm import FM

    empty = score_symbols(
        ["AAPL", "GLD"],
        route=[0, 0],
        kind=[0, 1],
        side=[float(SIDE_LIQUID), float(SIDE_ILLIQUID)],
    )
    assert empty["AAPL"] == pytest.approx(-2 * float(SIDE_LIQUID))
    assert empty["GLD"] == pytest.approx(-2 * float(SIDE_ILLIQUID))
    # 30 finished samples in AAA's cell, 10 in another cell, so the
    # cell mean and the overall fallback are not the same number.
    segments = tuple((0, 0.40, float(SIDE_LIQUID), 0.0, 0) for _ in range(30))
    segments += tuple((3, 0.0, float(SIDE_LIQUID), 0.0, 0) for _ in range(10))
    scored = score_symbols(
        ["AAA", "BBB"],
        route=[0, 0],
        kind=[0, 1],
        side=[float(SIDE_LIQUID), float(SIDE_LIQUID)],
        segments=segments,
    )
    direct = FM.score(_book(segments), "hist", 0, np.arange(2), 1.0)
    assert scored["AAA"] == pytest.approx(float(direct[0]))
    assert scored["BBB"] == pytest.approx(float(direct[1]))
    assert scored["AAA"] != pytest.approx(scored["BBB"])


def _book(segments: tuple[tuple[int, float, float, float, int], ...]):
    import numpy as np
    from upleg_bt.costs import SIDE_LIQUID

    from bakeoff.fixed20.hist import load_hist_view

    view_cls = load_hist_view()
    cid, ret, side, halves, exits = zip(*segments)
    view = view_cls()
    view.T = 0
    view._cells = {}
    view._sg = (
        np.asarray(cid, dtype=np.int64),
        np.asarray(ret, dtype=np.float64),
        np.asarray(side, dtype=np.float64),
        np.asarray(halves, dtype=np.float64),
        np.asarray(exits, dtype=np.int64),
    )
    view.ROUTE = np.zeros((1, 2), dtype=np.int64)
    view.kind = np.asarray([0, 1], dtype=np.int64)
    view.MKT = np.zeros(1, dtype=np.int64)
    view.SIDE = np.full((1, 2), float(SIDE_LIQUID))
    return view


def test_commodity_priority_only_when_two_are_in_the_pool() -> None:
    scores = {"GLD": 0.1, "USO": 0.2, "AAPL": 0.9, "MSFT": 0.8}
    ranked = rank_pool(["AAPL", "USO", "GLD", "MSFT"], scores)
    assert ranked[:2] == ["GLD", "USO"]
    assert ranked[2:] == ["AAPL", "MSFT"]
    alone = rank_pool(["USO", "AAPL", "MSFT"], {"USO": 0.1, "AAPL": 0.2, "MSFT": 0.9})
    assert alone == ["MSFT", "AAPL", "USO"]


def test_bh_equal_weight_then_hold_and_skips_missing_names() -> None:
    day = date(2024, 1, 2)
    view = _view({"AAA": (10.0,), "BBB": (10.0,)}, session=day)
    # CCC is in the universe but has no bar on t0.
    first = bh_targets(view, universe=("AAA", "BBB", "CCC"), first=True)
    assert first.hold is False
    assert first.weights == {"AAA": pytest.approx(0.5), "BBB": pytest.approx(0.5)}
    assert abs(sum(first.weights.values()) - 1.0) < 1e-9
    assert first.reasons["CCC"] == "missing_at_t0"
    held = _view(
        {"AAA": (10.0,), "BBB": (11.0,)},
        session=day,
        prev=first.weights,
        has_book=True,
    )
    second = bh_targets(held, universe=("AAA", "BBB", "CCC"), first=False)
    assert second.hold is True
    assert second.weights == first.weights


def test_bh_full_fixed20_is_one_twentieth() -> None:
    day = date(2024, 1, 2)
    view = _view({symbol: (10.0,) for symbol in FIXED20}, session=day)
    book = bh_targets(view, first=True)
    assert set(book.weights) == set(FIXED20)
    assert book.weights["META"] == pytest.approx(0.05)
    assert abs(sum(book.weights.values()) - 1.0) < 1e-9


def test_n1_takes_three_and_hard_stop_beats_rank() -> None:
    source = inspect.getsource(n1_targets)
    body = source.split('"""', 2)[-1]
    assert "fib_stages" not in body
    assert "FM" not in body
    assert "score_symbols" not in body
    assert "stage_snapshot" not in body
    rising = tuple(float(100 + index) for index in range(60))
    below = tuple(float(100 - index) for index in range(60))
    view = _view({"LEAD": rising, "SOFT": below, "WEAK": below})
    book = n1_targets(view, universe=("LEAD", "SOFT", "WEAK"))
    assert book.weights["LEAD"] == pytest.approx(1.0)
    assert "WEAK" not in book.weights
    soft = tuple(float(80 + index * 0.2) for index in range(60))
    crashed = _view(
        {"LEAD": rising, "SOFT": soft},
        avg_cost={"LEAD": rising[-1] / (1.0 + HARD_STOP - 0.01)},
        prev={"LEAD": 0.5, "SOFT": 0.5},
        has_book=True,
    )
    stopped = n1_targets(crashed, universe=("LEAD", "SOFT"))
    assert "LEAD" not in stopped.weights
    assert stopped.reasons["LEAD"] == "hard_stop"
    assert stopped.weights["SOFT"] == pytest.approx(1.0)


def test_n1_does_not_halve_when_price_breaks_the_average() -> None:
    # Last close is under the 50-day mean, so the name is a full exit.
    closes = tuple(float(100 - index) for index in range(60))
    view = _view({"AAA": closes}, prev={"AAA": 1.0}, has_book=True)
    book = n1_targets(view, universe=("AAA",))
    assert book.weights == {}
    assert book.reasons["AAA"] == "break_ma"


def test_f_refuses_without_upleg() -> None:
    if probe() is None:
        pytest.skip("upleg_bt 已安装，F 规则改由下面的测试覆盖")
    closes = tuple(float(100 + index) for index in range(60))
    with pytest.raises(UplegUnavailable):
        f_targets(_view({"AAA": closes}), universe=("AAA",))


def test_f_stop_zeroes_before_halve(monkeypatch: pytest.MonkeyPatch) -> None:
    _need_upleg()
    monkeypatch.setattr(
        "bakeoff.fixed20.targets.stage_snapshot",
        lambda view, symbol: ("grind", 0),
    )
    monkeypatch.setattr(
        "bakeoff.fixed20.targets.score_symbols",
        lambda symbols, **kwargs: {symbol: 0.0 for symbol in symbols},
    )
    closes = tuple(float(100 + index) for index in range(60))
    view = _view(
        {"AAA": closes},
        avg_cost={"AAA": closes[-1] / 0.90},
        prev={"AAA": 1.0},
        has_book=True,
    )
    book = f_targets(view, universe=("AAA",))
    assert book.weights == {}
    assert book.reasons["AAA"] == "hard_stop"
    assert book.hist_calls == 1


def test_f_fish_does_not_open_a_new_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _need_upleg()
    monkeypatch.setattr(
        "bakeoff.fixed20.targets.stage_snapshot",
        lambda view, symbol: ("fish", 3),
    )
    monkeypatch.setattr(
        "bakeoff.fixed20.targets.score_symbols",
        lambda symbols, **kwargs: {symbol: 1.0 for symbol in symbols},
    )
    closes = tuple(float(50 + index) for index in range(60))
    fresh = f_targets(_view({"AAA": closes}), universe=("AAA",))
    assert fresh.weights == {}
    assert fresh.reasons["AAA"] == "fish_no_new"
    held = f_targets(
        _view({"AAA": closes}, prev={"AAA": 1.0}, has_book=True),
        universe=("AAA",),
    )
    assert held.weights["AAA"] == pytest.approx(1.0)
    assert held.reasons["AAA"] == "f_select"


def test_down_confirm_needs_three_falling_days() -> None:
    falling = tuple(float(200 - index) for index in range(60))
    rising = tuple(float(100 + index) for index in range(60))
    assert down_confirmed(falling)
    assert not down_confirmed(rising)
