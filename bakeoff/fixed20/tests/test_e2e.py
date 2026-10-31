"""Synthetic three-leg run. Toy prices only; not a Fixed20 history result."""

from __future__ import annotations

import ast
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from bakeoff.fixed20.engine import run_bakeoff, run_synthetic
from bakeoff.fixed20.models import BookView
from bakeoff.fixed20.pit import Bar, BarQuery, MembershipInterval
from bakeoff.fixed20.stages import STAGE_FUNCTION, stage_snapshot
from bakeoff.fixed20.synthetic import (
    LEADER_SYMBOLS,
    TOY_START_INDEX,
    build_material,
    provider_from_material,
    sessions,
    toy_window,
)
from bakeoff.fixed20.universe import FIXED20
from bakeoff.fixed20.upleg_link import UplegUnavailable, probe

_PACKAGE = Path(__file__).resolve().parents[1]
_BANNED = {"requests", "urllib", "httpx", "yfinance", "socket", "aiohttp"}
_REAL_START = date(2019, 1, 2)
_REAL_END = date(2025, 10, 30)


def _need_upleg() -> None:
    reason = probe()
    if reason:
        pytest.skip(reason)


def _bar(symbol: str, session: date, close: str) -> Bar:
    price = Decimal(close)
    return Bar(
        symbol=symbol,
        session_date=session,
        open=price - Decimal("0.10"),
        high=price + Decimal("0.10"),
        low=price - Decimal("0.20"),
        close=price,
        volume=200_000,
    )


def _provider(bars: tuple[Bar, ...], universe: tuple[str, ...]) -> object:
    start = min(bar.session_date for bar in bars)
    intervals = tuple(
        MembershipInterval("fixed20-v1", symbol, start, None) for symbol in universe
    )
    return provider_from_material(bars, intervals, fixture_id="fixed20-unit")


def test_sources_do_not_download_or_write_tables() -> None:
    files = sorted(_PACKAGE.glob("*.py"))
    assert {path.name for path in files} >= {"engine.py", "hist.py", "targets.py", "synthetic.py"}
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert ".to_csv" not in text
        assert "equity_BH" not in text
        assert "summary.csv" not in text
        assert "nav_compare" not in text
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".", 1)[0]]
            else:
                names = []
            leaked = sorted(set(names) & _BANNED)
            assert leaked == [], f"{path.name} imports {leaked}"
            if isinstance(node, ast.FunctionDef) and node.name == "fib_stages_weekly":
                raise AssertionError(f"{path.name} 重写了 fib_stages_weekly")
            if isinstance(node, ast.ClassDef) and node.name == "FM":
                raise AssertionError(f"{path.name} 重写了 FM")


def test_toy_window_is_not_the_real_window() -> None:
    start, end = toy_window()
    assert end < _REAL_START
    assert start != _REAL_START
    assert end != _REAL_END
    assert (end - start).days < 40


def test_shell_fails_closed_without_upleg() -> None:
    if probe() is None:
        pytest.skip("upleg_bt 已安装，三腿合成跑改由确定性测试覆盖")
    with pytest.raises(UplegUnavailable):
        run_synthetic(20)


def test_synthetic_three_legs_are_deterministic() -> None:
    _need_upleg()
    first = run_synthetic(20)
    second = run_synthetic(20)
    assert first.meta["synthetic"] is True
    assert first.meta["real_window"] is False
    assert first.meta["emits_performance_tables"] is False
    assert first.meta["delay"] == 1
    assert first.meta["pit_signal_query_delay"] == 0
    assert first.meta["meta_fb_splice"] is True
    assert first.meta["vol_target"] is None
    assert first.meta["f_hist_calls"] >= 1
    assert first.meta["hist_function"].endswith("FM.score")
    assert first.meta["stage_function"] == STAGE_FUNCTION
    cost = first.meta["cost"]
    assert cost["function"] == "trade_fee"
    assert cost["cost_mult"] == 1.0
    assert first.meta["missing_at_t0"] == []
    assert first.meta["forced_exits"] == []
    assert set(first.legs) == {"BH", "F", "N1"}
    start, end = toy_window()
    assert first.meta["t0"] == start.isoformat()
    assert first.meta["t_end"] == end.isoformat()
    assert start != _REAL_START
    assert end != _REAL_END
    for leg in first.legs.values():
        assert leg.rows
        assert leg.rows[0].date > start
        assert leg.rows[-1].date == end
        assert leg.rows == second.legs[leg.leg].rows
        assert leg.fills == second.legs[leg.leg].fills
    other = run_synthetic(21)
    assert other.legs["BH"].rows != first.legs["BH"].rows


def test_bh_buys_once_and_fee_hits_the_first_close() -> None:
    _need_upleg()
    from upleg_bt.costs import SIDE_LIQUID

    result = run_synthetic(20)
    bh = result.legs["BH"]
    assert bh.fills
    assert len(bh.fills) == len(FIXED20)
    assert {fill.symbol for fill in bh.fills} == set(FIXED20)
    assert {fill.date for fill in bh.fills} == {bh.rows[0].date}
    assert all(fill.side == "buy" for fill in bh.fills)
    assert abs(sum(fill.weight_delta for fill in bh.fills) - 1.0) < 1e-9
    provider = provider_from_material(*build_material(20), fixture_id="fixed20-synthetic-20")
    opened = provider.bars(
        BarQuery(as_of=bh.rows[0].date, symbols=FIXED20, delay=0, start=bh.rows[0].date, end=bh.rows[0].date)
    )
    by_symbol = {bar.symbol: bar for bar in opened.bars}
    gross_move = sum((float(by_symbol[symbol].close) / float(by_symbol[symbol].open) - 1.0) / 20 for symbol in FIXED20)
    expected = (1.0 - float(SIDE_LIQUID)) * (1.0 + gross_move)
    assert bh.rows[0].equity == pytest.approx(expected, abs=1e-9)
    assert bh.rows[0].gross == pytest.approx(1.0, abs=1e-9)
    assert bh.rows[0].cash == pytest.approx(0.0, abs=1e-9)


def test_f_and_n1_trade_the_leaders_only() -> None:
    _need_upleg()
    result = run_synthetic(20)
    for leg in ("F", "N1"):
        buys = [fill for fill in result.legs[leg].fills if fill.side == "buy"]
        assert buys
        assert {fill.symbol for fill in buys} == set(LEADER_SYMBOLS)
        assert {fill.date for fill in buys} == {result.legs[leg].rows[0].date}
        assert abs(sum(fill.weight_delta for fill in buys) - 1.0) < 1e-9
    # Locked stage on the signal day, history cut at t0.
    start = date.fromisoformat(result.meta["t0"])
    provider = provider_from_material(*build_material(20), fixture_id="fixed20-synthetic-20")
    snap = provider.bars(BarQuery(as_of=start, symbols=("AAPL",), delay=0, end=start))
    bars = snap.bars
    view = BookView(
        session=start,
        dates={"AAPL": tuple(bar.session_date for bar in bars)},
        opens={"AAPL": tuple(float(bar.open) for bar in bars)},
        highs={"AAPL": tuple(float(bar.high) for bar in bars)},
        lows={"AAPL": tuple(float(bar.low) for bar in bars)},
        closes={"AAPL": tuple(float(bar.close) for bar in bars)},
        volumes={"AAPL": tuple(bar.volume for bar in bars)},
        avg_cost={},
        prev_weights={},
    )
    assert stage_snapshot(view, "AAPL")[0] == "trend"
    assert all(bar.session_date <= start for bar in bars)


def test_later_close_does_not_change_earlier_marks() -> None:
    _need_upleg()
    bars, intervals = build_material(20)
    calendar = sessions()
    bumped_day = calendar[TOY_START_INDEX + 8]
    changed = []
    for bar in bars:
        if bar.symbol == "AAPL" and bar.session_date == bumped_day:
            close = bar.close + Decimal("1.00")
            changed.append(
                Bar(
                    symbol=bar.symbol,
                    session_date=bar.session_date,
                    open=bar.open,
                    high=max(bar.high, close),
                    low=bar.low,
                    close=close,
                    volume=bar.volume,
                )
            )
        else:
            changed.append(bar)
    start, end = toy_window()
    base = run_bakeoff(
        provider_from_material(bars, intervals, fixture_id="fixed20-base"),
        window_start=start,
        window_end=end,
        synthetic=True,
        seed=20,
    )
    alt = run_bakeoff(
        provider_from_material(tuple(changed), intervals, fixture_id="fixed20-bumped"),
        window_start=start,
        window_end=end,
        synthetic=True,
        seed=20,
    )
    for leg in ("BH", "F", "N1"):
        left = [row for row in base.legs[leg].rows if row.date < bumped_day]
        right = [row for row in alt.legs[leg].rows if row.date < bumped_day]
        assert left and left == right
        on_day = [row for row in base.legs[leg].rows if row.date == bumped_day]
        moved = [row for row in alt.legs[leg].rows if row.date == bumped_day]
        assert on_day and on_day != moved


def test_missing_name_at_t0_is_not_bought_later() -> None:
    _need_upleg()
    origin = date(2024, 3, 4)
    days = [origin + timedelta(days=offset) for offset in range(4)]
    bars = []
    for day in days:
        bars.append(_bar("AAA", day, "50.00"))
        if day != days[0]:
            bars.append(_bar("CCC", day, "40.00"))
    result = run_bakeoff(
        _provider(tuple(bars), ("AAA", "CCC")),
        window_start=days[0],
        window_end=days[-1],
        universe=("AAA", "CCC"),
    )
    assert result.meta["missing_at_t0"] == ["CCC"]
    buys = [fill.symbol for fill in result.legs["BH"].fills if fill.side == "buy"]
    assert buys == ["AAA"]
    assert abs(result.legs["BH"].fills[0].weight_delta - 1.0) < 1e-9


def test_five_missing_sessions_force_an_exit() -> None:
    _need_upleg()
    origin = date(2024, 4, 1)
    days = [origin + timedelta(days=offset) for offset in range(8)]
    bars = [_bar("AAA", day, "30.00") for day in days]
    bars.extend(_bar("BBB", day, "20.00") for day in days[:2])
    result = run_bakeoff(
        _provider(tuple(bars), ("AAA", "BBB")),
        window_start=days[0],
        window_end=days[-1],
        universe=("AAA", "BBB"),
    )
    forced = [item for item in result.meta["forced_exits"] if item["symbol"] == "BBB"]
    assert len(forced) == 1
    assert forced[0]["reason"] == "halt_5"
    bbb = [fill for fill in result.legs["BH"].fills if fill.symbol == "BBB"]
    assert [fill.side for fill in bbb] == ["buy", "sell"]
    assert bbb[-1].reason == "forced_halt"
