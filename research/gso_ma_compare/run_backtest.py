"""Run the GLD/SLV/USO 10/20/30-week up-leg study.

Usage (from the repo root):
    python research/gso_ma_compare/run_backtest.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gso.config import (  # noqa: E402
    IS_END,
    LONG_START,
    MAIN_START,
    OOS_START,
    SUB_2022_START,
    SYMBOLS,
)
from gso.data import load_universe  # noqa: E402
from gso.indicators import fib_stage_frame, sma_upleg, to_weekly, wilder_atr  # noqa: E402
from gso.simulate import run_strategy  # noqa: E402
from gso.study import (  # noqa: E402
    attach_gold_windows,
    build_capture_table,
    detect_rallies,
    metric_row,
    plot_three,
    render_summary,
    slice_nav,
)

VERSIONS = (
    ("10周", "W", 10),
    ("20周", "W", 20),
    ("30周", "W", 30),
    ("10周日频", "D", 10),
    ("20周日频", "D", 20),
    ("30周日频", "D", 30),
)


def _diagnose(daily_map: dict[str, pd.DataFrame]) -> None:
    for symbol in SYMBOLS:
        weekly = to_weekly(daily_map[symbol])
        atr = wilder_atr(weekly["High"], weekly["Low"], weekly["Close"])
        fib = fib_stage_frame(weekly["High"], weekly["Low"], weekly["Close"], atr)
        counts = fib["fib_stage"].value_counts().to_dict()
        print(f"{symbol} weeks={len(weekly)} fib={counts}", flush=True)
        for length in (10, 20, 30):
            up = sma_upleg(weekly["Close"], length, 2, 4)
            print(
                f"  MA{length} upleg={up['in_upleg'].mean():.3f} halved_while_in="
                f"{(up['halve'] & up['in_upleg']).mean():.3f}",
                flush=True,
            )
        rets = daily_map[symbol]["Close"].pct_change()
        jumps = rets[rets.abs() > 0.45]
        if len(jumps):
            print(f"  {symbol} daily moves beyond 45%: {len(jumps)} (check split adjustment)", flush=True)


def main() -> None:
    cache = ROOT / "cache"
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    daily_map = load_universe(SYMBOLS, cache)
    _diagnose(daily_map)

    first = {}
    last_dates = []
    for symbol in SYMBOLS:
        frame = daily_map[symbol]
        first[symbol] = frame.index.min().strftime("%Y-%m-%d")
        last_dates.append(frame.index.max())
        print(
            f"{symbol} {frame.index.min().date()} .. {frame.index.max().date()} rows={len(frame)}",
            flush=True,
        )
    common = daily_map["GLD"].index
    for symbol in SYMBOLS[1:]:
        common = common.intersection(daily_map[symbol].index)
    common = pd.DatetimeIndex(common).sort_values()
    long_start = max(pd.Timestamp(LONG_START), common.min())
    last = min(last_dates)
    print(f"book start {long_start.date()} last common session {common.max().date()}", flush=True)

    runs = {}
    for name, freq, length in VERSIONS:
        print(f"running {name}", flush=True)
        result = run_strategy(daily_map, freq, length, long_start.strftime("%Y-%m-%d"))
        result["nav"] = result["nav"].loc[:last]
        result["participation"] = result["participation"].loc[:last]
        result["closes"] = result["closes"].loc[:last]
        result["opens"] = result["opens"].loc[:last]
        runs[name] = result
        print(
            f"  end nav={result['nav'].iloc[-1]:.4f} max gross={result['max_gross']:.3f} "
            f"entries={sum(1 for ep in result['episodes'] if not ep['open'])}+"
            f"{sum(1 for ep in result['episodes'] if ep['open'])} open",
            flush=True,
        )
        if result["max_gross"] > 0.85:
            raise RuntimeError(f"{name} gross exposure {result['max_gross']:.3f} broke the no-leverage bound")

    periods = (
        ("主样本", MAIN_START, last),
        ("2022年以来", SUB_2022_START, last),
        ("长历史", long_start, last),
        ("样本内", MAIN_START, IS_END),
        ("样本外", OOS_START, last),
    )
    rows = []
    for name, freq, length in VERSIONS:
        for period, start, end in periods:
            rows.append(metric_row(name, freq, length, period, runs[name]["nav"], runs[name]["episodes"], start, end))
    metrics = pd.DataFrame(rows)

    weekly = {symbol: to_weekly(daily_map[symbol]) for symbol in SYMBOLS}
    rallies = []
    for symbol in SYMBOLS:
        rallies.extend(detect_rallies(weekly[symbol], symbol))
    rallies = attach_gold_windows(rallies, weekly)
    capture = build_capture_table(rallies, runs)

    meta = {"first": first, "last": last.strftime("%Y-%m-%d"), "long_start": long_start.strftime("%Y-%m-%d")}
    hashes = {}
    for symbol in SYMBOLS:
        meta_path = cache / f"{symbol}.json"
        if meta_path.exists():
            hashes[symbol] = json.loads(meta_path.read_text(encoding="utf-8")).get("sha256", "")
    meta["sha256"] = hashes

    summary = render_summary(metrics, capture, [name for name, _, _ in VERSIONS], meta)
    summary_path = results / "summary_zh.md"
    summary_path.write_text(summary, encoding="utf-8")
    metrics.to_csv(results / "metrics.csv", index=False)
    capture.to_csv(results / "capture.csv", index=False)

    chart_navs = {}
    for length, name in ((10, "10周"), (20, "20周"), (30, "30周")):
        window = slice_nav(runs[name]["nav"], MAIN_START, last)
        chart_navs[length] = window / window.iloc[0]
    plot_three(
        chart_navs,
        last,
        results / "equity_main.png",
        results / "drawdown_main.png",
    )

    artifacts = Path("/opt/cursor/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    for filename in ("summary_zh.md", "metrics.csv", "capture.csv", "equity_main.png", "drawdown_main.png"):
        shutil.copy(results / filename, artifacts / filename)

    print(summary, flush=True)
    if not np.isfinite(metrics["年化收益"]).all():
        raise RuntimeError("a performance row has a non-finite annualized return")


if __name__ == "__main__":
    main()
