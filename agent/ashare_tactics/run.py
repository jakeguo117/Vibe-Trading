"""Run the four A-share tactic backtests and write the gitignored report.

    PYTHONPATH=agent python -m ashare_tactics.run

Raw bars go to ``data/ashare_tactics_2015`` (gitignored). Tables and charts go
to ``results/ashare_tactics_2015`` (also gitignored). Copy the report out if
you need to attach it; do not commit the parquet cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_tactics.bars import arrays_from_frame, load_frame, symbol_bars_from_frame
from ashare_tactics.constants import INDEX_ALL, INDEX_CSI300, INITIAL_CASH
from ashare_tactics.download import download_all
from ashare_tactics.metrics import benchmark_equity, slope_breakdown, summarize, yearly_returns
from ashare_tactics.plotting import plot_equity
from ashare_tactics.portfolio import simulate
from ashare_tactics.report import write_summary
from ashare_tactics.signals import TACTICS, tactic_dual_indicator


def _guard(path: Path, allowed: tuple[str, ...]) -> Path:
    resolved = path.resolve()
    if not any(part in allowed for part in resolved.parts):
        raise SystemExit(f"refusing to write outside {allowed}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _name_is_st(name: str) -> bool:
    return "ST" in str(name).upper()


def _load_index(cache: Path, symbol: str) -> pd.Series:
    frame = pd.read_parquet(cache / f"index_{symbol}.parquet")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    series = frame.drop_duplicates("date").set_index("date")["close"].astype(float).sort_index()
    return series


def _calendar(index_close: pd.Series, start: np.datetime64, end: np.datetime64) -> pd.DatetimeIndex:
    days = index_close.index
    mask = (days.values.astype("datetime64[D]") >= np.datetime64(start, "D")) & (
        days.values.astype("datetime64[D]") <= np.datetime64(end, "D")
    )
    calendar = pd.DatetimeIndex(days[mask]).normalize()
    if len(calendar) == 0:
        raise SystemExit("benchmark calendar is empty")
    return calendar


def _delist_pnl_text(trades_by_key: dict[str, pd.DataFrame], delist_symbols: set[str]) -> str:
    if not delist_symbols:
        return "（没有可对照的上交所退市代码）"
    names = {
        "tactic5": "战术5",
        "tactic13": "战术13",
        "tactic19": "战术19",
        "tactic25": "战术25",
        "tactic25_no_st_filter": "战术25（不做ST过滤）",
    }
    bits = []
    for key, trades in trades_by_key.items():
        if trades is None or len(trades) == 0:
            bits.append(f"{names.get(key, key)} 0 笔")
            continue
        part = trades[trades["symbol"].isin(delist_symbols)]
        pnl = float(part["pnl"].sum()) if len(part) else 0.0
        bits.append(f"{names.get(key, key)} {len(part)} 笔、盈亏 {pnl:,.0f} 元")
    return "；".join(bits)


def run(cache: Path, out: Path, start: str, end: str, workers: int, max_symbols: int, skip_download: bool) -> None:
    cache = _guard(cache, ("data", "tmp"))
    out = _guard(out, ("results", "tmp"))
    if not skip_download:
        download_all(cache, workers=workers, max_symbols=max_symbols)
    universe = pd.read_parquet(cache / "universe.parquet")
    if max_symbols > 0:
        universe = universe.head(max_symbols).copy()
    bar_dir = cache / "bars"
    start_d = np.datetime64(start, "D")
    end_d = np.datetime64(end, "D")
    csi300 = _load_index(cache, INDEX_CSI300)
    csi_all = _load_index(cache, INDEX_ALL)
    calendar = _calendar(csi300, start_d, end_d)
    print(f"calendar {calendar[0].date()} → {calendar[-1].date()} ({len(calendar)} sessions)", flush=True)

    names = {row.symbol: row.name for row in universe.itertuples(index=False)}
    delist_dates = {}
    delist_symbols = set()
    for row in universe.itertuples(index=False):
        if row.source == "sse_delist" and pd.notna(row.delist_date):
            delist_dates[row.symbol] = np.datetime64(pd.Timestamp(row.delist_date), "D")
            delist_symbols.add(row.symbol)
    st_flags = {symbol: _name_is_st(name) for symbol, name in names.items()}

    batches = {key: [] for key, _label, _fn in TACTICS}
    batches["tactic25_no_st_filter"] = []
    signal_counts = {key: 0 for key in list(batches) + ["limit_up_skips"]}
    present_jan = 0
    bars_ok = 0
    sse_delist_bars = 0
    symbols = [row.symbol for row in universe.itertuples(index=False) if (bar_dir / f"{row.symbol}.parquet").exists()]
    for n, symbol in enumerate(symbols, start=1):
        try:
            frame = load_frame(bar_dir / f"{symbol}.parquet", delist_dates.get(symbol))
        except Exception as exc:  # noqa: BLE001 - one bad cache file should not abort
            print(f"skip {symbol}: {exc}", flush=True)
            continue
        if frame.empty:
            continue
        bars_ok += 1
        if symbol in delist_symbols:
            sse_delist_bars += 1
        dates = frame["date"].to_numpy(dtype="datetime64[D]")
        if dates.min() <= np.datetime64("2015-01-31") and dates.max() >= np.datetime64("2015-01-05"):
            present_jan += 1
        arrays = arrays_from_frame(frame)
        is_st = st_flags.get(symbol, False)
        for key, _label, fn in TACTICS:
            batch = fn(symbol, is_st=is_st, **arrays)
            signal_counts[key] += batch.signals
            signal_counts["limit_up_skips"] += batch.limit_up_skips
            batches[key].extend(
                c for c in batch.candidates if start_d <= np.datetime64(c.entry_date, "D") <= end_d
            )
        extra = tactic_dual_indicator(symbol, is_st=is_st, apply_st_filter=False, **arrays)
        batches["tactic25_no_st_filter"].extend(
            c for c in extra.candidates if start_d <= np.datetime64(c.entry_date, "D") <= end_d
        )
        if n % 400 == 0 or n == len(symbols):
            sizes = {key: len(val) for key, val in batches.items()}
            print(f"signals {n}/{len(symbols)} {sizes}", flush=True)

    def loader(symbol: str):
        return symbol_bars_from_frame(load_frame(bar_dir / f"{symbol}.parquet", delist_dates.get(symbol)))

    bench_300 = benchmark_equity(csi300, calendar, INITIAL_CASH)
    bench_all = benchmark_equity(csi_all, calendar, INITIAL_CASH)
    summary_rows = []
    yearly = {"年份": [int(y) for y in sorted(set(calendar.year))]}
    trades_by_key: dict[str, pd.DataFrame] = {}
    labels = {key: label for key, label, _fn in TACTICS}
    labels["tactic25_no_st_filter"] = "战术25（不做ST过滤）"
    chart_titles = {key: f"{label}：净值曲线" for key, label in labels.items()}

    for key, _label, _fn in list(TACTICS) + [("tactic25_no_st_filter", "", None)]:
        print(f"simulate {key} candidates={len(batches[key])}", flush=True)
        result = simulate(batches[key], calendar, loader, INITIAL_CASH)
        result.equity.to_frame("equity").to_parquet(out / f"equity_{key}.parquet")
        if len(result.trades):
            result.trades.to_csv(out / f"trades_{key}.csv", index=False)
        trades_by_key[key] = result.trades
        row = summarize(labels[key], result.equity, result.trades, INITIAL_CASH)
        summary_rows.append(row)
        returns = yearly_returns(result.equity, INITIAL_CASH)
        yearly[labels[key]] = [float(returns.get(year, np.nan)) for year in yearly["年份"]]
        if key in {item[0] for item in TACTICS}:
            plot_equity(
                out / f"equity_{key}.png",
                chart_titles[key],
                labels[key],
                result.equity,
                bench_300.reindex(result.equity.index).ffill(),
                bench_all.reindex(result.equity.index).ffill(),
            )
        print(
            f"  ann={row['annualized_return']:.3%} mdd={row['max_drawdown']:.3%} "
            f"sharpe={row['sharpe']:.2f} trades={row['trade_count']}",
            flush=True,
        )

    for key, series, label in (
        ("csi300", bench_300, "沪深300"),
        ("csi_all", bench_all, "中证全指"),
    ):
        row = summarize(label, series, pd.DataFrame(), INITIAL_CASH)
        row["win_rate"] = np.nan
        row["avg_holding_days"] = np.nan
        row["trade_count"] = 0
        summary_rows.append(row)
        returns = yearly_returns(series, INITIAL_CASH)
        yearly[label] = [float(returns.get(year, np.nan)) for year in yearly["年份"]]

    summary = pd.DataFrame(summary_rows)
    yearly_frame = pd.DataFrame(yearly)
    slope = slope_breakdown(trades_by_key["tactic5"])
    summary.to_csv(out / "summary.csv", index=False)
    yearly_frame.to_csv(out / "yearly_returns.csv", index=False)
    slope.to_csv(out / "tactic5_slope.csv", index=False)

    failures = cache / "failures.txt"
    n_fail = 0 if not failures.exists() else sum(1 for line in failures.read_text().splitlines() if line.strip())
    spot_n = int((universe["source"] == "sina_hs_a").sum()) if "source" in universe.columns else len(universe)
    added = int((universe["source"] == "sse_delist").sum()) if "source" in universe.columns else 0
    coverage = {
        "spot": spot_n,
        "sse_delist_added": added,
        "sse_delist_bars": sse_delist_bars,
        "bars": bars_ok,
        "failed": n_fail,
        "present_jan_2015": present_jan,
    }
    (out / "coverage.json").write_text(
        json.dumps({**coverage, "signals": signal_counts, "end": str(calendar[-1].date())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(
        out / "summary_zh.md",
        {
            "summary": summary,
            "yearly": yearly_frame.rename(columns={"年份": "年份"}),
            "slope": slope,
            "coverage": coverage,
            "end": str(calendar[-1].date()),
            "delist_pnl": _delist_pnl_text(trades_by_key, delist_symbols),
        },
    )
    print(f"wrote {out}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Backtest four A-share short-term tactics.")
    parser.add_argument("--cache", type=Path, default=root / "data" / "ashare_tactics_2015")
    parser.add_argument("--out", type=Path, default=root / "results" / "ashare_tactics_2015")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-10-16")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run(
        cache=args.cache,
        out=args.out,
        start=args.start,
        end=args.end,
        workers=args.workers,
        max_symbols=args.max_symbols,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
