"""Run A-share tactic backtests (daily-bar testable set) and write reports.

    PYTHONPATH=research python -m ashare_tactics.run

Raw bars: /Volumes/Disk/项目/data-cache/ashare_tactics (gitignored).
Results: research/ashare_tactics/results/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_tactics.bars import arrays_from_frame, load_frame, symbol_bars_from_frame
from ashare_tactics.constants import INDEX_ALL, INDEX_CSI300, INITIAL_CASH, SKIPPED, START
from ashare_tactics.download import download_all
from ashare_tactics.metrics import benchmark_equity, slope_breakdown, summarize, yearly_returns
from ashare_tactics.plotting import plot_equity, plot_top2_vs_csi300
from ashare_tactics.portfolio import simulate
from ashare_tactics.report import write_summary
from ashare_tactics.signals import CORE_TACTICS, TACTICS, tactic_dual_indicator


def _guard(path: Path, allowed: tuple[str, ...]) -> Path:
    resolved = path.resolve()
    parts = set(resolved.parts)
    if not any(a in parts or a in str(resolved) for a in allowed):
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


def run(cache: Path, out: Path, start: str, end: str, workers: int, max_symbols: int, skip_download: bool) -> None:
    t_run = time.time()
    cache = _guard(cache, ("data", "tmp", "data-cache"))
    out = _guard(out, ("results", "tmp", "research", "ashare_tactics"))
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
        if getattr(row, "source", "") in {"sse_delist", "szse_delist"} and pd.notna(row.delist_date):
            delist_dates[row.symbol] = np.datetime64(pd.Timestamp(row.delist_date), "D")
            delist_symbols.add(row.symbol)
    st_flags = {symbol: _name_is_st(name) for symbol, name in names.items()}

    batches = {key: [] for key, _label, _fn in TACTICS}
    batches["tactic25_no_st_filter"] = []
    signal_counts = {key: 0 for key in list(batches) + ["limit_up_skips"]}
    present_jan = 0
    bars_ok = 0
    sse_delist_bars = 0
    szse_delist_bars = 0
    symbols = [row.symbol for row in universe.itertuples(index=False) if (bar_dir / f"{row.symbol}.parquet").exists()]
    t_sig = time.time()
    for n, symbol in enumerate(symbols, start=1):
        try:
            frame = load_frame(bar_dir / f"{symbol}.parquet", delist_dates.get(symbol))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {symbol}: {exc}", flush=True)
            continue
        if frame.empty:
            continue
        bars_ok += 1
        src = universe.loc[universe["symbol"] == symbol, "source"]
        src_val = str(src.iloc[0]) if len(src) else ""
        if src_val == "sse_delist":
            sse_delist_bars += 1
        elif src_val == "szse_delist":
            szse_delist_bars += 1
        dates = frame["date"].to_numpy(dtype="datetime64[D]")
        if dates.min() <= np.datetime64("2016-01-31") and dates.max() >= np.datetime64("2016-01-05"):
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
    print(f"signal pass {time.time()-t_sig:.0f}s", flush=True)

    def loader(symbol: str):
        return symbol_bars_from_frame(load_frame(bar_dir / f"{symbol}.parquet", delist_dates.get(symbol)))

    bench_300 = benchmark_equity(csi300, calendar, INITIAL_CASH)
    bench_all = benchmark_equity(csi_all, calendar, INITIAL_CASH)
    summary_rows = []
    yearly = {"年份": [int(y) for y in sorted(set(calendar.year))]}
    trades_by_key: dict[str, pd.DataFrame] = {}
    equity_by_key: dict[str, pd.Series] = {}
    labels = {key: label for key, label, _fn in TACTICS}
    labels["tactic25_no_st_filter"] = "战术25（不做ST过滤）"

    run_list = list(TACTICS) + [("tactic25_no_st_filter", "", None)]
    for key, _label, _fn in run_list:
        print(f"simulate {key} candidates={len(batches[key])}", flush=True)
        result = simulate(batches[key], calendar, loader, INITIAL_CASH)
        result.equity.to_frame("equity").to_parquet(out / f"equity_{key}.parquet")
        if len(result.trades):
            result.trades.to_csv(out / f"trades_{key}.csv", index=False)
        trades_by_key[key] = result.trades
        equity_by_key[key] = result.equity
        row = summarize(labels[key], result.equity, result.trades, INITIAL_CASH)
        summary_rows.append(row)
        returns = yearly_returns(result.equity, INITIAL_CASH)
        yearly[labels[key]] = [float(returns.get(year, np.nan)) for year in yearly["年份"]]
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
    slope = slope_breakdown(trades_by_key.get("tactic5", pd.DataFrame()))
    summary.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    yearly_frame.to_csv(out / "yearly_returns.csv", index=False)
    slope.to_csv(out / "tactic5_slope.csv", index=False)

    # Top 2 tactics by annualized return (exclude benchmarks / sensitivity row)
    tactic_rows = summary[
        summary["strategy"].astype(str).str.startswith("战术")
        & ~summary["strategy"].astype(str).str.contains("不做")
    ].copy()
    tactic_rows = tactic_rows.sort_values("annualized_return", ascending=False)
    top2 = tactic_rows.head(2)
    named = []
    for _, r in top2.iterrows():
        # find key
        key = next(k for k, lab in labels.items() if lab == r["strategy"])
        named.append((r["strategy"], equity_by_key[key]))
    if named:
        plot_top2_vs_csi300(
            out / "top2_vs_csi300.png",
            "表现最好的两套战术 vs 沪深300",
            named,
            bench_300.reindex(named[0][1].index).ffill(),
        )

    # Also keep one chart for each of the original four (sanity)
    for key, label, _fn in CORE_TACTICS:
        plot_equity(
            out / f"equity_{key}.png",
            f"{label}：净值曲线",
            label,
            equity_by_key[key],
            bench_300.reindex(equity_by_key[key].index).ffill(),
            bench_all.reindex(equity_by_key[key].index).ffill(),
        )

    failures = cache / "failures.txt"
    n_fail = 0 if not failures.exists() else sum(1 for line in failures.read_text().splitlines() if line.strip())
    spot_n = int((universe["source"] == "sina_hs_a").sum()) if "source" in universe.columns else len(universe)
    sse_added = int((universe["source"] == "sse_delist").sum()) if "source" in universe.columns else 0
    szse_added = int((universe["source"] == "szse_delist").sum()) if "source" in universe.columns else 0
    coverage = {
        "spot": spot_n,
        "sse_delist_added": sse_added,
        "szse_delist_added": szse_added,
        "sse_delist_bars": sse_delist_bars,
        "szse_delist_bars": szse_delist_bars,
        "bars": bars_ok,
        "failed": n_fail,
        "present_jan_2016": present_jan,
        "runtime_sec": round(time.time() - t_run),
    }
    (out / "coverage.json").write_text(
        json.dumps(
            {
                **coverage,
                "signals": signal_counts,
                "end": str(calendar[-1].date()),
                "start": str(calendar[0].date()),
                "skipped": [{"id": i, "reason": r} for i, r in SKIPPED],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(
        out / "summary_zh.md",
        {
            "summary": summary,
            "yearly": yearly_frame,
            "slope": slope,
            "coverage": coverage,
            "end": str(calendar[-1].date()),
            "start": str(calendar[0].date()),
            "skipped": SKIPPED,
            "delist_pnl": _delist_pnl_text(trades_by_key, delist_symbols, labels),
            "top2": [(r["strategy"], float(r["annualized_return"])) for _, r in top2.iterrows()],
        },
    )
    print(f"wrote {out} in {time.time()-t_run:.0f}s", flush=True)


def _delist_pnl_text(trades_by_key: dict[str, pd.DataFrame], delist_symbols: set[str], labels: dict) -> str:
    if not delist_symbols:
        return "（没有可对照的退市代码）"
    bits = []
    for key, trades in trades_by_key.items():
        if key.endswith("no_st_filter"):
            continue
        if trades is None or len(trades) == 0:
            bits.append(f"{labels.get(key, key)} 0 笔")
            continue
        part = trades[trades["symbol"].isin(delist_symbols)]
        pnl = float(part["pnl"].sum()) if len(part) else 0.0
        bits.append(f"{labels.get(key, key)} {len(part)} 笔、盈亏 {pnl:,.0f} 元")
    return "；".join(bits)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Backtest daily-bar A-share tactics.")
    parser.add_argument("--cache", type=Path, default=Path("/Volumes/Disk/项目/data-cache/ashare_tactics"))
    parser.add_argument("--out", type=Path, default=root / "research" / "ashare_tactics" / "results")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-10-16")
    parser.add_argument("--workers", type=int, default=6)
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
