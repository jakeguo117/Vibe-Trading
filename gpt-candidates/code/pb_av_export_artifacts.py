"""Write market-data artifacts using the Drive warehouse file names.

Does not upload anywhere and does not call Databento.
The account field is the env var name DATABENTO_API_KEY, never the secret.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

G = Path("/workspace/gpt-candidates")
OUT = G / "out"
MIN_DIR = G / "data" / "minutes"
YH = G / "data" / "yahoo" / "daily.parquet"
MANIFEST = OUT / "pb_av_vs_pb_manifest.json"
ART = Path("/opt/cursor/artifacts")
MIN_OUT = ART / "databento" / "ohlcv-1m"
BARS_NAME = "databento_EQUS.MINI_ohlcv-1m_20230328-20261023_pb-minute-ticker-days.parquet"
LIST_NAME = "databento_EQUS.MINI_ohlcv-1m_20230328-20261023_pb-minute-ticker-days-list.csv"
MANIFEST_NAME = "databento_EQUS.MINI_ohlcv-1m_20230328-20261023_pb-minute-ticker-days_manifest.json"
KEY = os.environ.get("DATABENTO_API_KEY", "")


def scrub(text: str) -> str:
    if KEY:
        text = text.replace(KEY, "[REDACTED]")
    return text


def export_minutes() -> dict:
    """Monthly shards. The artifact store rejects a single ~200MB object."""
    MIN_OUT.mkdir(parents=True, exist_ok=True)
    by_month: dict[str, list[Path]] = {}
    for path in sorted(MIN_DIR.glob("*.parquet")):
        if path.name.startswith("."):
            continue
        by_month.setdefault(path.stem[:7], []).append(path)
    shards = []
    n_rows = 0
    symbols: set[str] = set()
    n_empty = 0
    rows_by_date: dict[str, int] = {}
    for month, paths in sorted(by_month.items()):
        frames = []
        for path in paths:
            df = pd.read_parquet(path)
            rows_by_date[path.stem] = int(len(df))
            if len(df) == 0:
                n_empty += 1
                continue
            df["session_date"] = path.stem
            df["symbol"] = df["symbol"].astype(str)
            frames.append(df[["symbol", "ts_event", "open", "high", "low", "close", "volume", "session_date"]])
        if not frames:
            continue
        all_df = pd.concat(frames, ignore_index=True)
        d0 = min(p.stem for p in paths).replace("-", "")
        d1 = max(p.stem for p in paths).replace("-", "")
        name = f"databento_EQUS.MINI_ohlcv-1m_{d0}-{d1}_pb-minute-ticker-days.parquet"
        all_df.to_parquet(MIN_OUT / name, index=False, compression="zstd")
        n = int(len(all_df))
        n_rows += n
        symbols.update(all_df.symbol.unique())
        shards.append({"file": f"databento/ohlcv-1m/{name}", "start": d0, "end": d1, "n_rows": n, "n_session_dates": len(paths)})
        print(f"shard {name} rows={n}", flush=True)
    return {
        "shards": shards,
        "n_rows": n_rows,
        "n_session_files": sum(len(v) for v in by_month.values()),
        "n_empty_session_files": n_empty,
        "n_symbols_with_rth_bars": len(symbols),
        "rows_by_session_date": rows_by_date,
        "note": "One object for the full window is about 212MB and the artifact store rejected it. Monthly shards are the complete RTH archive.",
    }


def export_yahoo() -> dict | None:
    if not YH.exists():
        return None
    df = pd.read_parquet(YH)
    d0 = str(df.date.min()).replace("-", "")[:8]
    d1 = str(df.date.max()).replace("-", "")[:8]
    name = f"yahoo_yfinance_ohlcv-1d_{d0}-{d1}_pb-signal-universe.parquet"
    dest_dir = ART / "yahoo" / "ohlcv-1d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = G / "data" / "export"
    local.mkdir(parents=True, exist_ok=True)
    staged = local / name
    df.to_parquet(staged, index=False, compression="zstd")
    shutil.copy2(staged, dest_dir / name)
    return {
        "file": f"yahoo/ohlcv-1d/{name}",
        "n_rows": int(len(df)),
        "n_symbols": int(df.symbol.nunique()),
        "start": str(df.date.min())[:10],
        "end": str(df.date.max())[:10],
        "cost_usd": 0,
        "source": "Yahoo Finance via yfinance, auto_adjust=False. Not a Databento purchase.",
        "account_env": None,
    }


def copy_list() -> str:
    src = OUT / "pb_minute_ticker_days_2023plus.csv"
    dest = MIN_OUT / LIST_NAME
    MIN_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return f"databento/ohlcv-1m/{LIST_NAME}"


def copy_results() -> list[str]:
    dest = ART / "pb_av_vs_pb"
    dest.mkdir(parents=True, exist_ok=True)
    names = [
        "trades_compare.csv",
        "equity_compare.csv",
        "metrics_compare.csv",
        "same_bar_minute_review.csv",
        "pb_av_vs_pb_equity_10bp.png",
        "README_结论_EXPLORATORY.md",
        "pb_av_signals.csv",
        "pb_av_vs_pb_facts.json",
        "pb_av_resume_checkpoint.json",
        "pb_av_downloaded_ticker_days.csv",
        "pb_av_remaining_ticker_days.csv",
        "yahoo_match.json",
    ]
    copied = []
    for name in names:
        src = OUT / name
        if not src.exists() and name.startswith("README"):
            src = G / name
        if src.exists():
            shutil.copy2(src, dest / name)
            copied.append(name)
    return copied


def batch_rows(raw: dict, rows_by_date: dict) -> list[dict]:
    out = []
    probe = raw.get("probe") or {}
    if probe:
        out.append(
            {
                "batch_id": probe.get("batch_id"),
                "purpose": "schema_probe",
                "dataset": "EQUS.MINI",
                "schema": "ohlcv-1m",
                "all_symbols": False,
                "date": "2026-09-29",
                "n_symbols": 1,
                "cost_usd": probe.get("cost_usd"),
                "downloaded": True,
                "note": "Inside the 48635 list. The same ticker-day is included again in that day's batch.",
            }
        )
    for b in raw.get("batches", []):
        if b.get("purpose") != "avwap_window":
            continue
        out.append(
            {
                "batch_id": b.get("batch_id"),
                "purpose": "avwap_window",
                "dataset": "EQUS.MINI",
                "schema": "ohlcv-1m",
                "all_symbols": False,
                "date": b.get("date"),
                "start": b.get("start"),
                "end": b.get("end"),
                "n_symbols": b.get("n_symbols"),
                "cost_usd": b.get("cost_usd"),
                "quoted": b.get("quoted"),
                "downloaded": b.get("downloaded"),
                "n_rows_rth_retained": rows_by_date.get(b.get("date")),
                "error": b.get("error"),
            }
        )
    for b in raw.get("restore_batches", []):
        out.append(
            {
                "batch_id": b.get("batch_id"),
                "purpose": "restore_overwritten_day",
                "dataset": "EQUS.MINI",
                "schema": "ohlcv-1m",
                "all_symbols": False,
                "date": b.get("date"),
                "n_symbols": b.get("n_symbols"),
                "cost_usd": b.get("cost_usd"),
                "downloaded": True,
                "n_rows_rth_retained": b.get("rows_rth"),
                "note": "Review pull had replaced the day file. This batch bought the original symbol list again.",
            }
        )
    for b in raw.get("review_batches", []):
        out.append(
            {
                "batch_id": b.get("batch_id"),
                "purpose": "same_bar_review",
                "dataset": "EQUS.MINI",
                "schema": "ohlcv-1m",
                "all_symbols": False,
                "date": b.get("date"),
                "n_symbols": b.get("n_symbols"),
                "cost_usd": b.get("cost_usd"),
                "downloaded": b.get("downloaded"),
                "n_rows_rth_retained": b.get("rows_rth"),
            }
        )
    return out


def write_data_manifest(raw: dict, minute_info: dict, yahoo_info: dict | None, list_file: str) -> None:
    batches = batch_rows(raw, minute_info.pop("rows_by_session_date"))
    probe = float((raw.get("probe") or {}).get("cost_usd") or 0)
    avwap_spent = sum(float(b["cost_usd"] or 0) for b in batches if b["purpose"] == "avwap_window" and b.get("downloaded"))
    review_spent = sum(float(b["cost_usd"] or 0) for b in batches if b["purpose"] == "same_bar_review" and b.get("downloaded"))
    restore_spent = sum(float(b["cost_usd"] or 0) for b in batches if b["purpose"] == "restore_overwritten_day")
    doc = {
        "dataset": "EQUS.MINI",
        "schema": "ohlcv-1m",
        "stype_in": "raw_symbol",
        "all_symbols": False,
        "account_env": "DATABENTO_API_KEY",
        "window": ["2023-03-28", "2026-10-23"],
        "session": "RTH only. Bars whose start is in [09:30, 16:00) America/New_York. Extended hours were not retained.",
        "price_adjustment": "none",
        "coverage": "pb-minute-ticker-days",
        "ticker_days_file": list_file,
        "n_ticker_days_requested": 48635,
        "bars_shards": minute_info["shards"],
        "n_rows": minute_info["n_rows"],
        "n_session_files": minute_info["n_session_files"],
        "n_empty_session_files": minute_info["n_empty_session_files"],
        "n_symbols_with_rth_bars": minute_info["n_symbols_with_rth_bars"],
        "yahoo_daily": yahoo_info,
        "probe_usd": probe,
        "cum_quoted_avwap_usd": raw.get("cum_quoted_avwap_usd"),
        "cum_spent_usd": probe + avwap_spent + review_spent + restore_spent,
        "cum_downloaded_usd_ledger": raw.get("cum_downloaded_usd"),
        "stop_reason": raw.get("stop_reason"),
        "batches": batches,
        "label": "EXPLORATORY",
    }
    text = scrub(json.dumps(doc, ensure_ascii=False, indent=1))
    (MIN_OUT / MANIFEST_NAME).write_text(text)
    print(f"manifest spent={doc['cum_spent_usd']:.4f} rows={doc['n_rows']}", flush=True)


def main() -> None:
    raw = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    minute_info = export_minutes()
    yahoo_info = export_yahoo()
    list_file = copy_list()
    write_data_manifest(raw, minute_info, yahoo_info, list_file)
    copied = copy_results()
    print("results", copied, flush=True)


if __name__ == "__main__":
    main()
