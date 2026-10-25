"""PB (daily pullback) vs PB-AV (same rules plus h→t RTH AVWAP) for 2023-03-28..2026-10-23.

Minute data: EQUS.MINI ohlcv-1m for the pre-screened ticker-days only.
Each batch is metadata.get_cost'd before download. No ALL_SYMBOLS. No key logging.
Daily path: Yahoo Finance auto_adjust=False, the same source as frozen_config.yaml.
Results are EXPLORATORY.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/gpt-candidates/code")
import engine as E
import pb_av_lib as L
import pb_av_lib
import test_pb_av as T

G = Path("/workspace/gpt-candidates")
OUT = G / "out"
DATA = G / "data"
MIN_DIR = DATA / "minutes"
YH_DIR = DATA / "yahoo"
LOG_DIR = G / "logs"
MANIFEST = OUT / "pb_av_vs_pb_manifest.json"

DATASET = "EQUS.MINI"
SCHEMA = "ohlcv-1m"
# Portal balance is not exposed by the historical API. 2026-10-22 reconstruction
# was about $52.3 remaining. Stop $12 under that, which is also $5 under a $45
# working balance. This is a ceiling, not a spend target.
CEILING_USD = 40.0
PROBE_USD = 0.000213414431
MAX_BATCH_USD = 1.0
WINDOW_START = "2023-03-28"
WINDOW_END = "2026-10-23"
COSTS = (10, 30, 50)
KEY = os.environ.get("DATABENTO_API_KEY", "")

OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
MIN_DIR.mkdir(parents=True, exist_ok=True)
YH_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def scrub(text: str) -> str:
    if KEY:
        text = text.replace(KEY, "[REDACTED]")
    return text


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + scrub(msg)
    print(line, flush=True)
    with open(LOG_DIR / "pb_av.log", "a") as fh:
        fh.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def write_json(path: Path, obj) -> None:
    raw = scrub(json.dumps(obj, ensure_ascii=False, indent=1, default=json_default))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw)
    tmp.replace(path)


class CreditStop(RuntimeError):
    pass


class SymbolError(RuntimeError):
    pass


_tls = threading.local()


def client():
    import databento as db

    if not hasattr(_tls, "c"):
        _tls.c = db.Historical(KEY)
    return _tls.c


def classify(exc: BaseException) -> str:
    status = getattr(exc, "http_status", None)
    msg = scrub(str(exc)).lower()
    if status in (401, 402, 403) or any(k in msg for k in ("credit", "quota", "insufficient", "billing", "payment", "unauthorized")):
        return "credit"
    if "data_end" in msg or "available_end" in msg:
        return "range"
    if status == 422 or "symbology" in msg or "could not be resolved" in msg or "none of the symbols" in msg:
        return "symbol"
    if status == 429 or (isinstance(status, int) and status >= 500):
        return "retry"
    return "other"


def _call(fn, what: str):
    delay = 1.0
    last = None
    for attempt in range(5):
        try:
            return fn()
        except Exception as exc:
            kind = classify(exc)
            last = exc
            if kind == "credit":
                raise CreditStop(scrub(str(exc))) from None
            if kind == "range":
                raise RuntimeError("dataset range rejected: " + scrub(str(exc))) from None
            if kind == "symbol":
                raise SymbolError(scrub(str(exc))) from None
            if kind == "retry" or kind == "other":
                log(f"retry {what} attempt {attempt} {type(exc).__name__} {scrub(str(exc))[:180]}")
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
    raise RuntimeError(f"{what} failed: {scrub(str(last))[:300]}")


def day_end(date: str) -> str:
    return (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def get_cost(date: str, symbols: list[str]) -> float:
    end = day_end(date)
    L.assert_one_calendar_day(date, end, symbols)

    def fn():
        return float(
            client().metadata.get_cost(
                dataset=DATASET,
                symbols=list(symbols),
                schema=SCHEMA,
                start=date,
                end=end,
                stype_in="raw_symbol",
            )
        )

    return _call(fn, f"get_cost {date} n={len(symbols)}")


def quote_symbols(date: str, symbols: list[str]) -> tuple[float, list[str], list[str]]:
    if not symbols:
        return 0.0, [], []
    try:
        return get_cost(date, symbols), list(symbols), []
    except SymbolError:
        if len(symbols) == 1:
            log(f"unresolved {symbols[0]} {date}")
            return 0.0, [], list(symbols)
        mid = len(symbols) // 2
        c1, ok1, bad1 = quote_symbols(date, symbols[:mid])
        c2, ok2, bad2 = quote_symbols(date, symbols[mid:])
        return c1 + c2, ok1 + ok2, bad1 + bad2


def download_day(date: str, symbols: list[str], db2yh: dict[str, str]) -> tuple[int, float]:
    """Download one quoted day. Returns (rth_rows, cost_usd)."""
    end = day_end(date)
    L.assert_one_calendar_day(date, end, symbols)
    cost = get_cost(date, symbols)
    if cost > MAX_BATCH_USD:
        raise CreditStop(f"batch {date} quote ${cost:.4f} exceeds ${MAX_BATCH_USD:.0f} breaker")
    if cost <= 0:
        _save_bars(date, pd.DataFrame(columns=["symbol", "ts_event", "open", "high", "low", "close", "volume"]))
        return 0, 0.0

    def fn():
        return client().timeseries.get_range(
            dataset=DATASET,
            symbols=list(symbols),
            schema=SCHEMA,
            start=date,
            end=end,
            stype_in="raw_symbol",
        )

    store = _call(fn, f"download {date} n={len(symbols)}")
    df = store.to_df()
    if len(df) == 0:
        _save_bars(date, pd.DataFrame(columns=["symbol", "ts_event", "open", "high", "low", "close", "volume"]))
        return 0, cost
    df = df.reset_index()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.loc[df["ts_event"].dt.strftime("%Y-%m-%d") == date]
    df = df.loc[df["symbol"].isin(symbols)]
    if len(df):
        keep = L.rth_mask(pd.DatetimeIndex(df["ts_event"]))
        df = df.loc[keep]
    df["symbol"] = df["symbol"].map(lambda s: db2yh.get(s, s))
    out = df[["symbol", "ts_event", "open", "high", "low", "close", "volume"]].copy()
    _save_bars(date, out)
    return int(len(out)), cost


def _save_bars(date: str, df: pd.DataFrame) -> None:
    """Merge into an existing day file. A later one-symbol pull must not erase the day."""
    path = MIN_DIR / f"{date}.parquet"
    if path.exists() and len(df):
        old = pd.read_parquet(path)
        if len(old):
            old = old[~old["symbol"].isin(set(df["symbol"].astype(str)))]
            df = pd.concat([old, df], ignore_index=True)
    tmp = MIN_DIR / f".{date}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_inputs():
    sig = pd.read_csv(OUT / "pb_signals_2023plus.csv")
    td = pd.read_csv(OUT / "pb_minute_ticker_days_2023plus.csv")
    blk = pd.read_csv(OUT / "pb_signals_h_before_equs_blocked.csv")
    for df, col in ((sig, "t"), (sig, "h"), (td, "date")):
        df[col] = df[col].astype(str).str.slice(0, 10)
    if len(sig) != 12227 or len(td) != 48635 or len(blk) != 95:
        raise SystemExit(f"row count mismatch signals={len(sig)} ticker_days={len(td)} blocked={len(blk)}")
    if int((sig.h_before_equs == 1).sum()) != 95:
        raise SystemExit("h_before_equs flag is not 95")
    if td.duplicated().any():
        raise SystemExit("ticker-day list has duplicates")
    return sig, td, blk


def build_symbol_maps(symbols: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    yh2db = {s: L.map_symbol_yahoo_to_databento(s) for s in symbols}
    db2yh = {v: k for k, v in yh2db.items()}
    if len(db2yh) != len(yh2db):
        raise SystemExit("databento symbol map is not one-to-one")
    return yh2db, db2yh


def quote_all(td: pd.DataFrame, yh2db: dict[str, str], manifest: dict) -> dict:
    groups = {}
    for date, g in td.groupby("date", sort=True):
        groups[str(date)] = sorted({yh2db[s] for s in g.symbol})
    dates = list(groups)
    log(f"quoting {len(dates)} days / {int(td.shape[0])} ticker-days")
    done = {b["date"]: b for b in manifest.get("batches", []) if b.get("purpose") == "avwap_window" and b.get("quoted")}
    batches = []
    cum = 0.0

    def work(date):
        if date in done and done[date].get("ok_symbols"):
            b = done[date]
            return b
        cost, ok, bad = quote_symbols(date, groups[date])
        if cost > MAX_BATCH_USD:
            raise CreditStop(f"{date} quote ${cost:.4f} > breaker")
        return {
            "batch_id": f"avwap_{date}",
            "purpose": "avwap_window",
            "dataset": DATASET,
            "schema": SCHEMA,
            "all_symbols": False,
            "date": date,
            "start": date,
            "end": day_end(date),
            "n_symbols_requested": len(groups[date]),
            "n_symbols": len(ok),
            "ok_symbols": ok,
            "unresolved_symbols": bad,
            "cost_usd": cost,
            "quoted": True,
            "downloaded": False,
        }

    # A handful of days can run together; get_cost does not bill.
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(work, d): d for d in dates}
        n = 0
        for fut in as_completed(futs):
            date = futs[fut]
            try:
                batch = fut.result()
            except CreditStop:
                raise
            except Exception as exc:
                raise RuntimeError(f"quote failed {date}: {scrub(str(exc))[:300]}") from None
            batches.append(batch)
            n += 1
            if n % 50 == 0 or n == len(dates):
                log(f"quoted {n}/{len(dates)} last={date} cost={batch['cost_usd']:.6f}")
    batches.sort(key=lambda b: b["date"])
    cum = float(sum(b["cost_usd"] for b in batches))
    manifest["batches"] = [b for b in manifest.get("batches", []) if b.get("purpose") != "avwap_window"] + batches
    manifest["cum_quoted_avwap_usd"] = cum
    manifest["cum_quoted_usd"] = cum + PROBE_USD + float(manifest.get("cum_quoted_review_usd") or 0)
    write_json(MANIFEST, manifest)
    log(f"quote sum avwap ${cum:.4f} with probe ${cum + PROBE_USD:.4f}")
    if cum + PROBE_USD > CEILING_USD:
        manifest["stop_reason"] = "quote_exceeds_ceiling"
        manifest["ceiling_usd"] = CEILING_USD
        write_json(MANIFEST, manifest)
        raise CreditStop(f"quoted ${cum + PROBE_USD:.4f} > ceiling ${CEILING_USD:.2f}")
    unresolved = sorted({s for b in batches for s in b["unresolved_symbols"]})
    if unresolved:
        log(f"unresolved symbols: {unresolved}")
    return manifest


def download_all(manifest: dict, db2yh: dict[str, str]) -> dict:
    stop = threading.Event()
    lock = threading.Lock()
    spent = PROBE_USD
    for b in manifest["batches"]:
        if b.get("purpose") == "avwap_window" and b.get("downloaded"):
            spent += float(b.get("cost_usd") or 0)

    def work(batch):
        if stop.is_set():
            return batch["date"], 0.0, "skipped_after_credit_stop"
        path = MIN_DIR / f"{batch['date']}.parquet"
        # A file already on disk was paid for. Do not request it again.
        if path.exists():
            extra = 0.0 if batch.get("downloaded") else float(batch["cost_usd"])
            return batch["date"], extra, "cached"
        syms = batch["ok_symbols"]
        if not syms:
            _save_bars(batch["date"], pd.DataFrame(columns=["symbol", "ts_event", "open", "high", "low", "close", "volume"]))
            return batch["date"], 0.0, "no_resolved_symbols"
        try:
            rows, cost = download_day(batch["date"], syms, db2yh)
            return batch["date"], cost, f"rows={rows}"
        except CreditStop as exc:
            stop.set()
            return batch["date"], 0.0, "CREDIT " + str(exc)
        except Exception as exc:
            msg = scrub(str(exc))[:300]
            if "credit" in msg.lower() or "quota" in msg.lower() or "insufficient" in msg.lower():
                stop.set()
                return batch["date"], 0.0, "CREDIT " + msg
            return batch["date"], 0.0, "ERROR " + msg

    todo = [b for b in manifest["batches"] if b.get("purpose") == "avwap_window"]
    log(f"downloading {len(todo)} quoted days")
    by_date = {b["date"]: b for b in todo}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(work, b): b["date"] for b in todo}
        n = 0
        for fut in as_completed(futs):
            date, cost, status = fut.result()
            n += 1
            with lock:
                b = by_date[date]
                if status.startswith("CREDIT") or status.startswith("skipped"):
                    b["downloaded"] = False
                    b["error"] = status
                    if status.startswith("CREDIT"):
                        manifest["stop_reason"] = "download_credit_failure"
                    manifest["cum_downloaded_usd"] = spent
                    write_json(MANIFEST, manifest)
                elif status.startswith("ERROR"):
                    b["downloaded"] = False
                    b["error"] = status
                else:
                    b["downloaded"] = True
                    b["error"] = None
                    b["download_note"] = status
                    if cost:
                        b["cost_usd"] = float(cost) if status.startswith("rows") else b["cost_usd"]
                    spent += float(cost or 0)
                    manifest["cum_downloaded_usd"] = spent
                if n % 20 == 0 or n == len(todo) or (b.get("error")):
                    log(f"download {n}/{len(todo)} {date} {status} spent≈${spent:.4f}")
                    write_json(MANIFEST, manifest)
            if stop.is_set() and str(status).startswith("CREDIT"):
                # Let in-flight finish; do not raise until the pool drains.
                pass
    manifest["cum_downloaded_usd"] = spent
    write_json(MANIFEST, manifest)
    if manifest.get("stop_reason") == "download_credit_failure":
        raise CreditStop("download stopped: credit or quota")
    failed = [b["date"] for b in todo if not b.get("downloaded")]
    manifest["download_failed_dates"] = failed
    manifest["n_download_failed"] = len(failed)
    write_json(MANIFEST, manifest)
    if len(failed) > 15:
        manifest["stop_reason"] = "download_incomplete"
        write_json(MANIFEST, manifest)
        raise RuntimeError(f"{len(failed)} days failed to download; not continuing into a partial comparison")
    if failed:
        log(f"missing {len(failed)} days; signals that needed them are excluded from PB-AV")
    return manifest


def download_yahoo(tickers: list[str]) -> pd.DataFrame:
    import logging

    logging.getLogger("yfinance").setLevel(logging.ERROR)
    import yfinance as yf

    tickers = sorted(set(tickers))
    frames = []
    chunk = 80
    for i in range(0, len(tickers), chunk):
        part = tickers[i : i + chunk]
        cache = YH_DIR / f"chunk_{i:04d}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        last = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    part,
                    start="2022-01-01",
                    end="2026-10-24",
                    auto_adjust=False,
                    actions=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                flat = _flatten_yahoo(raw, part)
                flat.to_parquet(cache, index=False)
                frames.append(flat)
                log(f"yahoo {i + len(part)}/{len(tickers)} rows={len(flat)}")
                last = None
                break
            except Exception as exc:
                last = exc
                log(f"yahoo chunk {i} attempt {attempt} {type(exc).__name__}")
                time.sleep(2 + attempt * 3)
        if last is not None:
            raise RuntimeError(f"yahoo chunk {i} failed: {last}") from None
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = daily["date"].astype(str).str.slice(0, 10)
    daily = daily.drop_duplicates(["symbol", "date"], keep="last")
    daily.to_parquet(YH_DIR / "daily.parquet", index=False)
    return daily


def _flatten_yahoo(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "dividends", "split"])
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        if multi:
            if t not in raw.columns.get_level_values(0):
                continue
            sub = raw[t].copy()
        else:
            sub = raw.copy()
        sub = sub.reset_index()
        date_col = "Date" if "Date" in sub.columns else "Datetime" if "Datetime" in sub.columns else sub.columns[0]
        keep = {
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "split",
        }
        have = {k: v for k, v in keep.items() if k in sub.columns}
        sub = sub[list(have)].rename(columns=have)
        for col in ("open", "high", "low", "close", "volume", "dividends", "split"):
            if col not in sub.columns:
                sub[col] = np.nan if col != "split" else 0.0
        sub["symbol"] = t
        sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")
        rows.append(sub[["symbol", "date", "open", "high", "low", "close", "volume", "dividends", "split"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "dividends", "split"])
    return pd.concat(rows, ignore_index=True)


def compute_avwap(sig: pd.DataFrame, td: pd.DataFrame) -> pd.DataFrame:
    base = sig.copy()
    base["sid"] = np.arange(len(base))
    fetch = base[base.h_before_equs == 0]
    index: dict[tuple[str, str], list[int]] = {}
    by_sym = {s: g.date.tolist() for s, g in td.groupby("symbol", sort=False)}
    for sid, sym, h, t in zip(fetch.sid, fetch.symbol, fetch.h, fetch.t):
        for d in by_sym.get(sym, []):
            if h <= d <= t:
                index.setdefault((sym, d), []).append(int(sid))
    n = len(base)
    sum_tpv = np.zeros(n)
    sum_v = np.zeros(n)
    n_bars = np.zeros(n, np.int32)
    n_days = np.zeros(n, np.int32)
    files = sorted(MIN_DIR.glob("*.parquet"))
    log(f"avwap over {len(files)} day files")
    for i, path in enumerate(files):
        date = path.stem
        df = pd.read_parquet(path)
        if len(df) == 0:
            continue
        tp = (df["high"].to_numpy(np.float64) + df["low"].to_numpy(np.float64) + df["close"].to_numpy(np.float64)) / 3.0
        vol = df["volume"].to_numpy(np.float64)
        ok = np.isfinite(tp) & np.isfinite(vol) & (vol >= 0) & (df["high"].to_numpy(np.float64) >= df["low"].to_numpy(np.float64))
        tmp = pd.DataFrame({"symbol": df["symbol"].to_numpy(), "tpv": np.where(ok, tp * vol, 0.0), "vol": np.where(ok, vol, 0.0), "bar": ok.astype(np.int32)})
        agg = tmp.groupby("symbol", sort=False).sum(numeric_only=True)
        for sym, row in agg.iterrows():
            if int(row.bar) <= 0:
                continue
            for sid in index.get((sym, date), []):
                sum_tpv[sid] += float(row.tpv)
                sum_v[sid] += float(row.vol)
                n_bars[sid] += int(row.bar)
                n_days[sid] += 1
        if (i + 1) % 200 == 0:
            log(f"avwap files {i + 1}/{len(files)}")
    base["sum_vol"] = sum_v
    base["n_bars"] = n_bars
    base["n_days_got"] = n_days
    base["n_days_expected"] = np.where(base.h_before_equs == 1, 0, base.k_lag + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        av = np.where(sum_v > 0, sum_tpv / sum_v, np.nan)
    base["avwap"] = av
    base["C_raw"] = base["C_adj"] * base["F"]
    return base


def apply_split_flags(av: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    splits = daily[(daily["split"].fillna(0) > 0) & ((daily["split"] - 1).abs() > 1e-6)]
    split_map = {s: set(g.date) for s, g in splits.groupby("symbol")}
    reasons = []
    for sym, h, t, h_before, n_got, n_exp, sum_vol, c_raw, avwap in zip(
        av.symbol, av.h, av.t, av.h_before_equs, av.n_days_got, av.n_days_expected, av.sum_vol, av.C_raw, av.avwap
    ):
        if int(h_before) == 1:
            reasons.append("h_before_equs")
            continue
        hit = [d for d in split_map.get(sym, ()) if h <= d <= t]
        if hit:
            reasons.append("split_in_window")
            continue
        if int(n_got) != int(n_exp) or int(n_got) == 0:
            reasons.append("missing_minute_days")
            continue
        if not (sum_vol > 0) or not math.isfinite(float(avwap)):
            reasons.append("zero_volume")
            continue
        if not (float(c_raw) > float(avwap)):
            reasons.append("below_avwap")
            continue
        reasons.append("")
    av = av.copy()
    av["exclude_reason"] = reasons
    av["pass_avwap"] = av["exclude_reason"].eq("")
    return av


def build_matrices(daily: pd.DataFrame, symbols: list[str]):
    qqq = daily[daily.symbol == "QQQ"].dropna(subset=["close"])
    cal = pd.DatetimeIndex(pd.to_datetime(sorted(qqq.date.unique())))
    if pd.Timestamp(WINDOW_START) not in cal or pd.Timestamp(WINDOW_END) not in cal:
        raise RuntimeError(f"QQQ calendar missing window endpoints; has {cal.min().date()}..{cal.max().date()} n={len(cal)}")
    syms = list(symbols)
    col = {s: i for i, s in enumerate(syms)}
    date_to_i = {ts.strftime("%Y-%m-%d"): i for i, ts in enumerate(cal)}
    T, N = len(cal), len(syms)
    O = np.full((T, N), np.nan)
    H = np.full((T, N), np.nan)
    lo = np.full((T, N), np.nan)
    C = np.full((T, N), np.nan)
    V = np.full((T, N), np.nan)
    DIV = np.full((T, N), np.nan)
    SP = np.zeros((T, N))
    sub = daily[daily.symbol.isin(col)]
    ii = sub.date.map(date_to_i)
    ok = ii.notna()
    sub = sub.loc[ok]
    ii = ii.loc[ok].astype(int).to_numpy()
    jj = sub.symbol.map(col).to_numpy()
    O[ii, jj] = sub.open.to_numpy(np.float64)
    H[ii, jj] = sub.high.to_numpy(np.float64)
    Lc = sub.low.to_numpy(np.float64)
    lo[ii, jj] = Lc
    C[ii, jj] = sub.close.to_numpy(np.float64)
    V[ii, jj] = sub.volume.to_numpy(np.float64)
    DIV[ii, jj] = sub.dividends.fillna(0).to_numpy(np.float64)
    SP[ii, jj] = sub.split.fillna(0).to_numpy(np.float64)
    F = np.ones((T, N))
    for s, j in col.items():
        ratios = pd.Series(SP[:, j], index=np.arange(T))
        F[:, j] = pb_av_lib.future_split_factor(ratios).to_numpy(np.float64)
    # Engine cash dividend uses adjusted shares. Store dividend per adjusted share.
    with np.errstate(invalid="ignore", divide="ignore"):
        DIV = np.where(np.isfinite(DIV), DIV / F, 0.0)
    DIV = np.nan_to_num(DIV)
    SP = np.where((SP > 0) & (np.abs(SP - 1) > 1e-6), SP, 0.0)
    fin = np.isfinite(C)
    lastvalid = np.where(fin.any(0), T - 1 - np.argmax(fin[::-1], axis=0), -1)
    d = type("D", (), {})()
    d.O, d.H, d.L, d.C, d.V, d.F, d.SP, d.DIV = O, H, lo, C, V, F, SP, DIV
    d.cal = cal
    d.names = np.array(syms)
    d.T = T
    d.lastvalid = lastvalid
    f = type("F", (), {})()
    f.d = d
    f.E = {9: E.ema_sma_init(C, 9)}
    f.VOL20 = E.roll(V, 20, "mean")
    i0 = int(cal.searchsorted(pd.Timestamp(WINDOW_START)))
    i1 = int(cal.searchsorted(pd.Timestamp(WINDOW_END), side="right") - 1)
    return f, col, date_to_i, i0, i1


def match_report(sig: pd.DataFrame, daily: pd.DataFrame, f, col, date_to_i) -> dict:
    px = daily[["symbol", "date", "close"]].rename(columns={"date": "t", "close": "yh_close"})
    m = sig.merge(px, on=["symbol", "t"], how="left")
    rel = (m.yh_close - m.C_adj).abs() / m.C_adj.abs().clip(lower=1e-3)
    absd = (m.yh_close - m.C_adj).abs()
    ok = m.yh_close.notna() & ((rel <= 0.002) | (absd <= 0.02))
    # F match on the rebuilt matrix
    f_err = []
    for sym, t, f_csv in zip(sig.symbol, sig.t, sig.F):
        j = col.get(sym)
        i = date_to_i.get(t)
        if j is None or i is None:
            f_err.append(np.nan)
            continue
        f_err.append(abs(float(f.d.F[i, j]) - float(f_csv)) / max(abs(float(f_csv)), 1e-9))
    f_err = np.asarray(f_err, dtype=float)
    rep = {
        "n_signals": int(len(sig)),
        "n_with_yahoo_close": int(m.yh_close.notna().sum()),
        "n_close_match": int(ok.sum()),
        "close_match_rate": float(ok.mean()),
        "close_rel_p95": float(rel.quantile(0.95)) if rel.notna().any() else None,
        "n_f_match_1bp": int(np.nansum(f_err <= 1e-4)),
        "f_match_rate_1bp": float(np.nanmean(f_err <= 1e-4)),
        "f_rel_p95": float(np.nanquantile(f_err, 0.95)),
    }
    return rep


def orders_from(frame: pd.DataFrame, col: dict, date_to_i: dict) -> tuple[dict, dict]:
    cand = frame[frame.in_risk_range == 1]
    top = L.pick_top_orders(cand) if len(cand) else cand
    orders = {}
    meta = {}
    dropped = 0
    for row in top.itertuples(index=False):
        i = date_to_i.get(row.t)
        j = col.get(row.symbol)
        if i is None or j is None:
            dropped += 1
            continue
        # ncand recomputed below
        orders[int(i)] = (int(j), float(row.B), float(row.S), 0)
        meta[int(i)] = row.symbol
    counts = cand.groupby("t").size().to_dict() if len(cand) else {}
    for i, sym in list(meta.items()):
        # date string from the index is recovered via the order's signal date lookup
        pass
    # fill ncand
    inv_date = {v: k for k, v in date_to_i.items()}
    for i in list(orders):
        t = inv_date[i]
        j, b, s, _ = orders[i]
        orders[i] = (j, b, s, int(counts.get(t, 1)))
    return orders, {"n_signal_days": len(orders), "dropped_missing_symbol": dropped, "top_symbol_by_index": meta}


def run_arm(f, orders, i0, i1, bp: int):
    tr, eq, ol = E.simulate(f, orders, (0.01, 0.05), bp, i0, i1, risk_frac=0.005, cap0=16000.0, tag=f"{bp}")
    met = E.metrics(eq, tr)
    met["bootstrap_meanR_LB95"] = E.block_bootstrap_meanR(tr, eq)
    how = {}
    for row in ol:
        how[row[3]] = how.get(row[3], 0) + 1
    met["order_outcomes"] = how
    return tr, eq, met


def same_bar_jobs(trades: pd.DataFrame) -> pd.DataFrame:
    amb = trades[trades.ambiguous == 1][["sym", "entry_date", "B_adj", "S_adj"]].drop_duplicates()
    return amb


def ensure_review_minutes(jobs: pd.DataFrame, manifest: dict, yh2db: dict[str, str], db2yh: dict[str, str], have: set[tuple[str, str]]) -> dict:
    need = {}
    for sym, date in zip(jobs.sym, jobs.entry_date):
        if (sym, str(date)[:10]) in have:
            continue
        need.setdefault(str(date)[:10], set()).add(yh2db[sym])
    if not need:
        manifest["cum_quoted_review_usd"] = 0.0
        return manifest
    spent = float(manifest.get("cum_downloaded_usd") or 0)
    review_quote = 0.0
    batches = []
    for date, syms in sorted(need.items()):
        cost, ok, bad = quote_symbols(date, sorted(syms))
        review_quote += cost
        batches.append((date, ok, bad, cost))
        log(f"review quote {date} n={len(ok)} ${cost:.6f} running ${review_quote:.4f}")
        if spent + review_quote > CEILING_USD:
            manifest["stop_reason"] = "review_quote_exceeds_ceiling"
            manifest["cum_quoted_review_usd"] = review_quote
            manifest["review_downloaded"] = False
            write_json(MANIFEST, manifest)
            raise CreditStop(f"same-bar quote would take spend to ${spent + review_quote:.4f}")
    manifest["cum_quoted_review_usd"] = review_quote
    for date, ok, bad, cost in batches:
        if not ok:
            continue
        rows, charged = download_day(date, ok, db2yh)
        spent += charged
        manifest.setdefault("review_batches", []).append(
            {
                "batch_id": f"review_{date}",
                "purpose": "same_bar_review",
                "dataset": DATASET,
                "schema": SCHEMA,
                "all_symbols": False,
                "date": date,
                "start": date,
                "end": day_end(date),
                "n_symbols": len(ok),
                "unresolved_symbols": bad,
                "cost_usd": cost,
                "quoted": True,
                "downloaded": True,
                "rows_rth": rows,
            }
        )
        manifest["cum_downloaded_usd"] = spent
        write_json(MANIFEST, manifest)
        log(f"review download {date} rows={rows} spent≈${spent:.4f}")
    return manifest


def load_day_index() -> dict[tuple[str, str], pd.DataFrame]:
    """Lazy file map. Values loaded on demand via path."""
    out = {}
    for path in MIN_DIR.glob("*.parquet"):
        out[path.stem] = path
    return out


def bars_for(path_map, sym: str, date: str) -> pd.DataFrame:
    path = path_map.get(date)
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    df = pd.read_parquet(path, columns=["symbol", "ts_event", "open", "high", "low", "close"])
    df = df[df.symbol == sym]
    if len(df) == 0:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    df = df.rename(columns={"ts_event": "ts"})
    return df


def review_trades(trades: pd.DataFrame, f, col, date_to_i, path_map) -> pd.DataFrame:
    rows = []
    amb = trades[trades.ambiguous == 1]
    for rec in amb.itertuples(index=False):
        i = date_to_i.get(str(rec.entry_date)[:10])
        j = col.get(rec.sym)
        if i is None or j is None:
            b_raw = s_raw = float("nan")
        else:
            fac = float(f.d.F[i, j])
            b_raw = float(rec.B_adj) * fac
            s_raw = float(rec.S_adj) * fac
        bars = bars_for(path_map, rec.sym, str(rec.entry_date)[:10])
        rev = L.review_same_bar(bars, b_raw, s_raw)
        rows.append(
            {
                "arm": rec.arm,
                "cost_bp": int(rec.cost_bp),
                "symbol": rec.sym,
                "signal_date": rec.sig_date,
                "entry_date": rec.entry_date,
                "B_adj": rec.B_adj,
                "S_adj": rec.S_adj,
                "B_raw": b_raw,
                "S_raw": s_raw,
                "daily_exit_reason": rec.exit_reason,
                "daily_ambiguous": int(rec.ambiguous),
                "entry_px_raw": rec.entry_px_raw,
                "exit_px_raw": rec.exit_px_raw,
                "R": rec.R,
                **rev,
            }
        )
    return pd.DataFrame(rows)


def owned_pairs() -> set[tuple[str, str]]:
    have = set()
    for path in MIN_DIR.glob("*.parquet"):
        df = pd.read_parquet(path, columns=["symbol"])
        if len(df) == 0:
            continue
        for s in df.symbol.unique():
            have.add((str(s), path.stem))
    return have


def plot_equity(eq: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = eq[eq.cost_bp == 10]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for arm, color, lw, ls in (
        ("PB", "black", 2.4, "-"),
        ("PB-AV", "#E31C3D", 2.0, "--"),
    ):
        g = sub[sub.arm == arm]
        ax.plot(pd.to_datetime(g["date"]), g["equity"], color=color, lw=lw, ls=ls, label=arm)
    ax.set_title("EXPLORATORY  PB vs PB-AV  equity at 10bp   2023-03-28 to 2026-10-23")
    ax.set_ylabel("Equity (USD, start 16,000)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fmt_pct(x):
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "—"
    return f"{float(x) * 100:.2f}%"


def fmt_num(x, nd=3):
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def render_readme(facts: dict) -> str:
    m = facts["metrics"]
    lines = []
    lines.append("# PB vs PB-AV 结论（EXPLORATORY）")
    lines.append("")
    lines.append(f"窗口 **{WINDOW_START} → {WINDOW_END}**。两臂同一资金规则（16000 美元、单仓、每笔 0.5% 风险）、同一日线入场和止损，PB-AV 只多一条 `C(t) > AVWAP`。AVWAP 用 EQUS.MINI 未复权 1 分钟线，锚在 h 日开盘，常规时段 09:30–16:00 美东，typical=(H+L+C)/3。标签只能是 **EXPLORATORY**。")
    lines.append("")
    lines.append("## 结果")
    lines.append("")
    lines.append(facts["lead"])
    lines.append("")
    lines.append("| 成本 | 臂 | 笔数 | 均 R | 年化 | 最大回撤 | 终值 | 30/50 见下行 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for bp in COSTS:
        for arm in ("PB", "PB-AV"):
            r = m[f"{arm}|{bp}"]
            extra = ""
            if bp == 10:
                extra = "主结果"
            lines.append(
                f"| {bp}bp | {arm} | {r['n_trades']} | {fmt_num(r['mean_R'])} | {fmt_pct(r['CAGR'])} | {fmt_pct(r['MaxDD'])} | {fmt_num(r['final_equity'], 0)} | {extra} |"
            )
    lines.append("")
    lines.append("## 入场价和止损")
    lines.append("")
    lines.append(facts["price_paragraph"])
    lines.append("")
    lines.append("## 同 bar 复核")
    lines.append("")
    lines.append(facts["review_paragraph"])
    lines.append("")
    lines.append("主结果仍按日线保守止损。分钟复核只说明顺序，不把乐观路径写回净值。")
    lines.append("")
    lines.append("## 能不能信，样本够不够")
    lines.append("")
    lines.append(facts["trust_paragraph"])
    lines.append("")
    lines.append("## 实际花费的 Credit")
    lines.append("")
    lines.append(facts["credit_paragraph"])
    lines.append("")
    lines.append("全市场 EQUS.MINI ohlcv-1m 没有下载。密钥没有写入文件。")
    lines.append("")
    return "\n".join(lines)


def blank_manifest() -> dict:
    return {
        "label": "EXPLORATORY",
        "generated_for": "PB vs PB-AV 2023-03-28 to 2026-10-23",
        "dataset": DATASET,
        "schema": SCHEMA,
        "all_symbols_minute": False,
        "key_written": False,
        "portal_balance_usd": None,
        "portal_balance_note": "Historical API has no balance method. Ceiling uses the 2026-10-22 reconstructed remaining of about $52.3, minus more than $12.",
        "ceiling_usd": CEILING_USD,
        "prior_reconstruction_remaining_usd": 52.3,
        "probe": {
            "batch_id": "probe_A_2026-09-29",
            "purpose": "schema_probe",
            "dataset": DATASET,
            "schema": SCHEMA,
            "all_symbols": False,
            "symbols": ["A"],
            "n_symbols": 1,
            "start": "2026-09-29",
            "end": "2026-09-30",
            "cost_usd": PROBE_USD,
            "downloaded": True,
            "note": "Schema check. This ticker-day is inside the 48635 list and is requested again with that day's batch.",
        },
        "batches": [],
        "cum_downloaded_usd": PROBE_USD,
        "stop_reason": None,
        "inputs": {
            "signals": 12227,
            "ticker_days": 48635,
            "blocked_h": 95,
            "signals_sha256": sha256_file(OUT / "pb_signals_2023plus.csv"),
            "ticker_days_sha256": sha256_file(OUT / "pb_minute_ticker_days_2023plus.csv"),
        },
    }


def main():
    if not KEY:
        print("环境仍无 key")
        raise SystemExit(2)
    log(f"key_len={len(KEY)}")
    T.test_forbid_all_symbols()
    T.test_class_share_map()
    T.test_rth_and_avwap()
    T.test_same_bar_paths()
    T.test_rank_and_split_factor()
    log("self-test ok")
    sig, td, _blk = load_inputs()
    yh2db, db2yh = build_symbol_maps(sorted(set(sig.symbol) | {"QQQ"}))
    # QQQ is not in the minute pull.
    yh2db.pop("QQQ", None)
    db2yh = {v: k for k, v in yh2db.items()}

    manifest = blank_manifest()
    if MANIFEST.exists():
        prev = json.loads(MANIFEST.read_text())
        if prev.get("inputs", {}).get("ticker_days_sha256") == manifest["inputs"]["ticker_days_sha256"] and prev.get("batches"):
            manifest = prev
            manifest["key_written"] = False
            log("resuming from manifest")
    write_json(MANIFEST, manifest)

    yh_err = {}

    def yh_target():
        try:
            download_yahoo(sorted(set(sig.symbol) | {"QQQ"}))
        except Exception as exc:
            yh_err["err"] = exc
            log("yahoo failed " + scrub(str(exc))[:300])

    yh_thread = threading.Thread(target=yh_target, name="yahoo")
    yh_thread.start()
    try:
        if not manifest.get("cum_quoted_avwap_usd"):
            manifest = quote_all(td, yh2db, manifest)
        else:
            log(f"quotes present ${manifest['cum_quoted_avwap_usd']:.4f}")
            if float(manifest["cum_quoted_avwap_usd"]) + PROBE_USD > CEILING_USD:
                raise CreditStop("stored quote exceeds ceiling")
        manifest = download_all(manifest, db2yh)
    except CreditStop as exc:
        manifest["stop_reason"] = manifest.get("stop_reason") or "credit_ceiling"
        manifest["stop_detail"] = scrub(str(exc))
        write_json(MANIFEST, manifest)
        write_stop_readme(manifest, scrub(str(exc)))
        log("STOPPED " + scrub(str(exc)))
        raise SystemExit(3)
    finally:
        yh_thread.join()
    if "err" in yh_err:
        raise yh_err["err"]

    log("computing AVWAP")
    av = compute_avwap(sig, td)
    daily = pd.read_parquet(YH_DIR / "daily.parquet")
    av = apply_split_flags(av, daily)
    keep_cols = [
        "symbol", "t", "h", "k_lag", "B", "S", "B_raw", "S_raw", "C_adj", "C_raw", "F", "avwap", "sum_vol",
        "n_bars", "n_days_got", "n_days_expected", "exclude_reason", "pass_avwap", "in_risk_range", "R63", "ADV20", "j", "h_before_equs",
    ]
    av[keep_cols].to_csv(OUT / "pb_av_signals.csv", index=False)
    log(f"pass_avwap {(av.pass_avwap).sum()} / {len(av)} reasons {av.exclude_reason.value_counts().to_dict()}")

    log("building daily matrices")
    symbols = sorted(set(sig.symbol))
    f, col, date_to_i, i0, i1 = build_matrices(daily, symbols)
    report = match_report(sig, daily, f, col, date_to_i)
    log("match " + json.dumps(report))
    (OUT / "yahoo_match.json").write_text(json.dumps(report, indent=1))
    if report["close_match_rate"] < 0.98 or report["f_match_rate_1bp"] < 0.98:
        manifest["stop_reason"] = "yahoo_snapshot_mismatch"
        manifest["yahoo_match"] = report
        write_json(MANIFEST, manifest)
        text = (
            "# PB vs PB-AV 结论（EXPLORATORY）\n\n"
            "分钟线已按报价下载，但本环境重新拉取的 Yahoo 日线与信号表里的收盘价或拆股因子对不上，"
            f"收盘匹配率 {report['close_match_rate']:.3f}，拆股因子匹配率 {report['f_match_rate_1bp']:.3f}。"
            "在这个日线上跑出来的净值不能和原信号口径对照，所以没有发布两臂回测。\n\n"
            f"已下载 Credit 约 **${float(manifest.get('cum_downloaded_usd') or 0):.4f}**（含探针 ${PROBE_USD:.6f}）。"
            "全市场分钟没有下载。\n"
        )
        (OUT / "README_结论_EXPLORATORY.md").write_text(text)
        log("STOPPED yahoo mismatch")
        raise SystemExit(4)

    pb_orders, pb_meta = orders_from(sig, col, date_to_i)
    av_orders, av_meta = orders_from(av[av.pass_avwap], col, date_to_i)
    both = set(pb_meta["top_symbol_by_index"]) & set(av_meta["top_symbol_by_index"])
    differ = sum(1 for i in both if pb_meta["top_symbol_by_index"][i] != av_meta["top_symbol_by_index"][i])
    log(f"order days PB {pb_meta['n_signal_days']} PB-AV {av_meta['n_signal_days']} shared {len(both)} symbol-diff {differ}")

    trade_frames = []
    eq_frames = []
    metrics_rows = []
    metrics_map = {}
    for arm, orders in (("PB", pb_orders), ("PB-AV", av_orders)):
        for bp in COSTS:
            log(f"simulate {arm} {bp}bp")
            tr, eq, met = run_arm(f, orders, i0, i1, bp)
            tdf = pd.DataFrame(tr)
            tdf.insert(0, "arm", arm)
            tdf.insert(1, "cost_bp", bp)
            trade_frames.append(tdf)
            eq = eq.copy()
            eq.insert(0, "arm", arm)
            eq.insert(1, "cost_bp", bp)
            eq_frames.append(eq)
            row = {"arm": arm, "cost_bp": bp, **{k: met[k] for k in met if k != "order_outcomes"}}
            row["order_outcomes"] = json.dumps(met["order_outcomes"])
            metrics_rows.append(row)
            metrics_map[f"{arm}|{bp}"] = row
            log(f"{arm} {bp} n={row['n_trades']} meanR={row['mean_R']:.4f} CAGR={row['CAGR']:.4f} MaxDD={row['MaxDD']:.4f}")

    trades = pd.concat(trade_frames, ignore_index=True)
    equity = pd.concat(eq_frames, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    trades.to_csv(OUT / "trades_compare.csv", index=False)
    equity.to_csv(OUT / "equity_compare.csv", index=False)
    metrics.drop(columns=["order_outcomes"]).to_csv(OUT / "metrics_compare.csv", index=False)

    log("same-bar review")
    have = owned_pairs()
    jobs = same_bar_jobs(trades)
    try:
        manifest = ensure_review_minutes(jobs, manifest, yh2db, db2yh, have)
    except CreditStop as exc:
        manifest["stop_reason"] = "review_quote_exceeds_ceiling"
        manifest["stop_detail"] = scrub(str(exc))
        write_json(MANIFEST, manifest)
        log("review not downloaded: " + scrub(str(exc)))
    path_map = load_day_index()
    review = review_trades(trades, f, col, date_to_i, path_map)
    review.to_csv(OUT / "same_bar_minute_review.csv", index=False)

    # Price difference on the 10bp books.
    p10 = trades[trades.cost_bp == 10]
    a = p10[p10.arm == "PB"]
    b = p10[p10.arm == "PB-AV"]
    paired = a.merge(b, on=["sym", "sig_date"], suffixes=("_pb", "_av"))
    if len(paired):
        d_entry = (paired.entry_px_raw_pb - paired.entry_px_raw_av).abs()
        d_stop = (paired.stop_raw_at_entry_pb - paired.stop_raw_at_entry_av).abs()
        max_entry = float(d_entry.max())
        max_stop = float(d_stop.max())
    else:
        max_entry = max_stop = float("nan")
    price = {
        "pb_mean_entry": float(a.entry_px_raw.mean()) if len(a) else None,
        "av_mean_entry": float(b.entry_px_raw.mean()) if len(b) else None,
        "pb_mean_stop": float(a.stop_raw_at_entry.mean()) if len(a) else None,
        "av_mean_stop": float(b.stop_raw_at_entry.mean()) if len(b) else None,
        "pb_median_entry": float(a.entry_px_raw.median()) if len(a) else None,
        "av_median_entry": float(b.entry_px_raw.median()) if len(b) else None,
        "n_paired_same_signal": int(len(paired)),
        "n_pb_only": int(len(a) - len(paired)),
        "n_av_only": int(len(b) - len(paired)),
        "paired_max_abs_entry_diff": max_entry,
        "paired_max_abs_stop_diff": max_stop,
        "signal_days_pb": pb_meta["n_signal_days"],
        "signal_days_av": av_meta["n_signal_days"],
        "shared_days_top_symbol_differs": int(differ),
        "shared_days": int(len(both)),
    }
    rev_counts = review.resolution.value_counts().to_dict() if len(review) else {}
    facts = {
        "label": "EXPLORATORY",
        "metrics": {k: {kk: (None if isinstance(vv, float) and not math.isfinite(vv) else vv) for kk, vv in row.items() if kk != "order_outcomes"} for k, row in metrics_map.items()},
        "price": price,
        "review_counts": rev_counts,
        "n_review_rows": int(len(review)),
        "n_ambiguous_trade_rows": int((trades.ambiguous == 1).sum()),
        "avwap_reason_counts": av.exclude_reason.value_counts().to_dict(),
        "n_pass_avwap": int(av.pass_avwap.sum()),
        "n_pass_avwap_in_risk": int(((av.pass_avwap) & (av.in_risk_range == 1)).sum()),
        "n_pb_in_risk": int((sig.in_risk_range == 1).sum()),
        "yahoo_match": report,
        "credit_downloaded_usd": float(manifest.get("cum_downloaded_usd") or 0),
        "credit_quoted_avwap_usd": float(manifest.get("cum_quoted_avwap_usd") or 0),
        "credit_quoted_review_usd": float(manifest.get("cum_quoted_review_usd") or 0),
        "probe_usd": PROBE_USD,
        "ceiling_usd": CEILING_USD,
        "stop_reason": manifest.get("stop_reason"),
    }
    facts["lead"] = _lead(facts)
    facts["price_paragraph"] = _price_paragraph(facts)
    facts["review_paragraph"] = _review_paragraph(facts)
    facts["trust_paragraph"] = _trust_paragraph(facts)
    facts["credit_paragraph"] = _credit_paragraph(facts)
    text = render_readme(facts)
    (OUT / "README_结论_EXPLORATORY.md").write_text(text)
    (G / "README_结论_EXPLORATORY.md").write_text(text)
    write_json(OUT / "pb_av_vs_pb_facts.json", facts)
    fig = OUT / "pb_av_vs_pb_equity_10bp.png"
    plot_equity(equity, fig)
    manifest["stop_reason"] = manifest.get("stop_reason")
    manifest["completed"] = manifest.get("stop_reason") not in {"review_quote_exceeds_ceiling", "download_credit_failure"}
    manifest["yahoo_match"] = report
    manifest["outputs"] = {
        "trades": str(OUT / "trades_compare.csv"),
        "equity": str(OUT / "equity_compare.csv"),
        "metrics": str(OUT / "metrics_compare.csv"),
        "review": str(OUT / "same_bar_minute_review.csv"),
        "chart": str(fig),
        "readme": str(OUT / "README_结论_EXPLORATORY.md"),
        "signals": str(OUT / "pb_av_signals.csv"),
    }
    write_json(MANIFEST, manifest)
    log(f"done spent ${float(manifest.get('cum_downloaded_usd') or 0):.4f}")


def _lead(facts: dict) -> str:
    pb = facts["metrics"]["PB|10"]
    av = facts["metrics"]["PB-AV|10"]
    return (
        f"10bp 下 PB **{pb['n_trades']}** 笔、均 R **{fmt_num(pb['mean_R'])}**、年化 **{fmt_pct(pb['CAGR'])}**、最大回撤 **{fmt_pct(pb['MaxDD'])}**；"
        f"PB-AV **{av['n_trades']}** 笔、均 R **{fmt_num(av['mean_R'])}**、年化 **{fmt_pct(av['CAGR'])}**、最大回撤 **{fmt_pct(av['MaxDD'])}**。"
        f"日线 PB 风险带内信号 {facts['n_pb_in_risk']} 条，过 AVWAP 且仍在风险带内的 {facts['n_pass_avwap_in_risk']} 条"
        f"（全部过 AVWAP 的格子 {facts['n_pass_avwap']}，含后来排不上队的）。"
    )


def _price_paragraph(facts: dict) -> str:
    p = facts["price"]
    return (
        f"同一笔信号上，B 和 S 的公式没有分叉。10bp 账本里两臂都成交的同一信号有 **{p['n_paired_same_signal']}** 笔，"
        f"入场价最大绝对差 **{fmt_num(p['paired_max_abs_entry_diff'], 4)}** 美元，止损最大绝对差 **{fmt_num(p['paired_max_abs_stop_diff'], 4)}** 美元。"
        f"PB 独有 {p['n_pb_only']} 笔，PB-AV 独有 {p['n_av_only']} 笔。"
        f"两臂各自的平均入场价是 PB {fmt_num(p['pb_mean_entry'], 2)}、PB-AV {fmt_num(p['av_mean_entry'], 2)}；"
        f"平均初始止损是 PB {fmt_num(p['pb_mean_stop'], 2)}、PB-AV {fmt_num(p['av_mean_stop'], 2)}。"
        f"挂单日里两边都有候选的 {p['shared_days']} 天中，排第一的股票不同的有 {p['shared_days_top_symbol_differs']} 天。"
        "平均价的差别来自留下的股票不同，不是同一信号用了两套止损。"
    )


def _review_paragraph(facts: dict) -> str:
    c = facts["review_counts"]
    if facts["n_ambiguous_trade_rows"] == 0:
        return "三档成本里没有同日入场且最低价触及止损的笔，复核表只有表头。"
    parts = "，".join(f"{k} {v}" for k, v in sorted(c.items()))
    return (
        f"日线保守同 bar 记录共 **{facts['n_ambiguous_trade_rows']}** 行（三档成本都算，同一笔会重复出现）。"
        f"分钟分类：{parts}。"
        "同一根 1 分钟里既触发又碰到止损的，仍记保守止损。入场和止损落在不同分钟的，表里写了实际顺序。"
    )


def _trust_paragraph(facts: dict) -> str:
    pb = facts["metrics"]["PB|10"]
    av = facts["metrics"]["PB-AV|10"]
    bits = []
    n_ok = pb["n_trades"] >= 100 and av["n_trades"] >= 100
    bits.append(
        f"10bp 样本 PB {pb['n_trades']} 笔、PB-AV {av['n_trades']} 笔。"
        + ("两边都达到规格里 100 笔的参考线。" if n_ok else "至少有一臂不到规格里 100 笔的参考线，样本偏少。")
    )
    def lb(row):
        v = row.get("bootstrap_meanR_LB95")
        return v if isinstance(v, (int, float)) and math.isfinite(float(v)) else None
    pb_lb, av_lb = lb(pb), lb(av)
    bits.append(
        f"按月分块 bootstrap（12 个月、2000 次）平均 R 的 95% 下界：PB {fmt_num(pb_lb)}，PB-AV {fmt_num(av_lb)}。"
    )
    if (pb_lb is None or pb_lb <= 0) or (av_lb is None or av_lb <= 0):
        bits.append("下界没有同时站在 0 上方，这组均 R 不能当成稳定的正期望。")
    else:
        bits.append("下界在 0 上方，但下面的数据限制仍然在。")
    stress = []
    for bp in (30, 50):
        for arm in ("PB", "PB-AV"):
            r = facts["metrics"][f"{arm}|{bp}"]
            stress.append(f"{arm} {bp}bp 净利 {fmt_num(r['net_profit'], 0)}、PF {fmt_num(r.get('PF'))}")
    bits.append("成本压力：" + "；".join(stress) + "。")
    bits.append(
        "这批股票来自当时的上市清单快照，退市股不在池子里，有存活者偏差。"
        "分钟数据从 2023-03-28 才开始，看不到规格里的 2009–2022。"
        "2023 年以来的这段行情也不能代表别的市场。"
        "所以两臂可以互相比较，不能升成 RESEARCH_PASS，也不能拿去当已经证实的交易优势。"
    )
    return "".join(bits)


def _credit_paragraph(facts: dict) -> str:
    spent = facts["credit_downloaded_usd"]
    quoted = facts["credit_quoted_avwap_usd"]
    review = facts["credit_quoted_review_usd"]
    return (
        f"实际扣掉的 Credit 是 **${spent:.4f}**。其中探针（A，2026-09-29，1 个 ticker-day）${facts['probe_usd']:.6f}，"
        f"AVWAP 窗口报价合计 ${quoted:.4f}，同 bar 复核报价 ${review:.4f}。"
        f"门户余额接口读不到，本轮硬顶是 ${facts['ceiling_usd']:.0f}（旧账重建剩余约 $52.3，留出一截缓冲）。"
        "报价先累加，没有越过硬顶才下载。没有绑卡，没有充值。"
        + (f" 停止原因：{facts['stop_reason']}。" if facts.get("stop_reason") else " 报价和下载都在硬顶以内跑完。")
    )


def write_stop_readme(manifest: dict, detail: str) -> None:
    spent = float(manifest.get("cum_downloaded_usd") or 0)
    quoted = float(manifest.get("cum_quoted_avwap_usd") or 0)
    text = (
        "# PB vs PB-AV 结论（EXPLORATORY）\n\n"
        "比较没有跑完。Credit 护栏先停了下载，所以没有两臂净值。\n\n"
        f"已花费 **${spent:.4f}**。AVWAP 窗口已报价 **${quoted:.4f}**。硬顶 ${CEILING_USD:.0f}。"
        f"原因：{detail}\n\n"
        "已报价未下的部分没有形成账单。全市场分钟没有请求。密钥没有写入文件。\n"
    )
    (OUT / "README_结论_EXPLORATORY.md").write_text(text)
    (G / "README_结论_EXPLORATORY.md").write_text(text)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        err = scrub(traceback.format_exc())
        log(err)
        raise SystemExit(1)
