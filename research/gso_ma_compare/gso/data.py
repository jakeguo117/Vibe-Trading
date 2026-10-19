"""Yahoo daily OHLC with a local CSV cache. The cache is the study snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gso.indicators import normalize_ohlc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _from_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame is None or len(frame) == 0:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns={c: str(c).title() for c in frame.columns})
    return normalize_ohlc(frame)


def _from_chart_api(symbol: str, start: str, end: str) -> pd.DataFrame:
    import json as _json
    import urllib.request

    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?period1={p1}&period2={p2}&interval=1d&events=div%7Csplit"
        "&includeAdjustedClose=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 gso-ma-research"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = _json.loads(resp.read().decode())
    result = payload["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    if not stamps:
        raise RuntimeError(f"Yahoo chart API returned no timestamps for {symbol}")
    quote = result["indicators"]["quote"][0]
    idx = pd.to_datetime(stamps, unit="s", utc=True).tz_convert("America/New_York")
    dates = pd.DatetimeIndex(np.array([ts.date() for ts in idx], dtype="datetime64[ns]"))
    frame = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=dates,
    )
    adj = None
    adj_block = result["indicators"].get("adjclose") or []
    if adj_block and adj_block[0].get("adjclose") is not None:
        adj = pd.Series(adj_block[0]["adjclose"], index=dates)
        adj = adj[~adj.index.duplicated(keep="last")]
    frame = normalize_ohlc(frame)
    if adj is not None:
        ratio = (adj.reindex(frame.index) / frame["Close"]).replace([np.inf, -np.inf], np.nan)
        for col in ("Open", "High", "Low", "Close"):
            frame[col] = frame[col] * ratio
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        frame = frame[(frame["Open"] > 0) & (frame["Close"] > 0)]
    return frame


def load_symbol(symbol: str, cache_dir: Path, start: str = "2004-01-01", end: str | None = None) -> pd.DataFrame:
    """Read the cached Yahoo CSV, or download once and cache it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}.csv"
    meta_path = cache_dir / f"{symbol}.json"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame = frame.set_index("Date")
        frame = normalize_ohlc(frame)
    else:
        if end is None:
            end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        errors = []
        frame = None
        for loader in (_from_yfinance, _from_chart_api):
            try:
                frame = loader(symbol, start, end)
                break
            except Exception as exc:  # network / shape failures fall through to the other Yahoo path
                errors.append(f"{loader.__name__}: {exc}")
        if frame is None or frame.empty:
            raise RuntimeError(f"Yahoo download failed for {symbol}: {errors}")
        weekend = frame.index.weekday >= 5
        if float(weekend.mean()) > 0:
            raise RuntimeError(f"{symbol} cache has weekend sessions; timezone alignment is wrong")
        out = frame.copy()
        out.insert(0, "Date", out.index.strftime("%Y-%m-%d"))
        out.to_csv(path, index=False)
        meta = {
            "symbol": symbol,
            "rows": int(len(frame)),
            "first": frame.index.min().strftime("%Y-%m-%d"),
            "last": frame.index.max().strftime("%Y-%m-%d"),
            "sha256": _sha256(path),
            "source": "yahoo",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        frame = normalize_ohlc(frame)
    weekend = frame.index.weekday >= 5
    if float(weekend.mean()) > 0:
        raise RuntimeError(f"{symbol} has weekend sessions")
    return frame


def load_universe(symbols: tuple[str, ...], cache_dir: Path) -> dict[str, pd.DataFrame]:
    return {symbol: load_symbol(symbol, cache_dir) for symbol in symbols}
