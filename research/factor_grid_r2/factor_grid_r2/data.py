"""Yahoo price cache and the liquid panel used in place of the full-market scan.

trading-system PR #13 was not readable from this environment, so the study
cannot replay that branch's price cache. The panel is a fixed list of liquid
stocks, equity ETFs, the three priority commodities, three coins, and the
leveraged ETFs named in the round-2 brief. Survivorship bias is real and is
called out in the Chinese summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .engine import Pack
from .signals import build_features

EQUITY_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "XLB", "XLC", "EFA", "EEM", "SMH",
]
COMMODITY = ["GLD", "SLV", "USO"]
STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM",
    "UNH", "V", "MA", "HD", "PG", "KO", "PEP", "WMT", "COST", "AVGO",
    "ORCL", "CSCO", "INTC", "AMD", "NFLX", "DIS", "BA", "CAT", "GE", "IBM",
    "MCD", "NKE", "CRM", "ADBE", "BAC", "WFC", "CVX", "PFE", "MRK", "ABBV",
    "T", "VZ", "QCOM", "TXN", "AMAT", "LRCX", "MU", "AMGN", "HON", "UNP",
    "LOW", "SBUX", "GS",
]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
LEVERED = ["UGL", "AGQ", "UCO", "TQQQ", "SOXL"]
LEV_MAP = {"GLD": "UGL", "SLV": "AGQ", "USO": "UCO", "QQQ": "TQQQ", "SMH": "SOXL"}

KIND = {}
for s in STOCKS:
    KIND[s] = 0
for s in EQUITY_ETFS + LEVERED:
    KIND[s] = 1
for s in COMMODITY:
    KIND[s] = 2
for s in CRYPTO:
    KIND[s] = 3


def all_tickers() -> list[str]:
    out = []
    for group in (EQUITY_ETFS, COMMODITY, STOCKS, CRYPTO, LEVERED):
        for s in group:
            if s not in out:
                out.append(s)
    return out


def _flatten_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may return (field, ticker) or (ticker, field).
        level0 = set(map(str, raw.columns.get_level_values(0)))
        fields = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        if level0 & fields:
            raw = raw.swaplevel(0, 1, axis=1)
        frames = []
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            sub = raw[t].copy()
            sub = sub.rename(columns={c: str(c).lower().replace(" ", "_") for c in sub.columns})
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]
            if "close" not in keep:
                continue
            sub = sub[keep]
            sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()
    # single ticker
    sub = raw.rename(columns={c: str(c).lower().replace(" ", "_") for c in raw.columns})
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]
    sub = sub[keep]
    t = tickers[0]
    sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
    return sub.sort_index()


def load_yahoo(cache: Path, start: str = "1997-01-01", end: str = "2026-10-14") -> pd.DataFrame:
    import yfinance as yf

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "yahoo_panel.parquet"
    tickers = all_tickers()
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        have = set(df.columns.get_level_values(0))
        if set(tickers) <= have and len(df) > 1000:
            return df
    frames = []
    missing = []
    for i in range(0, len(tickers), 12):
        batch = tickers[i : i + 12]
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        flat = _flatten_download(raw, batch)
        got = set(flat.columns.get_level_values(0)) if len(flat.columns) else set()
        if len(flat):
            frames.append(flat)
        for t in batch:
            if t not in got:
                missing.append(t)
    for t in missing:
        raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        flat = _flatten_download(raw, [t])
        if len(flat):
            frames.append(flat)
    if not frames:
        raise RuntimeError("Yahoo download returned no prices")
    df = pd.concat(frames, axis=1).sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    # US session calendar: drop weekend rows that are crypto-only, then
    # broadcast crypto onto the next equity session without using a future close.
    spy = df[("SPY", "close")].dropna()
    cal = spy.index
    out = df.reindex(cal)
    for t in CRYPTO:
        if (t, "close") not in df.columns:
            continue
        src = df[t].copy()
        src.index = pd.DatetimeIndex(src.index).tz_localize(None).normalize()
        src = src[~src.index.duplicated(keep="last")].sort_index()
        # A crypto daily bar dated D is treated as known at the US close of D.
        # Decisions use the previous row, so the same-day close is only a mark.
        aligned = src.reindex(cal.union(src.index)).sort_index().ffill().reindex(cal)
        for field in aligned.columns:
            out[(t, field)] = aligned[field].to_numpy()
    out.to_parquet(path)
    return out


def load_stooq(symbols: dict[str, str], cache: Path) -> dict[str, pd.Series]:
    """symbols: yahoo name -> stooq symbol. Returns close series or an error string."""
    import io
    import urllib.request

    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for yahoo, sym in symbols.items():
        path = cache / f"stooq_{sym.replace('.', '_')}.csv"
        if path.exists():
            cached = path.read_text(errors="replace")
            if not cached.startswith("Date"):
                path.unlink()
        if not path.exists():
            url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; research-factor-grid-r2)"},
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                out[yahoo] = f"error:{type(exc).__name__}:{exc}"
                continue
            if not text.startswith("Date"):
                out[yahoo] = "error:stooq_browser_check"
                continue
            path.write_text(text)
        text = path.read_text()
        if not text.startswith("Date"):
            out[yahoo] = "error:not_csv"
            continue
        frame = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
        if "Close" not in frame.columns or frame.empty:
            out[yahoo] = "error:empty"
            continue
        s = frame.set_index("Date")["Close"].astype(float).sort_index()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        out[yahoo] = s
    return out


def probe_nasdaq(symbols: list[str], start: str = "2024-01-02", end: str = "2024-03-01") -> list[dict]:
    """Public Nasdaq historical endpoint. Empty tables are recorded, not invented."""
    import json
    import urllib.request

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; research-factor-grid-r2)",
        "Accept": "application/json",
    }
    rows = []
    for name in symbols:
        asset = "etf" if name in {"SPY", "QQQ", "GLD"} else "stocks"
        url = (
            "https://api.nasdaq.com/api/quote/"
            f"{name}/historical?assetclass={asset}&fromdate={start}&todate={end}&limit=80"
        )
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            table = ((payload.get("data") or {}).get("tradesTable") or {}).get("rows")
            n = 0 if table is None else len(table)
            rows.append({"name": name, "status": "ok" if n else "empty", "n": n, "start": start, "end": end})
        except Exception as exc:  # noqa: BLE001
            rows.append({"name": name, "status": f"error:{type(exc).__name__}", "n": 0})
    return rows


def make_pack(panel: pd.DataFrame) -> Pack:
    names = []
    for t in all_tickers():
        if (t, "close") in panel.columns and panel[(t, "close")].notna().sum() > 200:
            names.append(t)
    if "SPY" not in names:
        raise RuntimeError("SPY missing from the price panel")
    daily = panel.loc[:, panel.columns.get_level_values(0).isin(names)].copy()
    daily = daily.sort_index()
    feat = build_features(daily, names)
    spy = names.index("SPY")
    spy_trend = (feat["stage"][:, spy] == 2) | (feat["stage"][:, spy] == 3)
    n = len(names)
    ix = {s: i for i, s in enumerate(names)}
    kind = np.array([KIND.get(s, 0) for s in names], dtype=np.int8)
    is_crypto = np.array([s in CRYPTO for s in names])
    is_commodity = np.array([s in COMMODITY for s in names])
    is_levered = np.array([s in LEVERED for s in names])
    tradable_control = ~(is_crypto | is_levered)
    tradable_stock = ~(is_crypto | is_commodity | is_levered)
    tradable_sc = ~(is_commodity | is_levered)
    side = np.where(is_crypto, 0.0010, 0.0005).astype(np.float64)
    lev_index = np.full(n, -1, dtype=np.int64)
    for src, dst in LEV_MAP.items():
        if src in ix and dst in ix:
            lev_index[ix[src]] = ix[dst]
    return Pack(
        names=names,
        ix=ix,
        n=n,
        dates=pd.DatetimeIndex(feat["dates"]),
        O=feat["open"],
        H=feat["high"],
        L=feat["low"],
        C=feat["close"],
        V=feat["volume"],
        up=feat["up"],
        dn=feat["dn"],
        confirm=feat["confirm"],
        anchor=feat["anchor"],
        stage=feat["stage"],
        route=feat["route"],
        vol=feat["vol"],
        realized=feat["realized"],
        mom=feat["mom"],
        dv60=feat["dv60"],
        spy_trend=spy_trend,
        kind=kind,
        is_crypto=is_crypto,
        tradable_control=tradable_control,
        tradable_stock=tradable_stock,
        tradable_sc=tradable_sc,
        side=side,
        lev_index=lev_index,
    )
