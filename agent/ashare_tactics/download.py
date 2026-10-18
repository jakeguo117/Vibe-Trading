"""Cache A-share daily bars with AkShare's Sina source.

``stock_zh_a_hist`` (East Money) closes the connection from this environment.
The bars below are the same free series ``ak.stock_zh_a_daily`` decodes
(Sina ``hisdata_klc2`` plus the qfq factor file). The outstanding-share
request that function also makes is skipped: 成交额 is already on the kline,
and that extra call dominates the runtime.

Cached parquet stays under the gitignored ``data/`` directory. Nothing here
writes bars into the repository.
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ashare_tactics.constants import INDEX_ALL, INDEX_CSI300, WARMUP

HIST_URL = "https://finance.sina.com.cn/realstock/company/{}/hisdata_klc2/klc_kl.js"
QFQ_URL = "https://finance.sina.com.cn/realstock/company/{}/qfq.js"
COUNT_URL = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/"
    "json_v2.php/Market_Center.getHQNodeStockCount?node=hs_a"
)
LIST_URL = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/"
    "json_v2.php/Market_Center.getHQNodeData"
)
LIST_PARAMS = {
    "page": "1",
    "num": "80",
    "sort": "symbol",
    "asc": "1",
    "node": "hs_a",
    "symbol": "",
    "_s_r_a": "page",
}
_TLS = threading.local()
_HEADERS = {"User-Agent": "Mozilla/5.0 (research backtest; akshare sina daily)"}


def _session() -> requests.Session:
    sess = getattr(_TLS, "sess", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(_HEADERS)
        _TLS.sess = sess
    return sess


def _js():
    js = getattr(_TLS, "js", None)
    if js is None:
        from akshare.stock.cons import hk_js_decode
        from py_mini_racer import MiniRacer

        js = MiniRacer()
        js.eval(hk_js_decode)
        _TLS.js = js
    return js


def _get(url: str, params: dict | None = None, timeout: float = 20.0) -> requests.Response:
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = _session().get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - retry network errors
            last = exc
            time.sleep(0.4 * (2**attempt))
    raise RuntimeError(f"GET failed {url}") from last


def fetch_universe() -> pd.DataFrame:
    """Current A-share list from the Sina node AkShare's spot list uses."""
    raw_count = _get(COUNT_URL).text.strip().strip('"')
    count = int(raw_count)
    pages = max(1, math.ceil(count / 80))
    rows: list[dict] = []
    for page in range(1, pages + 1):
        params = dict(LIST_PARAMS)
        params["page"] = str(page)
        text = _get(LIST_URL, params=params).text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            from akshare.utils import demjson

            payload = demjson.decode(text)
        rows.extend(payload)
    frame = pd.DataFrame(rows)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["name"] = frame["name"].astype(str)
    frame = frame[frame["symbol"].str.match(r"^(sh|sz|bj)\d{6}$")]
    frame = frame.drop_duplicates("symbol")
    frame["source"] = "sina_hs_a"
    frame["delist_date"] = pd.NaT
    return frame[["symbol", "code", "name", "source", "delist_date"]].reset_index(drop=True)


def fetch_sse_delist() -> pd.DataFrame:
    """Shanghai terminated listings via the same query as ``stock_info_sh_delist``.

    Shenzhen's exchange endpoint resets the TLS handshake from this network,
    so SZ delistings are counted in the write-up rather than downloaded here.
    """
    try:
        import akshare as ak

        listed = ak.stock_info_sh_delist("全部")
        frame = pd.DataFrame(
            {
                "code": listed["公司代码"].astype(str).str.zfill(6),
                "name": listed["公司简称"].astype(str),
                "delist_date": pd.to_datetime(listed["暂停上市日期"], errors="coerce"),
            }
        )
    except Exception:
        frame = _fetch_sse_delist_direct()
    frame = frame[frame["code"].str.match(r"^6\d{5}$")]
    frame = frame[frame["delist_date"].notna()]
    frame = frame[frame["delist_date"] >= "2015-01-01"]
    frame["symbol"] = "sh" + frame["code"]
    frame["source"] = "sse_delist"
    return frame[["symbol", "code", "name", "source", "delist_date"]].drop_duplicates("symbol")


def _fetch_sse_delist_direct() -> pd.DataFrame:
    from curl_cffi import requests as cffi_requests

    url = "https://query.sse.com.cn/commonQuery.do"
    headers = {"Referer": "https://www.sse.com.cn/", "User-Agent": _HEADERS["User-Agent"]}
    params = {
        "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L",
        "isPagination": "true",
        "STOCK_CODE": "",
        "STOCK_TYPE": "1,2,8",
        "COMPANY_STATUS": "3",
        "type": "inParams",
        "pageHelp.cacheSize": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.pageSize": "500",
        "pageHelp.pageNo": "1",
        "pageHelp.endPage": "1",
    }
    response = cffi_requests.get(url, params=params, headers=headers, timeout=30, impersonate="chrome")
    data = response.json()["pageHelp"]["data"]
    return pd.DataFrame(
        {
            "code": [str(row.get("A_STOCK_CODE") or "").zfill(6) for row in data],
            "name": [str(row.get("COMPANY_ABBR") or "") for row in data],
            "delist_date": pd.to_datetime([row.get("DELIST_DATE") for row in data], errors="coerce"),
        }
    )


def _parse_factor(text: str) -> pd.Series | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    obj = json.loads(text[start : end + 1])
    data = obj.get("data") or []
    if not data:
        return None
    index = pd.to_datetime([row["d"] for row in data], errors="coerce")
    values = pd.to_numeric([row["f"] for row in data], errors="coerce")
    series = pd.Series(values, index=index).sort_index()
    series = series[np.isfinite(series.to_numpy()) & (series.to_numpy() > 0)]
    series = series[~series.index.isna()]
    if series.empty:
        return None
    return series


def fetch_daily(symbol: str, warmup: np.datetime64 = WARMUP) -> pd.DataFrame:
    """Unadjusted OHLCV plus the Sina qfq divisor (qfq = raw / factor)."""
    text = _get(HIST_URL.format(symbol)).text
    if "KLC_K2_" not in text:
        raise RuntimeError(f"no sina kline for {symbol}")
    payload = text.split("=", 1)[1].split(";", 1)[0].replace('"', "")
    rows = _js().call("d", payload)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"empty sina kline for {symbol}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None).dt.normalize()
    if "amount" not in frame.columns:
        frame["amount"] = frame["volume"] * frame["close"]
    for col in ("open", "high", "low", "close", "volume", "amount"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame[frame["date"].values.astype("datetime64[D]") >= np.datetime64(warmup, "D")]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    factor_text = _get(QFQ_URL.format(symbol)).text
    factor = _parse_factor(factor_text)
    if factor is None:
        frame["factor"] = 1.0
    else:
        factor.index = pd.to_datetime(factor.index).tz_localize(None).normalize()
        aligned = factor.reindex(frame["date"]).ffill().bfill()
        frame["factor"] = aligned.to_numpy(dtype=float)
        frame["factor"] = frame["factor"].fillna(1.0)
    frame = frame[frame["factor"] > 0]
    keep = ["date", "open", "high", "low", "close", "volume", "amount", "factor"]
    return frame[keep].reset_index(drop=True)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def download_symbol(symbol: str, dest: Path, warmup: np.datetime64 = WARMUP) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    frame = fetch_daily(symbol, warmup=warmup)
    if frame.empty:
        return "empty"
    _write_parquet(frame, dest)
    return "ok"


def download_index(symbol: str, dest: Path) -> None:
    """Index close via AkShare.

    Sina's ``stock_zh_index_daily`` series for CSI All Share (000985) stops in
    2016. When the Sina file does not cover 2015 through the latest session,
    fall back to ``stock_zh_index_daily_tx`` (Tencent, paged by year), which
    AkShare documents as the supplement for indexes Sina truncates.
    """
    if dest.exists() and dest.stat().st_size > 0:
        existing = pd.read_parquet(dest)
        existing["date"] = pd.to_datetime(existing["date"])
        if existing["date"].min() <= pd.Timestamp("2015-01-31") and existing["date"].max() >= pd.Timestamp("2026-10-01"):
            return
    import akshare as ak

    frame = ak.stock_zh_index_daily(symbol=symbol)
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].min() > pd.Timestamp("2015-06-01") or frame["date"].max() < pd.Timestamp("2026-01-01"):
        frame = ak.stock_zh_index_daily_tx(symbol=symbol, start_date="20140101", end_date="20261016")
        frame["date"] = pd.to_datetime(frame["date"])
    _write_parquet(frame, dest)


def build_universe(cache: Path) -> pd.DataFrame:
    spot = fetch_universe()
    try:
        delisted = fetch_sse_delist()
    except Exception as exc:  # noqa: BLE001 - delist list is best-effort
        print(f"sse delist list unavailable: {exc}")
        delisted = spot.iloc[0:0].copy()
    extra = delisted[~delisted["symbol"].isin(set(spot["symbol"]))]
    universe = pd.concat([spot, extra], ignore_index=True)
    universe_path = cache / "universe.parquet"
    _write_parquet(universe, universe_path)
    return universe


def download_all(cache: Path, workers: int = 4, max_symbols: int = 0, warmup: np.datetime64 = WARMUP) -> dict:
    """Download the universe and both benchmark indexes. Safe to resume."""
    cache.mkdir(parents=True, exist_ok=True)
    _assert_cache(cache)
    universe_path = cache / "universe.parquet"
    if universe_path.exists() and max_symbols == 0:
        universe = pd.read_parquet(universe_path)
    else:
        universe = build_universe(cache)
    if max_symbols > 0:
        universe = universe.head(max_symbols).copy()
    bar_dir = cache / "bars"
    bar_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(row.symbol, bar_dir / f"{row.symbol}.parquet") for row in universe.itertuples(index=False)]
    counts = {"ok": 0, "cached": 0, "empty": 0, "fail": 0}
    failed: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(download_symbol, symbol, dest, warmup): symbol for symbol, dest in jobs
        }
        done = 0
        for future in as_completed(future_map):
            symbol = future_map[future]
            done += 1
            try:
                status = future.result()
            except Exception as exc:  # noqa: BLE001 - one symbol must not stop the run
                status = "fail"
                failed.append(f"{symbol}\t{type(exc).__name__}: {exc}")
            counts[status] = counts.get(status, 0) + 1
            if done % 200 == 0 or done == len(jobs):
                elapsed = time.time() - started
                print(f"download {done}/{len(jobs)} {counts} {elapsed:.0f}s", flush=True)
    if failed:
        print(f"retrying {len(failed)} failures one at a time", flush=True)
        still: list[str] = []
        for line in failed:
            symbol = line.split("\t", 1)[0]
            dest = bar_dir / f"{symbol}.parquet"
            try:
                status = download_symbol(symbol, dest, warmup)
            except Exception as exc:  # noqa: BLE001 - record and continue
                still.append(f"{symbol}\t{type(exc).__name__}: {exc}")
                continue
            counts["fail"] -= 1
            counts[status] = counts.get(status, 0) + 1
        if still:
            (cache / "failures.txt").write_text("\n".join(still), encoding="utf-8")
        elif (cache / "failures.txt").exists():
            (cache / "failures.txt").unlink()
        counts["fail"] = len(still)
    for symbol in (INDEX_CSI300, INDEX_ALL):
        download_index(symbol, cache / f"index_{symbol}.parquet")
        print(f"index {symbol} ready", flush=True)
    counts["universe"] = int(len(universe))
    return counts


def _assert_cache(cache: Path) -> None:
    resolved = cache.resolve()
    if "data" not in resolved.parts and "tmp" not in resolved.parts:
        raise SystemExit(f"refusing to write market data outside data/: {resolved}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cache A-share daily bars (gitignored).")
    parser.add_argument("--cache", type=Path, default=Path("data/ashare_tactics_2015"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()
    print(download_all(args.cache, workers=args.workers, max_symbols=args.max_symbols))


if __name__ == "__main__":
    main()
