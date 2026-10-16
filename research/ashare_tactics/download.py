"""Cache A-share daily bars with AkShare East Money hist.

On Hermes, ``stock_zh_a_hist`` works; the prior cloud agent had to use Sina
because East Money reset TLS there. We store unadjusted OHLCV plus
``factor = raw_close / qfq_close`` so the rest of the engine is unchanged.

Cached parquet stays under the gitignored data-cache directory.
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
    """Unadjusted OHLCV plus a qfq divisor (qfq = raw / factor).

    Hermes can reach East Money ``stock_zh_a_hist`` (the cloud agent could not).
    We pull unadjusted and qfq series and set ``factor = raw_close / qfq_close``.
    """
    import akshare as ak

    code = "".join(ch for ch in symbol if ch.isdigit())[-6:].zfill(6)
    start = str(np.datetime64(warmup, "D")).replace("-", "")
    end = "20261016"
    last: Exception | None = None
    raw = qfq = None
    for attempt in range(4):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start, end_date=end, adjust=""
            )
            qfq = ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"
            )
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.5 * (2**attempt))
    if raw is None or qfq is None:
        raise RuntimeError(f"eastmoney hist failed for {symbol}") from last
    if raw.empty:
        raise RuntimeError(f"empty eastmoney hist for {symbol}")

    def _norm(frame: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        out = frame.rename(columns=rename).copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        for col in ("open", "high", "low", "close", "volume", "amount"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["date", "open", "high", "low", "close"])
        out = out.sort_values("date").drop_duplicates("date", keep="last")
        return out

    raw_n = _norm(raw)
    qfq_n = _norm(qfq)[["date", "close"]].rename(columns={"close": "qfq_close"})
    frame = raw_n.merge(qfq_n, on="date", how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = frame["close"] / frame["qfq_close"]
    factor = factor.replace([np.inf, -np.inf], np.nan)
    # When qfq is missing, treat as unadjusted.
    frame["factor"] = factor.fillna(1.0)
    frame.loc[frame["factor"] <= 0, "factor"] = 1.0
    frame = frame[frame["date"].values.astype("datetime64[D]") >= np.datetime64(warmup, "D")]
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
        if existing["date"].min() <= pd.Timestamp("2015-01-31") and existing["date"].max() >= pd.Timestamp("2026-09-01"):
            return
    import akshare as ak

    frame = ak.stock_zh_index_daily(symbol=symbol)
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].min() > pd.Timestamp("2015-06-01") or frame["date"].max() < pd.Timestamp("2026-01-01"):
        frame = ak.stock_zh_index_daily_tx(symbol=symbol, start_date="20140101", end_date="20261016")
        frame["date"] = pd.to_datetime(frame["date"])
    _write_parquet(frame, dest)


def fetch_szse_delist() -> pd.DataFrame:
    """Shenzhen terminated listings via AkShare ``stock_info_sz_delist``.

    Hermes can often reach the SZSE endpoint that the cloud agent could not.
    Falls back to an empty frame on failure so the run still proceeds.
    """
    import akshare as ak

    frames = []
    for tip in ("终止上市公司", "暂停上市公司"):
        try:
            listed = ak.stock_info_sz_delist(tip)
        except Exception as exc:  # noqa: BLE001
            print(f"szse delist ({tip}) unavailable: {exc}", flush=True)
            continue
        if listed is None or listed.empty:
            continue
        # Column names vary slightly across akshare versions.
        cols = {str(c): c for c in listed.columns}
        code_col = next((cols[k] for k in cols if "代码" in k or k.lower() == "code"), None)
        name_col = next((cols[k] for k in cols if "简称" in k or "名称" in k or "name" in k.lower()), None)
        date_col = next(
            (cols[k] for k in cols if "终止" in k or "暂停" in k or "日期" in k or "date" in k.lower()),
            None,
        )
        if code_col is None:
            print(f"szse delist ({tip}) unexpected columns: {list(listed.columns)}", flush=True)
            continue
        frame = pd.DataFrame(
            {
                "code": listed[code_col].astype(str).str.zfill(6),
                "name": listed[name_col].astype(str) if name_col else "",
                "delist_date": pd.to_datetime(listed[date_col], errors="coerce") if date_col else pd.NaT,
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["symbol", "code", "name", "source", "delist_date"])
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["code"].str.match(r"^[03]\d{5}$")]
    frame = frame[frame["delist_date"].notna()]
    frame = frame[frame["delist_date"] >= "2015-01-01"]
    frame["symbol"] = "sz" + frame["code"]
    frame["source"] = "szse_delist"
    return frame[["symbol", "code", "name", "source", "delist_date"]].drop_duplicates("symbol")


def build_universe(cache: Path) -> pd.DataFrame:
    spot = fetch_universe()
    pieces = [spot]
    try:
        sse = fetch_sse_delist()
        pieces.append(sse)
    except Exception as exc:  # noqa: BLE001 - delist list is best-effort
        print(f"sse delist list unavailable: {exc}", flush=True)
    try:
        szse = fetch_szse_delist()
        print(f"szse delist rows={len(szse)}", flush=True)
        pieces.append(szse)
    except Exception as exc:  # noqa: BLE001
        print(f"szse delist list unavailable: {exc}", flush=True)
    universe = pd.concat(pieces, ignore_index=True)
    universe = universe.drop_duplicates("symbol", keep="first")
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
    parts = set(resolved.parts)
    allowed = {"data", "tmp", "data-cache"}
    if not (parts & allowed) and "data-cache" not in str(resolved):
        raise SystemExit(f"refusing to write market data outside data/ or data-cache/: {resolved}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cache A-share daily bars (gitignored).")
    parser.add_argument("--cache", type=Path, default=Path("/Volumes/Disk/项目/data-cache/ashare_tactics"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()
    print(download_all(args.cache, workers=args.workers, max_symbols=args.max_symbols))


if __name__ == "__main__":
    main()
