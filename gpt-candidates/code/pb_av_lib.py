"""PB vs PB-AV helpers. No network. No secrets.

AVWAP is the regular-session minute VWAP from anchor day h through signal day t.
The daily entry and stop rules are unchanged; this module only filters and reviews.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

# "ALL" is Allstate's ticker. Databento's market-wide sentinel is ALL_SYMBOLS.
FORBIDDEN_SYMBOLS = {"ALL_SYMBOLS", "*"}
RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60  # bars whose start is in [09:30, 16:00)


def map_symbol_yahoo_to_databento(sym: str) -> str:
    """Yahoo class shares use a hyphen (BF-B). EQUS.MINI raw_symbol uses a dot (BF.B)."""
    if not isinstance(sym, str) or not sym or sym.strip() != sym:
        raise ValueError("empty symbol")
    if sym.upper() in FORBIDDEN_SYMBOLS or "ALL_SYMBOLS" in sym.upper():
        raise ValueError("refusing market-wide symbol")
    if any(ch.isspace() for ch in sym):
        raise ValueError("symbol has whitespace")
    return sym.replace("-", ".")


def assert_one_calendar_day(start: str, end: str, symbols: Iterable[str]) -> list[str]:
    """Reject anything other than an explicit one-day symbol list."""
    syms = list(symbols)
    if not syms:
        raise ValueError("empty symbol list")
    if len(syms) > 400:
        raise ValueError(f"symbol list too wide: {len(syms)}")
    for s in syms:
        if s.upper() in FORBIDDEN_SYMBOLS or "ALL_SYMBOLS" in s.upper():
            raise ValueError("refusing market-wide symbol")
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if (e - s) != pd.Timedelta(days=1):
        raise ValueError(f"minute request must span exactly one calendar day, got {start} -> {end}")
    if s.normalize() != s or e.normalize() != e:
        raise ValueError("start/end must be calendar dates")
    return syms


def rth_mask(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    """True for minute bars that start inside 09:30–16:00 America/New_York."""
    if ts_utc.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    ny = ts_utc.tz_convert("America/New_York")
    mins = ny.hour.to_numpy() * 60 + ny.minute.to_numpy()
    return (mins >= RTH_OPEN_MIN) & (mins < RTH_CLOSE_MIN)


def avwap_from_ohlcv(high, low, close, volume) -> tuple[float, float, int]:
    """Return (avwap, sum_volume, n_bars). avwap is NaN when volume sums to 0."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v) & (v >= 0) & (h >= l)
    h, l, c, v = h[ok], l[ok], c[ok], v[ok]
    sum_v = float(v.sum()) if len(v) else 0.0
    if sum_v <= 0 or not math.isfinite(sum_v):
        return float("nan"), sum_v, int(len(v))
    typical = (h + l + c) / 3.0
    return float((typical * v).sum() / sum_v), sum_v, int(len(v))


def review_same_bar(bars: pd.DataFrame, b_raw: float, s_raw: float) -> dict:
    """Classify an entry-day 1m path. Does not change the daily conservative result.

    bars columns: ts (UTC), open, high, low. Sorted by ts. Already RTH-filtered.
    """
    out = {
        "entry_ts": "",
        "stop_ts": "",
        "entry_at_open": 0,
        "ambiguous_1m": 0,
        "resolution": "no_bars",
        "n_rth_bars": 0,
        "scale_mismatch": 0,
    }
    if bars is None or len(bars) == 0 or not math.isfinite(b_raw) or not math.isfinite(s_raw):
        return out
    b = bars.sort_values("ts")
    out["n_rth_bars"] = int(len(b))
    med = float(np.nanmedian(b["close"])) if "close" in b.columns else float(np.nanmedian(b["high"]))
    if math.isfinite(med) and med > 0 and (b_raw < 0.5 * med or b_raw > 2.0 * med):
        out["scale_mismatch"] = 1
        out["resolution"] = "price_scale_mismatch"
        return out
    o = b["open"].to_numpy(dtype=np.float64)
    h = b["high"].to_numpy(dtype=np.float64)
    l = b["low"].to_numpy(dtype=np.float64)
    ts = [str(pd.Timestamp(t).isoformat()) for t in b["ts"]]
    entry_i = None
    for i in range(len(b)):
        if (math.isfinite(h[i]) and h[i] >= b_raw) or (math.isfinite(o[i]) and o[i] >= b_raw):
            entry_i = i
            break
    if entry_i is None:
        out["resolution"] = "no_minute_entry"
        return out
    out["entry_ts"] = ts[entry_i]
    out["entry_at_open"] = int(math.isfinite(o[entry_i]) and o[entry_i] >= b_raw)
    stop_i = None
    for j in range(entry_i, len(b)):
        if math.isfinite(l[j]) and l[j] <= s_raw:
            stop_i = j
            break
    if stop_i is None:
        prior = any(math.isfinite(l[k]) and l[k] <= s_raw for k in range(entry_i))
        out["resolution"] = "stop_before_entry" if prior else "no_stop_after_entry"
        out["ambiguous_1m"] = 0
        return out
    out["stop_ts"] = ts[stop_i]
    if stop_i == entry_i:
        out["ambiguous_1m"] = 1
        out["resolution"] = "same_minute_conservative"
    else:
        out["ambiguous_1m"] = 0
        out["resolution"] = "entry_then_stop"
    return out


def pick_top_orders(candidates: pd.DataFrame) -> pd.DataFrame:
    """One order per signal date. Rank: R63 desc, ADV20 desc, original column j asc."""
    need = {"t", "symbol", "R63", "ADV20", "j", "B", "S"}
    missing = need - set(candidates.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    if candidates.empty:
        return candidates.copy()
    df = candidates.dropna(subset=["R63", "ADV20", "B", "S", "j"]).copy()
    df = df.sort_values(["t", "R63", "ADV20", "j"], ascending=[True, False, False, True], kind="mergesort")
    return df.groupby("t", as_index=False, sort=True).head(1).reset_index(drop=True)


def future_split_factor(split_ratio: pd.Series) -> pd.Series:
    """F[t] = product of split ratios on dates strictly after t. Ratio 0 means no split."""
    r = split_ratio.astype(float).fillna(0.0)
    r = r.where(r > 0, 1.0)
    # product from t through the end, including t, then drop today's ratio
    rev = r.iloc[::-1].cumprod().iloc[::-1]
    f = rev.shift(-1)
    if len(f):
        f.iloc[-1] = 1.0
    return f.fillna(1.0)
