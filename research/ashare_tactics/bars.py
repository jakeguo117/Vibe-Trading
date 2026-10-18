"""Read cached daily bars. The cache is gitignored and is not source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ashare_tactics.portfolio import SymbolBars


def load_frame(path: Path, delist: np.datetime64 | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if delist is not None:
        frame = frame[frame["date"].values.astype("datetime64[D]") <= np.datetime64(delist, "D")]
    frame = frame[frame["volume"] > 0]
    frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)]
    return frame.reset_index(drop=True)


def repair_factor(raw_close: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Drop factor jumps that do not show up in the unadjusted price.

    On 112 names the Sina qfq factor steps (often down to 1) while the raw
    close barely moves, so ``raw / factor`` gaps by 30–200% and the dividend
    residual invents a cash credit. A real split moves the raw close by about
    the same ratio and is left alone. The tail of the factor is rescaled so
    later, genuine corporate actions stay intact.
    """
    factor = np.asarray(factor, dtype=float).copy()
    raw = np.asarray(raw_close, dtype=float)
    for i in range(1, len(factor)):
        if raw[i - 1] <= 0 or raw[i] <= 0 or factor[i - 1] <= 0 or factor[i] <= 0:
            continue
        raw_ret = raw[i] / raw[i - 1] - 1.0
        qfq_prev = raw[i - 1] / factor[i - 1]
        qfq_ret = (raw[i] / factor[i]) / qfq_prev - 1.0
        if abs(qfq_ret) > 0.30 and abs(raw_ret) < 0.20:
            factor[i:] *= factor[i - 1] / factor[i]
    return factor


def arrays_from_frame(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    factor = frame["factor"].to_numpy(dtype=float)
    factor = np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)
    factor = repair_factor(frame["close"].to_numpy(dtype=float), factor)
    return {
        "dates": frame["date"].to_numpy(dtype="datetime64[D]"),
        "raw_open": frame["open"].to_numpy(dtype=float),
        "raw_high": frame["high"].to_numpy(dtype=float),
        "raw_low": frame["low"].to_numpy(dtype=float),
        "raw_close": frame["close"].to_numpy(dtype=float),
        "volume": frame["volume"].to_numpy(dtype=float),
        "amount": frame["amount"].to_numpy(dtype=float),
        "factor": factor,
    }


def symbol_bars_from_frame(frame: pd.DataFrame) -> SymbolBars:
    arrays = arrays_from_frame(frame)
    return SymbolBars(
        dates=arrays["dates"],
        raw_open=arrays["raw_open"],
        raw_close=arrays["raw_close"],
        factor=arrays["factor"],
    )
