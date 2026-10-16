"""Trade candidates emitted by a tactic before portfolio sizing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Candidate:
    """One stock-level round trip. Prices are unadjusted and unslipped.

    ``factor_*`` is the Sina forward-adjustment divisor (qfq = raw / factor).
    The portfolio applies slippage, fees, lot size, and the factor residual.
    """

    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    raw_entry: float
    raw_exit: float
    factor_entry: float
    factor_exit: float
    liquidity: float
    slope: int | None = None
    take_profit: bool = False


@dataclass
class SignalBatch:
    """Candidates plus a few rejection counts for the write-up."""

    candidates: list[Candidate]
    signals: int = 0
    limit_up_skips: int = 0
