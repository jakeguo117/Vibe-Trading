"""Same fee hook as ``run_portfolio``: ``upleg_bt.costs.trade_fee``.

Rates are the repo defaults (5 bp liquid / 10 bp illiquid, $20M dollar-volume
cut). GLD/SLV/USO stay on the liquid side, matching the production overlay.
This module does not invent a second schedule. A zero-cost run is not a mode.
"""

from __future__ import annotations

import numpy as np

from bakeoff.fixed20.universe import FIXED20
from bakeoff.fixed20.upleg_link import require

COST_MULT = 1.0


def _fee_api():
    require()
    from upleg_bt.costs import (
        DV_LIQUID,
        SIDE_ILLIQUID,
        SIDE_LIQUID,
        side_cost_matrix,
        trade_fee,
    )
    from upleg_bt.engine.fib_locked import TICKERS

    return {
        "DV_LIQUID": DV_LIQUID,
        "SIDE_ILLIQUID": SIDE_ILLIQUID,
        "SIDE_LIQUID": SIDE_LIQUID,
        "side_cost_matrix": side_cost_matrix,
        "trade_fee": trade_fee,
        "TICKERS": TICKERS,
    }


def cost_meta() -> dict[str, object]:
    api = _fee_api()
    tickers = api["TICKERS"]
    forced = [ticker for ticker in tickers if ticker in FIXED20]
    return {
        "module": "upleg_bt.costs",
        "function": "trade_fee",
        "side_cost_function": "side_cost_matrix",
        "SIDE_LIQUID": float(api["SIDE_LIQUID"]),
        "SIDE_ILLIQUID": float(api["SIDE_ILLIQUID"]),
        "DV_LIQUID": float(api["DV_LIQUID"]),
        "cost_mult": COST_MULT,
        "commodity_forced_liquid_rule": "upleg_bt.engine.fib_locked.TICKERS",
        "commodity_forced_liquid": forced,
    }


def side_cost(symbol: str, dollar_volume: float) -> float:
    """One name's per-side fee fraction, via ``side_cost_matrix``."""
    api = _fee_api()
    if symbol in api["TICKERS"]:
        return float(api["SIDE_LIQUID"])
    matrix = api["side_cost_matrix"](np.asarray([dollar_volume], dtype=np.float64))
    return float(matrix[0])


def turnover_fee(
    target: dict[str, float],
    current: dict[str, float],
    side: dict[str, float],
    cost_mult: float = COST_MULT,
) -> float:
    """NAV fraction charged to move ``current`` weights to ``target``."""
    api = _fee_api()
    fee = float(api["trade_fee"](target, current, side, cost_mult))
    if fee < 0:
        raise RuntimeError("trade_fee 返回了负数")
    return fee
