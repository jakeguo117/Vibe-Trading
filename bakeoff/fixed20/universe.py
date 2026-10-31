"""Frozen Fixed20 universe. Order is part of the tie-break.

A1 META/FB: one continuous key, ``META``. Prints on or before 2022-06-08
are the then-traded ticker FB, relabeled onto META. From 2022-06-09 the
prints are META. See ``bakeoff.fixed20.splice``. Do not keep two series
and glue the returns together afterwards.

A2: GOOGL only. GOOG is not in this list.
"""

from __future__ import annotations

from datetime import date

SCHEME_ID = "fixed20_bakeoff_v1"
UNIVERSE_ID = "fixed20-v1"
ELIGIBILITY_ASOF = date(2018, 12, 31)

# Eligibility date is documentation only. The shell does not re-screen liquidity.
FIXED20: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "JPM",
    "BAC",
    "V",
    "XOM",
    "CVX",
    "JNJ",
    "UNH",
    "WMT",
    "HD",
    "CAT",
    "BA",
    "DIS",
    "GLD",
    "USO",
)

META_SYMBOL = "META"
FB_SYMBOL = "FB"
# Last session that still traded as FB. The next session, 2022-06-09, is META.
META_FB_LAST_FB = date(2022, 6, 8)
META_FB_SEAM = date(2022, 6, 9)

# Production fill order when more than one of these is eligible together.
# SLV is not in Fixed20; the pair that can conflict here is GLD then USO.
COMMODITY_PRIORITY: tuple[str, ...] = ("GLD", "SLV", "USO")

K = 3
SMA_N = 50
RET_N = 20
HARD_STOP = -0.08
DOWN_CONFIRM_DAYS = 3
# Same slope lookback the production up-leg uses (MA versus MA four bars ago).
SLOPE_LOOKBACK = 4
GROSS_CAP = 1.0
CASH_FLOOR = 0.0
INITIAL_NAV = 1.0
