"""Fixed research parameters. These follow the requested tactic notes."""

from __future__ import annotations

import numpy as np

START = np.datetime64("2015-01-01")
WARMUP = np.datetime64("2014-01-01")
# ChiNext (300/301) limit widened from 10% to 20% on this session.
CHINEXT_REFORM = np.datetime64("2020-08-24")
# Stamp tax on sells was halved from 0.1% to 0.05% effective this session.
STAMP_CUTOVER = np.datetime64("2023-08-28")

INITIAL_CASH = 1_000_000.0
MAX_POSITIONS = 10
WEIGHT = 0.10
LOT = 100

COMMISSION_RATE = 0.00025  # 0.025% each side
COMMISSION_MIN = 5.0  # CNY
SLIPPAGE = 0.001  # 0.1% each side
STAMP_BEFORE = 0.001  # 0.1% on sells
STAMP_AFTER = 0.0005  # 0.05% on sells from 2023-08-28

TAKE_PROFIT = 0.08  # midpoint of the noted 5–10% band
MAX_HOLD_521 = 20  # trading days, entry day counts as day 1
SAR_PERIOD = 4
SAR_STEP = 0.02
SAR_INCREMENT = 0.02
SAR_MAX = 0.06
SAR_MIN_TREND = 5
PRICE_CAP = 6.0  # nominal (unadjusted) yuan
# |MA20 change over 5 sessions| inside this band is "flat".
SLOPE_FLAT = 0.005
LIQUIDITY_WINDOW = 20
TRADING_DAYS = 252

INDEX_CSI300 = "sh000300"
INDEX_ALL = "sh000985"
