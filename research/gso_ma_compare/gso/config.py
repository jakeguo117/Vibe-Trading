"""Fixed study settings. The three weekly versions differ only by MA length."""

SYMBOLS = ("GLD", "SLV", "USO")

# Listed inception used to start the long reference book once all three exist.
# Actual first bars come from the Yahoo snapshot; the book starts on the later date.
INCEPTION = {
    "GLD": "2004-11-18",
    "SLV": "2006-04-28",
    "USO": "2006-04-10",
}
LONG_START = "2006-04-28"
MAIN_START = "2019-01-02"
SUB_2022_START = "2022-01-01"
IS_END = "2023-12-31"
OOS_START = "2024-01-01"

MA_WEEKS = (10, 20, 30)
CONFIRM_WEEKS = 2
SLOPE_LAG_WEEKS = 4
DAILY_BARS_PER_WEEK = 5

PIVOT_WEEKS = 3
ATR_WEEKS = 14
ATR_SWING_MULT = 1.5
FIB_BANDS = (0.618, 1.0, 1.618)
FIB_CONFIRM_WEEKS = 2

# 磨底 / 震荡 / 主升 / 鱼尾. Boundary prints go to the higher stage.
STAGE_WEIGHT = {
    "磨底": 0.3,
    "震荡": 0.5,
    "主升": 1.0,
    "鱼尾": 0.5,
}
DEFAULT_STAGE = "震荡"

CASH_FLOOR = 0.30
MAX_INVESTED = 1.0 - CASH_FLOOR
MAX_HOLDINGS = 3
SLOT = MAX_INVESTED / MAX_HOLDINGS

TARGET_VOL = 0.16
EWMA_SPAN = 32
CONST_CORR = 0.35
TRADING_DAYS_PER_YEAR = 252
WEEKS_PER_YEAR = 52

COST_BPS = 5.0
SLIPPAGE_BPS = 5.0
COST_RATE = (COST_BPS + SLIPPAGE_BPS) / 10000.0

RALLY_MIN_GAIN = 0.10
RALLY_MAX_WEEKS = 12
RALLY_START = "2019-01-02"
GOLD_WINDOWS = (("GLD", 2025), ("GLD", 2026))
