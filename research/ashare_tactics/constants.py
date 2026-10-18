"""Fixed research parameters. These follow the requested tactic notes."""

from __future__ import annotations

import numpy as np

START = np.datetime64("2016-01-01")
WARMUP = np.datetime64("2015-01-01")
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
TAKE_PROFIT_WIDE = 0.15  # default for limit-up pattern tactics without explicit TP
MAX_HOLD_521 = 20  # trading days, entry day counts as day 1
MAX_HOLD_DEFAULT = 10  # default hold for pattern tactics without explicit exit
SAR_PERIOD = 4
SAR_STEP = 0.02
SAR_INCREMENT = 0.02
SAR_MAX = 0.06
SAR_MIN_TREND = 5
PRICE_CAP = 6.0  # nominal (unadjusted) yuan
PRICE_CAP_17 = 10.0  # tactic 17 launch price gate
# |MA20 change over 5 sessions| inside this band is "flat".
SLOPE_FLAT = 0.005
LIQUIDITY_WINDOW = 20
TRADING_DAYS = 252
CONSOLIDATION_DAYS = 20  # ~1 month of trading days for "横盘一个月"
SMALL_BODY = 0.03  # |close/open-1| below this is a "small" candle
GIANT_VOL = 3.0  # volume > 3× prior = 巨量

INDEX_CSI300 = "sh000300"
INDEX_ALL = "sh000985"

# Skipped tactics (not daily-bar testable) with reasons for the Chinese report.
SKIPPED = (
    (1, "需要集合竞价数据（9:15–9:25 试盘/高开筛选）"),
    (6, "需要逐笔/分笔成交明细（庄家暗语）"),
    (10, "需要集合竞价与盘口数据（量比、内外盘）"),
    (11, "主观形态（龙回头支撑位判断无法客观编码）"),
    (14, "主观换手/情绪接力，日线无法可靠复现"),
    (15, "情绪周期框架，无客观买卖参数"),
    (18, "仅介绍 MACD/KDJ 概念，无具体买卖参数（具体规则见战术25）"),
    (20, "可转债标的，非 A 股日线回测范围"),
)
