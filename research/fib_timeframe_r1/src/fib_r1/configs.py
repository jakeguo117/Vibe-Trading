"""Frozen research configuration: 18 core configs, 15 primary comparisons, windows, gates.

Every value here is copied from the approved plan v0.1 (JAKE-FIB-TIMEFRAME-20261020-v0.1).
Nothing in this file may be edited after the first performance run; see preregistration.json.
"""
from __future__ import annotations

PLAN_SHA256 = "ca25225fed75769c7454b81b752f40c184a9c4a9f614c9233068f7e79365f0a5"
PLAN_FILE = "Jake_Fib_Timeframe_Test_Plan_v0.1_20261020.md"
REF_COMMIT = "2ac825b56c9121f7bb4616c4e5ab18a2619df657"
REF_BLOBS = {
    "backtest/upleg_bt/engine/fib_locked.py": "eb0391c8ebddf06e7b8cc6cdea6ee59789cc942f",
    "backtest/upleg_bt/engine/upleg.py": "f9f052cd43d46adcde6eaaaabe2dcdfd0cf4bddb",
    "backtest/upleg_bt/engine/portfolio.py": "488ba770cf92e6ccd9534721424fc5862c685f5a",
    "backtest/upleg_bt/engine/fm.py": "b6827ea5db3ef708a31e2233a37f905c2f99333c",
}

# §3.2 windows
WINDOWS = {
    "full": ("2019-01-02", "2026-09-30"),
    "design": ("2019-01-02", "2023-12-29"),
    "holdout": ("2024-01-02", "2026-09-30"),
    "slice_2019_2022": ("2019-01-02", "2022-12-30"),
    "slice_2023_end": ("2023-01-03", "2026-09-30"),
    "ytd_2026": ("2026-01-02", "2026-09-30"),
}
HOLDOUT_TAG = "OOS_REUSED"

# §3.1 pools
POOL_A = "D_LOCKED/U_FULL (formal locked snapshot, full US universe)"
POOL_BC = "D_COMMON/U_ETF (SPY QQQ IWM GLD SLV USO TLT EFA EEM)"

# §7.2 B/C cost tiers (per side, fraction of notional)
COST_TIER = {"liquid": {"fee": 1e-4, "slip": 4e-4, "min_dv60": 20e6},
             "mid": {"fee": 1e-4, "slip": 9e-4, "min_dv60": 5e6}}
MIN_DV60 = 5e6
MIN_HISTORY_DAYS = 252

CONFIGS = [
    # id, line, pool, status_hint
    ("A00", "A", POOL_A, "baseline formal v1.1.1, unmodified"),
    ("A10", "A", POOL_A, "g=1 only"),
    ("A01", "A", POOL_A, "DV60 rank only"),
    ("A11", "A", POOL_A, "g=1 + DV60 rank"),
    ("B-W-ON", "B", POOL_BC, "weekly clock, g official"),
    ("B-W-OFF", "B", POOL_BC, "weekly clock, g=1"),
    ("B-D-ON", "B", POOL_BC, "daily clock, g official"),
    ("B-D-OFF", "B", POOL_BC, "daily clock, g=1"),
    ("B-H-ON", "B", POOL_BC, "60m clock, g official"),
    ("B-H-OFF", "B", POOL_BC, "60m clock, g=1"),
    ("C-W-P", "C", POOL_BC, "weekly/weekly plain breakout, 1 position"),
    ("C-W-S", "C", POOL_BC, "weekly/weekly squeeze breakout, 1 position"),
    ("C-D-P", "C", POOL_BC, "daily/daily plain breakout, 1 position"),
    ("C-D-S", "C", POOL_BC, "daily/daily squeeze breakout, 1 position"),
    ("C-H-P", "C", POOL_BC, "60m/60m plain breakout, 1 position"),
    ("C-H-S", "C", POOL_BC, "60m/60m squeeze breakout, 1 position"),
    ("C-DH-P", "C", POOL_BC, "daily structure / 60m entry plain breakout"),
    ("C-DH-S", "C", POOL_BC, "daily structure / 60m entry squeeze breakout"),
]
CONFIG_IDS = [c[0] for c in CONFIGS]

# §10.3 15 primary comparisons (left minus right)
COMPARISONS = [
    ("A00-A10", "A"),
    ("A00-A01", "A"),
    ("A00-A11", "A"),
    ("B-W-ON-B-W-OFF", "B"),
    ("B-D-ON-B-D-OFF", "B"),
    ("B-H-ON-B-H-OFF", "B"),
    ("B-D-OFF-B-W-OFF", "B"),
    ("B-H-OFF-B-W-OFF", "B"),
    ("C-W-S-C-W-P", "C"),
    ("C-D-S-C-D-P", "C"),
    ("C-H-S-C-H-P", "C"),
    ("C-DH-S-C-DH-P", "C"),
    ("C-D-P-C-W-P", "C"),
    ("C-H-P-C-D-P", "C"),
    ("C-DH-P-C-D-P", "C"),
]
COMPARISON_IDS = [c[0] for c in COMPARISONS]

# §10.2 research gate
GATE = {
    "cagr_improvement_pp": 0.02,
    "maxdd_worsen_pp_max": 0.03,
    "maxdd_floor": -0.30,
    "min_holdout_trades": 50,
    "holdout_segments": 3,
    "holdout_segments_min_positive": 2,
    "cost_x2_growth_required": True,
    "trades_per_year_increase_min": 0.25,
    "adverse_rate_drop_pp_min": 0.05,
    "bootstrap_block": 20,
    "bootstrap_n": 10000,
    "bootstrap_seed": 20261020,
    "bootstrap_block_sensitivity": [5, 60],
    "alpha": 0.05,
}
