"""Daily candidate funnel + reject-reason aggregation for the three diagnostics."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qm.core import load_kit, bo_conditions, ps_conditions  # noqa: E402

RUN = "runs/qm-001-20261022-0041"
RES = f"{RUN}/deliverables"


def main():
    k = load_kit(f"{RUN}/data/panel.npz")
    d0, d1 = k.didx("2016-01-04"), k.didx("2026-10-20")
    idx = pd.DatetimeIndex(k.dates); cols = k.syms
    base = pd.DataFrame(k.base_ok, index=idx, columns=cols)
    cdf = pd.DataFrame(k.c, index=idx, columns=cols)
    hdf = pd.DataFrame(k.h, index=idx, columns=cols)
    ldf = pd.DataFrame(k.l, index=idx, columns=cols)
    s20 = pd.DataFrame(k.sma20, index=idx, columns=cols)
    s50 = pd.DataFrame(k.sma50, index=idx, columns=cols)

    leader = (pd.DataFrame(k.p21, index=idx, columns=cols) >= 98) | (pd.DataFrame(k.p63, index=idx, columns=cols) >= 98) | (pd.DataFrame(k.p126, index=idx, columns=cols) >= 98)
    prior = (cdf.shift(15) / cdf.shift(63) - 1) >= 0.30
    maxh = hdf.rolling(15, min_periods=15).max(); minl = ldf.rolling(15, min_periods=15).min()
    tight = (maxh / minl - 1) <= 0.18
    r5 = hdf.rolling(5, min_periods=5).max() - ldf.rolling(5, min_periods=5).min()
    r10p = hdf.shift(5).rolling(10, min_periods=10).max() - ldf.shift(5).rolling(10, min_periods=10).min()
    ratio = (r5 / r10p) <= 0.60
    hl = ldf.rolling(5, min_periods=5).min() > ldf.shift(5).rolling(10, min_periods=10).min()
    trend = (cdf >= s20) & (s20 >= s50) & (s20 > s20.shift(5))

    bo_steps = [("base_eligible", base), ("leader_pct98", leader & base), ("prior_up30", leader & prior & base),
                ("tight15", leader & prior & tight & base), ("range_ratio", leader & prior & tight & ratio & base),
                ("higher_low", leader & prior & tight & ratio & hl & base),
                ("trend_sma", leader & prior & tight & ratio & hl & trend & base)]
    rows = []
    for nm, m in bo_steps:
        a = m.to_numpy()[d0:d1 + 1]
        yrs = pd.Series(pd.DatetimeIndex(k.dates[d0:d1 + 1]).year)
        rows.append({"strategy": "BO", "step": nm, "candidate_days": int(a.sum()),
                     "avg_names_per_day": round(float(a.sum(axis=1).mean()), 2)})
    # triggers
    trig = bo_conditions(k).shift(1, fill_value=False).to_numpy() & (k.c > maxh.shift(1).to_numpy())
    rows.append({"strategy": "BO", "step": "breakout_close_signal", "candidate_days": int(trig[d0:d1 + 1].sum()),
                 "avg_names_per_day": round(float(trig[d0:d1 + 1].sum(axis=1).mean()), 3)})

    # PS funnel
    ps_m = ps_conditions(k)
    s10 = pd.DataFrame(k.sma10, index=idx, columns=cols)
    r10 = cdf / cdf.shift(10) - 1
    up = (cdf > cdf.shift(1)) & (cdf.shift(1) > cdf.shift(2)) & (cdf.shift(2) > cdf.shift(3))
    ext = (cdf / s10 - 1) >= 0.25
    for nm, m in [("base_eligible", base), ("r10_ge_50", (r10 >= 0.5) & base),
                  ("plus_3up", (r10 >= 0.5) & up & base), ("plus_extension", ps_m)]:
        a = m.to_numpy()[d0:d1 + 1]
        rows.append({"strategy": "PS", "step": nm, "candidate_days": int(a.sum()),
                     "avg_names_per_day": round(float(a.sum(axis=1).mean()), 2)})
    weak = (k.c < k.o) & (k.c < (k.h + k.l + k.c) / 3.0)
    trig_ps = ps_m.to_numpy() & weak
    rows.append({"strategy": "PS", "step": "weak_close_signal", "candidate_days": int(trig_ps[d0:d1 + 1].sum()),
                 "avg_names_per_day": round(float(trig_ps[d0:d1 + 1].sum(axis=1).mean()), 3)})

    # EP funnel from event table
    ev = pd.read_csv(f"{RUN}/data/edgar_8k202.csv")
    ev = ev[ev.acceptance.notna() & (~ev.accn.isin(["FETCH_FAIL", "NO_CIK"]))]
    try:
        from run_all import ep_event_table
        tbl = ep_event_table(k, f"{RUN}/data/edgar_8k202.csv")
        inwin = tbl[(tbl.t >= d0) & (tbl.t <= d1)]
        for nm, m in [("all_8k_202_events", pd.Series(True, index=inwin.index)),
                      ("gap_ge_10", inwin.gap >= 0.10),
                      ("plus_not_extended", (inwin.gap >= 0.10) & (inwin.r63p <= 0.25) & (inwin.drawup <= 0.40)),
                      ("plus_liquidity_base", (inwin.gap >= 0.10) & (inwin.r63p <= 0.25) & (inwin.drawup <= 0.40) & inwin.base),
                      ("plus_volume_and_close", (inwin.gap >= 0.10) & (inwin.r63p <= 0.25) & (inwin.drawup <= 0.40) & inwin.base & inwin.up & (inwin.volr >= 1.0))]:
            rows.append({"strategy": "EP", "step": nm, "candidate_days": int(m.sum()),
                         "avg_names_per_day": float("nan")})
    except Exception as e:  # noqa: BLE001
        rows.append({"strategy": "EP", "step": f"ERROR {e}", "candidate_days": 0, "avg_names_per_day": np.nan})

    pd.DataFrame(rows).to_csv(f"{RES}/funnel.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # reject reasons from the main config, all segments
    rj = []
    for s, cfg in [("BO", "BO-MAIN"), ("EP", "EP-MAIN"), ("PS", "PS-MAIN")]:
        d = f"{RES}/{s}/{cfg}"
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith("_rejects.csv"):
                t = pd.read_csv(os.path.join(d, f))
                if len(t):
                    t["segment"] = f.split("_")[0]
                    rj.append(t)
    if rj:
        allr = pd.concat(rj)
        summ = allr.groupby("reason").size().reset_index(name="n")
        allr.to_csv(f"{RES}/rejects_all.csv", index=False)
        summ.to_csv(f"{RES}/rejects_summary.csv", index=False)
        print(summ.sort_values("n", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
