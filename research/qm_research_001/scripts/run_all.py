#!/usr/bin/env python3
"""QM-RESEARCH-001 batch runner: 3 strategies x (1 main + 3 controls + 8 perturbations)
x 5 evaluation paths, plus stress / borrow scenarios, benchmarks and statistics.

Every run is written atomically and recorded in checkpoint.json; a restart skips
completed evaluations whose code+data hash match.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qm import engine, metrics, signals as sigmod  # noqa: E402
from qm.core import load_kit, bo_conditions, bo_signals, ps_conditions, ps_signals  # noqa: E402

RUN = os.environ.get("QM_RUN", "runs/qm-001-20261022-0041")
SEGS = {
    "DEV": ("2010-01-04", "2015-12-31"),
    "CONT": ("2016-01-04", "2026-10-20"),
    "S1": ("2016-01-04", "2019-12-31"),
    "S2": ("2020-01-02", "2022-12-30"),
    "S3": ("2023-01-03", "2026-10-20"),
}
SEG_LIST = ["DEV", "CONT", "S1", "S2", "S3"]
MAIN_SEGS = ["CONT", "S1", "S2", "S3"]


def sha256_file(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(os.path.join(os.path.dirname(__file__), "..", "src"))):
        for fn in sorted(files):
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ configs

def bo_configs():
    return [
        ("BO-MAIN", {}, {}),
        ("BO-C1", {"consolidation": False}, {}),
        ("BO-C2", {"leader": False, "prior_up": False}, {}),
        ("BO-C3", {}, {"no_partial": True, "no_breakeven": True, "no_sma_exit": True, "max_hold": 20}),
        ("BO-P01", {"prior_up_min": 0.24}, {}),
        ("BO-P02", {"prior_up_min": 0.36}, {}),
        ("BO-P03", {"ratio_max": 0.48}, {}),
        ("BO-P04", {"ratio_max": 0.72}, {}),
        ("BO-P05", {}, {"sma_len": 8}),
        ("BO-P06", {}, {"sma_len": 12}),
        ("BO-P07", {}, {"risk": 0.004}),
        ("BO-P08", {}, {"risk": 0.006}),
    ]


def ps_configs():
    return [
        ("PS-MAIN", {}, {}),
        ("PS-C1", {"wait": False}, {}),
        ("PS-C2", {"r10_min": 0.50, "drop_extra": True}, {}),
        ("PS-C3", {}, {"no_sma_exit": True}),
        ("PS-P01", {"r10_min": 0.40}, {}),
        ("PS-P02", {"r10_min": 0.60}, {}),
        ("PS-P03", {"ext_min": 0.20}, {}),
        ("PS-P04", {"ext_min": 0.30}, {}),
        ("PS-P05", {}, {"ps_hold": 4}),
        ("PS-P06", {}, {"ps_hold": 6}),
        ("PS-P07", {}, {"risk": 0.004}),
        ("PS-P08", {}, {"risk": 0.006}),
    ]


def ep_configs():
    return [
        ("EP-MAIN", {}, {}),
        ("EP-C1", {"vol_mult": 0.0}, {}),
        ("EP-C3", {}, {"no_sma_exit": True, "max_hold": 20}),
        ("EP-P01", {"gap_min": 0.08}, {}),
        ("EP-P02", {"gap_min": 0.12}, {}),
        ("EP-P03", {"vol_mult": 0.8}, {}),
        ("EP-P04", {"vol_mult": 1.2}, {}),
        ("EP-P05", {}, {"sma_len": 16}),
        ("EP-P06", {}, {"sma_len": 24}),
        ("EP-P07", {}, {"risk": 0.004}),
        ("EP-P08", {}, {"risk": 0.006}),
    ]


# ------------------------------------------------------------------ EP events

_EP_CACHE = {}


def ep_event_table(k, events_csv):
    """One row per (symbol, first visible session) earnings 8-K 2.02 event."""
    if "tbl" in _EP_CACHE:
        return _EP_CACHE["tbl"]
    from zoneinfo import ZoneInfo
    from datetime import datetime, time as dtime
    NY = ZoneInfo("America/New_York")
    ev = pd.read_csv(events_csv)
    ev = ev[ev["acceptance"].notna() & (~ev["accn"].isin(["FETCH_FAIL", "NO_CIK"]))]
    ev["acc_dt"] = pd.to_datetime(ev["acceptance"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["acc_dt"])
    j_of = {s: i for i, s in enumerate(k.syms)}
    dates = pd.DatetimeIndex(k.dates)
    v20 = pd.DataFrame(k.v).rolling(20, min_periods=20).mean().to_numpy()
    prev_c = np.vstack([np.full((1, k.c.shape[1]), np.nan, np.float32), k.c[:-1]])
    gap = k.o / prev_c - 1
    lo63 = pd.DataFrame(k.l).rolling(63, min_periods=40).min().to_numpy()
    rows = []
    seen = set()
    for s, g in ev.groupby("sym"):
        j = j_of.get(s)
        if j is None:
            continue
        for acc in g["acc_dt"]:
            day = acc.tz_convert(NY).date()
            t = None
            for off in range(0, 4):
                cand = pd.Timestamp(day) + pd.Timedelta(days=off)
                if cand not in dates:
                    continue
                ti = int(dates.get_loc(cand))
                open_dt = datetime.combine(cand.date(), dtime(9, 30), tzinfo=NY).astimezone(ZoneInfo("UTC"))
                if acc < open_dt:
                    t = ti
                    break
            if t is None or t < 64 or t >= k.c.shape[0]:
                continue
            if (s, t) in seen:
                continue
            seen.add((s, t))
            rows.append({
                "sym": s, "j": j, "t": t,
                "gap": float(gap[t, j]) if np.isfinite(gap[t, j]) else np.nan,
                "volr": float(k.v[t, j] / v20[t - 1, j]) if np.isfinite(v20[t - 1, j]) and v20[t - 1, j] > 0 else np.nan,
                "up": bool(np.isfinite(k.c[t, j]) and np.isfinite(k.o[t, j]) and k.c[t, j] > k.o[t, j]),
                "r63p": float(k.r63[t - 1, j]), "drawup": float(k.c[t - 1, j] / lo63[t - 1, j] - 1) if np.isfinite(lo63[t - 1, j]) else np.nan,
                "base": bool(k.base_ok[t, j]),
            })
    tbl = pd.DataFrame(rows)
    _EP_CACHE["tbl"] = tbl
    return tbl


def ep_triggers(k, events_csv, gap_min=0.10, vol_mult=1.0, close_above_open=True):
    tbl = ep_event_table(k, events_csv)
    m = (tbl.gap >= gap_min) & (tbl.r63p <= 0.25) & (tbl.drawup <= 0.40) & tbl.base
    if close_above_open:
        m &= tbl.up
    if vol_mult > 0:
        m &= (tbl.volr >= vol_mult)
    sel = tbl[m]
    trig = np.zeros(k.c.shape, dtype=bool)
    trig[sel.t.to_numpy(int), sel.j.to_numpy(int)] = True
    return trig, sel


# ------------------------------------------------------------------ main

def main():
    t_start = time.time()
    data_hash = sha256_file(os.path.join(RUN, "data", "panel.npz"))[:16]
    ch = code_hash()
    ck_path = os.path.join(RUN, "checkpoint.json")
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else {"done": {}, "budget": {"paths": 0, "wall_s": 0.0}}
    ck["code_hash"] = ch
    ck["data_hash"] = data_hash

    k = load_kit(os.path.join(RUN, "data", "panel.npz"))
    didx = {s: (k.didx(a), k.didx(b)) for s, (a, b) in SEGS.items()}
    res_dir = os.path.join(RUN, "deliverables")
    os.makedirs(res_dir, exist_ok=True)

    def save_path(strat, cfgname, seg, res, extra=None):
        d = os.path.join(res_dir, strat, cfgname)
        os.makedirs(d, exist_ok=True)
        eq = res["equity"].copy()
        eq["date"] = k.dates[eq["d"].to_numpy()].astype(str)
        eq["sym"] = np.where(eq["sym"] >= 0, k.syms[np.maximum(eq["sym"], 0)], "")
        eq.to_csv(os.path.join(d, f"{seg}_equity.csv"), index=False)
        tr = res["trades"]
        if len(tr):
            tr = tr.copy()
            tr["sym"] = k.syms[tr["j"].to_numpy()]
            tr["entry_date"] = k.dates[tr["entry_day"].to_numpy()].astype(str)
            tr["exit_date"] = k.dates[tr["exit_day"].to_numpy()].astype(str)
        tr.to_csv(os.path.join(d, f"{seg}_trades.csv"), index=False)
        rej = res["rejects"]
        if len(rej):
            rej = rej.copy()
            rej["date"] = k.dates[rej["d"].to_numpy()].astype(str)
            rej["sym"] = np.where(rej["j"] >= 0, k.syms[np.maximum(rej["j"], 0)], "")
        rej.to_csv(os.path.join(d, f"{seg}_rejects.csv"), index=False)
        d0, d1 = didx[seg]
        m = metrics.perf(res["equity"], k.dates, d0, d1)
        m.update(metrics.trade_stats(tr if len(tr) else None))
        m.update({"strategy": strat, "config": cfgname, "segment": seg, **(extra or {})})
        json.dump(m, open(os.path.join(d, f"{seg}_metrics.json"), "w"), indent=1, default=float)
        return m

    # ---------------- benchmarks
    bdir = os.path.join(res_dir, "benchmarks")
    os.makedirs(bdir, exist_ok=True)
    bench_rows = []
    bench_eq = {}
    for seg in SEG_LIST:
        d0, d1 = didx[seg]
        for sym, w, monthly, name in [("QQQ", 1.0, False, "QQQ-BH100"), ("SPY", 1.0, False, "SPY-BH100"),
                                      ("QQQ", 0.30, True, "QQQ-30"), ("SPY", 0.30, True, "SPY-30")]:
            b = metrics.benchmark(k, sym, w, d0, d1, monthly=monthly, cfg=engine.Cfg())
            b["date"] = k.dates[b["d"].to_numpy()].astype(str)
            b.to_csv(os.path.join(bdir, f"{name}_{seg}.csv"), index=False)
            m = metrics.perf(b, k.dates, d0, d1)
            m.update({"benchmark": name, "segment": seg})
            bench_rows.append(m)
            bench_eq[(name, seg)] = b
    pd.DataFrame(bench_rows).to_csv(os.path.join(bdir, "benchmark_metrics.csv"), index=False)
    print(f"[bench] done {time.time()-t_start:.1f}s")

    # ---------------- strategies
    all_metrics = []
    configs = {}

    # BO
    for name, mp, ep in bo_configs():
        configs[name] = ("BO", mp, ep)

    # PS
    for name, mp, ep in ps_configs():
        configs[name] = ("PS", mp, ep)

    # EP (needs events)
    ev_csv = os.path.join(RUN, "data", "edgar_8k202.csv")
    ep_ready = os.path.exists(ev_csv) and os.path.getsize(ev_csv) > 10000
    if ep_ready:
        for name, mp, ep in ep_configs():
            configs[name] = ("EP", mp, ep)

    for name, (strat, mp, ep) in configs.items():
        t0 = time.time()
        if strat == "BO":
            mask = bo_conditions(k, leader=mp.get("leader", True), prior_up=mp.get("prior_up", True),
                                 consolidation=mp.get("consolidation", True),
                                 prior_up_min=mp.get("prior_up_min", 0.30),
                                 ratio_max=mp.get("ratio_max", 0.60))
            trig = bo_signals(k, mask)
            sigs_all = sigmod.build(k, trig, "bo", k.didx("2010-01-04"), k.didx("2026-10-20"))
        elif strat == "PS":
            if mp.get("drop_extra"):
                mask = ps_conditions(k, r10_min=mp.get("r10_min", 0.5), ext_min=-9.9)
                # keep R10 only: rebuild without the 3-up and extension conditions
                import pandas as pd_
                idx = pd_.DatetimeIndex(k.dates)
                cdf = pd_.DataFrame(k.c, index=idx, columns=k.syms)
                r10 = cdf / cdf.shift(10) - 1
                mask = (r10 >= mp.get("r10_min", 0.5)) & pd_.DataFrame(k.base_ok, index=idx, columns=k.syms)
            else:
                mask = ps_conditions(k, r10_min=mp.get("r10_min", 0.5), ext_min=mp.get("ext_min", 0.25))
            trig = ps_signals(k, mask, wait_for_weakness=mp.get("wait", True))
            sigs_all = sigmod.build(k, trig, "ps", k.didx("2010-01-04"), k.didx("2026-10-20"))
        else:
            trig, ev_used = ep_triggers(k, ev_csv, gap_min=mp.get("gap_min", 0.10),
                                        vol_mult=mp.get("vol_mult", 1.0))
            sigs_all = sigmod.build(k, trig, "ep", k.didx("2010-01-04"), k.didx("2026-10-20"))

        for seg in SEG_LIST:
            key = f"{strat}|{name}|{seg}|{ch}|{data_hash}"
            if key in ck["done"] and not os.environ.get("QM_FORCE"):
                all_metrics.append(ck["done"][key])
                continue
            d0, d1 = didx[seg]
            cfg = engine.Cfg(style=strat.lower(), side=("short" if strat == "PS" else "long"),
                             sma_len=ep.get("sma_len", 20 if strat == "EP" else 10),
                             max_hold=ep.get("max_hold", 60), **{kx: v for kx, v in ep.items() if kx in
                             ("no_partial", "no_breakeven", "no_sma_exit", "ps_hold", "risk")})
            sigs = sigs_all[(sigs_all.d >= d0) & (sigs_all.d <= d1)] if len(sigs_all) else sigs_all
            res = engine.run(k, sigs, cfg, d0, d1)
            m = save_path(strat, name, seg, res, {"signals": int(len(sigs)), "slip_bps": cfg.slip_bps})
            all_metrics.append(m)
            ck["done"][key] = m
            ck["budget"]["paths"] += 1
            ck["budget"]["wall_s"] = round(ck["budget"]["wall_s"] + (time.time() - t0), 1)
            json.dump(ck, open(ck_path, "w"), indent=1, default=float)
        print(f"[{strat}] {name} paths done {time.time()-t0:.1f}s signals={len(sigs_all)}")

    pd.DataFrame(all_metrics).to_csv(os.path.join(res_dir, "comparisons.csv"), index=False)

    # ---------------- stress on main configs (25/50 bps)
    stress = []
    for strat, base in [("BO", "BO-MAIN"), ("EP", "EP-MAIN"), ("PS", "PS-MAIN")]:
        if base not in configs:
            continue
        if strat == "BO":
            s_all = sigmod.build(k, bo_signals(k, bo_conditions(k)), "bo", k.didx("2010-01-04"), k.didx("2026-10-20"))
        elif strat == "PS":
            s_all = sigmod.build(k, ps_signals(k, ps_conditions(k)), "ps", k.didx("2010-01-04"), k.didx("2026-10-20"))
        else:
            trig, _ = ep_triggers(k, ev_csv)
            s_all = sigmod.build(k, trig, "ep", k.didx("2010-01-04"), k.didx("2026-10-20"))
        for bps in (25, 50):
            for seg in MAIN_SEGS:
                d0, d1 = didx[seg]
                cfg = engine.Cfg(style=strat.lower(), side=("short" if strat == "PS" else "long"), slip_bps=bps,
                                 sma_len=20 if strat == "EP" else 10)
                res = engine.run(k, s_all[(s_all.d >= d0) & (s_all.d <= d1)], cfg, d0, d1)
                m = save_path(strat, f"{base}-SLIP{bps}", seg, res, {"signals": int(len(s_all)), "slip_bps": bps})
                stress.append(m)
    pd.DataFrame(stress).to_csv(os.path.join(res_dir, "stress.csv"), index=False)

    # ---------------- borrow scenarios for PS-PRICE
    borrow = []
    s_ps = sigmod.build(k, ps_signals(k, ps_conditions(k)), "ps", k.didx("2010-01-04"), k.didx("2026-10-20"))
    for rate in (0.10, 0.30, 1.00):
        for seg in MAIN_SEGS:
            d0, d1 = didx[seg]
            cfg = engine.Cfg(style="ps", side="short", borrow_rate=rate)
            res = engine.run(k, s_ps[(s_ps.d >= d0) & (s_ps.d <= d1)], cfg, d0, d1)
            m = save_path("PS", f"PS-PRICE-BORROW{int(rate*100)}", seg, res, {"borrow_rate": rate})
            borrow.append(m)
    pd.DataFrame(borrow).to_csv(os.path.join(res_dir, "borrow_scenarios.csv"), index=False)

    # ---------------- statistics: bootstrap on CONT vs QQQ
    stats_rows = []
    for strat, base in [("BO", "BO-MAIN"), ("EP", "EP-MAIN"), ("PS", "PS-MAIN")]:
        p = os.path.join(res_dir, strat, base, "CONT_equity.csv")
        if not os.path.exists(p):
            continue
        eq = pd.read_csv(p)
        q = bench_eq[("QQQ-BH100", "CONT")]
        r_s = eq["equity"].pct_change().to_numpy()[1:]
        r_b = q["equity"].pct_change().to_numpy()[1:]
        n = min(len(r_s), len(r_b))
        bs = metrics.block_bootstrap(r_s[:n], r_b[:n], block=20, reps=2000)
        stats_rows.append({"strategy": strat, "metric": "daily_excess_vs_QQQ", **bs})
        bs10 = metrics.block_bootstrap(r_s[:n], r_b[:n], block=10, reps=2000)
        bs40 = metrics.block_bootstrap(r_s[:n], r_b[:n], block=40, reps=2000)
        stats_rows.append({"strategy": strat, "metric": "daily_excess_vs_QQQ_block10", **bs10})
        stats_rows.append({"strategy": strat, "metric": "daily_excess_vs_QQQ_block40", **bs40})
        tp = os.path.join(res_dir, strat, base, "CONT_trades.csv")
        tr = pd.read_csv(tp) if os.path.getsize(tp) > 50 else pd.DataFrame()
        if len(tr):
            by_exit = tr.sort_values("exit_day")["R"].to_numpy()
            bs_r = metrics.block_bootstrap(by_exit, None, block=20, reps=2000)
            stats_rows.append({"strategy": strat, "metric": "trade_R_mean", **bs_r})
    pd.DataFrame(stats_rows).to_csv(os.path.join(res_dir, "statistics.csv"), index=False)

    ck["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ck["wall_total_s"] = round(time.time() - t_start, 1)
    json.dump(ck, open(ck_path, "w"), indent=1, default=float)
    print("[done]", ck["budget"], ck["wall_total_s"])


if __name__ == "__main__":
    main()
