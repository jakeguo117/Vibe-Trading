#!/usr/bin/env python3
"""QM-EOD-v2 runner: combined BO+EP multi-position portfolio + diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qm_eod.kit_core import load_kit  # noqa: E402
from qm_eod import engine, metrics, signals as sigmod  # noqa: E402

RUN_ID = os.environ.get("QM_EOD_RUN_ID") or (ROOT / "results" / ".current_run_id").read_text().strip()
RUN = ROOT / "results" / RUN_ID
DATA = RUN / "data"
OUT = RUN  # deliverables at run root per spec §7
SEGS = metrics.year_segments()


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
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def annotate(df, kit):
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    if "j" in out.columns:
        jj = out["j"].to_numpy(int)
        out["sym"] = np.where(jj >= 0, kit.syms[np.clip(jj, 0, len(kit.syms) - 1)], "")
    for col, src in [("entry_date", "entry_day"), ("exit_date", "exit_day"), ("date", "d"),
                     ("signal_date", "signal_d"), ("entry_date", "entry_d")]:
        if src in out.columns and col not in out.columns:
            vals = out[src].to_numpy(int)
            ok = (vals >= 0) & (vals < len(kit.dates))
            dates = np.full(len(vals), "", dtype=object)
            dates[ok] = kit.dates[vals[ok]].astype(str)
            out[col] = dates
    return out


def slice_sigs(sigs, d0, d1):
    if len(sigs) == 0:
        return sigs
    return sigs[(sigs.d >= d0 - 1) & (sigs.d <= d1)].copy()  # include d0-1 for next_open


def run_one(kit, sigs, cfg, d0, d1, e0, qqq_filter, label):
    t0 = time.time()
    res = engine.run(kit, sigs, cfg, d0, d1, e0=e0, qqq_filter=qqq_filter)
    wall = time.time() - t0
    m = metrics.perf(res["equity"], kit.dates, d0, d1)
    tr = res["trades"]
    m.update(metrics.trade_stats(tr if len(tr) else None))
    n_sig = int(len(sigs[(sigs.d >= d0) & (sigs.d <= d1)])) if len(sigs) else 0
    # fills = completed trades (approx); also count from audit
    n_fills = int(len(tr)) if len(tr) else 0
    if len(res["audit"]):
        n_fills_audit = int((res["audit"]["outcome"] == "fill").sum())
    else:
        n_fills_audit = n_fills
    m.update(metrics.signal_layer_stats(tr if len(tr) else None, n_sig, n_fills_audit))
    m["label"] = label
    m["e0"] = e0
    m["slip_bps"] = cfg.slip_bps
    m["wall_s"] = round(wall, 2)
    m["max_positions"] = cfg.max_positions
    return res, m


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = DATA / "panel.npz"
    edgar = DATA / "edgar_8k202.csv"
    assert panel.exists(), panel
    data_hash = sha256_file(panel)
    edgar_hash = sha256_file(edgar) if edgar.exists() else None
    ch = code_hash()

    print(f"[load] kit from {panel}", flush=True)
    kit = load_kit(str(panel))
    # clamp FULL end to available data
    last = str(kit.dates[-1])[:10]
    segs = {}
    for name, (a, b) in SEGS.items():
        b2 = min(b, last)
        if kit.didx(a) >= len(kit.dates) - 1:
            continue
        d0, d1 = kit.didx(a), min(kit.didx(b2), len(kit.dates) - 1)
        if d1 <= d0:
            continue
        segs[name] = (a, str(kit.dates[d1])[:10], d0, d1)
    print(f"[segs] { {k: (v[0], v[1]) for k,v in segs.items()} }", flush=True)

    print("[signals] BO...", flush=True)
    t0 = time.time()
    bo = sigmod.bo_signals_v2(kit)
    print(f"[signals] BO n={len(bo)} in {time.time()-t0:.1f}s", flush=True)
    print("[signals] EP...", flush=True)
    t0 = time.time()
    ep = sigmod.ep_signals_v2(kit, edgar_csv=str(edgar) if edgar.exists() else None)
    print(f"[signals] EP n={len(ep)} tagged={int(ep.earnings_tagged.sum()) if len(ep) else 0} in {time.time()-t0:.1f}s", flush=True)
    both = sigmod.combine_signals(bo, ep)
    bo.to_csv(OUT / "signals_bo.csv", index=False)
    ep.to_csv(OUT / "signals_ep.csv", index=False)

    qqq_ok = sigmod.qqq_rising_filter(kit)
    print(f"[filter] QQQ rising days={int(qqq_ok.sum())}/{len(qqq_ok)}", flush=True)

    # configs to run
    # Primary: COMBINED BO+EP, filter ON, $100k, 10bps
    # Also: filter OFF, 25/50 bps, $1k, BO-only, EP-only
    jobs = []
    for slip in (10, 25, 50):
        jobs.append(("COMBINED", "filter_on", both, 100_000.0, slip, False, True))
        jobs.append(("COMBINED", "filter_off", both, 100_000.0, slip, False, False))
    jobs.append(("COMBINED", "filter_on_1k", both, 1_000.0, 10, True, True))
    jobs.append(("BO_ONLY", "filter_on", bo, 100_000.0, 10, False, True))
    jobs.append(("EP_ONLY", "filter_on", ep, 100_000.0, 10, False, True))

    # Primary segment for full equity curve: FULL (or data end)
    primary_seg = "FULL" if "FULL" in segs else "CONT"
    all_metrics = []
    ck = {"done": {}, "code_hash": ch[:16], "data_hash": data_hash[:16]}

    # QQQ benchmark for primary
    d0p, d1p = segs[primary_seg][2], segs[primary_seg][3]
    qqq_bh = metrics.benchmark(kit, "QQQ", 1.0, d0p, d1p, e0=100_000.0, cfg=engine.Cfg(slip_bps=0))
    qqq_bh["date"] = kit.dates[qqq_bh["d"].to_numpy()].astype(str)
    qqq_m = metrics.perf(qqq_bh, kit.dates, d0p, d1p)
    qqq_m.update({"label": "QQQ-BH", "segment": primary_seg})
    all_metrics.append(qqq_m)

    primary_res = None
    primary_label = None

    for strat, tag, sigs_all, e0, slip, frac, use_filter in jobs:
        cfg = engine.Cfg(
            risk=0.005, maxw=0.25, max_positions=5, slip_bps=float(slip),
            allow_frac=frac, allow_reentry_same_day=True,
            label=f"{strat}|{tag}|slip{slip}|e{int(e0)}",
        )
        for seg, (a, b, d0, d1) in segs.items():
            # Skip non-primary segs for non-main jobs to save time — still run all segs for main configs
            is_main = (strat == "COMBINED" and tag in ("filter_on", "filter_off") and slip == 10 and e0 == 100_000)
            is_cost = (strat == "COMBINED" and tag == "filter_on" and e0 == 100_000)
            is_1k = (tag == "filter_on_1k")
            is_diag = strat in ("BO_ONLY", "EP_ONLY")
            # Always run FULL/CONT/year segs for main + cost + 1k + diag on FULL+CONT+years
            if not (is_main or is_cost or is_1k or is_diag):
                if seg not in (primary_seg, "CONT"):
                    continue
            if is_diag and seg not in (primary_seg, "CONT", "Y2015_19", "Y2020_21", "Y2022", "Y2023_25"):
                continue

            label = f"{strat}|{tag}|slip{slip}|e{int(e0)}|{seg}"
            key = label
            if key in ck["done"] and not os.environ.get("QM_FORCE"):
                all_metrics.append(ck["done"][key])
                continue

            sigs = slice_sigs(sigs_all, d0, d1)
            qf = qqq_ok if use_filter else None
            # For filter: apply on entry day. BO enters d+1 so filter checked at d+1 in engine.
            res, m = run_one(kit, sigs, cfg, d0, d1, e0, qf, label)
            m["segment"] = seg
            m["seg_start"] = a
            m["seg_end"] = b
            m["strategy"] = strat
            m["filter"] = "on" if use_filter else "off"
            m["n_signals_bo"] = int(((sigs["style"] == "bo") & (sigs.d >= d0) & (sigs.d <= d1)).sum()) if len(sigs) else 0
            m["n_signals_ep"] = int(((sigs["style"] == "ep") & (sigs.d >= d0) & (sigs.d <= d1)).sum()) if len(sigs) else 0
            all_metrics.append(m)
            ck["done"][key] = {k: v for k, v in m.items() if not isinstance(v, (pd.DataFrame,))}

            # Persist primary & important equity/trades
            save = is_main or (is_cost and seg == primary_seg) or (is_1k and seg == primary_seg) or (is_diag and seg == primary_seg)
            if save or seg == primary_seg:
                sub = OUT / "runs" / label.replace("|", "__")
                sub.mkdir(parents=True, exist_ok=True)
                eq = res["equity"].copy()
                eq["date"] = kit.dates[eq["d"].to_numpy()].astype(str)
                eq.to_csv(sub / "equity.csv", index=False)
                annotate(res["trades"], kit).to_csv(sub / "trades.csv", index=False)
                annotate(res["rejects"], kit).to_csv(sub / "rejects.csv", index=False)
                annotate(res["audit"], kit).to_csv(sub / "audit.csv", index=False)
                json.dump(m, open(sub / "metrics.json", "w"), indent=1, default=float)

            if label == f"COMBINED|filter_on|slip10|e100000|{primary_seg}":
                primary_res = res
                primary_label = label
                # also write top-level deliverable names
                eq = res["equity"].copy()
                eq["date"] = kit.dates[eq["d"].to_numpy()].astype(str)
                eq.to_csv(OUT / "daily_equity.csv", index=False)
                annotate(res["trades"], kit).to_csv(OUT / "trade_ledger.csv", index=False)
                annotate(res["audit"], kit).to_csv(OUT / "signal_audit.csv", index=False)
                annotate(res["rejects"], kit).to_csv(OUT / "rejects_raw.csv", index=False)

            print(f"[done] {label} cagr={m.get('cagr', float('nan')):.4f} "
                  f"n={m.get('n_trades', 0)} exec={m.get('execution_rate', float('nan')):.2f} "
                  f"{m['wall_s']}s", flush=True)
            json.dump(ck, open(OUT / "checkpoint.json", "w"), indent=1, default=float)

    # rejects summary
    if primary_res is not None and len(primary_res["rejects"]):
        rej = primary_res["rejects"]
        rs = rej.groupby("reason").size().reset_index(name="count").sort_values("count", ascending=False)
        rs.to_csv(OUT / "rejects_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["reason", "count"]).to_csv(OUT / "rejects_summary.csv", index=False)

    # QQQ equity for chart
    qqq_bh.to_csv(OUT / "qqq_equity.csv", index=False)

    # filter-off primary equity for chart overlay
    off_label = f"COMBINED|filter_off|slip10|e100000|{primary_seg}"
    off_dir = OUT / "runs" / off_label.replace("|", "__")
    if (off_dir / "equity.csv").exists():
        pass  # already saved

    pd.DataFrame(all_metrics).to_csv(OUT / "metrics.csv", index=False)

    frozen = {
        "run_id": RUN_ID,
        "spec": "QM_EOD_v2_Spec_20261022.md",
        "code_hash": ch,
        "data_hash_panel": data_hash,
        "data_hash_edgar": edgar_hash,
        "account": {
            "initial_equity_main": 100000,
            "initial_equity_stress": 1000,
            "risk_per_trade": 0.005,
            "max_notional_weight": 0.25,
            "max_positions": 5,
            "commission_per_share": 0.005,
            "commission_min": 1.0,
            "slippage_bps": [10, 25, 50],
            "allow_reentry_same_day": True,
            "partial_frac": "1/3",
            "sma_exit": 10,
        },
        "ep_vol_mult": 3.0,
        "bo_adr_min": 4.0,
        "qqq_filter": "10SMA>20SMA and 20SMA rising (5d)",
        "segments": {k: [v[0], v[1]] for k, v in segs.items()},
        "data_cutoff": last,
        "universe_flag": "UNIVERSE_SURVIVORSHIP_BIAS",
        "nature": "EOD-DIAG",
        "no_short": True,
        "ep_c2": "NOT_RUN_DATA_GAP",
    }
    json.dump(frozen, open(OUT / "frozen_config.json", "w"), indent=2)
    manifest = {
        **frozen,
        "primary_label": primary_label,
        "wall_total_s": round(time.time() - t_start, 1),
        "n_bo_signals_full": int(len(bo)),
        "n_ep_signals_full": int(len(ep)),
        "deviations": [
            "BO consolidation scanned discrete lengths {10,15,20,25,30,35,40} rather than continuous 10-40 search per bar (speed).",
            "EP main uses price-action gaps; 8-K tags optional (per spec).",
            "Exit-day re-entry ENABLED (v2 delta vs qm_research_001).",
            "Partial sell uses 1/3 (spec) vs prior 1/2.",
            "Combined BO+EP shared 5-slot book is primary; BO_ONLY/EP_ONLY are diagnostics.",
        ],
    }
    json.dump(manifest, open(OUT / "run_manifest.json", "w"), indent=2)
    print(f"[ALL DONE] {time.time()-t_start:.1f}s run_id={RUN_ID}", flush=True)


if __name__ == "__main__":
    main()
