"""Hourly batch driver: the six configs that were BLOCKED_NO_HOURLY_DATA, plus the shared-window
controls the plan §3.1 requires before any 60-minute result may be compared to a daily one.

Writes into `--out` (the h_line dir):
  hourly_preflight.json   coverage, cross-source reconciliation, frozen common window, grid rule
  equity_daily_<cfg>.csv  daily equity (net), one file per config and stress variant
  orders_trades_<cfg>.csv completed round trips; orders_log_<cfg>.csv every order incl. cancels
  entry_experience.csv    §9.3 per-trade table for the C-line configs
  h_line.json             every metric this driver measured, keyed by config and variant
and into `--control-out` (cw_line dir): the eight W/D configs re-run on the same common window.

Nothing is interpolated: a missing bar is a missing bar, an order without a fillable open is
cancelled, and the two 30-minute grid limitations (no 09:30-10:00 bar, no short bars) are printed
into the preflight instead of being absorbed silently.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import beng, ceng, configs
from . import hourly as H
from .data import load_etf_daily, coverage_report
from .metrics import perf, yearly, trade_stats, mae_mfe_stats

END = configs.WINDOWS["full"][1]
NEW = {"B-H-ON": ("B", "official"), "B-H-OFF": ("B", "ones"),
       "C-H-P": ("C", "P"), "C-H-S": ("C", "S"),
       "C-DH-P": ("C", "DH-P"), "C-DH-S": ("C", "DH-S")}
CONTROLS = {"B-W-ON": ("W", "official"), "B-W-OFF": ("W", "ones"),
            "B-D-ON": ("D", "official"), "B-D-OFF": ("D", "ones"),
            "C-W-P": ("W", "P"), "C-W-S": ("W", "S"),
            "C-D-P": ("D", "P"), "C-D-S": ("D", "S")}


def sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def to_long(daily: dict) -> pd.DataFrame:
    rows = []
    for s, d in daily.items():
        x = d.reset_index()
        x["sym"] = s
        rows.append(x[["date", "sym", "open", "high", "low", "close", "volume", "dv60"]]
                    .rename(columns={"dv60": "dv"}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["close"])


def mask_panel(P: ceng.Panel, j: int) -> ceng.Panel:
    """Drop one symbol from the tradeable pool (plan §10.2 gate 7 stress)."""
    Q = ceng.Panel(**{**P.__dict__})
    Q.O, Q.H, Q.L, Q.C, Q.V = (Q.O.copy(), Q.H.copy(), Q.L.copy(), Q.C.copy(), Q.V.copy())
    for a in (Q.O, Q.H, Q.L, Q.C, Q.V):
        a[:, j] = np.nan
    Q.ELIG = Q.ELIG.copy()
    Q.ELIG[:, j] = False
    return Q


def mask_bpanel(P: dict, j: int) -> dict:
    Q = dict(P)
    for k in ("O", "H", "L", "C"):
        Q[k] = P[k].copy()
        Q[k][:, j] = np.nan
    Q["ELIG"] = P["ELIG"].copy()
    Q["ELIG"][:, j] = False
    Q["code"] = P["code"].copy()
    Q["code"][:, j] = 0
    return Q


def mask_setup(S: dict, j: int) -> dict:
    Q = dict(S)
    for k in ("brk", "atr", "atr_prev", "sq", "sigma_prev", "dv_prev", "elig_prev"):
        Q[k] = S[k].copy()
        if Q[k].ndim == 2:
            if k == "elig_prev":
                Q[k][:, j] = False
            elif k == "sq":
                Q[k][:, j] = False
            else:
                Q[k][:, j] = np.nan
    return Q


def scene_breakout(P, squeeze, kind="H", **kw):
    cost_mult = kw.get("cost_mult", 1.0)
    start = kw.get("start", configs.WINDOWS["full"][0])
    end = kw.get("end", configs.WINDOWS["full"][1])
    if kind == "DH":
        r = H.run_dh(P, kw.pop("setup"), squeeze=squeeze, **kw)
    else:
        r = ceng.run_breakout(P, squeeze=squeeze, **kw)
    if getattr(P, "sess", None) is not None:
        r["eq_intraday_rows"] = len(r["eq"])
        r["eq"] = H.daily_marks(r["eq"], P.dates, P.is_last, start, end)
    eq = r["eq"]["equity"]
    m = perf(eq, r["eq"]["exposure"])
    tr = r["trades"] if len(r["trades"]) else pd.DataFrame()
    m["trades"] = trade_stats(tr) if len(tr) else {"n_trades": 0}
    ex = pd.DataFrame()
    if len(tr):
        ex = (H.entry_experience_bars(P, tr, fee_r=ceng.FEE * cost_mult)
              if getattr(P, "sess", None) is not None
              else ceng.entry_experience(P, tr, fee_r=ceng.FEE * cost_mult))
    m["entry"] = mae_mfe_stats(ex) if len(ex) else {"n": 0}
    if len(ex):
        m["entry"]["failed_breakout_rate"] = float(ex.failed_breakout_5bars.mean())
        m["entry"]["adverse_rate_1d"] = float((ex.mae_1d_atr.dropna() <= -0.5).mean()) \
            if ex.mae_1d_atr.notna().any() else None
        m["entry"]["net_1d_mean"] = float(ex.net_1d.mean())
        m["entry"]["net_5d_mean"] = float(ex.net_5d.mean())
        if "exited_early_1d" in ex:
            m["entry"]["exited_early_1d"] = float(ex.exited_early_1d.mean())
            m["entry"]["censored_1d"] = float(ex.censored_1d.mean())
    m["yearly"] = yearly(eq)
    m["n_rebalances"] = int((r["eq"]["n_hold"].diff().fillna(0) != 0).sum())
    m["stale_bars"] = int(r["eq"]["stale"].sum())
    m["open_position"] = None if r["_pos_open"] is None else {
        k: (str(v) if isinstance(v, (pd.Timestamp, np.datetime64)) else v)
        for k, v in r["_pos_open"].items()}
    return {"res": r, "metrics": m, "trades": tr, "exp": ex}


def scene_b(P, gm, **kw):
    r = beng.run_b(P, g_mode=gm, **kw)
    rt = beng.round_trips(r["orders"], P["syms"])
    if P.get("is_last") is not None:
        r["eq_intraday_rows"] = len(r["eq"])
        r["eq"] = H.daily_marks(r["eq"], P["dates"], P["is_last"],
                                kw.get("start", configs.WINDOWS["full"][0]),
                                kw.get("end", configs.WINDOWS["full"][1]))
    m = perf(r["eq"].equity, r["eq"].exposure)
    m["trades"] = trade_stats(rt)
    m["yearly"] = yearly(r["eq"].equity)
    m["n_rebal_days"] = int(len(r["rebal"]))
    m["rebal_notional"] = float(r["rebal"].notional.sum()) if len(r["rebal"]) else 0.0
    m["n_event_enter"] = int((r["events"].what == "enter").sum()) if len(r["events"]) else 0
    return {"res": r, "metrics": m, "trades": rt}


def b_entry_experience(P, trades, fee_r=ceng.FEE):
    """B-line §9.3 on the hourly grid: price MAE/MFE after the fill, 1d/5d in sessions."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    dates = P["dates"]
    sess = P["sess"]
    rows = []
    for r in trades.itertuples():
        j = P["syms"].index(r.sym)
        e = int(dates.searchsorted(pd.Timestamp(r.entry_date)))
        x = int(dates.searchsorted(pd.Timestamp(r.exit_date)))
        if x <= e:
            continue
        E = float(P["O"][e, j])
        lo, hi, c = P["L"][e:x, j], P["H"][e:x, j], P["C"][e:x, j]
        d0 = int(sess[e])
        out = dict(sym=r.sym, entry_date=r.entry_date, exit_date=r.exit_date,
                   bars_held=int(x - e), days_held=int(sess[x] - d0),
                   mae=float(np.nanmin(lo) / E - 1), mfe=float(np.nanmax(hi) / E - 1),
                   net_ret=r.net_ret)
        for k in (1, 5):
            n = int(((sess[e:x] - d0) < k).sum())
            if n == 0:
                out[f"mae_{k}d"] = np.nan
                out[f"net_{k}d"] = np.nan
            else:
                out[f"mae_{k}d"] = float(np.nanmin(lo[:n]) / E - 1)
                out[f"net_{k}d"] = float(c[n - 1]) / E - 1 - fee_r
        rows.append(out)
    return pd.DataFrame(rows)


def write_eq(r, out: Path, cid: str, suffix=""):
    eq = r["eq"].copy()
    eq.index.name = "date"
    eq.to_csv(out / f"equity_daily_{cid}{suffix}.csv", float_format="%.12g")


def hour_distribution(P: ceng.Panel, trades: pd.DataFrame, out: Path, cid: str):
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    et = pd.to_datetime(trades.sig_date)
    d = et.dt.hour.value_counts().sort_index()
    df = pd.DataFrame({"hour_et": d.index.astype(int), "n": d.values})
    df["config"] = cid
    df["share"] = df.n / df.n.sum()
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--control-out", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--backtest-dir", required=True)
    ap.add_argument("--hourly-csv", default=None)
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--gate7", action="store_true")
    ap.add_argument("--gate7-only", action="store_true",
                    help="reuse h_line.json/cw_line.json already on disk and only redo the "
                         "drop-top-contributor stress")
    ap.add_argument("--configs", default="all", help="comma list of the 6 new config ids, or 'all'")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.backtest_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cw_out = Path(a.control_out); cw_out.mkdir(parents=True, exist_ok=True)

    daily = load_etf_daily(a.raw_dir)
    cov_daily = coverage_report(daily); cov_daily.to_csv(out / "etf_coverage_daily.csv", index=False)

    bundle = H.load_hourly(daily, a.hourly_csv)
    cov = H.coverage(bundle); cov.to_csv(out / "hourly_coverage.csv", index=False)
    rec = H.reconcile(bundle, daily)
    rec.to_csv(out / "cross_source_hourly.csv", index=False)
    bars = bundle["bars"]

    Ld = to_long(daily)
    P_h = H.hourly_panel(bundle, Ld)
    S = H.setup_table(Ld, bundle["sessions"])
    cw = H.common_window(P_h, S)
    start = cw["start"]
    pre_path = out / "hourly_preflight.json"
    if a.gate7_only and pre_path.exists():
        prior = json.loads(pre_path.read_text())
        start = prior["common_window"]["start"]
        cw = prior["common_window"]
    print(f"[preflight] {len(P_h.dates)} bars / {cw['n_sessions']} sessions; common window starts {start}")
    print(cov.to_string(index=False))
    print(rec.to_string(index=False), flush=True)

    # ---- degraded-day symptoms we can actually see in the file (the vendor flag is not in the CSV)
    reg = bars
    vol = reg.groupby(["sym", "sess"]).v.sum().reset_index()
    vol["med"] = vol.groupby("sym").v.transform(
        lambda s: s.rolling(21, center=True, min_periods=11).median())
    vol["ratio"] = vol.v / vol.med
    low = vol[vol.ratio < 0.25].copy()
    low["date"] = [str(bundle["sessions"][int(s)].date()) for s in low.sess]
    per_sess = reg.groupby("sess").size().reindex(range(len(bundle["sessions"]))).fillna(0).astype(int)
    thin = [(str(bundle["sessions"][i].date()), int(n)) for i, n in per_sess.items() if n < 9]

    hourly_pre = {
        "feed": {
            "vendor": "Databento", "schema": "ohlcv-1h",
            "csv": str(a.hourly_csv or (H.HOURLY_ROOT / "csv" / H.CSV_NAME)),
            "csv_sha256": sha(a.hourly_csv or (H.HOURLY_ROOT / "csv" / H.CSV_NAME)),
            "manifest": str(H.HOURLY_ROOT / "meta" / "MANIFEST.json"),
            "manifest_sha256": sha(H.HOURLY_ROOT / "meta" / "MANIFEST.json"),
            "rows_raw": int(len(pd.read_csv(a.hourly_csv or (H.HOURLY_ROOT / "csv" / H.CSV_NAME),
                                            usecols=["symbol"]))),
            "rows_used": int(len(bars)), "rows_dropped_outside_regular_grid": int(len(bundle["dropped"])),
            "venues": H.VENUE,
            "start": "2019-01-02", "end_exclusive": "2026-10-01",
            "quoted_usd": 0.403400212526,
            "purchased_for": "JAKE-FIB-TIMEFRAME hourly unblock; no further purchase made",
        },
        "grid_rule": {
            "bar_label": "start of interval, UTC hour aligned (verified two ways: 15:00 bar close "
                         "within ~1bp of the official daily close; 13:00 bar on 2025-04-09 contains "
                         "the 13:18 ET tariff-pause move)",
            "regular_session_et": "09:30-16:00",
            "kept": "whole vendor hours fully inside the regular session -> 10:00..15:00 ET (6 bars)",
            "early_close": "13:00 ET -> 10:00..12:00 ET (3 bars)",
            "short_bars": 0,
            "limitation_opening_half_hour": "no 09:30-10:00 bar exists on this grid: opening half "
                                            "hour is not tradable and every next-open fill is at "
                                            "10:00 ET at the earliest (30 min lag, all six configs)",
            "limitation_short_bar": "no partial bars, so the §3.3 short_bar diagnostic is degenerate; "
                                    "replaced by a per-hour signal distribution and a post-hoc "
                                    "no-last-bar-entry rerun, both labelled diagnostics",
            "early_close_sessions": int(sum(1 for d in bundle["sessions"] if H.early_close(d))),
        },
        "coverage": cov.to_dict("records"),
        "coverage_min": float(cov.coverage.min()),
        "coverage_threshold_995": bool(cov.coverage.min() >= 0.995),
        "sessions_missing_by_symbol": {
            s: [str(bundle["sessions"][i].date()) for i in
                range(len(bundle["sessions"]))
                if i not in set(bars.loc[bars.sym == s, "sess"])]
            for s in H.SYMS},
        "cross_source": rec.to_dict("records"),
        "cross_source_verdict": ("PASS" if float(rec.med_abs_rel_close.max()) < 5e-4
                                 and float(rec.p99_abs_rel_close.max()) < 5e-3
                                 else "BLOCKED_CROSS_PERIOD"),
        "cross_source_note": "levels are aligned: after mapping the single-venue tape onto the "
                             "snapshot's adjusted scale with the split table, the median close "
                             "difference is a few basis points on every symbol. High/low "
                             "containment is 72-87%: the venue's own extremes are narrower than the "
                             "consolidated tape's, so intraday ranges are slightly understated. "
                             "That is a property of single-venue OHLCV, quoted here rather than "
                             "repaired by borrowing another vendor's highs.",
        "sessions_missing_all_symbols": thin,
        "thin_volume_days": low[["sym", "date", "ratio"]].to_dict("records"),
        "degraded_flag_note": "Databento's per-day degraded flag is not present in the CSV or the "
                              "manifest, so it cannot be read offline. 2020-04-06 was checked "
                              "explicitly: 6/6 regular bars for all nine symbols and volume within "
                              "normal range, so no symptom of it appears in this grid. Days with "
                              "visible symptoms are listed in thin_volume_days / "
                              "sessions_missing_all_symbols instead of being dropped.",
        "common_window": cw,
        "window_used": {"start": start, "end": END, "holdout": list(configs.WINDOWS["holdout"]),
                        "design": [start, configs.WINDOWS["design"][1]],
                        "holdout_tag": configs.HOLDOUT_TAG},
        "eligibility": {"daily_bars_min": ceng.MIN_DAYS, "dv60_min": configs.MIN_DV60,
                        "daily_inputs": "prior completed session only (§3.3)",
                        "bar_index_required": cw["bar_index_required"]},
    }
    (out / "hourly_preflight.json").write_text(json.dumps(hourly_pre, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: hourly_pre[k] for k in ("coverage_min", "coverage_threshold_995",
                                                 "cross_source_verdict")}, indent=1), flush=True)

    # ---- B-line hourly panel (Fib stages on 60m bars)
    Pb = H.build_b_hourly(P_h, stage_cache=out / "b_stage_cache.npz")
    which = [] if a.gate7_only else (list(NEW) if a.configs == "all" else a.configs.split(","))

    summ, eqs, tr_all, exp_all, hrs = {}, [], [], [], []
    csumm = {}
    if a.gate7_only:
        summ = json.loads((out / "h_line.json").read_text())
        csumm = json.loads((cw_out / "cw_line.json").read_text())
    for cid in which:
        line, mode = NEW[cid]
        common = dict(start=start, end=END)
        want_p = mode if mode in ("P", "S") else None
        if line == "B":
            r = scene_b(Pb, mode, **common)
            m = r["metrics"]
        elif mode in ("P", "S"):
            r = scene_breakout(P_h, mode == "S", kind="H", **common)
            m = r["metrics"]
        else:
            r = scene_breakout(P_h, mode.endswith("S"), kind="DH", setup=S, **common)
            m = r["metrics"]
        m.update(config=cid, tf=("H" if line == "B" or mode in ("P", "S") else "DH"))
        summ[cid] = {"base": m}
        write_eq(r["res"], out, cid)
        if len(r["trades"]):
            t = r["trades"].copy(); t.insert(0, "config", cid); tr_all.append(t)
            t.to_csv(out / f"orders_trades_{cid}.csv", index=False)
        if "res" in r and "orders" in r["res"] and len(r["res"]["orders"]):
            o = r["res"]["orders"].copy(); o.insert(0, "config", cid)
            o.to_csv(out / f"orders_log_{cid}.csv", index=False)
        if r.get("exp") is not None and len(r.get("exp", [])):
            e = r["exp"].copy(); e.insert(0, "config", cid); exp_all.append(e)
        if line == "B" and len(r["trades"]):
            bx = b_entry_experience(Pb, r["trades"])
            if len(bx):
                bx.insert(0, "config", cid); bx["line"] = "B"; exp_all.append(bx)
        if line == "B":
            hd = hour_distribution(P_h, r["trades"].rename(columns={"entry_date": "sig_date"})
                                   if len(r["trades"]) else r["trades"], out, cid)
        else:
            hd = hour_distribution(P_h, r["trades"], out, cid)
        if len(hd):
            hrs.append(hd)
        eqs.append(r["res"]["eq"].equity.rename(cid))
        print(f"{cid}: cagr={m['cagr']:.4f} maxdd={m['maxdd']:.4f} sharpe={m['sharpe']:.3f} "
              f"trades={m['trades'].get('n_trades')} expo={m.get('avg_exposure', float('nan')):.3f}",
              flush=True)

        if not a.stress:
            continue
        for cm, tag in ((2.0, "cost_x2"), (3.0, "cost_x3")):
            kw = {**common, "cost_mult": cm}
            rr = scene_b(Pb, mode, **kw) if line == "B" else (
                scene_breakout(P_h, mode == "S", kind="H", **kw) if mode in ("P", "S")
                else scene_breakout(P_h, mode.endswith("S"), kind="DH", setup=S, **kw))
            if cm == 2.0:
                write_eq(rr["res"], out, cid, "_costx2")
            summ[cid][tag] = {**rr["metrics"], "config": cid, "cost_mult": cm}
        for ds, tag in ((0, "delay_1bar_session_calendar"),):
            kw = {**common, "delay_sessions": 1 + ds}
            rr = scene_b(Pb, mode, **kw) if line == "B" else (
                scene_breakout(P_h, mode == "S", kind="H", **kw) if mode in ("P", "S")
                else scene_breakout(P_h, mode.endswith("S"), kind="DH", setup=S, **kw))
            write_eq(rr["res"], out, cid, "_delay1day")
            summ[cid][tag] = {**rr["metrics"], "config": cid, "delay_sessions": 1 + ds}
        kw = {**common, "delay_bars": 1}
        rr = scene_b(Pb, mode, **kw) if line == "B" else (
            scene_breakout(P_h, mode == "S", kind="H", **kw) if mode in ("P", "S")
            else scene_breakout(P_h, mode.endswith("S"), kind="DH", setup=S, **kw))
        write_eq(rr["res"], out, cid, "_delay1")
        summ[cid]["delay_1bar"] = {**rr["metrics"], "config": cid, "delay_bars": 1}
        if line == "C":
            for cap in (5000, 10000, 20000):
                kw = {**common, "integer": True, "capital": cap}
                rr = scene_breakout(P_h, mode == "S", kind="H", **kw) if mode in ("P", "S") else \
                    scene_breakout(P_h, mode.endswith("S"), kind="DH", setup=S, **kw)
                summ[cid][f"int_{cap}"] = {**rr["metrics"], "config": cid, "capital": cap,
                                           "integer": True}
            # post-hoc diagnostic (§3.3 short_bar proxy): no new entries on the session's last bar
            kw = {**common, "ban_last_bar_entry": True}
            rr = scene_breakout(P_h, mode == "S", kind="H", **kw) if mode in ("P", "S") else \
                scene_breakout(P_h, mode.endswith("S"), kind="DH", setup=S, **kw)
            summ[cid]["diag_no_lastbar_entry"] = {**rr["metrics"], "config": cid,
                                                  "diagnostic": "post_hoc_not_preregistered"}

    if eqs:
        pd.concat(eqs, axis=1).to_csv(out / "equity_h_all.csv", float_format="%.12g")
    if tr_all:
        pd.concat(tr_all, ignore_index=True, sort=False).to_csv(out / "orders_trades.csv", index=False)
    if exp_all:
        pd.concat(exp_all, ignore_index=True, sort=False).to_csv(out / "entry_experience.csv", index=False)
    if hrs:
        pd.concat(hrs, ignore_index=True).to_csv(out / "signal_hour_distribution.csv", index=False)

    # ---- controls: the eight W/D configs on the same common window (§3.1)
    panels = {"D": ceng.build_panel(Ld, "D"), "W": ceng.build_panel(Ld, "W")}
    epix = {tf: ceng.signal_episodes(P) for tf, P in panels.items()}
    pb = {"W": beng.build_b(daily, "W"), "D": beng.build_b(daily, "D")}
    csumm = csumm or {}
    ceqs = []
    cexp = []
    for cid, (tf, mode) in ({} if a.gate7_only else CONTROLS).items():
        if cid.startswith("B-"):
            r = scene_b(pb[tf], mode, start=start, end=END)
            m = r["metrics"]
        else:
            r = scene_breakout(panels[tf], mode == "S", kind=tf, start=start, end=END,
                               episodes=epix[tf][0], above=epix[tf][1])
            m = r["metrics"]
        m.update(config=cid, tf=tf, window="common")
        csumm[cid] = {"base": m}
        write_eq(r["res"], cw_out, cid)
        if len(r["trades"]):
            t = r["trades"].copy(); t.insert(0, "config", cid)
            t.to_csv(cw_out / f"orders_trades_{cid}.csv", index=False)
        if r.get("exp") is not None and len(r.get("exp", [])):
            e = r["exp"].copy(); e.insert(0, "config", cid); cexp.append(e)
        ceqs.append(r["res"]["eq"].equity.rename(cid))
        print(f"[control] {cid}: cagr={m['cagr']:.4f} maxdd={m['maxdd']:.4f} "
              f"sharpe={m['sharpe']:.3f} trades={m['trades'].get('n_trades')}", flush=True)
        if a.stress:
            rr = scene_b(pb[tf], mode, start=start, end=END, cost_mult=2.0) if cid.startswith("B-") else \
                scene_breakout(panels[tf], mode == "S", kind=tf, start=start, end=END,
                               cost_mult=2.0, episodes=epix[tf][0], above=epix[tf][1])
            write_eq(rr["res"], cw_out, cid, "_costx2")
            csumm[cid]["cost_x2"] = {**rr["metrics"], "config": cid, "cost_mult": 2.0}
    if ceqs:
        pd.concat(ceqs, axis=1).to_csv(cw_out / "equity_controls_all.csv", float_format="%.12g")
    if cexp:
        pd.concat(cexp, ignore_index=True, sort=False).to_csv(cw_out / "entry_experience.csv", index=False)

    if not a.gate7_only:
        (out / "h_line.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=str))
        (cw_out / "cw_line.json").write_text(json.dumps(csumm, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v["base"].items() if kk in ("cagr", "maxdd", "sharpe")}
                      for k, v in summ.items()}, indent=1, default=str), flush=True)

    # ---- gate 7: drop the top-contributing symbol in the holdout and re-run both sides
    if a.gate7:
        h0, h1 = pd.Timestamp(configs.WINDOWS["holdout"][0]), pd.Timestamp(configs.WINDOWS["holdout"][1])
        pairs = [("B-H-ON", "B-H-OFF"), ("B-H-OFF", "B-W-OFF"), ("C-H-S", "C-H-P"),
                 ("C-DH-S", "C-DH-P"), ("C-H-P", "C-D-P"), ("C-DH-P", "C-D-P")]
        trmap = {}
        for f in list(out.glob("orders_trades_*.csv")) + list(cw_out.glob("orders_trades_*.csv")):
            t = pd.read_csv(f)
            t["entry_date"] = pd.to_datetime(t.entry_date)
            trmap[f.stem.replace("orders_trades_", "")] = t[(t.entry_date >= h0) & (t.entry_date <= h1)]
        rows = []
        for left, right in pairs:
            if left not in summ and left not in csumm:
                continue
            both = pd.concat([trmap.get(left, pd.DataFrame()), trmap.get(right, pd.DataFrame())])
            if len(both) == 0:
                rows.append({"comparison": f"{left}-{right}", "status": "INSUFFICIENT_TRADES"})
                continue
            both = both.copy()
            if "net_dollars" in both and both.net_dollars.notna().any():
                both["contrib"] = both.net_dollars
            elif "notional" in both and "net_ret" in both:
                both["contrib"] = both.net_ret * both.notional       # C-line trades: dollars = ret x notional
            else:
                both["contrib"] = both.net_ret
            top = both.groupby("sym").contrib.sum().sort_values(ascending=False)
            drop = str(top.index[0])
            j = P_h.syms.index(drop)
            eq = {}
            for cid in (left, right):
                if cid in NEW:
                    line, mode = NEW[cid]
                    if line == "B":
                        rr = scene_b(mask_bpanel(Pb, j), mode, start=start, end=END)["res"]
                    elif mode in ("P", "S"):
                        rr = ceng.run_breakout(mask_panel(P_h, j), squeeze=mode == "S",
                                               start=start, end=END)
                    else:
                        rr = H.run_dh(mask_panel(P_h, j), mask_setup(S, j),
                                      squeeze=mode.endswith("S"), start=start, end=END)
                    eq[cid] = H.daily_marks(rr["eq"], P_h.dates, P_h.is_last, start, END)["equity"]
                else:
                    tf, mode = CONTROLS[cid]
                    if cid.startswith("B-"):
                        rr = scene_b(mask_bpanel(pb[tf], j), mode, start=start, end=END)["res"]
                        eq[cid] = rr["eq"]["equity"]
                    else:
                        rr = ceng.run_breakout(mask_panel(panels[tf], j), squeeze=mode == "S",
                                               start=start, end=END)
                        eq[cid] = rr["eq"]["equity"]
            a0 = eq[left].loc[(eq[left].index >= h0) & (eq[left].index <= h1)]
            b0 = eq[right].loc[(eq[right].index >= h0) & (eq[right].index <= h1)]
            from .compare import cagr_window
            rows.append({"comparison": f"{left}-{right}", "dropped_symbol": drop,
                         "cagr_left": cagr_window(eq[left], *configs.WINDOWS["holdout"]),
                         "cagr_right": cagr_window(eq[right], *configs.WINDOWS["holdout"]),
                         "cagr_diff_pp": (cagr_window(eq[left], *configs.WINDOWS["holdout"])
                                          - cagr_window(eq[right], *configs.WINDOWS["holdout"])) * 100,
                         "status": "DONE", "note": "post-hoc stress (plan §10.2 gate 7), not a new validation"})
        pd.DataFrame(rows).to_csv(out / "gate7_drop_top.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
