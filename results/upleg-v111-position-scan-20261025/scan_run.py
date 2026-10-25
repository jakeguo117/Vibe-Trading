"""Position-sizing scan on the locked v1.1.1 engine. Research only.

Does not edit upleg_bt. Official path is run_portfolio(D). This runner copies that
loop so a leave-up-leg cut other than 50% and an EWMA span other than 32 can be
measured. cut=0.50 and span=32 must match run_portfolio equity.
"""
from __future__ import annotations

import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HANDOFF = Path("/tmp/ca-v111/handoff/repo_handoff_backtest_20261010T185418+0800")
sys.path.insert(0, str(HANDOFF))

from upleg_bt.engine.fib_locked import ewma_var, to_weekly
from upleg_bt.engine.fm import G_CODE, FM
from upleg_bt.engine.portfolio import pvol, run_portfolio, sub_index, weekly_days
from upleg_bt.engine.upleg import ma_leg, name_signals
from upleg_bt.engine.v105 import daily_feats, legs_for
from upleg_bt.metrics import headline, perf

DATA = Path("/tmp/ca-v111/upleg_work/data")
OUT = Path("/tmp/ca-v111/scan_out")
END = "2026-09-30"
START_2019 = "2019-01-02"
SLEEVE = (("GLD", "UGL"), ("SLV", "AGQ"), ("USO", "UCO"))
LEV2_CSV = Path("/tmp/ca-v111/lev2/yahoo_yfinance_daily_UGL-AGQ-UCO_20081125-20261023_lev2-sleeve.csv")


def ewma_sigma_span(C: np.ndarray, span: int) -> np.ndarray:
    """Same recurrence as upleg.ewma_sigma, with alpha = 2/(span+1). Span 32 is alpha 2/33."""
    a = 2.0 / (span + 1.0)
    T, N = C.shape
    Cf = pd.DataFrame(C).ffill().values
    v = np.full(N, np.nan)
    out = np.full((T, N), np.nan, np.float32)
    for t in range(1, T):
        x = Cf[t] / Cf[t - 1] - 1.0
        ok = np.isfinite(x)
        pv = np.isfinite(v)
        v = np.where(ok, np.where(pv, a * x * x + (1.0 - a) * v, x * x), v)
        out[t] = v
    return (np.sqrt(out) * math.sqrt(252)).astype(np.float32)


def ensure_ma(D, n: int, processes: int = 2) -> None:
    if (n, 1) in D.SIG:
        return
    T, N = D.T, D.N
    arr = np.zeros((T, N), np.int8)
    args, js = [], []
    for j in range(N):
        pos = np.flatnonzero(np.isfinite(D.C[:, j]))
        if len(pos) == 0:
            continue
        args.append((pos, D.C[pos, j], D.ST[pos, j], D.week, (n,)))
        js.append(j)
    with get_context("fork").Pool(processes) as pool:
        for j, (pos, out) in zip(js, pool.imap(name_signals, args, chunksize=64)):
            arr[pos, j] = out[(n, 1)]
    legs = legs_for(D.cdaily, n)
    feat = daily_feats(D.cdaily, D.cW, legs, 1)
    for t, j in D.cj.items():
        f1 = feat[t].reindex(D.cal)
        o = f1["open"].to_numpy(dtype=float)
        w = to_weekly(D.cdaily[t])
        on, ma = ma_leg(w["close"], n)
        c2 = (w["close"] < ma) & (ma < ma.shift(4))
        dn = (c2 & c2.shift(1, fill_value=False)).astype(float).shift(1)
        dnd = pd.Series(dn.to_numpy(), index=w.index.to_period("W-FRI")).groupby(level=0).last()
        dd = dnd.reindex(D.cal.to_period("W-FRI")).fillna(0).to_numpy() > 0.5
        lg = f1["LEG"].reindex(D.cal).fillna(False).to_numpy(dtype=bool)
        arr[:, j] = (lg.astype(np.int8) | (dd.astype(np.int8) << 1)) * np.isfinite(o)
    D.SIG[(n, 1)] = arr


def vol_for_span(D, span: int, cache: dict) -> np.ndarray:
    if span in cache:
        return cache[span]
    if span == 32:
        cache[32] = D.VOL
        return D.VOL
    V = ewma_sigma_span(D.C, span)
    for t, j in D.cj.items():
        ret = D.cdaily[t]["close"].pct_change().fillna(0.0)
        sig = pd.Series(np.sqrt(ewma_var(ret, span).to_numpy()) * math.sqrt(252), index=ret.index)
        V[:, j] = sig.reindex(D.cal).to_numpy(dtype=float)
    cache[span] = V
    return V


def run_scan(D, start, end, K=3, reserve=0.30, gross_cap=0.70, vt=0.16, ma=30, delay=1,
             cut=0.50, rank="hist", vol=None):
    """Official loop, plus leave-up-leg keep-fraction (1-cut) and a vol matrix override.

    cut=1 sells the whole name when it leaves the up-leg (engine exit='full').
    cut=0 keeps the full slot until the down-leg exit. 0<cut<1 scales that name
    and hands the freed slot budget to names still in the up-leg, same as halve.
    """
    cal = D.cal
    t0 = int(cal.searchsorted(pd.Timestamp(start)))
    t1 = int(cal.searchsorted(pd.Timestamp(end), side="right")) - 1
    if t0 < 1:
        t0 = 1
    LEGM = D.SIG[(ma, delay)]
    STM = D.ST if delay == 1 else D.ST2
    O, C, SIDE = D.O, D.C, D.SIDE
    V = D.VOL if vol is None else vol
    cset = list(D.cj.values())
    capv = gross_cap
    slot = (1.0 - reserve) / K
    sub = sub_index(D)
    P = lambda j: sub.get(j, j)
    comp = {}
    for a, b in sub.items():
        comp[a] = b
        comp[b] = b
    keep = 1.0 - cut
    full_exit = cut >= 1.0 - 1e-12
    eq = 1.0
    w, held, lastc = {}, {}, {}
    rec = []
    for t in range(t0, t1 + 1):
        S = t - 1
        R = 0.0
        ret_o = {}
        for j, wj in w.items():
            p = O[t, j]
            r = (p / lastc[j] - 1) if np.isfinite(p) else 0.0
            ret_o[j] = r
            R += wj * r
        eq *= 1 + R
        if w:
            w = {j: wj * (1 + ret_o[j]) / (1 + R) for j, wj in w.items()}
        for j in w:
            if np.isfinite(O[t, j]):
                lastc[j] = O[t, j]
        changed = False
        for j in list(held):
            code = int(LEGM[S, j])
            leg = bool(code & 1)
            dn = bool(code & 2)
            pj = P(j)
            if t > D.lastvalid[j] or t > D.lastvalid[pj]:
                del held[j]
                changed = True
                continue
            if not np.isfinite(O[t, j]) or not np.isfinite(O[t, pj]):
                continue
            if full_exit:
                if not leg:
                    del held[j]
                    changed = True
            elif not leg and dn:
                del held[j]
                changed = True
            elif not leg and not held[j] and cut > 0:
                held[j] = True
                changed = True
            elif leg and held[j]:
                held[j] = False
                changed = True
        hcomp = {comp.get(j, j) for j in held}
        free = K - len(held)
        if free > 0:
            for j in cset:
                if free == 0:
                    break
                if j not in held and (int(LEGM[S, j]) & 1) and np.isfinite(O[t, j]):
                    held[j] = False
                    free -= 1
                    changed = True
            if free > 0:
                ok = D.EL[S] & D.POOL & ((LEGM[S] & 1) > 0) & np.isfinite(O[t])
                js = np.flatnonzero(ok)
                if held:
                    js = js[~np.isin(js, np.fromiter(held.keys(), np.int64))]
                if len(js):
                    order = np.lexsort((-D.DV[S, js], -D.score(rank, S, js, 1.0)))
                    for j in js[order]:
                        if free == 0:
                            break
                        j = int(j)
                        cj = comp.get(j, j)
                        if cj in hcomp or not np.isfinite(O[t, P(j)]):
                            continue
                        held[j] = False
                        hcomp.add(cj)
                        free -= 1
                        changed = True
        hj = list(held)
        ate_cash_floor = 1.0 - capv
        ate_cand = ate_stage = ate_exit = ate_vol = ate_cap = scale_up = 0.0
        raw_sum = 0.0
        if hj:
            g = np.array([G_CODE[max(int(STM[S, j]), 0)] for j in hj])
            half = np.array([held[j] for j in hj])
            stage_sum = float((slot * g).sum())
            raw = slot * g * np.where(half, keep, 1.0)
            if (not full_exit) and half.any() and (~half).any() and cut > 0:
                freed = (slot * g * cut)[half].sum()
                raw[~half] += freed * raw[~half] / raw[~half].sum()
            raw_sum = float(raw.sum())
            n_held = len(hj)
            ate_cand = (K - n_held) * slot
            ate_stage = n_held * slot - stage_sum
            ate_exit = stage_sum - raw_sum
            sig = np.array([V[S, P(j)] for j in hj])
            sp = pvol(raw, sig)
            scale = (vt / sp) if sp > 0 else 1.0
            uncapped = raw_sum * scale
            ate_vol = max(0.0, raw_sum - uncapped)
            scale_up = max(0.0, uncapped - raw_sum)
            ate_cap = max(0.0, uncapped - capv)
            tg = raw * scale
            if tg.sum() > capv + 1e-12:
                tg *= capv / tg.sum()
            tgt = {}
            for j, v in zip(hj, tg):
                tgt[P(j)] = tgt.get(P(j), 0.0) + v
        else:
            ate_cand = capv
            tgt = {}
        keys = set(tgt) | set(w)
        to = sum(abs(tgt.get(j, 0.0) - w.get(j, 0.0)) for j in keys)
        fee = 0.0
        if changed or to > 0.02:
            fee = sum(abs(tgt.get(j, 0.0) - w.get(j, 0.0)) * (SIDE[S, j] if np.isfinite(SIDE[S, j]) else 1e-3) for j in keys)
            eq *= 1 - fee
            w = {j: v for j, v in tgt.items() if v > 0}
            for j in w:
                if np.isfinite(O[t, j]):
                    lastc[j] = O[t, j]
        R = 0.0
        ret_c = {}
        for j, wj in w.items():
            c = C[t, j]
            r = (c / lastc[j] - 1) if np.isfinite(c) else 0.0
            ret_c[j] = r
            R += wj * r
        eq *= 1 + R
        if w:
            w = {j: wj * (1 + ret_c[j]) / (1 + R) for j, wj in w.items()}
        for j in w:
            if np.isfinite(C[t, j]):
                lastc[j] = C[t, j]
        gross_tgt = float(sum(tgt.values()))
        rec.append((cal[t], eq, sum(w.values()), gross_tgt, to if (changed or to > 0.02) else 0.0, fee,
                    ate_cash_floor + ate_cap, ate_vol, ate_exit, ate_cand, ate_stage, scale_up))
    df = pd.DataFrame(rec, columns=[
        "date", "equity", "exposure", "target", "turnover", "fee",
        "ate_cash", "ate_vol", "ate_exit", "ate_candidate", "ate_stage", "vol_scale_up",
    ]).set_index("date")
    return df


def slice_metrics(df, a, b):
    sl = df.loc[a:b]
    if len(sl) < 5:
        return None
    m = perf(df, a, b)
    m["trade_days"] = int(sl["fee"].gt(0).sum())
    for col in ("ate_cash", "ate_vol", "ate_exit", "ate_candidate", "ate_stage", "vol_scale_up", "target"):
        m[col] = float(sl[col].mean())
    ident = (sl["ate_cash"] + sl["ate_vol"] + sl["ate_exit"] + sl["ate_candidate"] + sl["ate_stage"] - sl["vol_scale_up"])
    m["ate_identity_maxabs"] = float((ident - (1.0 - sl["target"])).abs().max())
    return m


def pack_row(name, step, params, df2019, dflong, long_a):
    periods = {
        "2019_2022": ("2019-01-02", "2022-12-31"),
        "2023_plus": ("2023-01-01", END),
        "2019_plus": (START_2019, END),
    }
    row = {"name": name, "step": step, **params}
    for label, (a, b) in periods.items():
        m = slice_metrics(df2019, a, b)
        for k, v in m.items():
            row[f"{label}_{k}"] = v
        row[f"{label}_headline"] = json.dumps(headline(m), ensure_ascii=False)
    mlong = slice_metrics(dflong, long_a, END)
    for k, v in mlong.items():
        row[f"long_{k}"] = v
    row["long_start"] = long_a
    row["long_headline"] = json.dumps(headline(mlong), ensure_ascii=False)
    gold = slice_metrics(dflong, "2011-01-01", "2015-12-31")
    if gold:
        for k in ("cagr", "maxdd", "sortino", "sharpe", "avg_exposure", "ann_vol", "n_days"):
            row[f"goldbear_{k}"] = gold[k]
    return row


def apply_lev2(D, csv_path: Path):
    raw = pd.read_csv(csv_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    saved = {}
    for sig, exe in SLEEVE:
        j = D.cj[sig]
        g = raw[raw["Ticker"] == exe].drop_duplicates("Date").set_index("Date").sort_index()
        px = pd.to_numeric(g["Close"], errors="coerce")
        adj = pd.to_numeric(g["Adj Close"], errors="coerce")
        op = pd.to_numeric(g["Open"], errors="coerce")
        factor = (adj / px).where(np.isfinite(adj / px) & (px.to_numpy() > 0))
        o = (op * factor).reindex(D.cal)
        c = (px * factor).reindex(D.cal)
        ret = (px * factor).pct_change().fillna(0.0)
        sig_ann = pd.Series(np.sqrt(ewma_var(ret, 32).to_numpy()) * math.sqrt(252), index=(px * factor).index)
        vol = sig_ann.reindex(D.cal).ffill()
        saved[j] = (D.O[:, j].copy(), D.C[:, j].copy(), D.VOL[:, j].copy(), D.EL[:, j].copy(), int(D.lastvalid[j]))
        D.O[:, j] = o.to_numpy(dtype=float)
        D.C[:, j] = c.to_numpy(dtype=float)
        D.VOL[:, j] = vol.to_numpy(dtype=float)
        D.EL[:, j] = np.isfinite(o.to_numpy(dtype=float))
        fin = np.isfinite(D.C[:, j])
        D.lastvalid[j] = int(D.T - 1 - np.argmax(fin[::-1])) if fin.any() else -1
    return saved


def restore_lev2(D, saved):
    for j, (o, c, v, el, lv) in saved.items():
        D.O[:, j] = o
        D.C[:, j] = c
        D.VOL[:, j] = v
        D.EL[:, j] = el
        D.lastvalid[j] = lv


def one(D, spec, vol_cache, long_a):
    vol = vol_for_span(D, spec["span"], vol_cache)
    kw = dict(K=spec["K"], reserve=spec["reserve"], gross_cap=spec["gross_cap"], vt=spec["vt"],
              ma=spec["ma"], cut=spec["cut"], vol=vol)
    df2019 = run_scan(D, START_2019, END, **kw)
    dflong = run_scan(D, long_a, END, **kw)
    return df2019, dflong


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("LOAD", flush=True)
    D = FM(str(DATA), processes=2)
    long_a = str(pd.Timestamp(D.cal[1]).date())
    print(json.dumps({"cal0": str(D.cal[0].date()), "cal1": long_a, "cal_last": str(D.cal[-1].date()), "T": int(D.T), "N": int(D.N)}), flush=True)
    print("MA40", flush=True)
    ensure_ma(D, 40, processes=2)
    print("VERIFY", flush=True)
    official = run_portfolio(D, record=False)
    mine = run_scan(D, START_2019, END)
    same = len(official["eq"]) == len(mine) and bool((official["eq"].index == mine.index).all())
    maxabs = float(np.max(np.abs(official["eq"]["equity"].to_numpy() - mine["equity"].to_numpy()))) if same else None
    exp_abs = float(np.max(np.abs(official["eq"]["exposure"].to_numpy() - mine["exposure"].to_numpy()))) if same else None
    m = perf(official["eq"])
    trade_days = int(official["eq"]["fee"].gt(0).sum())
    verify = {
        "same_index": same,
        "equity_maxabs": maxabs,
        "exposure_maxabs": exp_abs,
        "official": {**m, "trade_days": trade_days, "headline": headline(m)},
        "PASS": bool(same and maxabs is not None and maxabs < 1e-9 and abs(m["cagr"] - 0.18024379719089412) < 1e-9
                     and abs(m["maxdd"] + 0.11700656773292528) < 1e-9 and abs(m["avg_exposure"] - 0.5914710808789267) < 1e-9
                     and trade_days == 496),
    }
    (OUT / "verify.json").write_text(json.dumps(verify, indent=1, ensure_ascii=False))
    print(json.dumps(verify, ensure_ascii=False), flush=True)
    if not verify["PASS"]:
        raise SystemExit("baseline does not match locked v1.1.1")
    base = {"K": 3, "reserve": 0.30, "gross_cap": 0.70, "vt": 0.16, "ma": 30, "cut": 0.50, "span": 32}
    specs = [("v1.1.1", "baseline", dict(base))]
    for floor, cap in ((0.30, 0.70), (0.20, 0.80), (0.10, 0.90), (0.00, 1.00)):
        if floor == 0.30:
            continue
        specs.append((f"cash_{int(floor*100)}", "cash_floor", {**base, "reserve": floor, "gross_cap": cap}))
    for vt in (0.16, 0.18, 0.20, 0.22):
        if vt == 0.16:
            continue
        specs.append((f"vt_{int(vt*100)}", "vol_target", {**base, "vt": vt}))
    for k in (2, 3, 4):
        if k == 3:
            continue
        specs.append((f"k_{k}", "k", {**base, "K": k}))
    for cut, tag in ((0.0, "0"), (0.25, "25"), (0.50, "50"), (1.0, "100")):
        if cut == 0.50:
            continue
        specs.append((f"cut_{tag}", "exit_cut", {**base, "cut": cut}))
    for ma in (20, 30, 40):
        if ma == 30:
            continue
        specs.append((f"ma_{ma}", "ma", {**base, "ma": ma}))
    for span in (16, 32, 64):
        if span == 32:
            continue
        specs.append((f"span_{span}", "vol_span", {**base, "span": span}))
    for cap in (0.70, 0.80, 0.90, 1.00):
        for vt in (0.16, 0.18, 0.20, 0.22):
            if cap == 0.70 and vt == 0.16:
                continue
            specs.append((f"grid_c{int(cap*100)}_v{int(vt*100)}", "grid", {**base, "reserve": round(1 - cap, 2), "gross_cap": cap, "vt": vt}))
    vol_cache = {}
    rows = []
    curves = {}
    done = {}
    for name, step, spec in specs:
        print("RUN", name, flush=True)
        key = tuple(sorted((k, spec[k]) for k in ("K", "reserve", "gross_cap", "vt", "ma", "cut", "span")))
        if key not in done:
            done[key] = one(D, spec, vol_cache, long_a)
        df2019, dflong = done[key]
        rows.append(pack_row(name, step, spec, df2019, dflong, long_a))
        if name == "v1.1.1":
            curves["v1.1.1"] = df2019
            dflong.to_csv(OUT / "equity_v111_long.csv", float_format="%.10g")
        df2019.to_csv(OUT / f"equity_{name}.csv", float_format="%.10g")
    pd.DataFrame(rows).to_csv(OUT / "sweep.csv", index=False)
    print("SWEEP_CORE", len(rows), flush=True)
    print("LEV2", flush=True)
    saved = apply_lev2(D, LEV2_CSV)
    lev = run_portfolio(D, record=False)
    lev_long = run_portfolio(D, start=long_a, end=END, record=False)
    restore_lev2(D, saved)
    lev_df = lev["eq"].copy()
    for col in ("ate_cash", "ate_vol", "ate_exit", "ate_candidate", "ate_stage", "vol_scale_up", "target"):
        lev_df[col] = np.nan
    lev_df["target"] = np.nan
    # Recompute LEV2 with the scanner at baseline params on the mutated book, so attribution exists.
    saved = apply_lev2(D, LEV2_CSV)
    vol_cache.clear()
    lev_s, lev_sl = one(D, base, vol_cache, long_a)
    restore_lev2(D, saved)
    off = float(np.max(np.abs(lev["eq"]["equity"].to_numpy() - lev_s["equity"].to_numpy())))
    off_long = float(np.max(np.abs(lev_long["eq"]["equity"].to_numpy() - lev_sl["equity"].to_numpy())))
    print(json.dumps({"lev2_vs_run_portfolio_maxabs": off, "lev2_long_maxabs": off_long}), flush=True)
    if off >= 1e-9 or off_long >= 1e-9:
        raise SystemExit("LEV2 scanner copy drifted from run_portfolio")
    rows.append(pack_row("LEV2", "control", {**base, "sleeve": "UGL/AGQ/UCO"}, lev_s, lev_sl, long_a))
    lev_s.to_csv(OUT / "equity_LEV2.csv", float_format="%.10g")
    pd.DataFrame(rows).to_csv(OUT / "sweep.csv", index=False)
    print("SWEEP_DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
