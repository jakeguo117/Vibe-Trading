"""Research-only FREQ and LEV2 runs. Does not modify locked upleg_bt / live trading code.

FREQ (upleg_v111_freq_event_scale), relative to v1.1.1 run_portfolio:
  * official L1 deadband stays > 2% and is not the event threshold;
  * extra trade when a current holding's Fib stage code changes (STM[S] != STM[S-1]);
  * extra trade when an execution name already in the book has |close-to-close return| on day S
    >= ret_thr (0.02 or 0.03). Known at S close, filled at t open (S = t-1);
  * leave-up-leg / halve / exit already sets changed and is not re-implemented;
  * new enters only (not restores) scale to 40% / 70% / 100% of the post-gross-cap target
    over three sessions. Unfilled weight stays cash. Halve / exit / data_end drop the
    scale state so the cut trades in full the same session. Tranche days force a fill
    even inside the deadband.

LEV2 (upleg_v111_lev2_etf): after FM overlay, replace GLD/SLV/USO open, close, σ, eligibility
and lastvalid with adjusted UGL/AGQ/UCO. SIG / ST / ST2 / POOL / stock columns stay.
σ = sqrt(ewma_var(native 2x close-to-close, span 32)) * sqrt(252), ffilled onto the strategy calendar.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HANDOFF = Path("/tmp/ca-v111/handoff/repo_handoff_backtest_20261010T185418+0800")
sys.path.insert(0, str(HANDOFF))

from upleg_bt.engine.fib_locked import ewma_var
from upleg_bt.engine.fm import G_CODE, FM
from upleg_bt.engine.portfolio import pvol, run_portfolio, sub_index, weekly_days
from upleg_bt.metrics import headline, perf, perf_periods, yearly

CSV = Path("/tmp/ca-v111/lev2/yahoo_yfinance_daily_UGL-AGQ-UCO_20081125-20261023_lev2-sleeve.csv")
CSV_SHA = "cc9063d763d1cd92f40cf1f78a4fb5b8cc5ecc517d90863f4424e01d342cd3e8"
GOLD = HANDOFF / "runs/20261025T172821+0800_reproduce_v1.1.1/daily_equity.csv"
OUT = Path("/tmp/ca-v111/research/out")
SLEEVE = (("GLD", "UGL"), ("SLV", "AGQ"), ("USO", "UCO"))


def _fac(day):
    if day is None or day >= 2:
        return 1.0
    if day <= 0:
        return 0.40
    return 0.70


def run_freq(D, start="2019-01-02", end="2026-09-30", ma=30, delay=1, exit="halve", rank="hist", seed=None, cost_mult=1.0,
             K=3, reserve=0.30, gross_cap=None, vt=0.16, sub=None, single_cap=None, cap_mode="weekly", er=None, record=True,
             ret_thr=None, scale_in=False):
    """Copy of run_portfolio. ret_thr=None and scale_in=False must match the official loop."""
    cal = D.cal
    t0 = int(cal.searchsorted(pd.Timestamp(start)))
    t1 = int(cal.searchsorted(pd.Timestamp(end), side="right")) - 1
    rng = np.random.default_rng(seed) if rank == "random" else None
    LEGM = D.SIG[(ma, delay)]
    STM = D.ST if delay == 1 else D.ST2
    O, C, V, SIDE = D.O, D.C, D.VOL, D.SIDE
    cset = list(D.cj.values())
    capv = (1 - reserve) if gross_cap is None else gross_cap
    slot = (1 - reserve) / K
    if sub is None:
        sub = sub_index(D)
    P = lambda j: sub.get(j, j)
    Q = {b: a for a, b in sub.items()}
    comp = {}
    for a, b in sub.items():
        comp[a] = b
        comp[b] = b
    isw = weekly_days(cal)
    erh = {}
    scale_day = {}
    eq = 1.0
    w = {}
    held = {}
    lastc = {}
    rec = []
    events = []
    wlog = []
    gaps = []
    for t in range(t0, t1 + 1):
        S = t - 1
        R = 0.0
        ret_o = {}
        for j, wj in w.items():
            p = O[t, j]
            r = (p / lastc[j] - 1) if np.isfinite(p) else 0.0
            ret_o[j] = r
            R += wj * r
        if record:
            for j, wj in w.items():
                gaps.append((t, j, ret_o[j], wj, wj * ret_o[j]))
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
                erh.pop(j, None)
                scale_day.pop(j, None)
                changed = True
                events.append((t, j, "data_end"))
                continue
            if not np.isfinite(O[t, j]) or not np.isfinite(O[t, pj]):
                continue
            if exit == "full":
                if not leg:
                    del held[j]
                    erh.pop(j, None)
                    scale_day.pop(j, None)
                    changed = True
                    events.append((t, j, "exit"))
            elif not leg and dn:
                del held[j]
                erh.pop(j, None)
                scale_day.pop(j, None)
                changed = True
                events.append((t, j, "exit_dn"))
            elif not leg and not held[j]:
                held[j] = True
                scale_day.pop(j, None)
                changed = True
                events.append((t, j, "halve"))
            elif leg and held[j]:
                held[j] = False
                changed = True
                events.append((t, j, "restore"))
        hcomp = {comp.get(j, j) for j in held}
        free = K - len(held)
        if free > 0:
            for j in cset:
                if free == 0:
                    break
                if j not in held and (int(LEGM[S, j]) & 1) and np.isfinite(O[t, j]):
                    held[j] = False
                    if scale_in:
                        scale_day[j] = 0
                    free -= 1
                    changed = True
                    events.append((t, j, "enter"))
            if free > 0:
                ok = D.EL[S] & D.POOL & ((LEGM[S] & 1) > 0) & np.isfinite(O[t])
                js = np.flatnonzero(ok)
                if held:
                    js = js[~np.isin(js, np.fromiter(held.keys(), np.int64))]
                if len(js):
                    order = rng.permutation(len(js)) if rng is not None else np.lexsort((-D.DV[S, js], -D.score(rank, S, js, cost_mult)))
                    for j in js[order]:
                        if free == 0:
                            break
                        j = int(j)
                        cj = comp.get(j, j)
                        if cj in hcomp or not np.isfinite(O[t, P(j)]):
                            continue
                        held[j] = False
                        if scale_in:
                            scale_day[j] = 0
                        hcomp.add(cj)
                        free -= 1
                        changed = True
                        events.append((t, j, "enter"))
        if er is not None:
            for j in list(erh):
                if j not in held:
                    erh.pop(j)
                    continue
                if t >= erh[j]:
                    erh.pop(j)
                    changed = True
                    events.append((t, j, "er_restore"))
            for j in held:
                if j in erh or j not in er:
                    continue
                for h, r, rest in er[j]:
                    if h <= t <= r - 1:
                        erh[j] = rest
                        changed = True
                        events.append((t, j, "er_halve"))
                        break
        event_force = False
        if ret_thr is not None and S >= 1:
            for j in list(held):
                if int(STM[S, j]) != int(STM[S - 1, j]):
                    event_force = True
                    events.append((t, j, "stage_switch"))
            for j in list(w):
                c1 = C[S, j]
                c0 = C[S - 1, j]
                if np.isfinite(c1) and np.isfinite(c0) and c0 != 0.0 and abs(c1 / c0 - 1.0) >= ret_thr:
                    event_force = True
                    events.append((t, j, "ret_thr"))
        tranche = bool(scale_in and any(scale_day.get(j, 99) <= 2 for j in held))
        hj = list(held)
        if hj:
            g = np.array([G_CODE[max(int(STM[S, j]), 0)] for j in hj])
            half = np.array([held[j] for j in hj])
            raw = slot * g * np.where(half, 0.5, 1.0)
            if half.any() and (~half).any():
                freed = (slot * g * 0.5)[half].sum()
                raw[~half] += freed * raw[~half] / raw[~half].sum()
            sig = np.array([V[S, P(j)] for j in hj])
            sp = pvol(raw, sig)
            tg = raw * (vt / sp if sp > 0 else 1.0)
            if tg.sum() > capv + 1e-12:
                tg *= capv / tg.sum()
            if single_cap is not None:
                lim = np.full(len(hj), single_cap) if (cap_mode == "every" or isw[t]) else np.array([max(single_cap, w.get(P(j), 0.0)) for j in hj])
                tg = np.minimum(tg, lim)
            if er is not None:
                tg = tg * np.array([0.5 if j in erh else 1.0 for j in hj])
            if scale_in:
                tg = tg * np.array([_fac(scale_day.get(j)) for j in hj])
            tgt = {}
            for j, v in zip(hj, tg):
                tgt[P(j)] = tgt.get(P(j), 0.0) + v
        else:
            tgt = {}
        trig_cap = single_cap is not None and cap_mode == "weekly" and isw[t] and any(v > single_cap + 1e-12 for v in w.values())
        keys = set(tgt) | set(w)
        to = sum(abs(tgt.get(j, 0.0) - w.get(j, 0.0)) for j in keys)
        traded = False
        fee = 0.0
        if changed or to > 0.02 or trig_cap or event_force or tranche:
            fee = sum(abs(tgt.get(j, 0.0) - w.get(j, 0.0)) * (SIDE[S, j] if np.isfinite(SIDE[S, j]) else 1e-3) for j in keys) * cost_mult
            eq *= 1 - fee
            w = {j: v for j, v in tgt.items() if v > 0}
            traded = True
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
        if scale_in:
            for j in list(scale_day):
                if j not in held:
                    scale_day.pop(j, None)
                    continue
                scale_day[j] += 1
                if scale_day[j] > 2:
                    scale_day.pop(j, None)
        if record:
            for j in set(w) | set(tgt):
                sj = Q.get(j, j)
                wlog.append((t, j, w.get(j, 0.0), tgt.get(j, 0.0), bool(held.get(sj, False)), int(STM[S, sj]), traded, int(sj)))
        rec.append((cal[t], eq, sum(w.values()), sum(v for j, v in w.items() if j in cset), len(w), sum(1 for j in held if held[j]), to if traded else 0.0, fee))
    df = pd.DataFrame(rec, columns=["date", "equity", "exposure", "comm_exposure", "n_hold", "n_half", "turnover", "fee"]).set_index("date")
    df["stock_exposure"] = df.exposure - df.comm_exposure
    out = {"eq": df}
    nm = lambda a: D.names[np.asarray(a, int)]
    EV = pd.DataFrame(events, columns=["t", "j", "what"])
    EV["date"] = cal[EV.t.values]
    EV["name"] = nm(EV.j)
    EV["exec_name"] = nm([P(j) for j in EV.j.values]) if len(EV) else []
    out["events"] = EV
    if record:
        WL = pd.DataFrame(wlog, columns=["t", "j", "w_close", "w_target", "halved", "stage_code", "traded_today", "j_signal"])
        WL["date"] = cal[WL.t.values]
        WL["name"] = nm(WL.j)
        WL["signal_name"] = nm(WL.j_signal)
        GP = pd.DataFrame(gaps, columns=["t", "j", "gap", "w_prev_close", "nav_impact"])
        GP["date"] = cal[GP.t.values]
        GP["name"] = nm(GP.j)
        out["weights"], out["gaps"] = WL, GP
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pack(name, res):
    m = perf(res["eq"])
    ev = res["events"]
    return {
        "name": name,
        "metrics": m,
        "headline": headline(m),
        "periods": perf_periods(res["eq"]),
        "yearly": yearly(res["eq"]),
        "n_events": int(len(ev)),
        "event_counts": {k: int(v) for k, v in ev["what"].value_counts().items()},
        "trade_days": int(res["eq"]["fee"].gt(0).sum()),
    }


def apply_lev2(D, csv_path: Path):
    raw = pd.read_csv(csv_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    names = set(D.names)
    for bad in ("UGL", "AGQ", "UCO"):
        if bad in names:
            raise SystemExit(f"LEV2 ticker {bad} is already in the universe")
    diag = {}
    saved_vol = {}
    for sig, exe in SLEEVE:
        j = D.cj[sig]
        g = raw[raw["Ticker"] == exe].drop_duplicates("Date").set_index("Date").sort_index()
        px = pd.to_numeric(g["Close"], errors="coerce")
        adj = pd.to_numeric(g["Adj Close"], errors="coerce")
        op = pd.to_numeric(g["Open"], errors="coerce")
        factor = adj / px
        ok = np.isfinite(factor.to_numpy()) & (px.to_numpy() > 0)
        factor = factor.where(ok)
        adj_o = (op * factor).astype(float)
        adj_c = (px * factor).astype(float)
        o = adj_o.reindex(D.cal)
        c = adj_c.reindex(D.cal)
        ret = adj_c.pct_change().fillna(0.0)
        sig_ann = pd.Series(np.sqrt(ewma_var(ret, 32).to_numpy()) * math.sqrt(252), index=adj_c.index)
        vol = sig_ann.reindex(D.cal).ffill()
        saved_vol[sig] = D.VOL[:, j].copy()
        D.O[:, j] = o.to_numpy(dtype=float)
        D.C[:, j] = c.to_numpy(dtype=float)
        D.VOL[:, j] = vol.to_numpy(dtype=float)
        D.EL[:, j] = np.isfinite(o.to_numpy(dtype=float))
        fin = np.isfinite(D.C[:, j])
        D.lastvalid[j] = int(D.T - 1 - np.argmax(fin[::-1])) if fin.any() else -1
        sl = (D.cal >= "2019-01-02") & (D.cal <= "2026-09-30")
        base = saved_vol[sig][sl]
        new = D.VOL[sl, j]
        both = np.isfinite(base) & np.isfinite(new) & (base > 0)
        diag[sig] = {
            "exec": exe,
            "rows": int(len(g)),
            "bt_days": int(sl.sum()),
            "bt_open_finite": int(np.isfinite(D.O[sl, j]).sum()),
            "median_vol_2x": float(np.nanmedian(new)),
            "median_vol_1x": float(np.nanmedian(base)),
            "median_vol_ratio": float(np.median(new[both] / base[both])) if both.any() else None,
        }
    return diag


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sha = _sha(CSV)
    if sha != CSV_SHA:
        raise SystemExit(f"LEV2 csv sha {sha} != {CSV_SHA}")
    print("LOAD_FM", flush=True)
    D = FM("/tmp/ca-v111/upleg_work/data", processes=2)
    for bad in ("UGL", "AGQ", "UCO"):
        if bad in D.ix:
            raise SystemExit(f"{bad} entered the name index")
    if any(D.POOL[j] for j in D.cj.values()):
        raise SystemExit("commodity columns are inside the selection pool")
    print("IDENTITY", flush=True)
    ident = run_freq(D, ret_thr=None, scale_in=False)
    gold = pd.read_csv(GOLD, index_col=0, parse_dates=True)
    same = len(gold) == len(ident["eq"]) and bool((gold.index == ident["eq"].index).all())
    maxabs = float(np.abs(ident["eq"].equity.values - gold.equity.values).max()) if same else None
    print(json.dumps({"identity_vs_reproduce_maxabs": maxabs, "same_idx": same}), flush=True)
    if not same or maxabs >= 1e-9:
        raise SystemExit("FREQ copy with flags off does not match the locked reproduce equity")
    print("FREQ_2", flush=True)
    f2 = run_freq(D, ret_thr=0.02, scale_in=True)
    print("FREQ_3", flush=True)
    f3 = run_freq(D, ret_thr=0.03, scale_in=True)
    js = [D.cj[k] for k, _ in SLEEVE]
    sig_before = {k: D.SIG[k][:, js].copy() for k in D.SIG}
    st_before = D.ST[:, js].copy()
    st2_before = D.ST2[:, js].copy()
    pool_before = D.POOL.copy()
    print("LEV2", flush=True)
    diag = apply_lev2(D, CSV)
    lev = run_portfolio(D)
    for k, arr in sig_before.items():
        if not np.array_equal(D.SIG[k][:, js], arr):
            raise SystemExit(f"SIG changed for {k}")
    if not np.array_equal(D.ST[:, js], st_before) or not np.array_equal(D.ST2[:, js], st2_before):
        raise SystemExit("stage arrays changed")
    if not np.array_equal(D.POOL, pool_before):
        raise SystemExit("POOL changed")
    rows = []
    packed = {}
    for key, res in (("v1.1.1", ident), ("FREQ_2pct", f2), ("FREQ_3pct", f3), ("LEV2", lev)):
        packed[key] = _pack(key, res)
        m = packed[key]["metrics"]
        rows.append({"variant": key, **{k: m[k] for k in m}, **{f"headline_{a}": b for a, b in packed[key]["headline"].items()}})
        res["eq"].to_csv(OUT / f"equity_{key}.csv", float_format="%.12g")
    pd.DataFrame(rows).to_csv(OUT / "comparison_table.csv", index=False)
    report = {
        "identity_vs_reproduce_maxabs": maxabs,
        "csv_sha256": sha,
        "lev2_diag": diag,
        "variants": packed,
    }
    (OUT / "freq_lev2_result.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print("DONE", json.dumps({k: packed[k]["headline"] for k in packed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
