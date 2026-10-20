"""A-line: bit-compatible copy of trading-system@2ac825b backtest/upleg_bt/engine/portfolio.py
with two research switches added (nothing else changed):

  g_mode="official"  -> G_CODE = [0.30, 0.50, 1.00, 0.50]   (A00 / A01)
  g_mode="ones"      -> G_CODE = [1, 1, 1, 1]                (A10 / A11)

  rank="hist"        -> official historical-ranking score     (A00 / A10)
  rank="dv"          -> DV60 desc, then symbol asc, no Fib-score (A01 / A11)

The official path MUST reproduce upleg_bt.engine.portfolio.run_portfolio to <1e-12;
tests/test_a00_identity.py enforces that. Any behavioural drift here is a bug, not a finding.

Provenance: copied 2026-10-20 from
  trading-system@2ac825b56c9121f7bb4616c4e5ab18a2619df657 backtest/upleg_bt/engine/portfolio.py
  git blob 488ba770cf92e6ccd9534721424fc5862c685f5a
Every line below the `# --- copied verbatim ---` marker is unchanged apart from the
`g` vector construction and the candidate-ordering branch, both marked `# SWITCH`.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

SHARE_CLASS_MAP = {"BRK-A": "BRK-B", "BH-A": "BH"}
G_ONES = np.ones(4, np.float64)


def _g_official():
    """Lazy: the pinned package is only on sys.path once the driver has added it."""
    from upleg_bt.engine.fm import G_CODE
    return G_CODE


def pvol(w, sig, corr=0.35):
    x = w * np.maximum(np.where(np.isfinite(sig), sig, 0.20), 1e-6)
    return math.sqrt(max(float((1 - corr) * (x * x).sum() + corr * x.sum() ** 2), 0.0))


def sub_index(D, mp=SHARE_CLASS_MAP):
    return {D.ix[a]: D.ix[b] for a, b in mp.items() if a in D.ix and b in D.ix}


def weekly_days(cal):
    wk = np.asarray(cal.to_period("W-FRI").asi8); isw = np.zeros(len(cal), bool)
    isw[2:] = wk[1:-1] != wk[:-2]; return isw


def _sym_rank(D):
    """Column order by symbol ascending: the 'code asc' tie-break for rank='dv'."""
    order = np.argsort(np.asarray(D.names, dtype=object).astype(str), kind="stable")
    r = np.empty(len(order), np.int64); r[order] = np.arange(len(order)); return r


def ordered_candidates(D, LEGM, S, t, held, rank, rng, cost_mult, symrank=None, gvec=None):
    ok = D.EL[S] & D.POOL & ((LEGM[S] & 1) > 0) & np.isfinite(D.O[t])
    js = np.flatnonzero(ok)
    if held:
        js = js[~np.isin(js, np.fromiter(held.keys(), dtype=np.int64))]
    if len(js) == 0:
        return js
    if rng is not None:
        order = rng.permutation(len(js))
    elif rank == "dv":
        # SWITCH: DV60 desc, then symbol asc. No historical Fib score, no cost term.
        if symrank is None: symrank = _sym_rank(D)
        order = np.lexsort((symrank[js], -D.DV[S, js]))
    else:
        order = np.lexsort((-D.DV[S, js], -D.score(rank, S, js, cost_mult)))
    return js[order]


def run_portfolio(D, start="2019-01-02", end="2026-09-30", ma=30, delay=1, exit="halve", rank="hist", seed=None,
                  cost_mult=1.0, K=3, reserve=0.30, gross_cap=None, vt=0.16, sub=None, single_cap=None,
                  cap_mode="weekly", er=None, record=True, g_mode="official", trace=False):
    # SWITCH: the only change to the coefficient input.
    G_CODE = _g_official() if g_mode == "official" else G_ONES
    cal = D.cal; t0 = int(cal.searchsorted(pd.Timestamp(start))); t1 = int(cal.searchsorted(pd.Timestamp(end), side="right")) - 1
    rng = np.random.default_rng(seed) if rank == "random" else None
    symrank = _sym_rank(D) if rank == "dv" else None
    LEGM = D.SIG[(ma, delay)]; STM = D.ST if delay == 1 else D.ST2
    O, C, V, SIDE = D.O, D.C, D.VOL, D.SIDE; cset = list(D.cj.values())
    capv = (1 - reserve) if gross_cap is None else gross_cap; slot = (1 - reserve) / K
    if sub is None: sub = sub_index(D)
    P = lambda j: sub.get(j, j); Q = {b: a for a, b in sub.items()}
    comp = {}
    for a, b in sub.items(): comp[a] = b; comp[b] = b
    isw = weekly_days(cal); erh = {}
    eq = 1.0; w = {}; held = {}; lastc = {}; rec = []; events = []; wlog = []; gaps = []; trc = []
    for t in range(t0, t1 + 1):
        S = t - 1
        R = 0.0; ret_o = {}
        for j, wj in w.items():
            p = O[t, j]; r = (p / lastc[j] - 1) if np.isfinite(p) else 0.0; ret_o[j] = r; R += wj * r
        if record:
            for j, wj in w.items(): gaps.append((t, j, ret_o[j], wj, wj * ret_o[j]))
        eq *= 1 + R
        if w: w = {j: wj * (1 + ret_o[j]) / (1 + R) for j, wj in w.items()}
        for j in w:
            if np.isfinite(O[t, j]): lastc[j] = O[t, j]
        changed = False
        for j in list(held):
            code = int(LEGM[S, j]); leg = bool(code & 1); dn = bool(code & 2); pj = P(j)
            if t > D.lastvalid[j] or t > D.lastvalid[pj]: del held[j]; erh.pop(j, None); changed = True; events.append((t, j, "data_end")); continue
            if not np.isfinite(O[t, j]) or not np.isfinite(O[t, pj]): continue
            if exit == "full":
                if not leg: del held[j]; erh.pop(j, None); changed = True; events.append((t, j, "exit"))
            elif not leg and dn: del held[j]; erh.pop(j, None); changed = True; events.append((t, j, "exit_dn"))
            elif not leg and not held[j]: held[j] = True; changed = True; events.append((t, j, "halve"))
            elif leg and held[j]: held[j] = False; changed = True; events.append((t, j, "restore"))
        hcomp = {comp.get(j, j) for j in held}
        free = K - len(held)
        if free > 0:
            for j in cset:
                if free == 0: break
                if j not in held and (int(LEGM[S, j]) & 1) and np.isfinite(O[t, j]):
                    held[j] = False; free -= 1; changed = True; events.append((t, j, "enter"))
            if free > 0:
                for j in ordered_candidates(D, LEGM, S, t, held, rank, rng, cost_mult, symrank=symrank):
                    if free == 0: break
                    j = int(j); cj = comp.get(j, j)
                    if cj in hcomp or not np.isfinite(O[t, P(j)]): continue
                    held[j] = False; hcomp.add(cj); free -= 1; changed = True; events.append((t, j, "enter"))
        if er is not None:
            for j in list(erh):
                if j not in held: erh.pop(j); continue
                if t >= erh[j]: erh.pop(j); changed = True; events.append((t, j, "er_restore"))
            for j in held:
                if j in erh or j not in er: continue
                for h, r, rest in er[j]:
                    if h <= t <= r - 1: erh[j] = rest; changed = True; events.append((t, j, "er_halve")); break
        hj = list(held)
        if hj:
            g = np.array([G_CODE[max(int(STM[S, j]), 0)] for j in hj]); half = np.array([held[j] for j in hj])
            raw = slot * g * np.where(half, 0.5, 1.0)
            if half.any() and (~half).any():
                freed = (slot * g * 0.5)[half].sum(); raw[~half] += freed * raw[~half] / raw[~half].sum()
            sig = np.array([V[S, P(j)] for j in hj]); sp = pvol(raw, sig)
            tg = raw * (vt / sp if sp > 0 else 1.0)
            tg_pre = tg.copy()                                   # TRACE: before the 70% cap
            cap_hit = bool(tg.sum() > capv + 1e-12)
            if tg.sum() > capv + 1e-12: tg *= capv / tg.sum()
            if single_cap is not None:
                lim = np.full(len(hj), single_cap) if (cap_mode == "every" or isw[t]) else np.array([max(single_cap, w.get(P(j), 0.0)) for j in hj])
                tg = np.minimum(tg, lim)
            if er is not None: tg = tg * np.array([0.5 if j in erh else 1.0 for j in hj])
            tgt = {}
            for j, v in zip(hj, tg): tgt[P(j)] = tgt.get(P(j), 0.0) + v
            if trace:
                for k, j in enumerate(hj):
                    trc.append((t, j, int(STM[S, j]), float(g[k]), bool(half[k]), float(raw[k]),
                                float(sig[k]) if np.isfinite(sig[k]) else np.nan, float(sp),
                                float(vt / sp) if sp > 0 else 1.0, float(tg_pre[k]), float(tg[k]),
                                int(len(hj)), cap_hit))
        else: tgt = {}
        trig_cap = single_cap is not None and cap_mode == "weekly" and isw[t] and any(v > single_cap + 1e-12 for v in w.values())
        keys = set(tgt) | set(w); to = sum(abs(tgt.get(j, 0.0) - w.get(j, 0.0)) for j in keys); traded = False; fee = 0.0
        if changed or to > 0.02 or trig_cap:
            fee = sum(abs(tgt.get(j, 0.0) - w.get(j, 0.0)) * (SIDE[S, j] if np.isfinite(SIDE[S, j]) else 1e-3) for j in keys) * cost_mult
            eq *= 1 - fee; w = {j: v for j, v in tgt.items() if v > 0}; traded = True
            for j in w:
                if np.isfinite(O[t, j]): lastc[j] = O[t, j]
        R = 0.0; ret_c = {}
        for j, wj in w.items():
            c = C[t, j]; r = (c / lastc[j] - 1) if np.isfinite(c) else 0.0; ret_c[j] = r; R += wj * r
        eq *= 1 + R
        if w: w = {j: wj * (1 + ret_c[j]) / (1 + R) for j, wj in w.items()}
        for j in w:
            if np.isfinite(C[t, j]): lastc[j] = C[t, j]
        if record:
            for j in set(w) | set(tgt):
                sj = Q.get(j, j)
                wlog.append((t, j, w.get(j, 0.0), tgt.get(j, 0.0), bool(held.get(sj, False)), int(STM[S, sj]), traded, int(sj)))
        rec.append((cal[t], eq, sum(w.values()), sum(v for j, v in w.items() if j in cset), len(w), sum(1 for j in held if held[j]), to if traded else 0.0, fee))
    df = pd.DataFrame(rec, columns=["date", "equity", "exposure", "comm_exposure", "n_hold", "n_half", "turnover", "fee"]).set_index("date")
    df["stock_exposure"] = df.exposure - df.comm_exposure
    out = {"eq": df}
    nm = lambda a: D.names[np.asarray(a, int)]
    if events:
        EV = pd.DataFrame(events, columns=["t", "j", "what"]); EV["date"] = cal[EV.t.values]; EV["name"] = nm(EV.j)
        EV["exec_name"] = nm([P(j) for j in EV.j.values]) if len(EV) else []
    else:
        EV = pd.DataFrame(columns=["t", "j", "what", "date", "name", "exec_name"])
    out["events"] = EV
    if record:
        if wlog:
            WL = pd.DataFrame(wlog, columns=["t", "j", "w_close", "w_target", "halved", "stage_code", "traded_today", "j_signal"])
            WL["date"] = cal[WL.t.values]; WL["name"] = nm(WL.j); WL["signal_name"] = nm(WL.j_signal)
        else:
            WL = pd.DataFrame(columns=["t", "j", "w_close", "w_target", "halved", "stage_code", "traded_today", "j_signal", "date", "name", "signal_name"])
        if gaps:
            GP = pd.DataFrame(gaps, columns=["t", "j", "gap", "w_prev_close", "nav_impact"]); GP["date"] = cal[GP.t.values]; GP["name"] = nm(GP.j)
        else:
            GP = pd.DataFrame(columns=["t", "j", "gap", "w_prev_close", "nav_impact", "date", "name"])
        out["weights"], out["gaps"] = WL, GP
    if trace:
        out["target_trace"] = pd.DataFrame(trc, columns=["t", "j", "stage_code", "g", "half", "raw", "sigma",
                                                         "port_vol", "scale", "tg_pre_cap", "tg_post_cap",
                                                         "n_hold", "cap_binding"])
    return out
