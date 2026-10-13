"""Daily open-trade engine for the v1.1.1 control and the round-2 candidates.

Stock and leveraged-ETF books use weight accounting (fractional shares, gross
capped at 1 minus the cash floor). LEAP calls and bull call spreads replace
the stock with a Black-Scholes approximation and dollar accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .options import call_price, clip_iv, strike_for_delta

CORR = 0.35
G_CODE = np.array([0.30, 0.50, 1.00, 0.50], dtype=np.float64)


@dataclass
class Cfg:
    name: str
    universe: str
    ma: int = 30
    vt: float = 0.16
    cash_floor: float = 0.30
    k: int = 3
    mom_n: int = 20
    exit_mode: str = "halve"
    priority: bool = True
    use_fib: bool = True
    rank: str = "hist"
    deriv: str = "none"
    crypto_cap: float = 1.0
    cost_mult: float = 1.0
    cell_min: int = 30
    opt: dict = field(default_factory=dict)


def cfg_control(**kw) -> Cfg:
    base = dict(
        name="control",
        universe="control",
        ma=30,
        vt=0.16,
        cash_floor=0.30,
        exit_mode="halve",
        priority=True,
        use_fib=True,
        rank="hist",
        deriv="none",
        crypto_cap=1.0,
        mom_n=20,
    )
    base.update(kw)
    return Cfg(**base)


def cfg_momentum(universe: str, name: str, **kw) -> Cfg:
    base = dict(
        name=name,
        universe=universe,
        ma=30,
        vt=0.25,
        cash_floor=0.0,
        exit_mode="anchor",
        priority=False,
        use_fib=False,
        rank="mom",
        deriv="none",
        crypto_cap=1.0 if universe == "SC" else 0.0,
        mom_n=20,
    )
    base.update(kw)
    return Cfg(**base)


class Pack:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _mask(pack: Pack, cfg: Cfg) -> np.ndarray:
    if cfg.universe == "control":
        m = pack.tradable_control.copy()
    elif cfg.universe == "S":
        m = pack.tradable_stock.copy()
    else:
        m = pack.tradable_sc.copy()
    if cfg.crypto_cap <= 0:
        m = m & ~pack.is_crypto
    return m


def _pvol(raw: np.ndarray, sig: np.ndarray) -> float:
    x = raw * np.maximum(sig, 1e-6)
    return math.sqrt(max((1.0 - CORR) * float(np.dot(x, x)) + CORR * float(x.sum()) ** 2, 0.0))


def _metrics(equity: np.ndarray, dates: np.ndarray, min_year_days: int) -> dict:
    eq = np.asarray(equity, dtype=np.float64)
    dt = pd.DatetimeIndex(dates)
    ok = np.isfinite(eq) & (eq > 0)
    if ok.sum() < 5:
        return {k: float("nan") for k in ("cagr", "total", "maxdd", "sharpe", "sortino", "vol", "worst_year", "worst_year_label", "max_exposure")} | {
            "years": {},
            "n_days": int(ok.sum()),
        }
    e = eq[ok]
    d = dt[ok]
    years_span = max((d[-1] - d[0]).days / 365.25, 1e-9)
    total = float(e[-1] / e[0] - 1.0)
    cagr = float((e[-1] / e[0]) ** (1.0 / years_span) - 1.0)
    peak = np.maximum.accumulate(e)
    dd = e / peak - 1.0
    maxdd = float(dd.min())
    r = np.diff(e) / e[:-1]
    r = r[np.isfinite(r)]
    if len(r) > 5 and r.std(ddof=1) > 1e-12:
        sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252.0))
        vol = float(r.std(ddof=1) * math.sqrt(252.0))
        down = r[r < 0]
        dsd = float(down.std(ddof=1)) if len(down) >= 2 else 1e-12
        sortino = float(r.mean() / max(dsd, 1e-12) * math.sqrt(252.0))
    else:
        sharpe = vol = sortino = float("nan")
    by_year = {}
    frame = pd.Series(e, index=d)
    year_end = frame.groupby(frame.index.year).tail(1)
    year_count = frame.groupby(frame.index.year).size()
    prev = float(e[0])
    worst = float("inf")
    worst_label = ""
    for ts, val in year_end.items():
        y = int(ts.year)
        if int(year_count.loc[y]) < min_year_days:
            prev = float(val)
            continue
        ret = float(val) / prev - 1.0 if prev > 0 else float("nan")
        by_year[str(y)] = ret
        if np.isfinite(ret) and ret < worst:
            worst = ret
            worst_label = str(y)
        prev = float(val)
    if worst == float("inf"):
        worst = float("nan")
    return {
        "cagr": cagr,
        "total": total,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "vol": vol,
        "worst_year": worst,
        "worst_year_label": worst_label,
        "years": by_year,
        "n_days": int(ok.sum()),
    }


def slice_metrics(curve: pd.Series, start, end, min_year_days: int) -> dict:
    s = curve.loc[(curve.index >= pd.Timestamp(start)) & (curve.index <= pd.Timestamp(end))].dropna()
    if len(s) < 5:
        out = _metrics(np.array([]), np.array([]), min_year_days)
        out["start"] = str(start)
        out["end"] = str(end)
        return out
    m = _metrics(s.to_numpy(), s.index.to_numpy(), min_year_days)
    m["start"] = str(pd.Timestamp(s.index[0]).date())
    m["end"] = str(pd.Timestamp(s.index[-1]).date())
    return m


def _score_at(j, kind, route, spy, side, cell_sum, cell_n, gsum, gn, wins, cnts, cell_min):
    cid = (int(kind) * 2 + (1 if spy else 0)) * 3 + int(route)
    cid = min(max(cid, 0), cell_sum.size - 1)
    if cell_n[cid] >= cell_min:
        base = cell_sum[cid] / cell_n[cid]
    elif gn > 0:
        base = gsum / gn
    else:
        base = 0.0
    return base - 2.0 * side


def run_backtest(pack: Pack, cfg: Cfg, start: str, end: str, min_year_days: int = 100) -> dict:
    dates = pack.dates
    i0 = int(dates.searchsorted(pd.Timestamp(start)))
    i1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    i0 = max(i0, 1)
    if i1 <= i0 + 5:
        empty = _metrics(np.array([]), np.array([]), min_year_days)
        empty.update({"curve": pd.Series(dtype=float), "trades": [], "max_exposure": float("nan"), "n_trades": 0})
        return empty
    tradable = _mask(pack, cfg)
    up = pack.up[cfg.ma]
    dn = pack.dn[cfg.ma]
    mom = pack.mom[cfg.mom_n]
    O, C = pack.O, pack.C
    n = pack.n
    side = pack.side
    opt = cfg.opt or {}
    iv_mult = float(opt.get("iv_multiplier_on_realized", 1.20))
    iv_floor = float(opt.get("iv_floor", 0.15))
    iv_cap = float(opt.get("iv_cap", 1.50))
    rate = float(opt.get("rate", 0.02))
    div = float(opt.get("dividend_yield", 0.0))
    leap_days = int(opt.get("leap_expiry_calendar_days", 365))
    leap_roll = int(opt.get("leap_roll_when_remaining_calendar_days_below", 182))
    spread_days = int(opt.get("spread_expiry_calendar_days", 180))
    spread_roll = int(opt.get("spread_roll_after_calendar_days", 90))
    leap_cost = float(opt.get("leap_side_cost_frac_of_premium", 0.04))
    spread_cost = float(opt.get("spread_side_cost_frac_of_debit", 0.06))
    dlong = float(opt.get("leap_delta", 0.70))
    dshort = float(opt.get("spread_short_delta", 0.40))

    held: dict[int, bool] = {}
    w: dict[int, float] = {}
    lastc: dict[int, float] = {}
    cash = 1.0
    shares: dict[int, float] = {}
    opts: dict[int, dict] = {}
    use_opt = cfg.deriv in ("leap", "spread")
    eq = 1.0
    curve = np.full(i1 + 1, np.nan)
    curve[i0] = 1.0
    max_exposure = 0.0
    lev_days = 0
    trades: list[float] = []
    entry_px: dict[int, float] = {}
    seg_on = np.zeros(n, dtype=bool)
    seg_px = np.zeros(n)
    wins = np.zeros(n)
    cnts = np.zeros(n)
    cell_sum = np.zeros(24)
    cell_n = np.zeros(24)
    gsum = 0.0
    gn = 0
    capv = 1.0 - cfg.cash_floor
    slot = capv / cfg.k
    priority = []
    if cfg.priority:
        for sym in ("GLD", "SLV", "USO"):
            if sym in pack.ix:
                priority.append(pack.ix[sym])

    def mark_options(px_row, when: pd.Timestamp) -> float:
        total = 0.0
        for j, pos in opts.items():
            spot = px_row[j]
            if not np.isfinite(spot) or spot <= 0:
                continue
            tau = max((pos["expiry"] - when).days, 0) / 365.0
            long_px = call_price(spot, pos["k_long"], tau, pos["iv"], rate, div)
            short_px = 0.0
            if pos["k_short"] is not None:
                short_px = call_price(spot, pos["k_short"], tau, pos["iv"], rate, div)
            total += pos["covered"] * (long_px - short_px)
        return total

    def equity_at(px_row, when: pd.Timestamp) -> float:
        if not use_opt:
            return eq
        stock_val = 0.0
        for j, q in shares.items():
            p = px_row[j]
            if np.isfinite(p):
                stock_val += q * p
        return cash + stock_val + mark_options(px_row, when)

    def close_option(j, spot, when, cost_frac):
        nonlocal cash
        pos = opts.pop(j)
        tau = max((pos["expiry"] - when).days, 0) / 365.0
        long_px = call_price(spot, pos["k_long"], tau, pos["iv"], rate, div)
        short_px = 0.0 if pos["k_short"] is None else call_price(spot, pos["k_short"], tau, pos["iv"], rate, div)
        gross = pos["covered"] * (long_px - short_px)
        cash += gross * (1.0 - cost_frac)
        if j in entry_px and entry_px[j] > 0 and np.isfinite(spot):
            trades.append((str(when.date()), spot / entry_px[j] - 1.0))
            entry_px.pop(j, None)

    def open_option(j, spot, when, budget):
        nonlocal cash
        if budget <= 1e-8 or not np.isfinite(spot) or spot <= 0:
            return
        iv = clip_iv(pack.realized[t, j], iv_mult, iv_floor, iv_cap)
        if cfg.deriv == "leap":
            expiry = when + pd.Timedelta(days=leap_days)
            tau = leap_days / 365.0
            k = strike_for_delta(spot, tau, iv, dlong, rate, div)
            px = call_price(spot, k, tau, iv, rate, div)
            delta = dlong
            if px <= 1e-8:
                return
            covered = budget / (spot * delta)
            prem = covered * px
            if prem > budget:
                covered *= budget / prem
                prem = covered * px
            pay = prem * (1.0 + leap_cost)
            if pay > cash:
                covered *= cash / pay
                pay = covered * px * (1.0 + leap_cost)
            cash -= pay
            opts[j] = {"k_long": k, "k_short": None, "iv": iv, "expiry": expiry, "covered": covered, "opened": when}
            entry_px[j] = spot
        else:
            expiry = when + pd.Timedelta(days=spread_days)
            tau = spread_days / 365.0
            k_l = strike_for_delta(spot, tau, iv, dlong, rate, div)
            k_s = strike_for_delta(spot, tau, iv, dshort, rate, div)
            if k_s < k_l:
                k_l, k_s = k_s, k_l
            debit = call_price(spot, k_l, tau, iv, rate, div) - call_price(spot, k_s, tau, iv, rate, div)
            if debit <= 1e-8:
                return
            net_delta = max(dlong - dshort, 0.05)
            covered = budget / (spot * net_delta)
            if covered * debit > budget:
                covered = budget / debit
            pay = covered * debit * (1.0 + spread_cost)
            if pay > cash:
                covered *= max(cash, 0.0) / pay if pay > 0 else 0.0
                pay = covered * debit * (1.0 + spread_cost)
            if covered <= 0:
                return
            cash -= pay
            opts[j] = {"k_long": k_l, "k_short": k_s, "iv": iv, "expiry": expiry, "covered": covered, "opened": when}
            entry_px[j] = spot

    for t in range(i0 + 1, i1 + 1):
        when = pd.Timestamp(dates[t])
        turned_on = up[t] & ~up[t - 1] & tradable & np.isfinite(C[t - 1])
        turned_off = up[t - 1] & ~up[t] & seg_on & np.isfinite(C[t - 1]) & (seg_px > 0)
        seg_px = np.where(turned_on, C[t - 1], seg_px)
        seg_on = np.where(turned_on, True, seg_on)
        for j in np.flatnonzero(turned_off):
            gross = C[t - 1, j] / seg_px[j] - 1.0
            net = gross - 2.0 * side[j]
            route = int(pack.route[t, j])
            if route == 3:
                route = 2
            route = min(max(route, 0), 2)
            kind = int(pack.kind[j])
            spy = bool(pack.spy_trend[t])
            cid = (kind * 2 + (1 if spy else 0)) * 3 + route
            cell_sum[cid] += net
            cell_n[cid] += 1
            gsum += net
            gn += 1
            cnts[j] += 1
            if net > 0:
                wins[j] += 1
            seg_on[j] = False

        if use_opt:
            eq = equity_at(O[t], when)
        else:
            port_r = 0.0
            rets = {}
            for j, wj in w.items():
                p = O[t, j]
                lc = lastc.get(j, np.nan)
                r = (p / lc - 1.0) if np.isfinite(p) and np.isfinite(lc) and lc > 0 else 0.0
                rets[j] = r
                port_r += wj * r
            eq *= 1.0 + port_r
            if w and abs(1.0 + port_r) > 1e-12:
                w = {j: wj * (1.0 + rets[j]) / (1.0 + port_r) for j, wj in w.items()}
            for j in list(w):
                if np.isfinite(O[t, j]):
                    lastc[j] = O[t, j]

        changed = False
        for j in list(held):
            leg = bool(up[t, j]) and bool(tradable[j]) and np.isfinite(O[t, j])
            hard = (
                cfg.exit_mode == "anchor"
                and np.isfinite(pack.anchor[cfg.ma][t, j])
                and np.isfinite(C[t - 1, j])
                and C[t - 1, j] < pack.anchor[cfg.ma][t, j]
            )
            if cfg.exit_mode == "anchor":
                if (not leg) or hard:
                    held.pop(j, None)
                    changed = True
            else:
                down = bool(dn[t, j])
                if not leg and down:
                    held.pop(j, None)
                    changed = True
                elif not leg and not held[j]:
                    held[j] = True
                    changed = True
                elif leg and held[j]:
                    held[j] = False
                    changed = True

        if len(held) < cfg.k:
            for j in priority:
                if len(held) >= cfg.k:
                    break
                if j not in held and bool(tradable[j]) and bool(up[t, j]) and np.isfinite(O[t, j]):
                    held[j] = False
                    changed = True
            if len(held) < cfg.k:
                cands = []
                for j in range(n):
                    if j in held or not tradable[j] or not bool(up[t, j]) or not np.isfinite(O[t, j]):
                        continue
                    if cfg.exit_mode == "anchor" and not bool(pack.confirm[cfg.ma][t, j]):
                        continue
                    cands.append(j)
                if cfg.rank == "mom":
                    cands.sort(key=lambda j: (-(mom[t, j] if np.isfinite(mom[t, j]) else -1e9), -(pack.dv60[t, j] if np.isfinite(pack.dv60[t, j]) else 0.0), pack.names[j]))
                else:
                    scored = []
                    for j in cands:
                        route = int(pack.route[t, j])
                        if route == 3:
                            route = 2
                        sc = _score_at(
                            j,
                            pack.kind[j],
                            min(max(route, 0), 2),
                            bool(pack.spy_trend[t]),
                            side[j],
                            cell_sum,
                            cell_n,
                            gsum,
                            gn,
                            wins,
                            cnts,
                            cfg.cell_min,
                        )
                        winp = (wins[j] + 1.0) / (cnts[j] + 2.0)
                        dv = pack.dv60[t, j] if np.isfinite(pack.dv60[t, j]) else 0.0
                        scored.append((-sc, -winp, -dv, pack.names[j], j))
                    scored.sort()
                    cands = [row[-1] for row in scored]
                for j in cands:
                    if len(held) >= cfg.k:
                        break
                    held[j] = False
                    changed = True

        exec_of = {}
        for j in held:
            ej = j
            if cfg.deriv == "lev":
                lj = int(pack.lev_index[j])
                if lj >= 0 and bool(up[t, lj]) and np.isfinite(O[t, lj]):
                    ej = lj
            exec_of[j] = ej
        if cfg.deriv == "lev" and any(exec_of[j] != j for j in held):
            lev_days += 1

        hj = list(held)
        tgt: dict[int, float] = {}
        if hj:
            raw_list = []
            sig_list = []
            half_list = []
            ejs = []
            for j in hj:
                ej = exec_of[j]
                if cfg.use_fib:
                    code = int(pack.stage[t, ej])
                    g = 0.30 if code < 0 or code > 3 else float(G_CODE[code])
                else:
                    g = 1.0
                raw_list.append(slot * g * (0.5 if held[j] else 1.0))
                sig = pack.vol[t, ej]
                sig_list.append(sig if np.isfinite(sig) and sig > 0 else 0.20)
                half_list.append(bool(held[j]))
                ejs.append(ej)
            raw = np.array(raw_list, dtype=np.float64)
            halves = np.array(half_list, dtype=bool)
            if halves.any() and (~halves).any() and raw[~halves].sum() > 0:
                full_raw = raw / np.where(halves, 0.5, 1.0)
                freed = float((full_raw * 0.5)[halves].sum())
                raw = raw.copy()
                raw[~halves] += freed * raw[~halves] / raw[~halves].sum()
            sig = np.array(sig_list, dtype=np.float64)
            sp = _pvol(raw, sig)
            tg = raw * ((cfg.vt / sp) if sp > 1e-8 else 1.0)
            if tg.sum() > capv + 1e-12:
                tg *= capv / tg.sum()
            if cfg.crypto_cap < 1.0:
                is_c = np.array([bool(pack.is_crypto[ej]) for ej in ejs])
                s = float(tg[is_c].sum()) if is_c.any() else 0.0
                if s > cfg.crypto_cap + 1e-12:
                    tg[is_c] *= cfg.crypto_cap / s
            for ej, v in zip(ejs, tg):
                tgt[int(ej)] = tgt.get(int(ej), 0.0) + float(v)
        exposure = float(sum(tgt.values()))
        if exposure > max_exposure:
            max_exposure = exposure

        if use_opt:
            if eq <= 0 or not np.isfinite(eq):
                curve[t] = eq
                break
            cost_frac = leap_cost if cfg.deriv == "leap" else spread_cost
            desired = set(tgt)
            for j in list(opts):
                pos = opts[j]
                remain = (pos["expiry"] - when).days
                age = (when - pos["opened"]).days
                roll = (cfg.deriv == "leap" and remain < leap_roll) or (cfg.deriv == "spread" and age >= spread_roll)
                if j not in desired or roll or not np.isfinite(O[t, j]):
                    if np.isfinite(O[t, j]):
                        close_option(j, float(O[t, j]), when, cost_frac)
                    else:
                        opts.pop(j, None)
            for j, weight in tgt.items():
                if j in opts:
                    continue
                if not np.isfinite(O[t, j]) or O[t, j] <= 0:
                    continue
                open_option(j, float(O[t, j]), when, weight * max(eq, 0.0))
            eq_close = equity_at(C[t], when)
            curve[t] = eq_close
            continue

        keys = set(tgt) | set(w)
        drift = sum(abs(tgt.get(j, 0.0) - w.get(j, 0.0)) for j in keys)
        if changed or drift > 0.02:
            fee = 0.0
            for j in keys:
                fee += abs(tgt.get(j, 0.0) - w.get(j, 0.0)) * side[j] * cfg.cost_mult
            eq *= max(0.0, 1.0 - fee)
            new_w = {j: v for j, v in tgt.items() if v > 1e-8}
            for j in set(entry_px) - set(new_w):
                px = O[t, j] if np.isfinite(O[t, j]) else lastc.get(j, np.nan)
                if j in entry_px and entry_px[j] > 0 and np.isfinite(px):
                    trades.append((str(when.date()), px / entry_px[j] - 1.0))
                entry_px.pop(j, None)
            for j in new_w:
                if j not in entry_px and np.isfinite(O[t, j]):
                    entry_px[j] = float(O[t, j])
            w = new_w
            for j in w:
                if np.isfinite(O[t, j]):
                    lastc[j] = float(O[t, j])

        port_r = 0.0
        rets = {}
        for j, wj in w.items():
            c = C[t, j]
            lc = lastc.get(j, np.nan)
            r = (c / lc - 1.0) if np.isfinite(c) and np.isfinite(lc) and lc > 0 else 0.0
            rets[j] = r
            port_r += wj * r
        eq *= 1.0 + port_r
        if w and abs(1.0 + port_r) > 1e-12:
            w = {j: wj * (1.0 + rets[j]) / (1.0 + port_r) for j, wj in w.items()}
        for j in w:
            if np.isfinite(C[t, j]):
                lastc[j] = float(C[t, j])
        curve[t] = eq

    sl = pd.Series(curve[i0 : i1 + 1], index=pd.DatetimeIndex(dates[i0 : i1 + 1]))
    out = _metrics(sl.dropna().to_numpy(), sl.dropna().index.to_numpy(), min_year_days)
    out["curve"] = sl
    out["trades"] = trades
    out["n_trades"] = len(trades)
    out["lev_days"] = int(lev_days)
    out["max_exposure"] = max_exposure
    out["start"] = str(pd.Timestamp(dates[i0]).date())
    out["end"] = str(pd.Timestamp(dates[i1]).date())
    return out
