"""B-line engine: the OLD three-position framework driven by a shorter decision clock.

Plan §5. Same bar-count parameters as the locked weekly system (MA30, slope look-back 4, 2-bar
confirm, Fib pivot 3 each side, simple ATR14, min swing 1.5 ATR, 2-bar stage confirm, 0.02 buffer),
but evaluated on the current signal period's own bars. Differences from A00, all fixed in advance:
  * pool = the fixed 9 ETFs, one uniform treatment (no commodity priority)   (§5.1)
  * candidates ranked only by DV60 desc then symbol asc
  * signal at bar close t -> execution at bar t+1 open; no extra calendar day
  * volatility = daily EWMA(span 32) as of the last completed daily bar     (§5.2)
  * cash/share ledger with per-side fee + slippage, not A's differential-fee approximation (§7.2)
Max 3 names, slot = 0.70/3, 70% cap, 16% target vol, corr 0.35, halving + redistribution kept.

The Fib stage machinery is the locked function, imported from the read-only reference checkout, so
the weekly B path is the same code A00 uses on weekly bars.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

from .aport import pvol
from .ceng import FEE, SLIP_LIQ, SLIP_MID, DV_LIQ

MAX_POS = 3
GROSS_CAP = 0.70
SLOT = GROSS_CAP / MAX_POS
TARGET_VOL = 0.16
CORR = 0.35
DEADBAND = 0.02
MA_N = 30
SLOPE_LB = 4
CONFIRM = 2
G_OFFICIAL = {"grind": 0.30, "chop": 0.50, "trend": 1.00, "fish": 0.50, "UNDEFINED": 0.30}


def _fib_stages(ohlc: pd.DataFrame):
    """Locked Fib stage engine (v1.0.5) on whatever bar period it is handed."""
    from upleg_bt.engine.fib_locked import fib_stages_weekly
    d = ohlc.rename(columns=str.lower)
    return fib_stages_weekly(d)


def _ma_leg(close: pd.Series, n=MA_N, slope_lb=SLOPE_LB, confirm=CONFIRM):
    from upleg_bt.engine.upleg import ma_leg
    return ma_leg(close, n, slope_lb, confirm)


def ewma_sigma_daily(Cd: np.ndarray) -> np.ndarray:
    """Annualised EWMA(span 32) vol on forward-filled daily closes (same formula as the engine)."""
    T, N = Cd.shape
    a = 2.0 / 33.0
    Cf = pd.DataFrame(Cd).ffill().values
    v = np.full(N, np.nan)
    out = np.full((T, N), np.nan)
    for t in range(1, T):
        x = Cf[t] / Cf[t - 1] - 1
        ok = np.isfinite(x)
        pv = np.isfinite(v)
        v = np.where(ok, np.where(pv, a * x * x + (1 - a) * v, x * x), v)
        out[t] = v
    return np.sqrt(out) * math.sqrt(252.0)


def build_b(daily: dict, tf: str):
    """Return per-bar arrays for the B clock.

    {dates, syms, O,H,L,C, DV, SIGMA, code (bit0 leg, bit1 dn, 2=stage idx), bars_per_week}
    """
    syms = sorted(daily)
    cal = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in daily.values()]))).sort_values()
    D = {k: np.full((len(cal), len(syms)), np.nan) for k in ("open", "high", "low", "close", "dv60")}
    for i, s in enumerate(syms):
        d = daily[s].reindex(cal)
        D["open"][:, i] = d.open.values
        D["high"][:, i] = d.high.values
        D["low"][:, i] = d.low.values
        D["close"][:, i] = d.close.values
        D["dv60"][:, i] = d.dv60.values
    sig = ewma_sigma_daily(D["close"])
    nval = np.cumsum(np.isfinite(D["close"]), axis=0)

    # aggregate to the signal clock
    per = pd.DatetimeIndex(cal).to_period(tf)
    groups = [g.values for _, g in pd.Series(np.arange(len(cal))).groupby(per.values)]
    bar_dates = pd.DatetimeIndex([cal[g[-1]] for g in groups])
    last_idx = np.array([g[-1] for g in groups])
    O = np.vstack([D["open"][g][np.isfinite(D["open"][g]).any(axis=1)][0] if np.isfinite(D["open"][g]).any()
                   else np.full(len(syms), np.nan) for g in groups])
    Hh = np.vstack([np.nanmax(D["high"][g], axis=0) for g in groups])
    Ll = np.vstack([np.nanmin(D["low"][g], axis=0) for g in groups])
    Cc = np.vstack([D["close"][g][np.isfinite(D["close"][g]).any(axis=1)][-1] if np.isfinite(D["close"][g]).any()
                    else np.full(len(syms), np.nan) for g in groups])

    codes = np.zeros((len(groups), len(syms)), np.int8)      # bit0 leg, bit1 dn
    stage_idx = np.zeros((len(groups), len(syms)), np.int8)
    stages = []
    STAGE_ORDER = ["grind", "chop", "trend", "fish"]
    for i, s in enumerate(syms):
        ohlc = pd.DataFrame({"open": O[:, i], "high": Hh[:, i], "low": Ll[:, i], "close": Cc[:, i]},
                            index=bar_dates)
        fib = _fib_stages(ohlc)
        st = fib["stage"].astype(str).values
        stages.append(st)
        stage_idx[:, i] = [STAGE_ORDER.index(x) if x in STAGE_ORDER else 0 for x in st]
        on, ma = _ma_leg(pd.Series(Cc[:, i], index=bar_dates))
        ma_lag = ma.shift(SLOPE_LB)
        dn = ((pd.Series(Cc[:, i], index=bar_dates) < ma) & (ma < ma_lag)).astype(int)
        dn = (dn.astype(bool) & dn.shift(1, fill_value=False).astype(bool)).astype(int)
        codes[:, i] = on.values.astype(np.int8) | (dn.values.astype(np.int8) << 1)
    return dict(dates=bar_dates, syms=syms, O=O, H=Hh, L=Ll, C=Cc,
                DV=D["dv60"][last_idx], SIGMA=sig[last_idx],
                ELIG=(nval[last_idx] >= 252) & (D["dv60"][last_idx] >= 5e6),
                code=codes, stage=stage_idx, stage_names=stages, last_idx=last_idx, tf=tf)


def _slip(dv):
    return SLIP_LIQ if (np.isfinite(dv) and dv >= DV_LIQ) else SLIP_MID


def run_b(P, g_mode="official", cost_mult=1.0, delay_bars=0, start="2019-01-02", end="2026-09-30",
          capital=10000.0, gvec=None):
    T, N = P["C"].shape
    i0 = int(P["dates"].searchsorted(pd.Timestamp(start)))
    i1 = int(P["dates"].searchsorted(pd.Timestamp(end), side="right")) - 1
    fee_r = FEE * cost_mult
    gcode = gvec if gvec is not None else np.array([G_OFFICIAL[s] for s in
                                                    ["grind", "chop", "trend", "fish"]])
    if g_mode == "ones":
        gcode = np.ones(4)
    cash = float(capital)
    shares = np.zeros(N)
    held = {}                      # j -> halved(True/False)
    pending_w = None               # target weight vector frozen at bar t close
    pending_bar = None
    pending_force = False          # a leg/state change overrides the 2% deadband
    eq_rows, events, wlog, orders, rebal = [], [], [], [], []
    last_close = P["C"][i0].copy()
    trades = []
    open_ep = {}                   # j -> dict(start_bar, start_date, invested)
    for t in range(i0, i1 + 1):
        # ---------- 1) execute the weights frozen at an earlier close, at this bar's open
        if pending_w is not None and pending_bar == t:
            px = P["O"][t]
            nav = cash + float(np.nansum(shares * np.where(np.isfinite(px), px, last_close)))
            tgt_val = pending_w * nav
            cur_val = shares * np.where(np.isfinite(px), px, last_close)
            delta = tgt_val - cur_val
            # §5.2: the 2% target-weight deadband is kept, and a leg/state change always trades
            if (not pending_force) and np.abs(delta).sum() <= DEADBAND * nav:
                delta[:] = 0.0
            traded_notional = 0.0
            for j in range(N):
                if delta[j] == 0.0 or not np.isfinite(px[j]):
                    continue
                s = _slip(P["DV"][max(t - 1, 0), j])
                if delta[j] > 0:
                    pr = px[j] * (1 + s)
                    dq = delta[j] / pr
                    cost = dq * pr
                    fee = cost * fee_r
                    if cost + fee > cash:
                        dq = max(0.0, cash / (pr * (1 + fee_r)))
                        cost = dq * pr
                        fee = cost * fee_r
                    if dq <= 0:
                        continue
                    cash -= cost + fee
                    shares[j] += dq
                    orders.append(dict(bar=t, date=P["dates"][t], sym=P["syms"][j], kind="buy",
                                       qty=dq, price=pr, fee=fee, slip_bps=s * 1e4, notional=cost))
                    traded_notional += cost
                else:
                    pr = px[j] * (1 - s)
                    dq = min(shares[j], -delta[j] / pr)
                    if dq <= 0:
                        continue
                    cost = dq * pr
                    fee = cost * fee_r
                    cash += cost - fee
                    shares[j] -= dq
                    orders.append(dict(bar=t, date=P["dates"][t], sym=P["syms"][j], kind="sell",
                                       qty=dq, price=pr, fee=fee, slip_bps=s * 1e4, notional=cost))
                    traded_notional += cost
                if shares[j] < 1e-12:
                    shares[j] = 0.0
            pending_w, pending_bar, pending_force = None, None, False
            if traded_notional > 0:
                rebal.append((t, P["dates"][t], traded_notional))
        # ---------- 2) mark to market
        px = P["C"][t]
        ok = np.isfinite(px)
        last_close = np.where(ok, px, last_close)
        mv = float(np.nansum(shares * last_close))
        nav = cash + mv
        eq_rows.append((P["dates"][t], nav, cash, mv / nav if nav > 0 else 0.0, int((shares > 0).sum())))
        # ---------- 3) state updates (bar t close)
        if t >= i1:
            break
        S = t
        code = P["code"][S]
        changed = False
        for j in list(held):
            c = int(code[j])
            leg, dn = bool(c & 1), bool(c & 2)
            if not leg and dn:
                del held[j]; changed = True; events.append((t, j, "exit_dn"))
            elif not leg and not held[j]:
                held[j] = True; changed = True; events.append((t, j, "halve"))
            elif leg and held[j]:
                held[j] = False; changed = True; events.append((t, j, "restore"))
        # entries: only by DV60 desc then symbol, only if the leg is on
        free = MAX_POS - len(held)
        if free > 0:
            cand = [j for j in range(N) if j not in held and (int(code[j]) & 1) and P["ELIG"][S, j]]
            cand.sort(key=lambda j: (-P["DV"][S, j], P["syms"][j]))
            for j in cand[:free]:
                held[j] = False; changed = True; events.append((t, j, "enter"))
        # ---------- 4) targets (only when no order is already in flight: latency freezes it)
        if pending_w is not None:
            wlog.append((t, *[np.nan] * N))
            continue
        if held:
            hj = list(held)
            g = np.array([gcode[int(P["stage"][S, j])] for j in hj])
            half = np.array([held[j] for j in hj])
            raw = SLOT * g * np.where(half, 0.5, 1.0)
            if half.any() and (~half).any():
                freed = (SLOT * g * 0.5)[half].sum()
                raw[~half] += freed * raw[~half] / raw[~half].sum()
            sig = P["SIGMA"][S, hj]
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, 0.20)
            sp = pvol(raw, sig, CORR)
            tg = raw * (TARGET_VOL / sp if sp > 0 else 1.0)
            if tg.sum() > GROSS_CAP + 1e-12:
                tg *= GROSS_CAP / tg.sum()
            tgt = np.zeros(N)
            for k, j in enumerate(hj):
                tgt[j] = tg[k]
        else:
            tgt = np.zeros(N)
        pending_w = tgt; pending_bar = t + 1 + delay_bars; pending_force = changed
        wlog.append((t, *[tgt[j] for j in range(N)]))
    eq = pd.DataFrame(eq_rows, columns=["date", "equity", "cash", "exposure", "n_hold"]).set_index("date")
    EV = pd.DataFrame(events, columns=["t", "j", "what"])
    if len(EV):
        EV["date"] = P["dates"][EV.t.values]
        EV["sym"] = [P["syms"][j] for j in EV.j]
    W = pd.DataFrame(wlog, columns=["t"] + P["syms"]).set_index("t")
    return {"eq": eq, "events": EV, "targets": W, "orders": pd.DataFrame(orders),
            "rebal": pd.DataFrame(rebal, columns=["bar", "date", "notional"]),
            "shares_left": shares.copy(), "nav_end": nav}


def round_trips(order_df: pd.DataFrame, syms) -> pd.DataFrame:
    """§8.1 completed trades: per symbol, from a zero position to the next zero position.

    Net P&L is the sum of every cash flow of the episode (buys negative, sells positive, fees
    always negative), so partial entries/exits inside one episode are accounted for once.
    """
    if order_df is None or len(order_df) == 0:
        return pd.DataFrame()
    out = []
    for s, g in order_df.groupby("sym"):
        pos, start, buy_cost, sell_proc, fees, qty = 0.0, None, 0.0, 0.0, 0.0, 0.0
        for r in g.sort_values("bar").itertuples():
            if r.kind == "buy":
                if pos <= 1e-12:
                    start, buy_cost, sell_proc, fees, qty = r.date, 0.0, 0.0, 0.0, 0.0
                buy_cost += float(r.notional)
                fees += float(r.fee)
                qty += float(r.qty)
                pos += float(r.qty)
            else:
                sell_proc += float(r.notional)
                fees += float(r.fee)
                pos -= float(r.qty)
                if pos <= 1e-9 and start is not None:
                    net = sell_proc - buy_cost - fees
                    out.append(dict(sym=s, entry_date=start, exit_date=r.date,
                                    buy_cost=buy_cost, sell_proceeds=sell_proc, fees=fees,
                                    net_dollars=net, net_ret=net / buy_cost if buy_cost > 0 else np.nan,
                                    qty=qty))
                    start, buy_cost, sell_proc, fees, qty, pos = None, 0.0, 0.0, 0.0, 0.0, 0.0
    return pd.DataFrame(out)
