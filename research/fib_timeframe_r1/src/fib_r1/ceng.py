"""C-line engine: single-position breakout / squeeze-breakout on the fixed 9-ETF pool.

Implements plan §6.2 (W/W, D/D), §7.1 (position size), §7.2 (cost ledger), §9 (MAE/MFE, windows).
Everything is bar-index based so the same code runs on daily and weekly structure bars; only
the panel handed in changes. No look-ahead: a signal at bar t close is executed at bar t+1 open,
the breakout window excludes t, the squeeze test reads bar t-1 only, ATR/percentile use bars <= t.

Cash/share ledger (not the A-line weight approximation):
  buy   shares at E = open*(1+slip), fee = notional*fee_rate charged to cash separately
  sell  shares at X = open*(1-slip), fee = notional*fee_rate
  NAV   = cash + shares*last_visible_close   (stale prices carried forward, flagged)

`delay_bars` adds N extra execution bars to every order (plan §12 latency stress): the order is
frozen at the decision bar and only valid for its own execution bar, never refreshed.
`integer=True` rounds shares down to whole shares and applies the $1 minimum ticket
(plan §7.2 small-capital check); the $1 minimum is a stress assumption, not a broker quote.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

TRADING_DAYS = 252
BREAK_WIN = 20
ATR_N = 14
SQ_WIN = 120
SQ_Q = 0.20
MAX_BARS = 40
STOP_ATR = 2.0
TRAIL_ATR = 3.0
TARGET_VOL = 0.16
RISK_BUDGET = 0.01
FEE = 1e-4          # 1bp per side
SLIP_LIQ = 4e-4     # DV60 >= $20m
SLIP_MID = 9e-4     # $5m .. $20m
DV_LIQ = 20e6
DV_MIN = 5e6
MIN_DAYS = 252
MIN_TICKET = 1.0    # integer-share stress only


@dataclass
class Panel:
    """Aligned bar panel: (T,N) arrays plus the daily panel used for sigma / DV60 / eligibility."""
    dates: pd.DatetimeIndex
    syms: list
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    DV: np.ndarray          # daily DV60 (median of Close*Volume over 60 sessions) aligned to bar dates
    SIGMA: np.ndarray       # 20-day sample std * sqrt(252), as of the last complete daily session of the bar
    ELIG: np.ndarray        # >=252 valid daily bars and DV60>=5m and all indicators warm
    last_daily_idx: np.ndarray   # bar -> daily row index of its last session
    tf: str = "D"
    # SWITCH (hourly batch): set only for the 60-minute grid; empty for D/W so the identity
    # path of the existing four C configs is bit-identical.
    sess: np.ndarray | None = None       # bar -> session ordinal (0..S-1)
    off: np.ndarray | None = None        # bar -> index of the bar inside its session
    is_last: np.ndarray | None = None    # bar -> True if it is the session's last bar
    prev_daily: np.ndarray | None = None  # bar -> daily row index of the PREVIOUS session


def wilder_atr(H, L, C, n=ATR_N):
    T, N = H.shape
    tr = np.full((T, N), np.nan)
    prev_c = np.vstack([np.full((1, N), np.nan), C[:-1]])
    tr[:] = np.nanmax(np.stack([H - L, np.abs(H - prev_c), np.abs(L - prev_c)]), axis=0)
    atr = np.full((T, N), np.nan)
    if T > n:
        atr[n - 1] = np.nanmean(tr[:n], axis=0)
        for t in range(n, T):
            atr[t] = (n - 1) * atr[t - 1] / n + tr[t] / n
    return atr


def breakout_line(H, win=BREAK_WIN):
    T, N = H.shape
    out = np.full((T, N), np.nan)
    for t in range(win, T):
        out[t] = np.nanmax(H[t - win:t], axis=0)     # excludes t
    return out


def squeeze_flags(atr, C, win=SQ_WIN, q=SQ_Q):
    """count(v_k <= v_{t-1}, k=t-win..t-1)/win <= q, where v = ATR/C. Bar t-1 only.

    The 120-value window must be fully defined (plan §3.2: "120 个有效 v"); a window with any
    missing ATR is not a valid squeeze sample and the flag stays False rather than being
    flattered by counting NaNs as "not above the reference".
    """
    T, N = C.shape
    v = atr / C
    ok = np.zeros((T, N), bool)
    for t in range(win + 1, T):
        ref = v[t - 1]
        hist = v[t - win:t]                            # t-win .. t-1 (120 values)
        fin = np.isfinite(hist).all(axis=0) & np.isfinite(ref)
        cnt = np.sum(hist <= ref[None, :], axis=0)
        ok[t] = fin & ((cnt / float(win)) <= q)
    return ok


def _rolling_daily_sigma(Cd, win=20):
    T, N = Cd.shape
    r = np.full((T, N), np.nan)
    r[1:] = Cd[1:] / Cd[:-1] - 1.0
    sig = np.full((T, N), np.nan)
    for t in range(win, T):
        w = r[t - win + 1:t + 1]
        sig[t] = np.nanstd(w, axis=0, ddof=1) * math.sqrt(TRADING_DAYS)
    return sig


def _rolling_dv60(dv, win=60):
    T, N = dv.shape
    out = np.full((T, N), np.nan)
    for t in range(win - 1, T):
        out[t] = np.nanmedian(dv[t - win + 1:t + 1], axis=0)
    return out


def _rolling_valid_count(valid, win=MIN_DAYS):
    T, N = valid.shape
    out = np.zeros((T, N), np.int32)
    for t in range(T):
        out[t] = valid[max(0, t - win + 1):t + 1].sum(axis=0)
    return out


def build_panel(daily: pd.DataFrame, tf: str = "D") -> Panel:
    """daily: long frame date,sym,open,high,low,close,volume,dv. Structure bars from `tf`."""
    syms = sorted(daily.sym.unique())
    cal = pd.DatetimeIndex(sorted(daily.date.unique()))
    ix = {s: i for i, s in enumerate(syms)}
    T0, N = len(cal), len(syms)
    D = {k: np.full((T0, N), np.nan) for k in ("open", "high", "low", "close", "volume", "dv")}
    for k, col in (("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"),
                   ("volume", "volume"), ("dv", "dv")):
        D[k][cal.get_indexer(daily.date.values), daily.sym.map(ix).values] = daily[col].values
    valid = np.isfinite(D["close"])
    sig = _rolling_daily_sigma(D["close"])
    dv60 = _rolling_dv60(D["dv"])
    nval = _rolling_valid_count(valid)

    if tf == "D":
        bars = D
        bar_dates = cal
        last_idx = np.arange(T0)
    else:
        per = pd.DatetimeIndex(cal).to_period(tf)
        grp = pd.Series(np.arange(T0)).groupby(per.values)
        idx = [g.values for _, g in grp]
        bar_dates = pd.DatetimeIndex([cal[g[-1]] for g in idx])
        bars = {}
        for k in ("open", "high", "low", "close"):
            bars[k] = np.vstack([np.nanmax(D[k][g], axis=0) if k == "high" else
                                 np.nanmin(D[k][g], axis=0) if k == "low" else
                                 D[k][g][np.isfinite(D[k][g]).any(axis=1)][0] if k == "open" and np.isfinite(D[k][g]).any() else
                                 (D[k][g][np.isfinite(D[k][g]).any(axis=1)][-1] if np.isfinite(D[k][g]).any() else np.full(N, np.nan))
                                 for g in idx])
        bars["volume"] = np.vstack([np.nansum(D["volume"][g], axis=0) for g in idx])
        bars["dv"] = np.vstack([np.nanmedian(D["dv"][g], axis=0) for g in idx])
        last_idx = np.array([g[-1] for g in idx])

    atr = wilder_atr(bars["high"], bars["low"], bars["close"])
    warm = np.isfinite(atr) & np.isfinite(breakout_line(bars["high"]))
    return Panel(
        dates=bar_dates, syms=syms,
        O=bars["open"], H=bars["high"], L=bars["low"], C=bars["close"], V=bars["volume"],
        DV=dv60[last_idx], SIGMA=sig[last_idx],
        ELIG=(nval[last_idx] >= MIN_DAYS) & (dv60[last_idx] >= DV_MIN) & warm,
        last_daily_idx=last_idx, tf=tf)


def _slip(dv):
    return SLIP_LIQ if (np.isfinite(dv) and dv >= DV_LIQ) else SLIP_MID


def signal_episodes(P: Panel):
    """§8.1: per symbol, an episode = consecutive bars with close > breakout line."""
    Brk = breakout_line(P.H)
    above = np.isfinite(P.C) & np.isfinite(Brk) & (P.C > Brk)
    ep = np.full(above.shape, -1, np.int64)
    T, N = above.shape
    cnt = 0
    for j in range(N):
        run = -1
        for t in range(T):
            if above[t, j]:
                if run < 0:
                    run = cnt; cnt += 1
                ep[t, j] = run
            else:
                run = -1
    return ep, above


def run_breakout(P: Panel, squeeze=False, cost_mult=1.0, start="2019-01-02", end="2026-09-30",
                 capital=10000.0, slippage=True, delay_bars=0, integer=False,
                 episodes=None, above=None, delay_sessions=0, ban_last_bar_entry=False,
                 thin_p=None, thin_seed=0):
    """Single position, cash allowed. Returns dict with equity, trades, orders."""
    A = wilder_atr(P.H, P.L, P.C)
    B = breakout_line(P.H)
    SQ = squeeze_flags(A, P.C) if squeeze else np.ones_like(A, bool)
    T, N = P.C.shape
    i0 = int(P.dates.searchsorted(pd.Timestamp(start)))
    i1 = int(P.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    # SWITCH (hourly batch): delay_sessions is the plan §12 "one extra trading day" calendar
    # stress; it needs the session/offset map and is inert (0) for every pre-existing config.
    sess_start = sess_end = None
    if delay_sessions and P.sess is not None:
        n_s = int(P.sess.max()) + 1
        sess_start = np.array([int(np.flatnonzero(P.sess == s)[0]) for s in range(n_s)])
        sess_end = np.array([int(np.flatnonzero(P.sess == s)[-1]) for s in range(n_s)])

    def target_bar(t):
        """Execution bar for a decision at bar t; None when the plan has no valid time left."""
        u = t + 1 + delay_bars
        if delay_sessions and sess_start is not None:
            s = int(P.sess[t]) + delay_sessions
            if s >= len(sess_start):
                return None
            u = int(sess_start[s]) + int(P.off[t])
            if u > sess_end[s]:
                u = int(sess_end[s])
        return u if u <= i1 else None

    fee_r = FEE * cost_mult
    cash = float(capital)
    pos = None
    pending = None                 # dict(kind, j, exec_bar, ...)
    trades, orders = [], []
    eq_rows = []
    last_px = {}
    for t in range(i0, i1 + 1):
        # ---- 1) execute an order frozen `delay_bars+1` bars earlier, at this bar's open
        if pending is not None and pending["exec_bar"] == t:
            kind, j = pending["kind"], pending["j"]
            px = P.O[t, j]
            if not np.isfinite(px):
                orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind=kind,
                                   status="cancelled_missing_open", qty=np.nan, price=np.nan,
                                   fee=0.0, slip_bps=0.0, reason="no open"))
            elif kind == "entry":
                s = _slip(P.DV[max(t - 1, 0), j]) if slippage else 0.0
                E = px * (1 + s)
                S0 = E - STOP_ATR * pending["ATR"]
                d = (E - S0) / E
                sig = pending["sigma"]
                if not (np.isfinite(S0) and S0 > 0 and np.isfinite(d) and d > 0):
                    orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry", status="rejected",
                                       qty=np.nan, price=np.nan, fee=0.0, slip_bps=s * 1e4, reason="s0_nonpositive"))
                elif not (np.isfinite(sig) and sig > 0):
                    orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry", status="rejected",
                                       qty=np.nan, price=np.nan, fee=0.0, slip_bps=s * 1e4, reason="sigma_invalid"))
                else:
                    w = min(1.0, TARGET_VOL / sig, RISK_BUDGET / d)
                    qty = w * cash / E
                    if integer:
                        qty = float(np.floor(qty))
                    notional = qty * E
                    fee_amt = max(notional * fee_r, MIN_TICKET) if integer else notional * fee_r
                    if notional + fee_amt > cash:                 # fee-inclusive budget (plan §7.1)
                        qty = max(0.0, (cash - (MIN_TICKET if integer else 0.0)) / (E * (1 + fee_r)))
                        if integer:
                            qty = float(np.floor(qty))
                        notional = qty * E
                        fee_amt = max(notional * fee_r, MIN_TICKET) if integer else notional * fee_r
                    if qty <= 0 or notional <= 0:
                        orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry", status="rejected",
                                           qty=0.0, price=E, fee=0.0, slip_bps=s * 1e4, reason="size_zero"))
                    else:
                        cash -= notional + fee_amt
                        pos = dict(j=j, qty=qty, E=E, S0=S0, S=S0, M=-np.inf, atr_entry=pending["ATR"],
                                   entry_bar=t, entry_date=P.dates[t], fee_entry=fee_amt, notional=notional,
                                   slip_bps=s * 1e4, sig_bar=pending["sig_bar"], sig_date=pending["sig_date"],
                                   B_line=pending["B"], sigma=sig, w=w, dv60=pending["DV60"],
                                   episode=pending.get("episode", -1))
                        pos["M"] = float(P.C[t, j]) if np.isfinite(P.C[t, j]) else E
                        last_px[j] = float(P.C[t, j]) if np.isfinite(P.C[t, j]) else E
                        orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry", status="filled",
                                           qty=qty, price=E, fee=fee_amt, slip_bps=s * 1e4,
                                           reason=pending["why"], exec_bar=t, sig_bar=pending["sig_bar"]))
            else:  # exit
                s = _slip(P.DV[max(t - 1, 0), j]) if slippage else 0.0
                X = px * (1 - s)
                notional = pos["qty"] * X
                fee_amt = max(notional * fee_r, MIN_TICKET) if integer else notional * fee_r
                cash += notional - fee_amt
                orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="exit", status="filled",
                                   qty=pos["qty"], price=X, fee=fee_amt, slip_bps=s * 1e4,
                                   reason=pending["why"], exec_bar=t, sig_bar=pending["sig_bar"]))
                _close_trade(trades, pos, pending, t, X, notional, fee_amt, P)
                last_px.pop(j, None)
                pos = None
            pending = None
        # ---- 2) mark to market at this bar's close
        if pos is not None:
            j = pos["j"]
            c = P.C[t, j]
            stale = 0
            if np.isfinite(c):
                last_px[j] = float(c)
            else:
                stale = 1
            mv = pos["qty"] * last_px[j]
            nav = cash + mv
            expo = mv / nav if nav > 0 else 0.0
        else:
            nav, expo, stale = cash, 0.0, 0
        eq_rows.append((P.dates[t], nav, cash, expo, 1 if pos is not None else 0, stale))
        # ---- 3) decision at this bar's close
        if t >= i1:
            break
        if pos is not None and pending is None:
            # an exit already frozen at an earlier close is never re-priced or pushed later
            j = pos["j"]
            c = P.C[t, j]
            if np.isfinite(c) and np.isfinite(A[t, j]):
                pos["M"] = max(pos["M"], float(c))
                pos["S"] = max(pos["S"], pos["S0"], pos["M"] - TRAIL_ATR * A[t, j])
                held = t - pos["entry_bar"] + 1
                stop_hit = c <= pos["S"]
                time_hit = held >= MAX_BARS
                if stop_hit or time_hit:
                    why = "stop+time" if (stop_hit and time_hit) else ("trail_stop" if stop_hit else "time_40")
                    u = target_bar(t)
                    if u is None:
                        pending = None
                    else:
                        pending = dict(kind="exit", j=j, why=why, held=held, exec_bar=u,
                                       sig_bar=t)
        elif pending is None:
            if ban_last_bar_entry and P.is_last is not None and bool(P.is_last[t]):
                continue
            ok = P.ELIG[t] & np.isfinite(P.C[t]) & np.isfinite(B[t])
            ok &= P.C[t] > B[t]
            if squeeze:
                ok &= SQ[t]
            js = np.flatnonzero(ok)
            # an order must be fillable at its own execution bar
            u = target_bar(t)
            js = np.array([j for j in js if u is not None and np.isfinite(P.O[min(u, T - 1), j])],
                          dtype=int)
            if len(js):
                js = js[np.lexsort((js, -P.DV[t, js]))]   # DV60 desc, then symbol (columns are sym-sorted)
                j = int(js[0])
                atr_s = A[t, j]
                sig = P.SIGMA[t, j]
                d = 2.0 * atr_s / P.C[t, j]      # (E-S0)/E with E ~ C_t: the signal-time risk distance
                why = []
                if not (np.isfinite(sig) and sig > 0):
                    why.append("sigma_invalid")
                if not (np.isfinite(d) and d > 0):
                    why.append("d_invalid")
                if why:
                    orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry", status="rejected",
                                       qty=np.nan, price=np.nan, fee=0.0, slip_bps=0.0, reason=",".join(why)))
                elif thin_p is not None and np.random.default_rng([thin_seed, j, t]).random() > thin_p:
                    # §8.4 opportunity thinning: the refusal does NOT fall through to the second name
                    orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry", status="thinned",
                                       qty=np.nan, price=np.nan, fee=0.0, slip_bps=0.0,
                                       reason=f"thin_p={thin_p}"))
                else:
                    pending = dict(kind="entry", j=j, ATR=atr_s, sigma=sig, B=float(B[t, j]),
                                   DV60=float(P.DV[t, j]), sig_bar=t, sig_date=P.dates[t],
                                   exec_bar=u,
                                   episode=int(episodes[t, j]) if episodes is not None else -1,
                                   why="breakout" + ("+squeeze" if squeeze else ""))
    eq = pd.DataFrame(eq_rows, columns=["date", "equity", "cash", "exposure", "n_hold", "stale"]).set_index("date")
    tr = pd.DataFrame(trades)
    od = pd.DataFrame(orders)
    return {"eq": eq, "trades": tr, "orders": od, "_pos_open": pos}


def _close_trade(trades, pos, pl, t, X, notional, fee_amt, P):
    j = pos["j"]
    gross = (X / pos["E"] - 1.0)
    fees = pos["fee_entry"] + fee_amt
    net_dollars = notional - pos["notional"] - fees
    trades.append(dict(
        sym=P.syms[j], entry_bar=pos["entry_bar"], exit_bar=t,
        entry_date=pos["entry_date"], exit_date=P.dates[t],
        entry_price=pos["E"], exit_price=X, qty=pos["qty"],
        # bars counted the plan's way: the entry bar is bar 1, the decision bar is the last held bar
        bars_held=int(pl.get("held", t - pos["entry_bar"])),
        fill_span_bars=t - pos["entry_bar"],
        gross_ret=gross, net_ret=net_dollars / pos["notional"],
        fee_entry=pos["fee_entry"], fee_exit=fee_amt, fees_total=fees,
        w_at_entry=pos["w"], sigma=pos["sigma"], atr_entry=pos["atr_entry"],
        S0=pos["S0"], exit_reason=pl["why"], sig_date=pos["sig_date"], entry_sig_bar=pos["sig_bar"],
        dv60=pos["dv60"], slip_bps=pos["slip_bps"], episode=pos.get("episode", -1),
        notional=pos["notional"], B_line=pos["B_line"],
    ))


def entry_experience(P: Panel, trades: pd.DataFrame, fee_r: float = FEE, slip: float = SLIP_LIQ,
                     brk: np.ndarray | None = None) -> pd.DataFrame:
    """§9.3 per trade: MAE/MFE from entry fill to exit (no bars outside the position), 1d/5d windows.

    Windows are counted in STRUCTURE bars from the entry bar (the plan's 1/5 交易日 window for the
    D and W configs); the entry bar is included as bar 1. Exit-bar highs/lows are excluded because
    the exit fills at that bar's open. net_1d/net_5d net out both sides' costs so they are account
    consequences, not price returns.
    """
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    if brk is None:
        brk = breakout_line(P.H)
    rows = []
    for r in trades.itertuples():
        j = P.syms.index(r.sym)
        e, x = int(r.entry_bar), int(r.exit_bar)
        E = float(r.entry_price)
        lo, hi, c = P.L[e:x, j], P.H[e:x, j], P.C[e:x, j]
        mae = float(np.nanmin(lo) / E - 1.0)
        mfe = float(np.nanmax(hi) / E - 1.0)
        n = len(c)
        k1 = min(1, n - 1)
        k5 = min(5, n - 1)
        mae1 = float(np.nanmin(lo[:k1 + 1]) / E - 1.0)
        mae5 = float(np.nanmin(lo[:k5 + 1]) / E - 1.0)
        cost1 = float(r.fee_entry) / float(r.notional) + fee_r + slip
        net1 = float(c[k1]) / E - 1.0 - cost1
        net5 = float(c[k5]) / E - 1.0 - cost1
        # failed breakout: a completed structure-bar close back below the FROZEN line within 5 bars
        kf = min(5, max(0, n - 1))
        if kf > 0:
            cc = c[1:kf + 1]
            failed = bool((np.isfinite(cc) & (cc < float(r.B_line))).any())
        else:
            failed = False
        rows.append(dict(
            sym=r.sym, entry_date=r.entry_date, exit_date=r.exit_date, bars_held=n,
            mae=mae, mfe=mfe, mae_atr=mae / (r.atr_entry / E) if r.atr_entry else np.nan,
            mae_1d=mae1, mae_1d_atr=mae1 / (r.atr_entry / E) if r.atr_entry else np.nan,
            mae_1d_pct=mae1, mae_5d=mae5, net_1d=net1, net_5d=net5,
            right_censored=bool(n < 5), exited_early=bool(n < 5),
            failed_breakout_5bars=failed,
            net_ret=r.net_ret, exit_reason=r.exit_reason,
        ))
    return pd.DataFrame(rows)
