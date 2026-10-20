"""Hourly data plane for the six formerly-BLOCKED configs (plan §3.1, §3.3, §6.2, §6.3).

Feed: Databento `ohlcv-1h`, two single-venue jobs (ARCX.PILLAR for SPY/GLD/SLV/USO/TLT/EFA/EEM,
XNAS.ITCH for QQQ/IWM), 2019-01-02 .. 2026-09-30, purchased for this study. Nothing here buys,
interpolates or proxies data: a missing bar stays missing and a missing order is cancelled.

Three facts drive every rule below, and all three are measured by `probe_hourly_preflight.py`:

1. Bar labels are the START of the interval, aligned to the UTC hour; minute-of-hour is 0 in
   America/New_York too, so a bar labelled 15:00 ET covers 15:00-16:00 ET. Two independent
   checks agree: the 15:00 bar's close sits within ~1bp of the official daily close, and the
   13:00 bar on 2025-04-09 contains the 13:18 ET tariff-pause move.
2. Therefore a session's real 60-minute grid starts at 09:30, which the vendor grid cannot
   express: the 09:00 bar mixes 30 minutes of pre-market into the regular open. The plan forbids
   pre/post-market prints (§3.3), so the grid used here is the set of whole vendor hours fully
   inside the regular session: 10:00..15:00 on a normal day (6 bars), 10:00..12:00 on a 13:00
   early close (3 bars). Consequences, stated in the report and not papered over:
     - no bar contains 09:30-10:00, so the opening half hour is not tradable in these configs;
     - every "next open" fill happens at 10:00 ET at the earliest, a systematic 30-minute lag
       versus the 09:30 official open, identical across all six configs;
     - no short/partial bars exist at all (the grid ends exactly at the close), so the plan's
       §3.3 short_bar diagnostic is degenerate. The plan's underlying concern (edge-of-session
       dependence) is instead reported as a per-hour signal distribution plus a post-hoc
       "no last-bar-of-day entry" rerun, both labelled as diagnostics.
3. Prices are single-venue and unadjusted, while every other line of this study uses the locked
   snapshot's dividend+split adjusted OHLC. Hourly OHLC is mapped onto exactly that scale with
   `hourly_adj = raw * (AdjClose/Close) / fut`, where `fut` is the product of splits AFTER the
   date (the same construction as `data.load_etf_daily`). `reconcile` then measures what is left:
   that residual is genuine single-venue vs consolidated-tape difference, and it is the number
   the report quotes. Deriving the factor from the ratio itself would have forced agreement and
   hidden the very difference being audited, so the split table is the only input used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

from . import ceng
from .data import SYMS

HOURLY_ROOT = Path("/Users/jake/projects/trading/data/hourly_databento_2019_2026")
CSV_NAME = "ALL9_ohlcv-1h_2019-20260930.csv"
VENUE = {"SPY": "ARCX.PILLAR", "GLD": "ARCX.PILLAR", "SLV": "ARCX.PILLAR", "USO": "ARCX.PILLAR",
         "TLT": "ARCX.PILLAR", "EFA": "ARCX.PILLAR", "EEM": "ARCX.PILLAR",
         "QQQ": "XNAS.ITCH", "IWM": "XNAS.ITCH"}
SESSION_OPEN = 9.5          # 09:30 ET
SESSION_CLOSE = 16.0        # 16:00 ET
EARLY_CLOSE = 13.0          # 13:00 ET


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    off = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=off + 7 * (n - 1))


def early_close(d) -> bool:
    """NYSE 13:00 ET early closes: day after Thanksgiving, Jul 3, Dec 24 (weekdays only).

    Independence Day / Christmas observed on those dates means the exchange is shut, not early.
    """
    d = pd.Timestamp(d)
    if d.weekday() >= 5:
        return False
    if d.month == 11 and d.weekday() == 4:
        thanksgiving = _nth_weekday(d.year, 11, 3, 4)
        return d.date() == thanksgiving + pd.Timedelta(days=1)
    if d.month == 7 and d.day == 3:
        j4 = date(d.year, 7, 4)
        return j4.weekday() < 5          # Jul 4 on a weekend -> Jul 3 IS the observed holiday
    if d.month == 12 and d.day == 24:
        xmas = date(d.year, 12, 25)
        return xmas.weekday() < 5
    return False


def expected_hours(d) -> list:
    """Whole vendor hours fully inside the regular session, labelled by their start (ET)."""
    last = int(EARLY_CLOSE - 1) if early_close(d) else int(SESSION_CLOSE - 1)
    return list(range(10, last + 1))


def session_frame(daily: dict, start="2019-01-02", end="2026-09-30") -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in daily.values()])))
    return cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]


def split_factor(daily: dict) -> dict:
    """fut(d) = product of splits strictly after d; hourly_raw / fut is the snapshot's Close scale."""
    out = {}
    for s, d in daily.items():
        spl = d.get("splits")
        if spl is None:
            spl = np.ones(len(d))
        spl = np.where(np.asarray(spl, float) > 0, np.asarray(spl, float), 1.0)
        fut = np.r_[np.cumprod(spl[::-1])[::-1][1:], 1.0]
        out[s] = pd.Series(fut, index=d.index)
    return out


def load_hourly(daily: dict, csv_path: str | Path | None = None, start="2019-01-02", end="2026-09-30",
                adjust=True) -> dict:
    """Long frame of regular-session hourly bars on the locked snapshot's adjusted scale.

    Returns dict(bars=DataFrame, sessions=DatetimeIndex, expected=dict(date->n_hours),
                 dropped=DataFrame of rows outside the regular grid).
    """
    path = Path(csv_path) if csv_path else HOURLY_ROOT / "csv" / CSV_NAME
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.ts_event, utc=True, format="mixed")
    et = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["et"] = et.values
    df["date"] = pd.to_datetime(et.dt.normalize().dt.tz_localize(None)).values
    df["hour"] = et.dt.hour.values
    df = df.rename(columns={"symbol": "sym", "open": "o", "high": "h", "low": "l", "close": "c",
                            "volume": "v"})
    sess = session_frame(daily, start, end)
    sset = pd.DatetimeIndex(sess)
    df = df[df.date.isin(sset)].copy()

    exp = {d: len(expected_hours(d)) for d in sset}
    hours_of = {d: expected_hours(d) for d in sset}
    keep = np.array([h in hours_of.get(d) for d, h in zip(df.date.values, df.hour.values)])
    dropped = df[~keep].copy()
    df = df[keep].copy()

    df["sess"] = sset.get_indexer(pd.DatetimeIndex(df.date))
    df["off"] = [hours_of[d].index(h) for d, h in zip(df.date.values, df.hour.values)]
    df["short_bar"] = False          # whole hours only; see module docstring point 2
    df = df.sort_values(["et", "sym"]).reset_index(drop=True)

    if adjust:
        futs = split_factor(daily)
        fac = np.ones(len(df))
        for s, g in df.groupby("sym"):
            f = futs[s].reindex(g.date.values)
            adjf = daily[s]["adj_ratio"].reindex(g.date.values).values
            fac[g.index.values] = adjf / f.values
        df["factor"] = fac
        for c in ("o", "h", "l", "c"):
            df[c + "_raw"] = df[c].values
            df[c] = df[c].values * fac
    return {"bars": df, "sessions": sset, "expected": exp, "dropped": dropped}


def coverage(bundle: dict) -> pd.DataFrame:
    """Per symbol: sessions present, expected bars, coverage vs the regular-session grid (§3.3)."""
    bars, sess = bundle["bars"], bundle["sessions"]
    exp_total = int(sum(bundle["expected"].values()))
    rows = []
    for s in SYMS:
        g = bars[bars.sym == s]
        per = g.groupby("sess").size().reindex(range(len(sess))).fillna(0).astype(int)
        has_session = (per > 0)
        rows.append({"sym": s, "venue": VENUE[s], "sessions_expected": int(len(sess)),
                     "sessions_present": int(has_session.sum()),
                     "sessions_missing": int(len(sess) - has_session.sum()),
                     "bars": int(per.sum()), "bars_expected": exp_total,
                     "coverage": float(per.sum() / exp_total),
                     "bars_expected_9x": int(exp_total * len(SYMS)),
                     "coverage_all9": float(per.sum() * len(SYMS) / exp_total / len(SYMS)),
                     "full_days": int((per == [len(expected_hours(d)) for d in sess]).sum()),
                     "partial_days": int(((per > 0) & (per < [len(expected_hours(d)) for d in sess])).sum()),
                     "first": str(sess[0].date()) if len(sess) else None,
                     "last": str(sess[-1].date()) if len(sess) else None})
    return pd.DataFrame(rows)


def daily_aggregate(bars: pd.DataFrame) -> pd.DataFrame:
    """Regular-session daily bars aggregated from the hourly grid (for the cross-source check)."""
    g = bars.sort_values("et").groupby(["sym", "date"]).agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"),
        n=("c", "size")).reset_index()
    return g


def reconcile(bundle: dict, daily: dict) -> pd.DataFrame:
    """Residual single-venue vs consolidated difference after the adjustment mapping (§3.1).

    Compared on the adjusted scale the study trades: close, open, and whether the hourly
    aggregated high/low actually contains the consolidated high/low. A difference here is data,
    not a bug — the report quotes it and marks the cross-period comparison accordingly.
    """
    agg = daily_aggregate(bundle["bars"])
    rows = []
    for s, g in agg.groupby("sym"):
        d = daily[s].reindex(g.date.values)
        rel_c = g.c.values / d["close"].values - 1.0
        rel_o = g.o.values / d["open"].values - 1.0
        ok = np.isfinite(rel_c)
        rows.append({
            "sym": s, "venue": VENUE[s], "n_days": int(ok.sum()),
            "med_abs_rel_close": float(np.nanmedian(np.abs(rel_c[ok]))),
            "p99_abs_rel_close": float(np.nanquantile(np.abs(rel_c[ok]), 0.99)),
            "max_abs_rel_close": float(np.nanmax(np.abs(rel_c[ok]))),
            "med_abs_rel_open": float(np.nanmedian(np.abs(rel_o[ok]))),
            "within_10bp": float((np.abs(rel_c[ok]) < 1e-3).mean()),
            "high_contained": float((g.h.values[ok] >= d["high"].values[ok] * (1 - 1e-3)).mean()),
            "low_contained": float((g.l.values[ok] <= d["low"].values[ok] * (1 + 1e-3)).mean()),
        })
    return pd.DataFrame(rows)


def hourly_panel(bundle: dict, Ld: pd.DataFrame, start="2019-01-02", end="2026-09-30") -> ceng.Panel:
    """Panel whose bars are the hourly grid, whose eligibility/vol/liquidity are prior-day daily.

    §3.3: an intraday decision may not see the day's own close, dollar volume or ATR. Every
    daily-derived column is therefore indexed at the previous completed session, and `ELIG`
    additionally requires the hourly indicators to be warm (ATR14, 20-bar breakout line).
    """
    bars = bundle["bars"]
    sess = bundle["sessions"]
    syms = sorted(SYMS)
    dates = pd.DatetimeIndex(sorted(bars.et.unique()))
    tix = {t: i for i, t in enumerate(dates)}
    six = {s: i for i, s in enumerate(syms)}
    T, N = len(dates), len(syms)
    A = {k: np.full((T, N), np.nan) for k in ("o", "h", "l", "c", "v")}
    ti = bars.et.map(tix).values
    si = bars.sym.map(six).values
    for k in ("o", "h", "l", "c", "v"):
        A[k][ti, si] = bars[k].values

    D = ceng.build_panel(Ld, "D")                      # daily panel over the full snapshot history
    pos = np.searchsorted(D.dates.values, dates.normalize().values)
    assert (D.dates.values[pos] == dates.normalize().values).all(), "hourly session not in daily calendar"
    prev = np.clip(pos - 1, 0, len(D.dates) - 1)
    atr = ceng.wilder_atr(A["h"], A["l"], A["c"])
    warm = np.isfinite(atr) & np.isfinite(ceng.breakout_line(A["h"]))

    sess_of_bar = bars.drop_duplicates("et").set_index("et").sess.reindex(dates).values
    off_of_bar = bars.drop_duplicates("et").set_index("et").off.reindex(dates).values
    last_in_sess = np.zeros(T, bool)
    last_in_sess[[int(np.flatnonzero(sess_of_bar == s)[-1]) for s in np.unique(sess_of_bar)]] = True

    return ceng.Panel(
        dates=dates, syms=syms,
        O=A["o"], H=A["h"], L=A["l"], C=A["c"], V=A["v"],
        DV=D.DV[prev], SIGMA=D.SIGMA[prev],
        ELIG=D.ELIG[prev] & warm,
        last_daily_idx=prev, tf="H",
        sess=sess_of_bar.astype(np.int64), off=off_of_bar.astype(np.int64),
        is_last=last_in_sess, prev_daily=prev)


def setup_table(Ld: pd.DataFrame, sessions: pd.DatetimeIndex | None = None) -> dict:
    """Per-day frozen setup for the D/H hybrid (§6.3), keyed on the session calendar.

    Frozen before the session opens, so every column is strictly d-1 except the breakout line,
    which is by definition the max of the 20 bars ending at d-1 (ceng.breakout_line excludes t),
    and the squeeze flag, which reads v(t-1) inside the 120 window ending at t-1. Both are
    computed from bars <= d-1 only; the test suite asserts that editing day d cannot move them.
    """
    D = ceng.build_panel(Ld, "D")
    atr = ceng.wilder_atr(D.H, D.L, D.C)
    sq = ceng.squeeze_flags(atr, D.C)
    brk = ceng.breakout_line(D.H)
    if sessions is None:
        keep = np.arange(len(D.dates))
    else:
        keep = np.searchsorted(D.dates.values, pd.DatetimeIndex(sessions).values)
        assert (D.dates.values[keep] == pd.DatetimeIndex(sessions).values).all()
    prev = np.clip(keep - 1, 0, len(D.dates) - 1)
    return {"dates": D.dates[keep], "atr": atr[keep], "atr_prev": atr[prev], "sq": sq[keep],
            "brk": brk[keep], "sigma_prev": D.SIGMA[prev], "dv_prev": D.DV[prev],
            "elig_prev": D.ELIG[prev], "close": D.C[keep], "idx_prev": prev}


def run_dh(P: ceng.Panel, S: dict, squeeze=False, cost_mult=1.0, delay_bars=0, delay_sessions=0,
           exit_delay_days=0, start="2019-01-02", end="2026-09-30", capital=10000.0,
           slippage=True, integer=False, ban_last_bar_entry=False):
    """C-DH: daily structure frozen before the session, hourly entry, daily exit (§6.3).

    Order of events inside a bar, exactly as C does it:
      1. an order frozen at an earlier close fills at this bar's open (never same-bar);
      2. mark to market at this bar's close;
      3. decision at this bar's close - a daily exit is only evaluated on the session's last bar
         (that bar closes at 16:00, so the daily close is known); entries are evaluated on every
         bar but read only the frozen setup of that session.
    Exits are daily: trailing stop off daily closes with daily ATR, 40 *daily* bars max hold,
    entry day = bar 1, executed at the next session's first available bar open (+exit_delay_days).
    """
    T, N = P.C.shape
    i0 = int(P.dates.searchsorted(pd.Timestamp(start)))
    i1 = int(P.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    fee_r = ceng.FEE * cost_mult
    sess, off, is_last = P.sess, P.off, P.is_last
    n_sess = int(sess.max()) + 1
    sess_start = np.array([int(np.flatnonzero(sess == s)[0]) for s in range(n_sess)])
    sess_end = np.array([int(np.flatnonzero(sess == s)[-1]) for s in range(n_sess)])

    def entry_bar(t):
        """Next hourly bar open (+delay_bars); delay_sessions shifts one full session, same hour."""
        if delay_sessions:
            s = int(sess[t]) + delay_sessions
            if s >= n_sess:
                return None
            u = int(sess_start[s]) + int(off[t])
            if u > sess_end[s]:
                u = int(sess_end[s])
            u += delay_bars
            return u if u <= i1 else None
        u = t + 1 + delay_bars
        return u if u <= i1 else None

    def exit_bar(d):
        """Daily close-triggered exit: the next session's first available bar (its 10:00 open).

        §6.3 asks for the next normal open. This feed's earliest regular bar opens at 10:00 ET
        (module docstring point 2), so that is the open every daily exit and every daily setup
        fill uses - a stated 30-minute lag, identical across all six configs.
        """
        s = d + 1 + exit_delay_days + delay_sessions
        if s >= n_sess:
            return None
        u = int(sess_start[s])
        return u if u <= i1 else None

    cash = float(capital)
    pos = None
    pending = None
    trades, orders, eq_rows = [], [], []
    submitted = {}
    last_px = {}
    for t in range(i0, i1 + 1):
        if pending is not None and pending["exec_bar"] == t:
            kind, j = pending["kind"], pending["j"]
            px = P.O[t, j]
            if not np.isfinite(px):
                orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind=kind,
                                   status="cancelled_missing_open", qty=np.nan, price=np.nan,
                                   fee=0.0, slip_bps=0.0, reason="no open"))
            elif kind == "entry":
                s = ceng._slip(P.DV[max(t - 1, 0), j]) if slippage else 0.0
                E = px * (1 + s)
                S0 = E - ceng.STOP_ATR * pending["ATR"]
                d = (E - S0) / E
                sig = pending["sigma"]
                if not (np.isfinite(S0) and S0 > 0 and np.isfinite(d) and d > 0):
                    orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry",
                                       status="rejected", qty=np.nan, price=np.nan, fee=0.0,
                                       slip_bps=s * 1e4, reason="s0_nonpositive"))
                elif not (np.isfinite(sig) and sig > 0):
                    orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry",
                                       status="rejected", qty=np.nan, price=np.nan, fee=0.0,
                                       slip_bps=s * 1e4, reason="sigma_invalid"))
                else:
                    w = min(1.0, ceng.TARGET_VOL / sig, ceng.RISK_BUDGET / d)
                    qty = w * cash / E
                    if integer:
                        qty = float(np.floor(qty))
                    notional = qty * E
                    fee_amt = max(notional * fee_r, ceng.MIN_TICKET) if integer else notional * fee_r
                    if notional + fee_amt > cash:
                        qty = max(0.0, (cash - (ceng.MIN_TICKET if integer else 0.0)) / (E * (1 + fee_r)))
                        if integer:
                            qty = float(np.floor(qty))
                        notional = qty * E
                        fee_amt = max(notional * fee_r, ceng.MIN_TICKET) if integer else notional * fee_r
                    if qty <= 0 or notional <= 0:
                        orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry",
                                           status="rejected", qty=0.0, price=E, fee=0.0,
                                           slip_bps=s * 1e4, reason="size_zero"))
                    else:
                        cash -= notional + fee_amt
                        pos = dict(j=j, qty=qty, E=E, S0=S0, S=S0, M=-np.inf, atr_entry=pending["ATR"],
                                   entry_bar=t, entry_date=P.dates[t], entry_day=int(sess[t]),
                                   fee_entry=fee_amt, notional=notional, slip_bps=s * 1e4,
                                   sig_bar=pending["sig_bar"], sig_date=pending["sig_date"],
                                   B_line=pending["B"], sigma=sig, w=w, dv60=pending["DV60"],
                                   episode=pending.get("episode", -1))
                        pos["M"] = float(P.C[t, j]) if np.isfinite(P.C[t, j]) else E
                        last_px[j] = float(P.C[t, j]) if np.isfinite(P.C[t, j]) else E
                        orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry",
                                           status="filled", qty=qty, price=E, fee=fee_amt,
                                           slip_bps=s * 1e4, reason=pending["why"], exec_bar=t,
                                           sig_bar=pending["sig_bar"]))
            else:
                s = ceng._slip(P.DV[max(t - 1, 0), j]) if slippage else 0.0
                X = px * (1 - s)
                notional = pos["qty"] * X
                fee_amt = max(notional * fee_r, ceng.MIN_TICKET) if integer else notional * fee_r
                cash += notional - fee_amt
                orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="exit", status="filled",
                                   qty=pos["qty"], price=X, fee=fee_amt, slip_bps=s * 1e4,
                                   reason=pending["why"], exec_bar=t, sig_bar=pending["sig_bar"]))
                ceng._close_trade(trades, pos, pending, t, X, notional, fee_amt, P)
                last_px.pop(j, None)
                pos = None
            pending = None

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
        if t >= i1:
            break

        d = int(sess[t])
        if pos is not None and pending is None and bool(is_last[t]):
            j = pos["j"]
            c = P.C[t, j]
            a = S["atr"][d, j]
            if np.isfinite(c) and np.isfinite(a):
                pos["M"] = max(pos["M"], float(c))
                pos["S"] = max(pos["S"], pos["S0"], pos["M"] - ceng.TRAIL_ATR * a)
                held = d - pos["entry_day"] + 1
                stop_hit = c <= pos["S"]
                time_hit = held >= ceng.MAX_BARS
                if stop_hit or time_hit:
                    why = "stop+time" if (stop_hit and time_hit) else ("trail_stop" if stop_hit else "time_40d")
                    u = exit_bar(d)
                    if u is None:
                        pending = None
                    else:
                        pending = dict(kind="exit", j=j, why=why, held=held, exec_bar=u, sig_bar=t)
        elif pos is None and pending is None:
            if ban_last_bar_entry and bool(is_last[t]):
                continue
            ok = S["elig_prev"][d] & np.isfinite(P.C[t]) & np.isfinite(S["brk"][d])
            ok &= P.C[t] > S["brk"][d]
            if squeeze:
                ok &= S["sq"][d]
            js = np.flatnonzero(ok)
            js = np.array([j for j in js if submitted.get((int(d), j), 0) == 0], dtype=int)
            u = entry_bar(t)
            if len(js):
                js = np.array([j for j in js if u is not None and np.isfinite(P.O[min(u, T - 1), j])],
                              dtype=int)
            if len(js):
                js = js[np.lexsort((js, -P.DV[t, js]))]
                j = int(js[0])
                submitted[(int(d), j)] = 1
                atr_d = S["atr_prev"][d, j]
                sig = S["sigma_prev"][d, j]
                dr = 2.0 * atr_d / P.C[t, j]
                why = []
                if not (np.isfinite(sig) and sig > 0):
                    why.append("sigma_invalid")
                if not (np.isfinite(dr) and dr > 0):
                    why.append("d_invalid")
                if why:
                    orders.append(dict(bar=t, date=P.dates[t], sym=P.syms[j], kind="entry",
                                       status="rejected", qty=np.nan, price=np.nan, fee=0.0,
                                       slip_bps=0.0, reason=",".join(why)))
                else:
                    pending = dict(kind="entry", j=j, ATR=atr_d, sigma=sig, B=float(S["brk"][d, j]),
                                   DV60=float(P.DV[t, j]), sig_bar=t, sig_date=P.dates[t],
                                   exec_bar=u, episode=int(d), why="dh_breakout" + ("+squeeze" if squeeze else ""))
    eq = pd.DataFrame(eq_rows, columns=["date", "equity", "cash", "exposure", "n_hold", "stale"]).set_index("date")
    return {"eq": eq, "trades": pd.DataFrame(trades), "orders": pd.DataFrame(orders), "_pos_open": pos}


def build_b_hourly(P_h: ceng.Panel, stage_cache: str | Path | None = None) -> dict:
    """B-line panel on the 60-minute clock (§5.2): locked Fib stages and MA30 run on hourly bars.

    Everything the B engine needs is the same shape as `beng.build_b`, except that volatility and
    dollar volume stay DAILY (prior completed session) and the Fib/trend state is recomputed on
    the hourly close series. That is exactly "same bar counts, different economic time": MA30 =
    30 hourly bars, slope look-back 4 hourly bars, 2-bar confirm, pivots 3 bars either side.
    """
    from upleg_bt.engine.fib_locked import fib_stages_weekly
    from upleg_bt.engine.upleg import ma_leg

    dates, syms = P_h.dates, P_h.syms
    T, N = P_h.C.shape
    cache = Path(stage_cache) if stage_cache else None
    codes = np.zeros((T, N), np.int8)
    stage_idx = np.zeros((T, N), np.int8)
    stage_names = []
    STAGE_ORDER = ["grind", "chop", "trend", "fish"]
    cached = {}
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        cached = {k: z[k] for k in z.files}
    for i, s in enumerate(syms):
        if f"stage_{s}" in cached:
            st = cached[f"stage_{s}"]
            on = cached[f"on_{s}"] if f"on_{s}" in cached else None
        else:
            ohlc = pd.DataFrame({"open": P_h.O[:, i], "high": P_h.H[:, i],
                                 "low": P_h.L[:, i], "close": P_h.C[:, i]}, index=dates)
            st = fib_stages_weekly(ohlc)["stage"].astype(str).values
            on, _ma = ma_leg(pd.Series(P_h.C[:, i], index=dates), 30, 4, 2)
            on = np.asarray(on, dtype=bool)
            if cache:
                np.savez(cache, **{**cached, f"stage_{s}": st, f"on_{s}": on})
                cached[f"stage_{s}"], cached[f"on_{s}"] = st, on
        stage_names.append(st)
        stage_idx[:, i] = [STAGE_ORDER.index(x) if x in STAGE_ORDER else 0 for x in st]
        c = pd.Series(P_h.C[:, i], index=dates)
        ma = c.rolling(30, min_periods=30).mean()
        ma_lag = ma.shift(4)
        dn = ((c < ma) & (ma < ma_lag)).astype(int)
        dn = (dn.astype(bool) & dn.shift(1, fill_value=False).astype(bool)).astype(int)
        codes[:, i] = np.asarray(on).astype(np.int8) | (dn.values.astype(np.int8) << 1)
    return dict(dates=dates, syms=list(syms), O=P_h.O, H=P_h.H, L=P_h.L, C=P_h.C,
                DV=P_h.DV, SIGMA=P_h.SIGMA, ELIG=P_h.ELIG, code=codes, stage=stage_idx,
                stage_names=stage_names, last_idx=P_h.last_daily_idx, tf="H",
                sess=P_h.sess, off=P_h.off, is_last=P_h.is_last)


def common_window(P_h: ceng.Panel, S: dict, min_sessions: int = 1) -> dict:
    """§3.2/§3.3: first session where every intraday indicator is warm, computed before any return.

    Binding constraint is the squeeze window: 120 valid v values + the breakout bar + ATR14, i.e.
    bar index 121 of the grid, and all nine symbols must have it. The result is frozen into
    hourly_preflight.json and used as the start of every r2 run, so no config gets a longer
    warm-up (or a longer performance window) than another.
    """
    T = len(P_h.dates)
    atr = ceng.wilder_atr(P_h.H, P_h.L, P_h.C)
    warm_atr = np.isfinite(atr)
    warm_brk = np.isfinite(ceng.breakout_line(P_h.H))
    v = atr / P_h.C
    warm_sq = np.zeros((T, len(P_h.syms)), bool)
    for t in range(ceng.SQ_WIN + 1, T):
        warm_sq[t] = np.isfinite(v[t - 1]) & np.isfinite(v[t - ceng.SQ_WIN:t]).all(axis=0)
    warm = warm_atr & warm_brk & warm_sq & np.isfinite(P_h.DV) & np.isfinite(P_h.SIGMA)
    allwarm = warm.all(axis=1)
    first_bar = int(np.argmax(allwarm)) if allwarm.any() else T - 1
    s0 = int(P_h.sess[first_bar])
    # start at the first bar of that session so every day is complete
    first_bar = int(np.flatnonzero(P_h.sess == s0)[0])
    return {"start": str(pd.Timestamp(P_h.dates[first_bar]).date()),
            "start_ts": str(pd.Timestamp(P_h.dates[first_bar])),
            "first_bar": first_bar, "session_ordinal": s0,
            "bar_index_required": int(ceng.SQ_WIN + 1),
            "n_bars": int(T), "n_sessions": int(P_h.sess.max()) + 1,
            "n_sessions_from_start": int(P_h.sess.max()) + 1 - s0,
            "months_from_start": float((P_h.dates[-1] - P_h.dates[first_bar]).days / 30.44),
            "all_symbols_warm": bool(allwarm[first_bar:first_bar + 1].all()),
            "binding_config": "C-H-S / C-DH-S (120-value squeeze window)"}


def daily_marks(eq: pd.DataFrame, dates: pd.DatetimeIndex, is_last, start, end) -> pd.DataFrame:
    """Collapse an intraday equity curve to one mark per session close.

    Plan §8.2/§9.1 require daily net returns and a daily-close drawdown for every comparison, and
    the hourly comparison must not be run on hourly returns with a sqrt(252) annualisation. The
    last bar of each session closes at the official close, so those rows are the daily marks; the
    index is normalised to the date so it intersects the daily configs exactly.
    """
    i0 = int(dates.searchsorted(pd.Timestamp(start)))
    i1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    n = min(len(eq), i1 - i0 + 1)
    m = np.asarray(is_last)[i0:i0 + n]
    out = eq.iloc[:n][m].copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


def entry_experience_bars(P: ceng.Panel, trades: pd.DataFrame, fee_r: float = ceng.FEE,
                          slip: float = ceng.SLIP_LIQ, brk: np.ndarray | None = None,
                          days: tuple = (1, 5)) -> pd.DataFrame:
    """§9.3 windows counted in TRADING DAYS for an hourly grid (6 bars = 1 day, half days 3).

    MAE/MFE use only bars after the fill and before the exit bar (§6.3: no day-low shortcut).
    net_1d/net_5d net both sides' costs; early exits stay early (no synthetic holding), and a
    window that runs past the end is flagged right_censored rather than counted as 0.
    """
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    if brk is None:
        brk = ceng.breakout_line(P.H)
    sess = P.sess
    rows = []
    for r in trades.itertuples():
        j = P.syms.index(r.sym)
        e, x = int(r.entry_bar), int(r.exit_bar)
        E = float(r.entry_price)
        d0 = int(sess[e])
        lo, hi, c = P.L[e:x, j], P.H[e:x, j], P.C[e:x, j]
        if len(c) == 0:
            continue
        mae = float(np.nanmin(lo) / E - 1.0)
        mfe = float(np.nanmax(hi) / E - 1.0)
        atr_rel = r.atr_entry / E if r.atr_entry else np.nan
        out = dict(sym=r.sym, entry_date=r.entry_date, exit_date=r.exit_date, bars_held=len(c),
                   days_held=int(sess[x] - d0), mae=mae, mfe=mfe,
                   mae_atr=mae / atr_rel if atr_rel else np.nan, entry_day=d0)
        cost1 = float(r.fee_entry) / float(r.notional) + fee_r + slip
        last_bar = len(P.dates) - 1
        for k in days:
            sel = (sess[e:x] - d0) < k
            n = int(sel.sum())
            last_held_day = int(sess[x - 1]) if x > e else d0
            truncated = last_held_day < d0 + (k - 1)
            out[f"exited_early_{k}d"] = bool(truncated)
            out[f"censored_{k}d"] = bool(truncated and x >= last_bar)
            if n == 0:
                out[f"mae_{k}d"] = np.nan
                out[f"net_{k}d"] = np.nan
                continue
            m = float(np.nanmin(lo[:n]) / E - 1.0)
            out[f"mae_{k}d"] = m
            out[f"mae_{k}d_atr"] = m / atr_rel if atr_rel else np.nan
            out[f"net_{k}d"] = float(c[n - 1]) / E - 1.0 - cost1
        kf = min(5, max(0, len(c) - 1))
        cc = c[1:kf + 1]
        out["failed_breakout_5bars"] = bool(len(cc) and (np.isfinite(cc) & (cc < float(r.B_line))).any())
        out["net_ret"] = r.net_ret
        out["exit_reason"] = r.exit_reason
        rows.append(out)
    return pd.DataFrame(rows)
