"""Account kernel: single position, 0.5% risk, 30% notional cap, daily bars."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Cfg:
    risk: float = 0.005
    maxw: float = 0.30
    slip_bps: float = 10.0
    comm_ps: float = 0.005
    comm_min: float = 1.0
    side: str = "long"
    style: str = "bo"          # bo | ep | ps
    partial_day: int = 4       # BO: D4 close -> D5 open
    sma_exit: int = 10
    max_hold: int = 60
    ps_hold: int = 5
    borrow_rate: float = 0.0   # annualised, /360, calendar days
    sma_len: int = 10
    no_sma_exit: bool = False
    no_partial: bool = False
    no_breakeven: bool = False
    cash_rate: float = 0.0
    label: str = "main"


def _comm(q, cfg):
    return max(cfg.comm_min, cfg.comm_ps * q)


def _fill(px, cfg, buying):
    s = cfg.slip_bps / 1e4
    return px * (1 + s) if buying else px * (1 - s)


def size_position(E, P, S, cfg, dv20, cash):
    side = cfg.side
    risk_ps = abs(P - S)
    if risk_ps <= 0:
        return 0, "stop_distance_zero"
    budget = cfg.risk * E
    # planned loss includes stop slippage and round-trip commission
    per_share = risk_ps + P * cfg.slip_bps / 1e4 + 2 * cfg.comm_ps
    q = budget / per_share
    q = min(q, cfg.maxw * E / P)
    if dv20 and np.isfinite(dv20) and dv20 > 0:
        q = min(q, 0.01 * dv20 / P)
    if side == "long":
        q = min(q, cash / P)
    q = int(math.floor(q))
    while q > 0 and (q * risk_ps + q * P * cfg.slip_bps / 1e4 + 2 * _comm(q, cfg)) > budget + 1e-9:
        q -= 1
    if q < 1:
        return 0, "size_lt_one_share"
    return q, "ok"


def run(kit, sigs, cfg: Cfg, d0, d1, e0=16000.0, rf_daily=None):
    """sigs: DataFrame with columns d, j, stop, atr, k1, k2, k3 (already ranked)."""
    by_day = {}
    if len(sigs):
        for d, g in sigs.groupby("d", sort=False):
            by_day[int(d)] = g.sort_values(["k1", "k2", "k3"], ascending=[False, False, True]).to_dict("records")

    o, h, l, c = kit.o, kit.h, kit.l, kit.c
    sma10, sma20 = kit.sma10, kit.sma20
    cash = float(e0)
    pos = None
    trades, rejects = [], []
    eq_rows = []
    exited_today = False
    entry_blocked_day = -1

    for d in range(d0, d1 + 1):
        exited_today = False

        # ---------- open: exits first
        if pos is not None:
            j = pos["j"]
            px = o[d, j]
            if not np.isfinite(px):
                # delisting / no tradable open: liquidate at last known close
                last = c[d - 1, j]
                px = last
                pos["flags"].append("delist_or_missing_open")
                cash = _close_pos(pos, px, cfg, cash, d, "delist_last_close", trades)
                pos = None
                exited_today = True
            else:
                side_long = cfg.side == "long"
                gap_stop = (px <= pos["stop"]) if side_long else (px >= pos["stop"])
                if gap_stop:
                    cash = _close_pos(pos, px, cfg, cash, d, "gap_stop", trades)
                    pos = None
                    exited_today = True
                elif pos["pending_full"]:
                    cash = _close_pos(pos, px, cfg, cash, d, pos["pending_reason"] or "signal_exit", trades)
                    pos = None
                    exited_today = True
                elif pos["pending_partial"]:
                    cash = _partial(pos, px, cfg, cash, d)
                    pos["pending_partial"] = False
                    if not cfg.no_breakeven:
                        pos["stop"] = max(pos["stop"], pos["entry"]) if side_long else min(pos["stop"], pos["entry"])
                elif pos["pending_trim"]:
                    cash = _trim(pos, px, cfg, cash, d)

        # ---------- intraday stop
        if pos is not None:
            j = pos["j"]
            if cfg.side == "long":
                if np.isfinite(l[d, j]) and l[d, j] <= pos["stop"]:
                    cash = _close_pos(pos, pos["stop"], cfg, cash, d, "stop", trades)
                    pos = None
                    exited_today = True
            else:
                # PS target check happens before stop only if target is above entry? adverse-first rule
                if np.isfinite(h[d, j]) and h[d, j] >= pos["stop"]:
                    cash = _close_pos(pos, pos["stop"], cfg, cash, d, "stop", trades)
                    pos = None
                    exited_today = True
                elif cfg.style == "ps" and (not cfg.no_sma_exit) and np.isfinite(sma10[d - 1, j]):
                    T = sma10[d - 1, j]
                    if T < o[d, j] and np.isfinite(l[d, j]) and l[d, j] <= T:
                        cash = _close_pos(pos, T, cfg, cash, d, "target_sma10", trades)
                        pos = None
                        exited_today = True

        # ---------- entry at open of d from signals at close d-1
        if pos is None and not exited_today and (d - 1) in by_day:
            for sg in by_day[d - 1]:
                j = int(sg["j"])
                P0 = o[d, j]
                if not np.isfinite(P0):
                    rejects.append({"d": d, "j": j, "reason": "no_open_price"})
                    continue
                P = _fill(P0, cfg, buying=(cfg.side == "long"))
                S = float(sg["stop"])
                dist = (P - S) if cfg.side == "long" else (S - P)
                if dist <= 0:
                    rejects.append({"d": d, "j": j, "reason": "gap_through_stop"})
                    continue
                if np.isfinite(sg["atr"]) and dist > sg["atr"]:
                    rejects.append({"d": d, "j": j, "reason": "stop_too_wide"})
                    continue
                E = cash
                q, why = size_position(E, P, S, cfg, kit.dv20[d - 1, j], cash)
                if q < 1:
                    rejects.append({"d": d, "j": j, "reason": why})
                    continue
                fee = _comm(q, cfg)
                if cfg.side == "long":
                    cost = q * P + fee
                    if cost > cash:
                        rejects.append({"d": d, "j": j, "reason": "cash"})
                        continue
                    cash -= cost
                else:
                    cash += q * P - fee
                pos = {
                    "j": j, "entry_day": d, "entry": P, "q0": q, "q": q,
                    "stop": S, "planned_risk": q * abs(P - S) + fee,
                    "fees": fee, "partial_done": False, "pending_full": False,
                    "pending_partial": False, "pending_trim": False, "pending_reason": None,
                    "mfe": 0.0, "mae": 0.0, "flags": [], "divs": 0.0, "borrow": 0.0,
                }
                break
            else:
                if by_day.get(d - 1):
                    rejects.append({"d": d, "j": -1, "reason": "none_executable"})

        # ---------- close-of-day accounting
        if pos is not None:
            j = pos["j"]
            px = c[d, j]
            if np.isfinite(px):
                side_long = cfg.side == "long"
                r = (px / pos["entry"] - 1) if side_long else (pos["entry"] / px - 1)
                pos["mfe"] = max(pos["mfe"], r)
                pos["mae"] = min(pos["mae"], r)
                # dividends
                if np.isfinite(kit.div[d, j]) and kit.div[d, j] > 0:
                    amt = kit.div[d, j] * kit.adjf[d, j] * pos["q"]
                    cash += amt if side_long else -amt
                    pos["divs"] += amt if side_long else -amt
                # borrow cost (short)
                if cfg.borrow_rate and not side_long:
                    days = 1
                    cash -= pos["q"] * px * cfg.borrow_rate / 360.0 * days
                    pos["borrow"] += pos["q"] * px * cfg.borrow_rate / 360.0 * days

                held = d - pos["entry_day"] + 1
                sma_arr = {10: kit.sma10, 16: kit.sma16, 20: kit.sma20, 24: kit.sma24}.get(cfg.sma_len, kit.sma10)
                if cfg.style == "bo":
                    if (not cfg.no_partial and not pos["partial_done"]
                            and held == cfg.partial_day and px > pos["entry"]):
                        pos["pending_partial"] = True
                    if (not cfg.no_sma_exit) and np.isfinite(sma_arr[d, j]) and px < sma_arr[d, j]:
                        pos["pending_full"] = True; pos["pending_reason"] = "sma_close"
                    if held >= cfg.max_hold:
                        pos["pending_full"] = True; pos["pending_reason"] = "max_hold"
                elif cfg.style == "ep":
                    if (not cfg.no_sma_exit) and np.isfinite(sma_arr[d, j]) and px < sma_arr[d, j]:
                        pos["pending_full"] = True; pos["pending_reason"] = "sma_close"
                    if (not cfg.no_sma_exit) and np.isfinite(sma_arr[d, j]) and sma_arr[d, j] > pos["stop"] and sma_arr[d, j] < px:
                        pos["stop"] = max(pos["stop"], sma_arr[d, j])
                    if held >= cfg.max_hold:
                        pos["pending_full"] = True; pos["pending_reason"] = "max_hold"
                elif cfg.style == "ps":
                    if held >= cfg.ps_hold:
                        pos["pending_full"] = True; pos["pending_reason"] = "max_hold"

                E = cash + pos["q"] * px * (1 if side_long else -1)
                if E > 0 and pos["q"] * px / E > cfg.maxw + 1e-9:
                    pos["pending_trim"] = True
                else:
                    pos["pending_trim"] = False
                eq = E
            else:
                eq = cash + pos["q"] * c[d - 1, j] * (1 if cfg.side == "long" else -1)
        else:
            eq = cash
            if cfg.cash_rate:
                cash *= (1 + cfg.cash_rate / 252.0)

        eq_rows.append((d, eq, cash, pos["q"] if pos else 0, (pos["j"] if pos else -1)))

    # final liquidation at last close
    if pos is not None:
        d = d1
        cash = _close_pos(pos, c[d, pos["j"]], cfg, cash, d, "final_liquidation", trades)
        pos = None
        eq_rows[-1] = (d, cash, cash, 0, -1)

    eq = pd.DataFrame(eq_rows, columns=["d", "equity", "cash", "shares", "sym"])
    return {"equity": eq, "trades": pd.DataFrame(trades), "rejects": pd.DataFrame(rejects), "cfg": asdict(cfg)}


def _close_pos(pos, px, cfg, cash, d, reason, trades):
    q = pos["q"]
    if q <= 0:
        return cash
    exit_px = _fill(px, cfg, buying=(cfg.side != "long"))
    fee = _comm(q, cfg)
    gross = (exit_px - pos["entry"]) * q if cfg.side == "long" else (pos["entry"] - exit_px) * q
    cash += (q * exit_px - fee) if cfg.side == "long" else (-q * exit_px - fee)
    fees = fee + pos["fees"]
    pnl = gross + pos.get("realised", 0.0) - fee - pos["fees"] - pos["borrow"] + pos["divs"]
    trades.append({
        "entry_day": pos["entry_day"], "exit_day": d, "j": pos["j"],
        "entry": pos["entry"], "exit": exit_px, "q0": pos["q0"], "q_exit": q,
        "planned_risk": pos["planned_risk"], "gross": gross, "fees": fees,
        "borrow": pos["borrow"], "divs": pos["divs"], "pnl": pnl,
        "R": pnl / pos["planned_risk"] if pos["planned_risk"] > 0 else np.nan,
        "hold_days": d - pos["entry_day"] + 1, "exit_reason": reason,
        "mfe": pos["mfe"], "mae": pos["mae"],
        "flags": ";".join(sorted(set(pos["flags"]))),
        "partial": pos["partial_done"],
    })
    return cash


def _partial(pos, px, cfg, cash, d):
    q = int(pos["q"] // 2)
    if q < 1 or q >= pos["q"]:
        return cash
    exit_px = _fill(px, cfg, buying=(cfg.side != "long"))
    fee = _comm(q, cfg)
    cash += (q * exit_px - fee) if cfg.side == "long" else (-q * exit_px - fee)
    pos["q"] -= q
    pos["fees"] += fee
    pos["partial_done"] = True
    # realised leg booked into planned-risk denominator at exit
    pos["realised"] = pos.get("realised", 0.0) + ((exit_px - pos["entry"]) * q if cfg.side == "long"
                                                  else (pos["entry"] - exit_px) * q) - fee
    return cash


def _trim(pos, px, cfg, cash, d):
    E = cash + pos["q"] * px * (1 if cfg.side == "long" else -1)
    target_q = int(math.floor(cfg.maxw * E / px))
    dq = pos["q"] - target_q
    if dq < 1:
        pos["pending_trim"] = False
        return cash
    exit_px = _fill(px, cfg, buying=(cfg.side != "long"))
    fee = _comm(dq, cfg)
    cash += (dq * exit_px - fee) if cfg.side == "long" else (-dq * exit_px - fee)
    pos["q"] -= dq
    pos["fees"] += fee
    pos["pending_trim"] = False
    return cash
