"""Multi-position account kernel for QM-EOD-v2 (max 5 longs, BO+EP)."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Cfg:
    risk: float = 0.005
    maxw: float = 0.25
    max_positions: int = 5
    slip_bps: float = 10.0
    comm_ps: float = 0.005
    comm_min: float = 1.0
    allow_frac: bool = False          # True for $1k stress
    partial_day: int = 4              # 4th holding day close -> next open sell 1/3
    partial_frac: float = 1.0 / 3.0
    sma_exit: int = 10
    max_hold: int = 60
    cash_rate: float = 0.0
    allow_reentry_same_day: bool = True  # exit day may open new entries (v2)
    label: str = "main"


def _comm(q: float, cfg: Cfg) -> float:
    return max(cfg.comm_min, cfg.comm_ps * abs(q))


def _fill(px: float, cfg: Cfg, buying: bool) -> float:
    s = cfg.slip_bps / 1e4
    return px * (1 + s) if buying else px * (1 - s)


def size_position(E: float, P: float, S: float, cfg: Cfg, dv20: float, cash: float):
    risk_ps = abs(P - S)
    if risk_ps <= 0 or P <= 0 or E <= 0:
        return 0.0, "stop_distance_zero"
    budget = cfg.risk * E
    per_share = risk_ps + P * cfg.slip_bps / 1e4 + 2 * cfg.comm_ps
    q = budget / per_share
    q = min(q, cfg.maxw * E / P)
    if dv20 and np.isfinite(dv20) and dv20 > 0:
        # 1% ADV in shares ≈ 0.01 * dollar_vol / price
        q = min(q, 0.01 * dv20 / P)
    q = min(q, cash / (P * (1 + cfg.slip_bps / 1e4) + cfg.comm_ps + 1e-9))
    if cfg.allow_frac:
        q = math.floor(q * 1000) / 1000.0  # milli-share precision
        if q < 0.001:
            return 0.0, "size_lt_min_frac"
    else:
        q = float(int(math.floor(q)))
        if q < 1:
            return 0.0, "size_lt_one_share"
    # shrink until planned loss fits budget
    for _ in range(50):
        fee = _comm(q, cfg)
        planned = q * risk_ps + q * P * cfg.slip_bps / 1e4 + 2 * fee
        if planned <= budget + 1e-6:
            break
        if cfg.allow_frac:
            q = math.floor(q * 0.95 * 1000) / 1000.0
            if q < 0.001:
                return 0.0, "size_lt_min_frac"
        else:
            q -= 1
            if q < 1:
                return 0.0, "size_lt_one_share"
    return q, "ok"


def _mark_equity(cash: float, positions: dict, c_row, d: int) -> float:
    eq = cash
    for pos in positions.values():
        j = pos["j"]
        px = c_row[j]
        if not np.isfinite(px):
            px = pos.get("last_px", pos["entry"])
        eq += pos["q"] * px
    return eq


def run(kit, sigs: pd.DataFrame, cfg: Cfg, d0: int, d1: int, e0: float = 100_000.0,
        qqq_filter: np.ndarray | None = None):
    """Multi-position long-only portfolio.

    sigs columns: d, j, style ('bo'|'ep'), entry_mode ('next_open'|'same_close'),
                  stop (may be NaN for BO until entry-day LOD), atr (ADR$ or ATR$),
                  max_stop_mult (1.0 BO / 1.5 EP), base_high (BO), k1,k2,k3 ranking,
                  adr_frac (ADR20 as fraction), earnings_tagged (bool, EP).
    qqq_filter: bool array length T; True = new entries allowed that day.
    """
    by_day: dict[int, list] = {}
    if len(sigs):
        for d, g in sigs.groupby("d", sort=False):
            by_day[int(d)] = g.sort_values(
                ["k1", "k2", "k3"], ascending=[False, False, True]
            ).to_dict("records")

    o, h, l, c = kit.o, kit.h, kit.l, kit.c
    sma10 = kit.sma10
    sma20 = kit.sma20
    cash = float(e0)
    positions: dict[int, dict] = {}  # j -> pos
    trades: list[dict] = []
    rejects: list[dict] = []
    audits: list[dict] = []  # every signal attempt
    eq_rows: list[tuple] = []

    def occupied():
        return len(positions)

    def try_enter(d: int, sg: dict, entry_px_raw: float, entry_kind: str):
        nonlocal cash
        j = int(sg["j"])
        if j in positions:
            audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "already_holding"})
            rejects.append({"d": d, "j": j, "reason": "already_holding", "style": sg.get("style")})
            return False
        if occupied() >= cfg.max_positions:
            audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "max_positions"})
            rejects.append({"d": d, "j": j, "reason": "max_positions", "style": sg.get("style")})
            return False
        if qqq_filter is not None and not bool(qqq_filter[d]):
            audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "qqq_filter"})
            rejects.append({"d": d, "j": j, "reason": "qqq_filter", "style": sg.get("style")})
            return False
        if not np.isfinite(entry_px_raw) or entry_px_raw <= 0:
            audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "no_entry_price"})
            rejects.append({"d": d, "j": j, "reason": "no_entry_price", "style": sg.get("style")})
            return False

        style = sg.get("style", "bo")
        adr_frac = float(sg.get("adr_frac", np.nan))
        atr = float(sg.get("atr", np.nan))  # dollar ADR approx = adr_frac * price

        # BO: abandon if open already > base_high + 1×ADR
        if style == "bo" and entry_kind == "open":
            bh = float(sg.get("base_high", np.nan))
            if np.isfinite(bh) and np.isfinite(adr_frac):
                if entry_px_raw > bh * (1.0 + adr_frac):
                    audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "gap_too_far"})
                    rejects.append({"d": d, "j": j, "reason": "gap_too_far", "style": style})
                    return False

        P = _fill(entry_px_raw, cfg, buying=True)

        # provisional stop
        if style == "ep":
            S = float(sg["stop"])
            dist = P - S
            max_mult = float(sg.get("max_stop_mult", 1.5))
            stop_cap = (adr_frac * P) if np.isfinite(adr_frac) else atr
            if dist <= 0:
                audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "gap_through_stop"})
                rejects.append({"d": d, "j": j, "reason": "gap_through_stop", "style": style})
                return False
            if np.isfinite(stop_cap) and dist > max_mult * stop_cap:
                audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "stop_too_wide"})
                rejects.append({"d": d, "j": j, "reason": "stop_too_wide", "style": style})
                return False
            temp_stop = S  # EP stop known; active next day
            stop_active = False
        else:
            # BO: temp protection = entry - 1×ADR; real stop = entry-day LOD set at close
            if np.isfinite(adr_frac):
                temp_stop = P * (1.0 - adr_frac)
                stop_cap = adr_frac * P
            elif np.isfinite(atr) and atr > 0:
                temp_stop = P - atr
                stop_cap = atr
            else:
                temp_stop = P * 0.95
                stop_cap = P * 0.05
            S = temp_stop  # provisional until LOD known
            stop_active = False  # LOD stop from next day; temp used entry day? spec: entry day uses temp, not LOD

        E = _mark_equity(cash, positions, c[d], d)
        q, why = size_position(E, P, S if style == "ep" else temp_stop, cfg, kit.dv20[d - 1 if d > 0 else d, j], cash)
        if q <= 0:
            audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": why})
            rejects.append({"d": d, "j": j, "reason": why, "style": style})
            return False
        fee = _comm(q, cfg)
        cost = q * P + fee
        if cost > cash + 1e-9:
            audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "cash"})
            rejects.append({"d": d, "j": j, "reason": "cash", "style": style})
            return False
        cash -= cost
        positions[j] = {
            "j": j, "entry_day": d, "entry": P, "q0": q, "q": q,
            "stop": S, "temp_stop": temp_stop, "stop_active": stop_active,
            "stop_cap": stop_cap, "style": style,
            "planned_risk": q * abs(P - (S if style == "ep" else temp_stop)) + fee,
            "fees": fee, "partial_done": False, "pending_full": False,
            "pending_partial": False, "pending_reason": None,
            "mfe": 0.0, "mae": 0.0, "flags": [], "divs": 0.0,
            "realised": 0.0, "last_px": P,
            "earnings_tagged": bool(sg.get("earnings_tagged", False)),
            "base_high": float(sg.get("base_high", np.nan)),
            "adr_frac": adr_frac,
            "max_stop_mult": float(sg.get("max_stop_mult", 1.0 if style == "bo" else 1.5)),
            "abandoned": False,
        }
        audits.append({**_audit_base(d, sg), "outcome": "fill", "reason": "ok",
                       "entry": P, "q": q, "stop": S})
        return True

    def _audit_base(d, sg):
        return {
            "signal_d": int(sg["d"]), "entry_d": d, "j": int(sg["j"]),
            "style": sg.get("style"), "k1": sg.get("k1"), "k2": sg.get("k2"),
            "earnings_tagged": bool(sg.get("earnings_tagged", False)),
        }

    def close_pos(pos, px, d, reason):
        nonlocal cash
        q = pos["q"]
        if q <= 0:
            return
        exit_px = _fill(px, cfg, buying=False)
        fee = _comm(q, cfg)
        gross = (exit_px - pos["entry"]) * q
        cash += q * exit_px - fee
        fees = fee + pos["fees"]
        pnl = gross + pos.get("realised", 0.0) - fee - pos["fees"] + pos["divs"]
        trades.append({
            "entry_day": pos["entry_day"], "exit_day": d, "j": pos["j"],
            "style": pos["style"], "entry": pos["entry"], "exit": exit_px,
            "q0": pos["q0"], "q_exit": q,
            "planned_risk": pos["planned_risk"], "gross": gross, "fees": fees,
            "divs": pos["divs"], "pnl": pnl,
            "R": pnl / pos["planned_risk"] if pos["planned_risk"] > 0 else np.nan,
            "hold_days": d - pos["entry_day"] + 1, "exit_reason": reason,
            "mfe": pos["mfe"], "mae": pos["mae"],
            "flags": ";".join(sorted(set(pos["flags"]))),
            "partial": pos["partial_done"],
            "earnings_tagged": pos.get("earnings_tagged", False),
        })
        pos["q"] = 0

    def partial_sell(pos, px, d):
        nonlocal cash
        q_sell = pos["q"] * cfg.partial_frac
        if not cfg.allow_frac:
            q_sell = float(int(math.floor(q_sell)))
        else:
            q_sell = math.floor(q_sell * 1000) / 1000.0
        if q_sell <= 0 or q_sell >= pos["q"]:
            return
        exit_px = _fill(px, cfg, buying=False)
        fee = _comm(q_sell, cfg)
        cash += q_sell * exit_px - fee
        pos["q"] -= q_sell
        pos["fees"] += fee
        pos["partial_done"] = True
        pos["realised"] = pos.get("realised", 0.0) + (exit_px - pos["entry"]) * q_sell - fee
        # breakeven stop
        pos["stop"] = max(pos["stop"], pos["entry"])
        pos["temp_stop"] = max(pos["temp_stop"], pos["entry"])

    # ---------------- day loop
    for d in range(d0, d1 + 1):
        # 1) Open: process pending exits / gap stops / partials
        for j in list(positions.keys()):
            pos = positions[j]
            px = o[d, j]
            if not np.isfinite(px):
                last = c[d - 1, j] if d > 0 else pos["entry"]
                close_pos(pos, last if np.isfinite(last) else pos["entry"], d, "delist_last_close")
                del positions[j]
                continue
            # gap through stop (only if stop_active or using temp on entry day? entry day stop not LOD)
            stop_px = pos["stop"] if pos["stop_active"] else pos["temp_stop"]
            if px <= stop_px:
                close_pos(pos, px, d, "gap_stop")
                del positions[j]
                continue
            if pos["pending_full"]:
                close_pos(pos, px, d, pos["pending_reason"] or "signal_exit")
                del positions[j]
                continue
            if pos["pending_partial"]:
                partial_sell(pos, px, d)
                pos["pending_partial"] = False

        # 2) Intraday stop (active stop from next day after entry; entry day uses temp_stop)
        for j in list(positions.keys()):
            pos = positions[j]
            stop_px = pos["stop"] if pos["stop_active"] else pos["temp_stop"]
            # Spec: entry day does NOT use LOD stop; temp protection = entry-ADR
            # We check low vs stop_px every day including entry (temp on entry day)
            if np.isfinite(l[d, j]) and l[d, j] <= stop_px:
                close_pos(pos, stop_px, d, "stop")
                del positions[j]

        # 3) Entries
        # BO next_open: signals from d-1
        # EP same_close: handled at close below; also allow BO from d-1 at open
        if (d - 1) in by_day:
            for sg in by_day[d - 1]:
                if sg.get("entry_mode", "next_open") != "next_open":
                    continue
                if occupied() >= cfg.max_positions:
                    audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "max_positions"})
                    rejects.append({"d": d, "j": int(sg["j"]), "reason": "max_positions", "style": sg.get("style")})
                    continue
                try_enter(d, sg, o[d, int(sg["j"])], "open")

        # 4) Close-of-day: BO stop finalization, MTM, exit signals, EP same-close entries
        for j in list(positions.keys()):
            pos = positions[j]
            px = c[d, j]
            if not np.isfinite(px):
                continue
            pos["last_px"] = px
            r = px / pos["entry"] - 1
            pos["mfe"] = max(pos["mfe"], r)
            pos["mae"] = min(pos["mae"], r)
            if np.isfinite(kit.div[d, j]) and kit.div[d, j] > 0:
                amt = kit.div[d, j] * kit.adjf[d, j] * pos["q"]
                cash += amt
                pos["divs"] += amt

            # BO: at end of entry day, set stop = LOD; abandon if too wide
            if pos["style"] == "bo" and d == pos["entry_day"] and not pos["stop_active"]:
                lod = l[d, j]
                if np.isfinite(lod):
                    dist = pos["entry"] - lod
                    cap = pos["stop_cap"]
                    if np.isfinite(cap) and dist > pos["max_stop_mult"] * cap:
                        # abandon signal: reverse entry at entry price (no slip round-trip), no trade booked
                        q = pos["q"]
                        cash += q * pos["entry"] + pos["fees"]  # refund entry+entry fee
                        rejects.append({"d": d, "j": j, "reason": "abandon_stop_wide", "style": "bo"})
                        audits.append({"signal_d": pos["entry_day"], "entry_d": d, "j": j,
                                       "style": "bo", "k1": None, "k2": None,
                                       "earnings_tagged": False, "outcome": "reject",
                                       "reason": "abandon_stop_wide"})
                        del positions[j]
                        continue
                    pos["stop"] = lod
                    # update planned risk with real stop
                    pos["planned_risk"] = pos["q"] * abs(pos["entry"] - lod) + pos["fees"]
                pos["stop_active"] = True  # from next day
            elif pos["style"] == "ep" and d == pos["entry_day"]:
                pos["stop_active"] = True  # from next day (no stop check same day after close entry)

            if j not in positions:
                continue
            held = d - pos["entry_day"] + 1
            sma_arr = sma10 if cfg.sma_exit == 10 else sma20
            if (not pos["partial_done"] and held == cfg.partial_day and px > pos["entry"]):
                pos["pending_partial"] = True
            if np.isfinite(sma_arr[d, j]) and px < sma_arr[d, j] and held >= 2:
                pos["pending_full"] = True
                pos["pending_reason"] = "sma_close"
            if held >= cfg.max_hold:
                pos["pending_full"] = True
                pos["pending_reason"] = "max_hold"

        # EP same_close entries at close of signal day d
        if d in by_day:
            for sg in by_day[d]:
                if sg.get("entry_mode") != "same_close":
                    continue
                if occupied() >= cfg.max_positions:
                    audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "max_positions"})
                    rejects.append({"d": d, "j": int(sg["j"]), "reason": "max_positions", "style": sg.get("style")})
                    continue
                # require close > open already encoded in signal gen; re-check
                jj = int(sg["j"])
                if not (np.isfinite(c[d, jj]) and np.isfinite(o[d, jj]) and c[d, jj] > o[d, jj]):
                    audits.append({**_audit_base(d, sg), "outcome": "reject", "reason": "close_le_open"})
                    rejects.append({"d": d, "j": jj, "reason": "close_le_open", "style": "ep"})
                    continue
                try_enter(d, sg, c[d, jj], "close")

        # equity snapshot
        npos = occupied()
        shares_held = sum(p["q"] for p in positions.values())
        eq = _mark_equity(cash, positions, c[d], d)
        if npos == 0 and cfg.cash_rate:
            cash *= (1 + cfg.cash_rate / 252.0)
            eq = cash
        eq_rows.append((d, eq, cash, shares_held, npos))

    # final liquidation
    if positions:
        for j in list(positions.keys()):
            pos = positions[j]
            px = c[d1, j]
            if not np.isfinite(px):
                px = pos["last_px"]
            close_pos(pos, px, d1, "final_liquidation")
            del positions[j]
        eq_rows[-1] = (d1, cash, cash, 0.0, 0)

    eq = pd.DataFrame(eq_rows, columns=["d", "equity", "cash", "shares", "n_pos"])
    return {
        "equity": eq,
        "trades": pd.DataFrame(trades),
        "rejects": pd.DataFrame(rejects),
        "audit": pd.DataFrame(audits),
        "cfg": asdict(cfg),
    }
