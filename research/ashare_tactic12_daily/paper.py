"""A股战法12（CCI月线）每日纸面交易单。无券商连接、不下单，只生成订单表。

与美股 Robinhood 策略/交易系统完全隔离：独立目录、独立状态目录、独立 LaunchAgent。
规则与 research/ashare_tactics 回测一致（信号用同一个 cci_signal_months 函数）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # mailer

from ashare_tactics.bars import arrays_from_frame  # noqa: E402
from ashare_tactics.costs import (  # noqa: E402
    buy_cash_out, code6, is_limit_down_open, is_limit_up_open, limit_band, sell_cash_in, size_shares,
)
from ashare_tactics.indicators import sma  # noqa: E402
from ashare_tactics.signals import _Book, _liquidity  # noqa: E402
from ashare_tactics.signals_patterns import cci_signal_months  # noqa: E402

MAC_STATE_DIR = "/Volumes/Disk/项目/data-cache/tactic12_paper"
MAC_CACHE_DIR = "/Volumes/Disk/项目/data-cache/ashare_tactics"


def config_from_env(env=None) -> dict:
    """路径/开关全部由环境变量决定；未设置时沿用 Mac 默认路径（兼容旧 LaunchAgent）。"""
    env = os.environ if env is None else env
    return {
        "state_dir": Path(env.get("STATE_DIR") or MAC_STATE_DIR),
        "cache_dir": Path(env.get("CACHE_DIR") or MAC_CACHE_DIR),
        "dry_run": str(env.get("DRY_RUN", "")).lower() in ("1", "true", "yes"),
        "min_coverage": float(env.get("MIN_COVERAGE") or 0.85),
        "min_universe": int(env.get("MIN_UNIVERSE") or 4000),
        "quiet": str(env.get("QUIET_SHEET", env.get("GITHUB_ACTIONS", ""))).lower() in ("1", "true", "yes"),
    }


_CFG = config_from_env()
STATE_DIR = _CFG["state_dir"]
CACHE = _CFG["cache_dir"]
INITIAL_CASH = 100_000.0
START_DATE = "2026-10-19"
MAX_POS = 10
WEIGHT = 0.10
MAX_HOLD = 40  # 与回测 _exit_ma(max_hold=40) 一致，买入当日为第1日


# ---------- 交易日历 ----------
def trade_dates() -> list[str]:
    import akshare as ak
    d = ak.tool_trade_date_hist_sina()
    return [str(pd.Timestamp(x).date()) for x in d["trade_date"]]


def prev_trade_date(cal: list[str], day: str) -> str:
    return [d for d in cal if d < day][-1]


# ---------- 数据增量刷新 ----------
def _fetch(code: str, start: str, end: str):
    """东方财富 hist；失败（海外IP限流等）时退回新浪日线。"""
    try:
        return _fetch_em(code, start, end)
    except Exception:  # noqa: BLE001
        return _fetch_sina(code, start, end)


def _fetch_sina(code: str, start: str, end: str):
    import akshare as ak
    sym = ("bj" if code.startswith(("4", "8", "92")) else "sh" if code.startswith(("6", "9")) else "sz") + code
    raw = ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust="")
    if raw is None or raw.empty:
        return None
    qfq = ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust="qfq")
    f = raw[["date", "open", "close", "high", "low", "volume", "amount"]].copy()
    f["volume"] = f["volume"] / 100.0  # 新浪为股，统一为手（与东财一致）
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    q = qfq[["date", "close"]].rename(columns={"close": "qc"})
    q["date"] = pd.to_datetime(q["date"]).dt.normalize()
    f = f.merge(q, on="date", how="left")
    f["factor"] = (f["close"] / f["qc"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return f.drop(columns="qc")


def _fetch_em(code: str, start: str, end: str):
    import akshare as ak
    raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
    qfq = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if raw is None or raw.empty:
        return None
    m = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}
    raw = raw.rename(columns=m)[list(m.values())]
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    q = qfq.rename(columns=m)[["date", "close"]].rename(columns={"close": "qc"})
    q["date"] = pd.to_datetime(q["date"]).dt.normalize()
    f = raw.merge(q, on="date", how="left")
    f["factor"] = (f["close"] / f["qc"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return f.drop(columns="qc")


def refresh_symbol(symbol: str, through: str) -> str:
    path = CACHE / "bars" / f"{symbol}.parquet"
    code = code6(symbol)
    end = through.replace("-", "")
    old = pd.read_parquet(path) if path.exists() else None
    if old is not None and len(old):
        old["date"] = pd.to_datetime(old["date"])
        # 已含 through 且文件写于收盘后 → 跳过
        last = old["date"].max()
        if str(last.date()) >= through and path.stat().st_mtime > pd.Timestamp(through + " 15:30", tz="Asia/Shanghai").timestamp():
            return "fresh"
        start = (last - pd.Timedelta(days=20)).strftime("%Y%m%d")
        new = _fetch(code, start, end)
        if new is None:
            return "nodata"
        ov = old.merge(new[["date", "factor"]], on="date", suffixes=("", "_n"))
        if len(ov) and np.allclose(ov["factor"], ov["factor_n"], rtol=1e-4):
            out = pd.concat([old[old["date"] < new["date"].min()], new], ignore_index=True)
        else:  # 除权导致前复权因子变化 → 全量重取
            out = _fetch(code, "20150101", end)
    else:
        out = _fetch(code, "20150101", end)
    if out is None:
        return "nodata"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return "ok"


def fetch_snapshot() -> pd.DataFrame:
    """全市场快照（开盘前即上一交易日收盘）。东财优先，失败退回新浪。列统一为
    symbol, open, high, low, close, prev_close, volume(手), amount。"""
    import akshare as ak
    last: Exception | None = None
    for k in range(3):
        try:
            d = ak.stock_zh_a_spot_em()
            code = d["代码"].astype(str).str.zfill(6)
            sym = np.where(code.str.startswith(("4", "8", "92")), "bj", np.where(code.str.startswith(("6", "9")), "sh", "sz")) + code
            out = pd.DataFrame({"symbol": sym, "open": d["今开"], "high": d["最高"], "low": d["最低"], "close": d["最新价"],
                                "prev_close": d["昨收"], "volume": d["成交量"], "amount": d["成交额"]})
            out.attrs["source"] = "eastmoney"
            return out
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(5 * 2 ** k)
    for k in range(3):
        try:
            d = ak.stock_zh_a_spot()
            out = pd.DataFrame({"symbol": d["代码"].astype(str), "open": d["今开"], "high": d["最高"], "low": d["最低"],
                                "close": d["最新价"], "prev_close": d["昨收"], "volume": d["成交量"] / 100.0, "amount": d["成交额"]})
            out.attrs["source"] = "sina"
            return out
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(5 * 2 ** k)
    raise RuntimeError("snapshot failed (eastmoney+sina)") from last


def apply_snapshot(symbol: str, snap_row, through: str, prev_day: str) -> str:
    """缓存最后一根K线=prev_day 且快照昨收与之吻合（无除权）→ 直接追加 through 这一根，沿用复权因子。
    否则返回 'need_fetch'，由逐只增量接口处理（除权、缺口、新股）。"""
    path = CACHE / "bars" / f"{symbol}.parquet"
    if not path.exists() or snap_row is None:
        return "need_fetch"
    old = pd.read_parquet(path)
    old["date"] = pd.to_datetime(old["date"])
    last = old.iloc[old["date"].values.argmax()]
    if str(last["date"].date()) >= through:
        return "fresh"
    vals = [snap_row[k] for k in ("open", "high", "low", "close", "prev_close", "volume", "amount")]
    if any(pd.isna(x) for x in vals):
        return "need_fetch"
    if float(snap_row["volume"]) <= 0:
        return "suspended"  # 当日停牌：不追加（与 hist 接口一致）
    if str(last["date"].date()) != prev_day or not np.isclose(float(snap_row["prev_close"]), float(last["close"]), rtol=1e-3):
        return "need_fetch"
    row = {"date": pd.Timestamp(through), "open": float(snap_row["open"]), "close": float(snap_row["close"]),
           "high": float(snap_row["high"]), "low": float(snap_row["low"]), "volume": float(snap_row["volume"]),
           "amount": float(snap_row["amount"]), "factor": float(last["factor"])}
    out = pd.concat([old, pd.DataFrame([row])[old.columns]], ignore_index=True)
    out.to_parquet(path, index=False)
    return "snap"


def snapshot_usable(today: str, now: pd.Timestamp | None = None) -> bool:
    """快照仅在 today 09:15（集合竞价）之前代表上一交易日收盘。"""
    now = now or pd.Timestamp.now(tz="Asia/Shanghai")
    return now < pd.Timestamp(today + " 09:15", tz="Asia/Shanghai")


def refresh_all(symbols: list[str], through: str, workers: int = int(os.environ.get("FETCH_WORKERS") or 4), snapshot: pd.DataFrame | None = None,
                prev_day: str | None = None) -> dict:
    counts: dict[str, int] = {}
    todo = list(symbols)
    if snapshot is not None and prev_day:
        rows = {r["symbol"]: r for r in snapshot.to_dict("records")}
        todo = []
        for s in symbols:
            try:
                r = apply_snapshot(s, rows.get(s), through, prev_day)
            except Exception:  # noqa: BLE001
                r = "need_fetch"
            if r == "need_fetch":
                todo.append(s)
            else:
                counts[r] = counts.get(r, 0) + 1

    def one(s):
        for k in range(4):  # 指数退避重试
            try:
                return refresh_symbol(s, through)
            except Exception:  # noqa: BLE001
                time.sleep(2 ** k)
        return "fail"

    symbols = todo

    with ThreadPoolExecutor(workers) as pool:
        for r in pool.map(one, symbols):
            counts[r] = counts.get(r, 0) + 1
    return counts


def load_universe() -> pd.DataFrame:
    from ashare_tactics.download import fetch_universe
    try:
        u = fetch_universe()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        u.to_parquet(STATE_DIR / "universe_live.parquet", index=False)
    except Exception:  # noqa: BLE001
        u = pd.read_parquet(STATE_DIR / "universe_live.parquet")
    return u


# ---------- 信号 ----------
@dataclass
class SymView:
    symbol: str
    name: str
    is_st: bool
    book: _Book
    ma5: np.ndarray
    ma20: np.ndarray


def load_view(symbol: str, name: str, through: str) -> SymView | None:
    path = CACHE / "bars" / f"{symbol}.parquet"
    if not path.exists():
        return None
    f = pd.read_parquet(path)
    f["date"] = pd.to_datetime(f["date"])
    f = f[(f["date"] <= pd.Timestamp(through)) & (f["volume"] > 0)].sort_values("date").reset_index(drop=True)
    if len(f) < 80:
        return None
    a = arrays_from_frame(f)
    is_st = "ST" in str(name).upper()
    b = _Book(symbol, a["dates"], a["raw_open"], a["raw_high"], a["raw_low"], a["raw_close"], a["volume"], a["amount"], a["factor"], is_st)
    return SymView(symbol, name, is_st, b, sma(b.q_close, 5), sma(b.q_close, 20))


def entry_signal(v: SymView, through: str) -> bool:
    """回测规则：信号月内，日线收盘由下而上突破 MA5 → 次日开盘买。"""
    b = v.book
    i = b.n - 1
    if str(pd.Timestamp(b.dates[i]).date()) != through or i < 30:
        return False  # 停牌：当日无K线
    months = cci_signal_months(b)
    if not months:
        return False
    if pd.Timestamp(b.dates[i]).to_period("M") not in months:
        return False
    return bool(np.isfinite(v.ma5[i]) and np.isfinite(v.ma5[i - 1]) and b.q_close[i] > v.ma5[i] and b.q_close[i - 1] <= v.ma5[i - 1])


def exit_reason(v: SymView, entry_date: str, through: str) -> str | None:
    """回测 _exit_ma：收盘跌破 MA20，或持有满 40 个交易日 → 次日开盘卖。"""
    b = v.book
    i = b.n - 1
    if str(pd.Timestamp(b.dates[i]).date()) != through:
        return None
    c, m = b.q_close[i], v.ma20[i]
    if np.isfinite(m) and c < m and not np.isclose(c, m, rtol=1e-9, atol=1e-6):
        return "收盘跌破20日均线"
    e = int(np.searchsorted(b.dates, np.datetime64(entry_date, "D")))
    if i >= e + MAX_HOLD - 1:
        return f"持有满{MAX_HOLD}个交易日"
    return None


# ---------- 状态 ----------
def new_state() -> dict:
    return {"start_date": START_DATE, "initial_cash": INITIAL_CASH, "cash": INITIAL_CASH,
            "positions": {}, "pending": None, "history": [], "trades": []}


def load_state(d: Path) -> dict:
    p = d / "state.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else new_state()


def save_state(d: Path, s: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(d / "state.json")


def bar_on(v: SymView | None, day: str):
    if v is None:
        return None
    idx = np.where(v.book.dates == np.datetime64(day, "D"))[0]
    return int(idx[0]) if len(idx) else None


def apply_fills(state: dict, views: dict, fill_day: str) -> list[str]:
    """按 fill_day 实际开盘价回填上一张订单表。先卖后买（与回测一致）。"""
    notes: list[str] = []
    pend = state.get("pending")
    if not pend or pend["date"] != fill_day:
        return notes
    carry_sells = []
    for o in pend["sells"]:
        v = views.get(o["symbol"]); i = bar_on(v, fill_day); pos = state["positions"].get(o["symbol"])
        if pos is None:
            continue
        if i is None or i == 0:
            notes.append(f"{o['symbol']} 停牌，卖出顺延"); carry_sells.append(o); continue
        b = v.book
        if is_limit_down_open(b.raw_open[i], b.raw_close[i - 1], b.band_at(i)):
            notes.append(f"{o['symbol']} 跌停开盘，卖出顺延"); carry_sells.append(o); continue
        cash_in = sell_cash_in(pos["shares"], float(b.raw_open[i]), pos["factor_entry"], float(b.factor[i]), np.datetime64(fill_day))
        state["cash"] += cash_in
        state["trades"].append({"date": fill_day, "side": "卖", "symbol": o["symbol"], "name": pos["name"], "shares": pos["shares"],
                                "price": float(b.raw_open[i]), "cash": cash_in, "pnl": cash_in - pos["cost"], "reason": o["reason"]})
        del state["positions"][o["symbol"]]
    for o in pend["buys"]:
        if len(state["positions"]) >= MAX_POS or o["symbol"] in state["positions"]:
            continue
        v = views.get(o["symbol"]); i = bar_on(v, fill_day)
        if i is None or i == 0:
            notes.append(f"{o['symbol']} 停牌，未买入"); continue
        b = v.book
        if is_limit_up_open(b.raw_open[i], b.raw_close[i - 1], b.band_at(i)):
            notes.append(f"{o['symbol']} 涨停开盘，未买入"); continue
        px = float(b.raw_open[i]); sh = o["shares"]
        while sh >= 100 and buy_cash_out(sh, px) > state["cash"] + 1e-6:
            sh -= 100
        if sh < 100:
            notes.append(f"{o['symbol']} 现金不足，未买入"); continue
        cost = buy_cash_out(sh, px)
        state["cash"] -= cost
        state["positions"][o["symbol"]] = {"name": o["name"], "shares": sh, "entry_date": fill_day, "entry_price": px,
                                           "factor_entry": float(b.factor[i]), "cost": cost}
        state["trades"].append({"date": fill_day, "side": "买", "symbol": o["symbol"], "name": o["name"], "shares": sh,
                                "price": px, "cash": -cost, "pnl": 0.0, "reason": "信号"})
    state["pending"] = {"date": None, "sells": carry_sells, "buys": []}
    return notes


def mark(pos: dict, v: SymView | None) -> float:
    if v is None:
        return pos["shares"] * pos["entry_price"]
    b = v.book
    return pos["shares"] * float(b.raw_close[-1]) * pos["factor_entry"] / float(b.factor[-1])


def build_orders(state: dict, views: dict, names: dict, through: str, today: str) -> dict:
    sells, sold = [], set()
    for o in (state.get("pending") or {}).get("sells", []):  # 顺延卖单
        sells.append({**o, "reason": o["reason"] + "（顺延）"}); sold.add(o["symbol"])
    for sym, pos in state["positions"].items():
        if sym in sold:
            continue
        r = exit_reason(views[sym], pos["entry_date"], through) if views.get(sym) else None
        if r:
            sells.append({"symbol": sym, "name": pos["name"], "shares": pos["shares"], "reason": r}); sold.add(sym)
    equity = state["cash"] + sum(mark(p, views.get(s)) for s, p in state["positions"].items())
    est_cash = state["cash"]
    for o in sells:
        v = views.get(o["symbol"]); p = state["positions"][o["symbol"]]
        est_cash += sell_cash_in(p["shares"], float(v.book.raw_close[-1]), p["factor_entry"], float(v.book.factor[-1]), np.datetime64(today)) if v else 0
    slots = MAX_POS - (len(state["positions"]) - len(sold))
    cands = []
    for sym, v in views.items():
        if sym in state["positions"] or v.is_st:  # 剔除ST
            continue
        if entry_signal(v, through):
            cands.append((-_liquidity(v.book.amount, v.book.n - 1), sym))
    cands.sort()
    buys = []
    for _, sym in cands:
        if len(buys) >= slots:
            break
        v = views[sym]; ref = float(v.book.raw_close[-1])
        sh = size_shares(equity, est_cash, ref, WEIGHT)
        if sh < 100:
            continue
        cost = buy_cash_out(sh, ref); est_cash -= cost
        buys.append({"symbol": sym, "name": v.name, "shares": sh, "ref": ref, "amount": cost})
    return {"date": today, "sells": sells, "buys": buys, "equity": equity, "signals": len(cands)}


def render(state: dict, orders: dict, views: dict, through: str, notes: list[str], dry: bool) -> str:
    eq = orders["equity"]; ini = state["initial_cash"]
    L = [f"A股战法12（CCI月线）纸面交易单 — {orders['date']} 开盘执行" + ("【演练】" if dry else ""),
         f"数据截至：{through} 收盘；纸面账户起始 {ini:,.0f} 元，起始日 {state['start_date']}",
         "说明：纸面交易，未连接券商、未下单。假设按当日开盘价成交；实际成交在下一交易日用真实开盘价回填（涨停开盘不买、跌停开盘不卖顺延、停牌顺延）。",
         "费用：佣金万2.5（最低5元）、卖出印花税0.05%、滑点0.1%；100股整数倍；单只目标仓位为权益10%，最多10只；剔除ST。", ""]
    L.append(f"一、买入（候选信号 {orders['signals']} 只，按20日成交额排序取前{len(orders['buys'])}只）")
    if orders["buys"]:
        L.append("代码 | 名称 | 股数 | 参考价(昨收) | 预计金额(含费)")
        L += [f"{code6(o['symbol'])} | {o['name']} | {o['shares']} | {o['ref']:.2f} | {o['amount']:,.2f}" for o in orders["buys"]]
    else:
        L.append("无")
    L += ["", "二、卖出"]
    if orders["sells"]:
        L.append("代码 | 名称 | 股数 | 原因")
        L += [f"{code6(o['symbol'])} | {o['name']} | {o['shares']} | {o['reason']}" for o in orders["sells"]]
    else:
        L.append("无")
    L += ["", "三、当前持仓（按昨收估值）"]
    if state["positions"]:
        L.append("代码 | 名称 | 股数 | 买入日 | 买入价 | 市值 | 浮动盈亏")
        for s, p in state["positions"].items():
            mv = mark(p, views.get(s))
            L.append(f"{code6(s)} | {p['name']} | {p['shares']} | {p['entry_date']} | {p['entry_price']:.2f} | {mv:,.2f} | {mv - p['cost']:+,.2f}")
    else:
        L.append("空仓")
    L += ["", f"现金：{state['cash']:,.2f} 元", f"总权益：{eq:,.2f} 元",
          f"累计盈亏：{eq - ini:+,.2f} 元（{(eq / ini - 1) * 100:+.2f}%）"]
    if notes:
        L += ["", "上一单回填备注："] + [f"- {n}" for n in notes]
    return "\n".join(L) + "\n"


# ---------- 幂等 / 质量闸门 ----------
class QualityError(RuntimeError):
    """数据覆盖不足：宁可不出单也不出错单。"""


def sent_marker(state_dir: Path, today: str) -> Path:
    return state_dir / "sent" / f"{today}.done"


def already_sent(state_dir: Path, today: str) -> bool:
    return sent_marker(state_dir, today).exists()


def mark_sent(state_dir: Path, today: str, info: str = "") -> None:
    p = sent_marker(state_dir, today)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(info or "sent", "utf-8")


def quality_check(n_universe: int, views: dict, through: str, positions: dict, min_cov: float, min_uni: int) -> str:
    fresh = sum(1 for v in views.values() if str(pd.Timestamp(v.book.dates[-1]).date()) == through)
    cov = fresh / n_universe if n_universe else 0.0
    msg = f"universe={n_universe} views={len(views)} 截至{through}有K线={fresh} 覆盖率={cov:.1%}"
    if n_universe < min_uni:
        raise QualityError(f"股票池过小（<{min_uni}）：{msg}")
    if cov < min_cov:
        raise QualityError(f"数据覆盖率低于{min_cov:.0%}：{msg}")
    missing = [s for s in positions if s not in views]
    if missing:
        raise QualityError(f"持仓缺少行情：{missing}；{msg}")
    return msg


# ---------- 主流程 ----------
def run(today: str, state_dir: Path, refresh: bool, send: bool, dry: bool, test_subject: bool,
        cfg: dict | None = None) -> str:
    cfg = cfg or _CFG
    if send and cfg["dry_run"]:
        print("DRY_RUN=1：不发邮件"); send = False
    if already_sent(state_dir, today):
        print(f"{today} 订单表已发送过（{sent_marker(state_dir, today)}），跳过"); return ""
    cal = trade_dates()
    if today not in cal:
        print(f"{today} 非交易日，跳过"); return ""
    through = prev_trade_date(cal, today)
    state = load_state(state_dir)
    if today < state["start_date"] and not dry:
        print("未到起始日"); return ""
    out = state_dir / "sheets" / f"order_sheet_{today}.txt"
    if out.exists() and (state.get("pending") or {}).get("date") == today:
        # 同日重跑（上次生成成功但未发出）：不重复回填/记账，只重发已生成的订单表
        text = out.read_text("utf-8")
        print(f"复用已生成的订单表 {out.name}")
        _deliver(text, today, state_dir, send, test_subject, cfg)
        return text
    uni = load_universe()
    names = dict(zip(uni["symbol"], uni["name"]))
    syms = list(names) + [s for s in state["positions"] if s not in names]
    if refresh:
        snap = None
        if snapshot_usable(today):
            try:
                snap = fetch_snapshot(); print("snapshot", snap.attrs.get("source"), len(snap), flush=True)
            except Exception as e:  # noqa: BLE001
                print("snapshot 失败，改逐只增量：", repr(e)[:200], flush=True)
        print("refresh", refresh_all(syms, through, snapshot=snap, prev_day=prev_trade_date(cal, through)), flush=True)
    views = {}
    for s in syms:
        v = load_view(s, names.get(s, state["positions"].get(s, {}).get("name", s)), through)
        if v is not None:
            views[s] = v
    print("quality", quality_check(len(names), views, through, state["positions"], cfg["min_coverage"], cfg["min_universe"]), flush=True)
    notes = apply_fills(state, views, through)
    orders = build_orders(state, views, names, through, today)
    text = render(state, orders, views, through, notes, dry)
    state["pending"] = {"date": today, "sells": orders["sells"], "buys": orders["buys"]}
    state["history"].append({"date": through, "equity": orders["equity"], "cash": state["cash"]})
    save_state(state_dir, state)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, "utf-8")
    _deliver(text, today, state_dir, send, test_subject, cfg)
    return text


def _deliver(text: str, today: str, state_dir: Path, send: bool, test_subject: bool, cfg: dict) -> None:
    if cfg["quiet"]:  # 公开仓库的 Actions 日志对外可见：不打印持仓/订单明细
        print(f"订单表已生成（{len(text.splitlines())} 行），日志中不打印明细")
    else:
        print(text)
    if send:
        from mailer import send_mail
        subj = ("【测试】" if test_subject else "") + f"【A股战法12纸面】{today} 订单表"
        r = send_mail(subj, text)
        print("email:", r)
        if r == "已发送":
            mark_sent(state_dir, today, r)
        else:
            raise RuntimeError(f"邮件未发送：{r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d"))
    ap.add_argument("--state-dir", type=Path, default=STATE_DIR)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--test-subject", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="演练：生成订单表但绝不发邮件")
    a = ap.parse_args()
    cfg = dict(_CFG)
    if a.dry_run:
        cfg["dry_run"] = True; a.dry = True
    try:
        run(a.today, a.state_dir, not a.no_refresh, a.send, a.dry, a.test_subject, cfg)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        print("FAILED", msg, flush=True)
        if a.send and not cfg["dry_run"]:
            from mailer import send_mail
            print("failure email:", send_mail(f"【A股战法12纸面】{a.today} 未出单（失败通知）",
                                              f"今日订单表未生成，以免发出错误订单。\n原因：{msg}\n请检查 GitHub Actions 日志。"))
        sys.exit(1)


if __name__ == "__main__":
    main()
