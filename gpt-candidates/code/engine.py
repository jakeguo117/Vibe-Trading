"""COMP 研究版引擎（研究用，不下单）。规格：spec_v1.1.md。价格空间：Yahoo 拆股调整价(adj)；原始价 = adj × F[t]（F 为 t 之后拆股连乘）。
全部筛选只用 t 及以前数据；订单最早 t+1。窗口含端点；缺失 bar 令窗口无效（不插值）。"""
from __future__ import annotations
import math, json, numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from pathlib import Path
G = Path("/workspace/gpt-candidates"); M = G / "data/mat"
WARM_START = "2008-01-02"

# ---------------- 指标 ----------------
def shift(X, k):
    if k == 0: return X
    Y = np.full_like(X, np.nan); Y[k:] = X[:-k]; return Y

def roll(X, w, fn):
    """[t-w+1, t] 含端点；任一缺失 -> NaN。"""
    Y = np.full_like(X, np.nan)
    if len(X) >= w: Y[w - 1:] = getattr(swv(X, w, axis=0), fn)(axis=-1)
    return Y

def ema_sma_init(X, n):
    """alpha=2/(n+1)；该证券第 n 个有效观测处以前 n 个有效值 SMA 初始化；缺失日不更新、输出 NaN。"""
    a = 2.0 / (n + 1); T, N = X.shape; out = np.full((T, N), np.nan)
    cnt = np.zeros(N); s = np.zeros(N); e = np.full(N, np.nan)
    for t in range(T):
        x = X[t]; ok = np.isfinite(x); cnt += ok
        s += np.where(ok & (cnt <= n), np.nan_to_num(x), 0.0)
        e = np.where(ok & (cnt == n), s / n, e)
        e = np.where(ok & (cnt > n), a * np.nan_to_num(x) + (1 - a) * e, e)
        out[t] = np.where(ok & (cnt >= n), e, np.nan)
    return out

def atr_wilder(H, L, C, n=14):
    """TR 用上一有效收盘；首 n 个 TR 简单均值，之后 Wilder 递推。"""
    T, N = C.shape; out = np.full((T, N), np.nan); pc = np.full(N, np.nan)
    cnt = np.zeros(N); s = np.zeros(N); a = np.full(N, np.nan)
    for t in range(T):
        h, l, c = H[t], L[t], C[t]; ok = np.isfinite(c) & np.isfinite(h) & np.isfinite(l)
        tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
        okt = ok & np.isfinite(pc); cnt += okt
        s += np.where(okt & (cnt <= n), np.nan_to_num(tr), 0.0)
        a = np.where(okt & (cnt == n), s / n, a)
        a = np.where(okt & (cnt > n), (a * (n - 1) + np.nan_to_num(tr)) / n, a)
        out[t] = np.where(okt & (cnt >= n), a, np.nan)
        pc = np.where(ok, c, pc)
    return out

def valid_count(C):
    return np.cumsum(np.isfinite(C), axis=0)

# ---------------- 数据 ----------------
class Data:
    def __init__(self, start=WARM_START, mat=M, arrays=None):
        if arrays is not None:   # 测试用合成数据
            self.__dict__.update(arrays); self._post(); return
        cal = pd.DatetimeIndex(np.load(mat / "calendar.npy")); i0 = int(np.searchsorted(cal, pd.Timestamp(start)))
        self.cal = cal[i0:]
        L = lambda k: np.load(mat / f"{k}.npy", mmap_mode="r")[i0:].astype(np.float64)
        self.O, self.H, self.L, self.C, self.V, self.DIV, self.SP, self.F = (L(k) for k in ("open", "high", "low", "close", "vol", "div", "split", "F"))
        nm = pd.read_csv(mat / "names.csv", keep_default_na=False); self.names = nm.sym.values.astype(str)
        self.is_etf = nm.is_etf.values.astype(bool); self._post()
    def _post(self):
        self.T, self.N = self.C.shape
        self.qqq = int(np.flatnonzero(self.names == "QQQ")[0]) if "QQQ" in list(self.names) else None
        self.spy = int(np.flatnonzero(self.names == "SPY")[0]) if "SPY" in list(self.names) else None
        self.DIV = np.nan_to_num(self.DIV); self.SP = np.nan_to_num(self.SP)
        import re
        pref = np.array([bool(re.search(r"-P[A-Z]?$", s)) for s in self.names])   # D12b：优先股后缀
        self.is_etf = self.is_etf | pref
        fin = np.isfinite(self.C); self.lastvalid = np.where(fin.any(0), self.T - 1 - np.argmax(fin[::-1], axis=0), -1)    # 不进股票池（沿用 is_etf 掩码表示“非普通股”）

# ---------------- 特征（全部族共用） ----------------
class Feat:
    def __init__(self, d: Data, emas=(9, 21, 50, 18, 24)):
        self.d = d; C, H, L, V, F = d.C, d.H, d.L, d.V, d.F
        self.E = {n: ema_sma_init(C, n) for n in emas}
        self.ATR = atr_wilder(H, L, C, 14)
        rawC = C * F; dvol = C * V            # 原始收盘×原始成交量 = 调整收盘×调整量（拆股不变）
        self.rawC = rawC; self.ADV20 = roll(dvol, 20, "mean"); self.VOL20 = roll(V, 20, "mean")
        self.HIST = valid_count(C); self.R63 = C / shift(C, 63) - 1
        stock = ~d.is_etf
        self.U = stock[None, :] & np.isfinite(C) & (rawC >= 5.0) & (self.ADV20 >= 20e6) & (self.HIST >= 252)
        self.U_reason = dict(price=(np.isfinite(C) & (rawC < 5.0) & stock).sum(1), adv=(np.isfinite(C) & ~(self.ADV20 >= 20e6) & stock).sum(1),
                             hist=(np.isfinite(C) & (self.HIST < 252) & stock).sum(1), bar=(np.isfinite(C) & stock).sum(1))
        # 63 日收益前 10%：在 U(t) 中按收益降序、ID(列序)升序，取 ceil(0.1N)
        T, N = C.shape; self.TOP = np.zeros((T, N), bool)
        for t in range(T):
            js = np.flatnonzero(self.U[t] & np.isfinite(self.R63[t]))
            if len(js) == 0: continue
            k = math.ceil(0.1 * len(js)); o = np.lexsort((js, -self.R63[t, js])); self.TOP[t, js[o[:k]]] = True
        if d.qqq is not None:
            q = C[:, d.qqq]; sma = roll(q[:, None], 200, "mean")[:, 0]; self.ENV = (q > sma) & np.isfinite(sma)
        else: self.ENV = np.ones(T, bool)
        E9, E21, E50 = self.E[9], self.E[21], self.E[50]
        self.TREND = (C > E50) & (E9 > E21) & (E21 > E50) & (E21 > shift(E21, 5))

# ---------------- 信号族 ----------------
def screen(f, env=True, mom=True):
    S = f.U & f.TREND
    if mom: S &= f.TOP
    if env: S &= f.ENV[:, None]
    return S

def tick_up(Xadj, F):  # 原始价 +0.01 再换回调整价
    return (Xadj * F + 0.01) / F

def sig_pb(f, env=True, mom=True, pullback=True, ema_pb=21):
    d = f.d; H, L, C, O = d.H, d.L, d.C, d.O; Epb = f.E[ema_pb]
    RM20 = roll(H, 20, "max")
    best = np.full_like(H, -np.inf); kb = np.zeros(H.shape, np.int16); nanw = np.zeros(H.shape, bool)
    for k in range(2, 11):
        cand = shift(H, k); nanw |= ~np.isfinite(cand); upd = cand > best; best = np.where(upd, cand, best); kb = np.where(upd, k, kb)
    hok = np.zeros(H.shape, bool); mL = np.full_like(L, np.nan); mLk = L.copy()
    for k in range(2, 11):
        mLk = np.minimum(mLk, shift(L, k - 1))          # min L[t-k+1, t] = [h+1, t]
        sel = kb == k
        hok |= sel & (shift(H, k) == shift(RM20, k))
        mL = np.where(sel, mLk, mL)
    ok = screen(f, env, mom) & hok & ~nanw & np.isfinite(mL)
    if pullback: ok &= (mL <= 1.01 * Epb) & (mL >= 0.95 * Epb)
    ok &= (C > Epb) & (C > O) & (C > shift(C, 1))
    B = tick_up(H, d.F); S = mL - 0.1 * f.ATR
    return ok & np.isfinite(S) & np.isfinite(B), B, S, (0.01, 0.05)

def sig_vc(f):
    d = f.d; H, L, C, V = d.H, d.L, d.C, d.V
    mx5, mn5, v5 = roll(H, 5, "max"), roll(L, 5, "min"), roll(V, 5, "mean")
    r = mx5 / mn5 - 1; rC, rB, rA = r, shift(r, 5), shift(r, 10); vC, vA = v5, shift(v5, 10)
    ok = screen(f) & (rA > rB) & (rB > rC) & (rC <= 0.06) & (vC <= 0.7 * vA) & (C >= 0.95 * roll(H, 20, "max"))
    B = tick_up(mx5, d.F); S = mn5 - 0.1 * f.ATR
    return ok & np.isfinite(S), B, S, (0.01, 0.05)

def sig_htf(f):
    d = f.d; H, L, C, V = d.H, d.L, d.C, d.V
    mn40, v20 = roll(L, 40, "min"), roll(V, 20, "mean")
    best = np.full_like(H, -np.inf); kb = np.zeros(H.shape, np.int16); nanw = np.zeros(H.shape, bool)
    for k in range(5, 16):
        cand = shift(H, k); nanw |= ~np.isfinite(cand); upd = cand > best; best = np.where(upd, cand, best); kb = np.where(upd, k, kb)
    Hh = best; pole = np.zeros(H.shape, bool); fmx = np.full_like(H, np.nan); fmn = np.full_like(H, np.nan); fv = np.full_like(H, np.nan); vpre = np.full_like(H, np.nan)
    rmx, rmn, rsv = H.copy(), L.copy(), V.copy()
    for k in range(2, 16):                       # 旗面 [h+1, t] = [t-k+1, t]
        rmx = np.maximum(rmx, shift(H, k - 1)); rmn = np.minimum(rmn, shift(L, k - 1)); rsv = rsv + shift(V, k - 1)
        if k < 5: continue
        sel = kb == k
        pole |= sel & (Hh / shift(mn40, k) - 1 >= 0.90)
        fmx = np.where(sel, rmx, fmx); fmn = np.where(sel, rmn, fmn); fv = np.where(sel, rsv / k, fv); vpre = np.where(sel, shift(v20, k), vpre)
    ok = screen(f) & pole & ~nanw & (fmx <= 1.02 * Hh) & (fmn >= 0.80 * Hh) & (C >= 0.90 * Hh) & (fv <= 0.7 * vpre)
    B = tick_up(fmx, d.F); S = fmn - 0.1 * f.ATR
    return ok & np.isfinite(S), B, S, (0.01, 0.10)

def sig_bo20(f):
    d = f.d; H, L = d.H, d.L
    mx20 = roll(H, 20, "max"); mn5 = roll(L, 5, "min")
    # D: 过去20日=[t-19,t]；要求 C(t) 未已越过 B 不另设；同筛选
    ok = screen(f) & np.isfinite(mx20) & np.isfinite(mn5)
    B = tick_up(mx20, d.F); S = mn5 - 0.1 * f.ATR
    return ok & np.isfinite(S), B, S, (0.01, 0.05)

def top_orders(f, ok, B, S, rng):
    """每个信号日 t：预期风险在范围内的候选中，按 63日收益降序、20日美元量降序、ID升序，取第一。返回 dict t->(j,B,S)。"""
    lo, hi = rng; risk = (B - S) / B; ok = ok & (risk >= lo) & (risk <= hi)
    out = {}; ncand = ok.sum(1)
    for t in np.flatnonzero(ncand):
        js = np.flatnonzero(ok[t]); o = np.lexsort((js, -f.ADV20[t, js], -f.R63[t, js])); j = js[o[0]]
        out[int(t)] = (int(j), float(B[t, j]), float(S[t, j]), int(len(js)))
    return out, ncand

def delay_orders(f, orders, env=True, mom=True):
    """PB 延迟版：t 的第一名，t+1 收盘重新验证共同筛选与风险（B/S 不变），t+2 才挂单。键改为 t+1（sim 中 lag=1 统一）。"""
    S = screen(f, env, mom); out = {}
    for t, (j, B, Sx, n) in orders.items():
        if t + 1 < f.d.T and S[t + 1, j]: out[t + 1] = (j, B, Sx, n)
    return out

# ---------------- 成交仿真（单仓） ----------------
def fee(sh, px):
    return max(1.0, 0.005 * sh) + 1e-4 * sh * px

def simulate(f, orders, rng, slip_bp, i0, i1, risk_frac=0.005, cap0=16000.0, lag_flat=0, tag="", pay_lag=15):
    """orders: 信号日 t -> (j,B_adj,S_adj,ncand)；在 t+1 执行。lag_flat=1 时额外要求 t-1 收盘也空仓（延迟版：键已是 t+1）。
    订单优先：开盘≤S 先止损；否则执行前一日安排的退出；否则盘中止损。返回 trades, equity, order_log。"""
    d = f.d; O, H, L, C, F, SP, DIV = d.O, d.H, d.L, d.C, d.F, d.SP, d.DIV; E9 = f.E[9]; sl = slip_bp * 1e-4; lo, hi = rng
    cash = cap0; recv = []; pos = None; trades = []; rows = []; flat = {}; olog = []
    for x in range(i0, i1 + 1):
        paid = sum(a for (p, a) in recv if p <= x); recv = [(p, a) for (p, a) in recv if p > x]; cash += paid
        exited_today = False
        if pos is not None:
            j = pos["j"]
            if SP[x, j] > 0 and np.isfinite(C[x, j]):          # 拆股：股数随 F 变化，非整数部分按当日收盘换现金
                raw_new = pos["ash"] / F[x, j]; fl = math.floor(raw_new + 1e-9)
                if raw_new - fl > 1e-9:
                    cash += (raw_new - fl) * C[x, j] * F[x, j]; pos["cil_cash"] += (raw_new - fl) * C[x, j] * F[x, j]; pos["ash"] = fl * F[x, j]
            if DIV[x, j] > 0 and pos["entry_day"] < x:          # 除息日前一收盘持有 -> 应收
                amt = pos["ash"] * DIV[x, j]; recv.append((x + pay_lag, amt)); pos["div"] += amt
            if np.isfinite(O[x, j]) and np.isfinite(L[x, j]):
                o, l = O[x, j], L[x, j]; px = None
                if o <= pos["S"]: px, why = o * (1 - sl), "stop_gap_open"
                elif pos["sched"]: px, why = o * (1 - sl), pos["sched"]
                elif l <= pos["S"]: px, why = pos["S"] * (1 - sl), "stop"
                if px is not None:
                    cash = _close(pos, x, px, why, F, cash, trades, sl); pos = None; exited_today = True
            elif x > d.lastvalid[j]:                             # 证券数据终止：无官方清算现金数据 -> 末收盘计价并标记数据阻断
                cash = _close(pos, x, pos["lastc"] * (1 - sl), "delisted_no_settlement_DATA_BLOCKED", F, cash, trades, sl, fx=d.lastvalid[j]); pos = None; exited_today = True
            else:
                pos["halt_days"] += 1                            # 停牌/缺 bar：不成交、不插值
        sd = x - 1
        if pos is None and not exited_today and sd in orders and sd >= i0 and flat.get(sd, False) and (lag_flat == 0 or flat.get(sd - 1, True)):
            j, B, S, nc = orders[sd]; how = "no_bar"
            if np.isfinite(O[x, j]) and np.isfinite(H[x, j]) and np.isfinite(L[x, j]):
                o, h, l = O[x, j], H[x, j], L[x, j]; fill = None
                if o >= B:
                    if o <= B * 1.02: fill, how = o * (1 + sl), "open_gap"
                    else: how = "cancel_gap"
                elif h >= B: fill, how = B * (1 + sl), "intraday"
                else: how = "not_triggered"
                if fill is not None and not (lo <= (fill - S) / fill <= hi): fill, how = None, "cancel_risk"
                if fill is not None:
                    eq = rows[-1]["equity"] if rows else cap0
                    fr = fill * F[x, j]; sr = S * F[x, j]
                    vcap = 0.01 * f.VOL20[sd, j] / F[x, j] if np.isfinite(f.VOL20[sd, j]) else 0.0   # 拆股调整量 -> x 日原始股数
                    q = math.floor(min(risk_frac * eq / (fr - sr), cash / fr, vcap))
                    while q > 0 and q * fr + fee(q, fr) > cash: q -= 1
                    if q > 0:
                        fe = fee(q, fr); cash -= q * fr + fe
                        pos = dict(j=j, sig_day=sd, entry_day=x, B=B, S=S, ash=q * F[x, j], q0=q, fill_raw=fr, S_raw=sr, fee_in=fe, slip_in=q * fr * sl / (1 + sl),
                                   how=how, R0=q * (fr - sr), div=0.0, days=0, sched=None, halt_days=0, ncand=nc, cil_cash=0.0, amb=int(how == "intraday" and l <= S), lastc=C[x, j])
                        if l <= S:                               # 同 bar 入场且触及止损：保守当日止损
                            cash = _close(pos, x, S * (1 - sl), "stop_same_bar", F, cash, trades, sl); pos = None; exited_today = True
                    else: how = "qty_zero"
            olog.append((sd, x, j, how))
        if pos is not None:
            j = pos["j"]; pos["days"] += 1
            if np.isfinite(C[x, j]):
                pos["lastc"] = C[x, j]
                if np.isfinite(E9[x, j]) and C[x, j] < E9[x, j]: pos["sched"] = "close_below_ema9"
            if pos["days"] >= 20 and not pos["sched"]: pos["sched"] = "time_20d"
            mv = pos["ash"] * pos["lastc"]
        else: mv = 0.0
        rv = sum(a for (_, a) in recv); eqv = cash + rv + mv
        rows.append(dict(date=d.cal[x], cash=cash, div_receivable=rv, position_mv=mv, equity=eqv, exposure=mv / eqv if eqv > 0 else 0.0, holding=d.names[pos["j"]] if pos else ""))
        flat[x] = pos is None
    if pos is not None:                                          # 样本结束：按末收盘减卖出成本强平，单列
        cash = _close(pos, i1, pos["lastc"] * (1 - sl), "end_of_sample", F, cash, trades, sl)
        rv = sum(a for (_, a) in recv); rows[-1].update(cash=cash, position_mv=0.0, equity=cash + rv, exposure=0.0, holding="")
    for t in trades:
        t["tag"] = tag; t["sym"] = d.names[t["j"]]
        for k in ("sig", "entry", "exit"): t[f"{k}_date"] = str(d.cal[t[f"{k}_day"]].date())
    return trades, pd.DataFrame(rows), olog

def _close(pos, x, px, why, F, cash, trades, sl, fx=None):
    j = pos["j"]; fx = x if fx is None else fx; sh = pos["ash"] / F[fx, j]; pr = px * F[fx, j]; fe = fee(sh, pr)
    cash += sh * pr - fe
    pnl = sh * pr - fe - pos["q0"] * pos["fill_raw"] - pos["fee_in"] + pos["div"] + pos["cil_cash"]
    trades.append(dict(j=j, sig_day=pos["sig_day"], entry_day=pos["entry_day"], exit_day=x, B_adj=pos["B"], S_adj=pos["S"], qty=pos["q0"], entry_px_raw=pos["fill_raw"],
                       stop_raw_at_entry=pos["S_raw"], exit_px_raw=pr, exit_shares_raw=sh, fees=pos["fee_in"] + fe, slippage_cost=pos["slip_in"] + sh * pr * sl / (1 - sl),
                       dividends=pos["div"], entry_type=pos["how"], exit_reason=why, pnl=pnl, R0=pos["R0"], R=pnl / pos["R0"], hold_days=max(pos["days"], 1),
                       ambiguous=pos["amb"], halt_days=pos["halt_days"], n_candidates=pos["ncand"]))
    return cash

# ---------------- 指标 ----------------
def metrics(eq: pd.DataFrame, trades: list, cap0=16000.0):
    e = eq.equity.values; r = np.r_[e[0] / cap0 - 1, e[1:] / e[:-1] - 1]
    yrs = len(e) / 252.0; cagr = (e[-1] / cap0) ** (1 / yrs) - 1 if yrs > 0 and e[-1] > 0 else np.nan
    peak = np.maximum.accumulate(np.r_[cap0, e])[1:]; mdd = (e / peak - 1).min()
    dd = math.sqrt(np.mean(np.minimum(r, 0) ** 2)); sortino = (r.mean() / dd * math.sqrt(252)) if dd > 0 else np.nan
    sd = r.std(ddof=1); sharpe = r.mean() / sd * math.sqrt(252) if sd > 0 else np.nan
    R = np.array([t["R"] for t in trades]); pnl = np.array([t["pnl"] for t in trades])
    gw, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    lose = (pnl <= 0).astype(int); mcl = 0; c = 0
    for v in lose: c = c + 1 if v else 0; mcl = max(mcl, c)
    srt = np.sort(pnl)[::-1]
    out = dict(start=str(eq.date.iloc[0].date()), end=str(eq.date.iloc[-1].date()), final_equity=e[-1], net_profit=e[-1] - cap0, CAGR=cagr, MaxDD=mdd, Sortino=sortino, Sharpe=sharpe,
               vol_ann=sd * math.sqrt(252), exposure_mean=eq.exposure.mean(), days_in_market=float((eq.position_mv > 0).mean()),
               n_trades=len(trades), mean_R=R.mean() if len(R) else np.nan, median_R=float(np.median(R)) if len(R) else np.nan, win_rate=(pnl > 0).mean() if len(pnl) else np.nan,
               PF=(gw / gl) if gl > 0 else (np.nan), max_consec_losses=mcl, avg_hold_days=np.mean([t["hold_days"] for t in trades]) if trades else np.nan,
               n_ambiguous=int(sum(t["ambiguous"] for t in trades)), n_gap_stop=int(sum(t["exit_reason"] == "stop_gap_open" for t in trades)),
               worst_R=R.min() if len(R) else np.nan, best_R=R.max() if len(R) else np.nan,
               net_profit_ex_top5=float(pnl.sum() - srt[:5].clip(min=0).sum()) if len(pnl) else np.nan,
               mean_R_ex_top5=float(np.mean(np.sort(R)[:-5])) if len(R) > 5 else np.nan,
               turnover_ann=(sum(t["qty"] * t["entry_px_raw"] for t in trades) / eq.equity.mean() / yrs) if yrs > 0 else np.nan,
               utilization_in_pos=float(eq.exposure[eq.position_mv > 0].mean()) if (eq.position_mv > 0).any() else 0.0,
               years_with_trades=len(set(t["exit_date"][:4] for t in trades)))
    return out

def block_bootstrap_meanR(trades, eq, B=2000, block=12, seed=20261024):
    if not trades: return np.nan
    months = pd.period_range(eq.date.iloc[0], eq.date.iloc[-1], freq="M"); m = len(months); idx = {p: i for i, p in enumerate(months)}
    s = np.zeros(m); n = np.zeros(m)
    for t in trades:
        i = idx[pd.Period(t["exit_date"], "M")]; s[i] += t["R"]; n[i] += 1
    rg = np.random.default_rng(seed); nb = math.ceil(m / block); res = []
    for _ in range(B):
        st = rg.integers(0, max(1, m - block + 1), nb); ii = np.concatenate([np.arange(a, min(a + block, m)) for a in st])[:m]
        res.append(s[ii].sum() / n[ii].sum() if n[ii].sum() > 0 else np.nan)
    return float(np.nanpercentile(res, 2.5))
