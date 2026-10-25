"""27 格 + 4 个 PB 敏感性 + 基准。每格独立检查点 out/cells/<exp>__<bp>bp.json；已存在则跳过（续跑只补缺格）。失败重试最多 2 次。
单命令复现：/workspace/.venv_yf/bin/python code/run_all.py"""
import sys, json, time, traceback, pickle, math, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/gpt-candidates/code"); import engine as E
G = E.G; OUT = G / "out"; CELLS = OUT / "cells"; CELLS.mkdir(parents=True, exist_ok=True); (OUT / "orders").mkdir(exist_ok=True)
LOG = open(G / "logs/run_all.log", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(map(str, a)); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
PERIODS = {"FULL": ("2009-01-01", "2026-10-23"), "DEV": ("2009-01-01", "2018-12-31"), "VAL": ("2019-01-01", "2022-12-31"), "TEST": ("2023-01-01", "2026-10-23")}
COSTS = (10, 30, 50)
EXPS = [  # (exp_id, 族, 说明, builder, rng 覆盖, 风险, lag)
    ("E01_PB_base", "PB", "PB 基线", lambda f: E.sig_pb(f), None, 0.005, 0),
    ("E02_VC_base", "VC", "VC 基线", lambda f: E.sig_vc(f), None, 0.005, 0),
    ("E03_HTF_base", "HTF", "HTF 基线", lambda f: E.sig_htf(f), None, 0.005, 0),
    ("E04_PB_noenv", "PB", "PB 去环境(QQQ>200SMA)", lambda f: E.sig_pb(f, env=False), None, 0.005, 0),
    ("E05_PB_nomom", "PB", "PB 去63日前10%动量", lambda f: E.sig_pb(f, mom=False), None, 0.005, 0),
    ("E06_PB_nopullback", "PB", "PB 去回踩条件(保留趋势与10日结构止损)", lambda f: E.sig_pb(f, pullback=False), None, 0.005, 0),
    ("E07_BO20", "BO", "普通20日突破对照", lambda f: E.sig_bo20(f), None, 0.005, 0),
    ("E08_PB_AV", "PB", "PB-AV 分钟AVWAP扩展", None, None, 0.005, 0),
    ("E09_PB_delay1", "PB", "PB 延迟一天(t+2挂单、t+1收盘复核)", "delay", None, 0.005, 1),
]
SENS = [
    ("S01_PB_risk025", "PB", "敏感性：每笔风险0.25%", "E01", 0.0025, 21),
    ("S02_PB_risk100", "PB", "敏感性：每笔风险1%", "E01", 0.01, 21),
    ("S03_PB_ema18", "PB", "敏感性：回踩均线21→18", None, 0.005, 18),
    ("S04_PB_ema24", "PB", "敏感性：回踩均线21→24", None, 0.005, 24),
]
t0 = time.time(); d = E.Data(); log("data", d.T, d.N, d.cal[0].date(), d.cal[-1].date())
f = E.Feat(d); log("feat done", round(time.time() - t0, 1), "s")
idx = lambda a, b: (int(np.searchsorted(d.cal, pd.Timestamp(a))), int(np.searchsorted(d.cal, pd.Timestamp(b), side="right")) - 1)
# 每日股票池覆盖与排除原因
cov = pd.DataFrame({"date": d.cal.strftime("%Y-%m-%d"), "stocks_with_bar": f.U_reason["bar"], "excl_price_lt5": f.U_reason["price"], "excl_adv_lt20m": f.U_reason["adv"],
                    "excl_hist_lt252": f.U_reason["hist"], "universe_U": f.U.sum(1), "top10pct": f.TOP.sum(1), "QQQ_env_on": f.ENV.astype(int)})
cov.to_csv(OUT / "audit/daily_universe_coverage.csv", index=False)
ORD = {}
def get_orders(eid, builder, ema=21):
    fn = OUT / "orders" / f"{eid}.pkl"
    if fn.exists(): return pickle.load(open(fn, "rb"))
    if builder == "delay": base = get_orders("E01_PB_base", lambda f: E.sig_pb(f)); o = E.delay_orders(f, base[0]); res = (o, base[1], base[2])
    else:
        ok, B, S, rng = builder(f); o, nc = E.top_orders(f, ok, B, S, rng); res = (o, rng, nc); del ok, B, S
    pickle.dump(res, open(fn, "wb")); return res
def run_cell(eid, fam, desc, orders, rng, risk, lag, bp):
    fn = CELLS / f"{eid}__{bp}bp.json"
    if fn.exists(): return json.load(open(fn))
    for attempt in range(3):
        try:
            res = dict(exp=eid, family=fam, desc=desc, cost_bp=bp, status="RAN", periods={})
            for pn, (a, b) in PERIODS.items():
                i0, i1 = idx(a, b); tr, eq, ol = E.simulate(f, orders, rng, bp, i0, i1, risk_frac=risk, lag_flat=lag, tag=f"{eid}|{bp}|{pn}")
                m = E.metrics(eq, tr); m["bootstrap_meanR_LB95"] = E.block_bootstrap_meanR(tr, eq)
                m["orders_seen"] = len(ol); m.update({f"orders_{k}": sum(1 for o in ol if o[3] == k) for k in ("cancel_gap", "cancel_risk", "not_triggered", "qty_zero", "no_bar")})
                res["periods"][pn] = m
                tdf = pd.DataFrame(tr); tdf.insert(0, "experiment", eid); tdf.insert(1, "cost_bp", bp); tdf.insert(2, "period", pn)
                tdf.to_csv(CELLS / f"{eid}__{bp}bp__{pn}_trades.csv", index=False)
                eq.to_csv(CELLS / f"{eid}__{bp}bp__{pn}_equity.csv", index=False)
            json.dump(res, open(fn, "w"), ensure_ascii=False, indent=1, default=float); return res
        except Exception as ex:
            log("FAIL", eid, bp, "attempt", attempt, repr(ex)); open(G / f"logs/error_{eid}_{bp}.txt", "a").write(traceback.format_exc())
    res = dict(exp=eid, family=fam, desc=desc, cost_bp=bp, status="FAILED_AFTER_2_RETRIES", recover="删除该格 json 后重跑 run_all.py"); json.dump(res, open(fn, "w"), ensure_ascii=False); return res
reg = []
for eid, fam, desc, builder, _, risk, lag in EXPS:
    if builder is None:
        for bp in COSTS:
            fn = CELLS / f"{eid}__{bp}bp.json"
            r = dict(exp=eid, family=fam, desc=desc, cost_bp=bp, status="DATA_BLOCKED", reason="无 2009–2026 全市场常规时段分钟线（box 上仅有 SPY/GLD 等少数 ETF 的 1 小时线，2019 起）；规格禁止用日线近似 AVWAP")
            json.dump(r, open(fn, "w"), ensure_ascii=False, indent=1); reg.append(r); log(eid, bp, "DATA_BLOCKED")
        continue
    t1 = time.time(); o, rng, nc = get_orders(eid, builder); log(eid, "orders", len(o), "signal-days", round(time.time() - t1, 1), "s")
    for bp in COSTS:
        r = run_cell(eid, fam, desc, o, rng, risk, lag, bp); reg.append(r); p = r.get("periods", {}).get("FULL", {})
        log(eid, bp, r["status"], "FULL n=", p.get("n_trades"), "CAGR=", round(p.get("CAGR", float("nan")), 4), "MDD=", round(p.get("MaxDD", float("nan")), 4))
# 敏感性
for eid, fam, desc, src, risk, ema in SENS:
    if src == "E01": o, rng, nc = get_orders("E01_PB_base", lambda f: E.sig_pb(f))
    else: o, rng, nc = get_orders(eid, (lambda e: (lambda f: E.sig_pb(f, ema_pb=e)))(ema))
    for bp in COSTS:
        r = run_cell(eid, fam, desc, o, rng, risk, 0, bp); r["sensitivity"] = True; reg.append(r); p = r.get("periods", {}).get("FULL", {})
        log(eid, bp, r["status"], "FULL n=", p.get("n_trades"), "CAGR=", round(p.get("CAGR", float("nan")), 4))
# 基准：SPY/QQQ 分红再投资持有（收盘+股息，无费用、现金利息0）
bm = {}
for nm, j in (("SPY", d.spy), ("QQQ", d.qqq)):
    c = d.C[:, j]; dv = d.DIV[:, j]; r = np.r_[0, (c[1:] + dv[1:]) / c[:-1] - 1]; bm[nm] = r
pd.DataFrame({"date": d.cal.strftime("%Y-%m-%d"), "SPY_tr_ret": bm["SPY"], "QQQ_tr_ret": bm["QQQ"], "QQQ_close": d.C[:, d.qqq], "QQQ_env_on": f.ENV.astype(int)}).to_csv(OUT / "benchmark_daily_returns.csv", index=False)
log("done", round(time.time() - t0, 1), "s")
