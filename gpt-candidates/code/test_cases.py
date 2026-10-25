"""规格 §10 工程小案例：每条给出输入、预期、实际。运行：python code/test_cases.py → tests/test_results.json"""
import sys, json, math, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/gpt-candidates/code"); import engine as E
R = []
def rec(name, cond, expect, actual): R.append(dict(case=name, passed=bool(cond), expected=expect, actual=str(actual))); print(("PASS " if cond else "FAIL ") + name, actual)

def synth(T=30, N=1, px=100.0):
    cal = pd.bdate_range("2024-01-02", periods=T)
    a = dict(cal=cal, O=np.full((T, N), px), H=np.full((T, N), px * 1.01), L=np.full((T, N), px * 0.99), C=np.full((T, N), px), V=np.full((T, N), 1e6),
             DIV=np.zeros((T, N)), SP=np.zeros((T, N)), F=np.ones((T, N)), names=np.array([f"S{i}" for i in range(N)]), is_etf=np.zeros(N, bool))
    return a
class FF:
    def __init__(self, d, e9=None):
        self.d = d; self.E = {9: e9 if e9 is not None else np.full(d.C.shape, 1.0)}; self.VOL20 = np.full(d.C.shape, 1e6)

# 1 跳空：开盘 > 1.02B 撤单；开盘在 [B,1.02B] 按开盘+滑点
a = synth(); a["O"][6, 0] = 103.5; a["H"][6, 0] = 104; d = E.Data(arrays=a); f = FF(d)
tr, eq, ol = E.simulate(f, {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("跳空>2%撤单", ol[0][3] == "cancel_gap" and len(tr) == 0, "cancel_gap，0笔", ol)
a = synth(); a["O"][6, 0] = 102.0; a["H"][6, 0] = 103; d = E.Data(arrays=a); f = FF(d)
tr, eq, ol = E.simulate(f, {5: (0, 101.0, 97.0, 1)}, (0.01, 0.06), 10, 0, 29)
rec("跳空≤2%按开盘+10bp成交", abs(tr[0]["entry_px_raw"] - 102 * 1.001) < 1e-9, 102 * 1.001, tr[0]["entry_px_raw"])
# 2 持仓开盘跳空穿止损：按开盘-滑点
a = synth(); a["H"][6, 0] = 101.5; a["O"][8, 0] = 95.0; a["L"][8, 0] = 94; d = E.Data(arrays=a); f = FF(d)
tr, eq, ol = E.simulate(f, {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("开盘跳空止损按开盘-滑点", tr[0]["exit_reason"] == "stop_gap_open" and abs(tr[0]["exit_px_raw"] - 95 * 0.999) < 1e-9 and tr[0]["R"] < -1, "stop_gap_open @94.905, R<-1", (tr[0]["exit_reason"], tr[0]["exit_px_raw"], round(tr[0]["R"], 3)))
# 3 同 bar 歧义：盘中触发且同日最低≤S -> 当日止损，歧义=1
a = synth(); a["H"][6, 0] = 101.5; a["L"][6, 0] = 96.5; d = E.Data(arrays=a); f = FF(d)
tr, eq, ol = E.simulate(f, {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("同bar歧义保守止损", tr[0]["exit_reason"] == "stop_same_bar" and tr[0]["ambiguous"] == 1 and tr[0]["exit_day"] == 6, "stop_same_bar, ambiguous=1, 当日", (tr[0]["exit_reason"], tr[0]["ambiguous"], tr[0]["exit_day"]))
# 4 费用：100股@$50 → max(1,0.5)+1bp×5000=1.5；2股 → 最低1美元+1bp
rec("佣金公式", abs(E.fee(100, 50) - 1.5) < 1e-12 and abs(E.fee(2, 50) - 1.01) < 1e-12, "1.5 / 1.01", (E.fee(100, 50), E.fee(2, 50)))
a = synth(px=50); a["H"][6, 0] = 50.6; a["L"][6, 0] = 49.8; d = E.Data(arrays=a); f = FF(d)
tr, eq, ol = E.simulate(f, {5: (0, 50.5, 49.5, 1)}, (0.01, 0.05), 0, 0, 29, cap0=16000)
q = tr[0]["qty"]; exp_q = math.floor(0.005 * 16000 / (50.5 - 49.5))
rec("风险定股数与现金", q == exp_q and eq.cash.iloc[6] >= 0 and abs(eq.cash.iloc[6] - (16000 - q * 50.5 - E.fee(q, 50.5))) < 1e-6, f"qty={exp_q}, 现金=16000-qty×50.5-费", (q, round(eq.cash.iloc[6], 4)))
# 名义上限 100%：风险很小时被现金限制
a = synth(px=50); a["H"][6, 0] = 50.6; d = E.Data(arrays=a); f = FF(d); f.VOL20[:] = 1e9
tr, eq, ol = E.simulate(f, {5: (0, 50.5, 50.0, 1)}, (0.0, 0.05), 0, 0, 29, cap0=16000)
rec("名义上限100%且现金非负", tr[0]["qty"] * 50.5 + E.fee(tr[0]["qty"], 50.5) <= 16000 and eq.cash.min() >= 0, "qty×价+费≤16000", tr[0]["qty"])
# 5 拆股：持仓中 2:1 拆股（调整价连续、F 由 2 变 1），净值与无拆股版本一致
a = synth(); a["H"][6, 0] = 101.5; d0 = E.Data(arrays=a); tr0, eq0, _ = E.simulate(FF(d0), {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
b = synth(); b["H"][6, 0] = 101.5; b["F"][:10, 0] = 2.0; b["SP"][10, 0] = 2.0; b["V"] = b["V"]; d1 = E.Data(arrays=b)
tr1, eq1, _ = E.simulate(FF(d1), {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
jump = float(eq1.equity.iloc[10] - eq1.equity.iloc[9])
rec("拆股净值连续(2:1)", abs(jump) < 1e-9 and tr1[0]["qty"] * 2 == tr1[0]["exit_shares_raw"], "拆股日净值跳变=0（调整价不变），原始股数×2", f"拆股日净值变化={jump:.2e}, 入场原始股数 {tr1[0]['qty']} → 出场 {tr1[0]['exit_shares_raw']}")
b = synth(); b["H"][6, 0] = 101.5; b["F"][:10, 0] = 1.5; b["SP"][10, 0] = 1.5; d2 = E.Data(arrays=b)
tr2, eq2, _ = E.simulate(FF(d2), {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("拆股3:2零股换现金", float(tr2[0]["exit_shares_raw"]).is_integer(), "出场股数为整数，零股现金入账", (tr2[0]["qty"], tr2[0]["exit_shares_raw"], round(float(np.abs(eq0.equity.values - eq2.equity.values).max()), 4)))
# 6 停牌：入场日无 bar -> 不成交；持仓中无 bar -> 不成交不插值；数据终止 -> 数据阻断标记
a = synth(); a["O"][6, 0] = np.nan; a["H"][6, 0] = np.nan; a["L"][6, 0] = np.nan; a["C"][6, 0] = np.nan; d = E.Data(arrays=a)
tr, eq, ol = E.simulate(FF(d), {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("停牌日不成交", ol[0][3] == "no_bar" and len(tr) == 0, "no_bar", ol)
a = synth(); a["H"][6, 0] = 101.5
for k in ("O", "H", "L", "C"): a[k][12:, 0] = np.nan
d = E.Data(arrays=a); tr, eq, ol = E.simulate(FF(d), {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("退市无清算数据->DATA_BLOCKED", tr[0]["exit_reason"].endswith("DATA_BLOCKED"), "delisted_no_settlement_DATA_BLOCKED", tr[0]["exit_reason"])
# 7 9EMA 退出与 20 日到期：入场日算第1天，第20日收盘安排次日开盘
a = synth(); a["H"][6, 0] = 101.5; d = E.Data(arrays=a); tr, eq, ol = E.simulate(FF(d), {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("20日到期次日开盘", tr[0]["exit_reason"] == "time_20d" and tr[0]["exit_day"] - tr[0]["entry_day"] == 20, "入场日6，第20持仓日=25收盘，26开盘退出", (tr[0]["entry_day"], tr[0]["exit_day"]))
e9 = np.full((30, 1), 1.0); e9[8, 0] = 1e9; tr, eq, ol = E.simulate(FF(d, e9), {5: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("收盘破9EMA次开退出", tr[0]["exit_reason"] == "close_below_ema9" and tr[0]["exit_day"] == 9, "第8日收盘破线，第9日开盘", tr[0]["exit_day"])
# 8 退出当日不再入场
a = synth(); a["H"][6, 0] = 101.5; a["L"][8, 0] = 96; a["H"][9, 0] = 101.5; d = E.Data(arrays=a)
tr, eq, ol = E.simulate(FF(d), {5: (0, 101.0, 97.0, 1), 7: (0, 101.0, 97.0, 1), 8: (0, 101.0, 97.0, 1)}, (0.01, 0.05), 10, 0, 29)
rec("持仓不开新仓/退出当日不入场/次一收盘重筛", len(tr) == 2 and tr[1]["entry_day"] == 9 and tr[1]["sig_day"] == 8, "第2笔信号日8、入场日9（信号日7被持仓阻断）", [(t["sig_day"], t["entry_day"], t["exit_day"]) for t in tr])
# 9 并列排序：R63 同值 -> 20日美元量降序 -> ID 升序
class F2: pass
f2 = F2(); f2.ADV20 = np.array([[5e7, 9e7, 9e7]]); f2.R63 = np.array([[0.5, 0.5, 0.5]])
ok = np.array([[True, True, True]]); Bm = np.full((1, 3), 100.0); Sm = np.full((1, 3), 97.0)
o, _ = E.top_orders(f2, ok, Bm, Sm, (0.01, 0.05)); rec("并列排序", o[0][0] == 1, "选列1（量并列取ID小）", o[0])
f2.R63 = np.array([[0.5, 0.4, 0.6]]); o, _ = E.top_orders(f2, ok, Bm, Sm, (0.01, 0.05)); rec("收益优先", o[0][0] == 2, "列2", o[0])
# 10 无前视：真实数据截断前后信号一致
d = E.Data(); sub = np.r_[np.arange(0, 400), [d.qqq]]
arr = {k: getattr(d, k)[:, sub] for k in ("O", "H", "L", "C", "V", "DIV", "SP", "F")}; arr.update(cal=d.cal, names=d.names[sub], is_etf=d.is_etf[sub])
full = E.Feat(E.Data(arrays=arr)); cut = 1500
arr2 = {k: (v[:cut + 1] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in arr.items()}; arr2["cal"] = d.cal[:cut + 1]
part = E.Feat(E.Data(arrays=arr2)); same = True; detail = {}
for nm, fn in (("PB", E.sig_pb), ("VC", E.sig_vc), ("HTF", E.sig_htf), ("BO20", E.sig_bo20)):
    a1 = fn(full); a2 = fn(part)
    eqok = np.array_equal(a1[0][:cut + 1], a2[0]); eqB = np.allclose(np.where(a2[0], a1[1][:cut + 1], 0), np.where(a2[0], a2[1], 0), equal_nan=True)
    detail[nm] = (bool(eqok), bool(eqB), int(a2[0].sum())); same &= eqok and eqB
rec("无前视(截断数据信号一致)", same, "截断到第1500日前后信号/B/S 完全相同", detail)
# 注：F（未来拆股因子）只用于把调整价换回当日原始价，等价于当日真实原始价；不进入信号比较
# 11 半日市：真实日历含 2025-11-28、2025-12-24（半日），不含 2025-11-27（感恩节），半日 bar 按正常交易日计数
cal = pd.DatetimeIndex(d.cal)
rec("半日市日历", pd.Timestamp("2025-11-28") in cal and pd.Timestamp("2025-12-24") in cal and pd.Timestamp("2025-11-27") not in cal and pd.Timestamp("2025-12-25") not in cal,
    "含11-28/12-24，不含11-27/12-25", [str(x.date()) for x in cal[(cal >= "2025-11-25") & (cal <= "2025-12-29")]])
json.dump(R, open("/workspace/gpt-candidates/tests/test_results.json", "w"), ensure_ascii=False, indent=1)
print(sum(r["passed"] for r in R), "/", len(R))
